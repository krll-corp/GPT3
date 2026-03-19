"""
GPT-3 Dev Model Implementation - Version-Agnostic

This module implements a custom GPT-3 style model (with pre-LayerNorm) that works
across multiple versions of the transformers library without version-specific code.

Key features for version compatibility:
- Runtime detection of attention mask formats (2D, 3D, 4D)
- Helper functions for consistent mask preparation
- Flexible parameter handling for API changes
- No hard dependencies on specific transformers versions

Tested with transformers >= 4.30.0
"""
import math
import torch
import torch.nn as nn
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2MLP

from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    PreTrainedModel,
)
from transformers.utils import logging

logger = logging.get_logger(__name__)


def _prepare_4d_attention_mask(mask, dtype, tgt_len=None):
    """
    Version-agnostic attention mask preparation.

    Creates a 4D attention mask from a 2D or 3D mask, compatible with all transformers versions.
    This helper handles the mask formatting that changed across transformers versions.

    Args:
        mask: Input mask tensor of shape [batch, seq_len] or [batch, 1, seq_len]
        dtype: Target dtype for the mask
        tgt_len: Target sequence length (for cross-attention, otherwise None)

    Returns:
        4D mask of shape [batch, 1, tgt_len, src_len] with 0 for valid positions and large negative for masked
    """
    batch_size, src_len = mask.shape[0], mask.shape[-1]
    tgt_len = tgt_len if tgt_len is not None else src_len

    # Expand mask to 4D: [batch, 1, tgt_len, src_len]
    if mask.dim() == 2:
        # [batch, src_len] -> [batch, 1, 1, src_len]
        expanded_mask = mask[:, None, None, :]
    elif mask.dim() == 3:
        # [batch, 1, src_len] -> [batch, 1, 1, src_len]
        expanded_mask = mask[:, :, None, :] if mask.shape[1] == 1 else mask[:, None, :, :]
    else:
        # Already 4D or higher, return as-is
        return mask

    # Broadcast to [batch, 1, tgt_len, src_len] if needed
    if tgt_len > 1 and expanded_mask.shape[2] == 1:
        expanded_mask = expanded_mask.expand(batch_size, 1, tgt_len, src_len)

    # Convert to additive mask: 1 (valid) -> 0, 0 (masked) -> large negative
    expanded_mask = expanded_mask.to(dtype=dtype)
    inverted_mask = 1.0 - expanded_mask
    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


def _make_causal_mask(seq_len, dtype, device):
    """
    Version-agnostic causal mask creation.

    Creates a causal attention mask that prevents attending to future positions.
    Works consistently across all transformers versions.

    Args:
        seq_len: Sequence length
        dtype: Data type for the mask
        device: Device to create the mask on

    Returns:
        Causal mask of shape [1, 1, seq_len, seq_len]
    """
    # Create upper triangular matrix (1s above diagonal, 0s on and below)
    mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)
    # Convert to additive mask: 0 for valid positions, large negative for masked positions
    mask = mask.to(dtype=dtype)
    mask = mask.masked_fill(mask.to(torch.bool), torch.finfo(dtype).min)
    return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]


class GPT3DevConfig(GPT2Config):
    model_type = "gpt3dev"
    
    def __init__(self, use_pre_layernorm=True, **kwargs):
        super().__init__(**kwargs)
        self.use_pre_layernorm = use_pre_layernorm
        self.d_head = kwargs.get("d_head", self.n_embd // self.n_head)


# Custom GPT-2 with Pre-LayerNorm and Biases aka GPT-3
from transformers.models.gpt2.modeling_gpt2 import (
    GPT2MLP,
)


class GPT3DevAttention(nn.Module):
    def __init__(self, config, is_cross_attention=False):
        super().__init__()
        self.is_cross_attention = is_cross_attention
        self.n_head = config.n_head
        self.d_head = config.d_head
        self.hidden_size = config.n_embd
        self.scale = 1 / math.sqrt(self.d_head)

        self.c_attn = nn.Linear(self.hidden_size, 3 * self.n_head * self.d_head, bias=True)
        self.c_proj = nn.Linear(self.n_head * self.d_head, self.hidden_size, bias=True)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.n_head, self.d_head)
        return x.transpose(1, 2)

    def merge_heads(self, x, batch_size):
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, -1, self.n_head * self.d_head)

    def forward(self, hidden_states, layer_past=None, attention_mask=None, head_mask=None,
                encoder_hidden_states=None, encoder_attention_mask=None,
                use_cache=False, output_attentions=False):
        batch_size, seq_len = hidden_states.size(0), hidden_states.size(1)
        qkv = self.c_attn(hidden_states)
        q, k, v = qkv.split(self.n_head * self.d_head, dim=-1)
        query = self.split_heads(q, batch_size)
        key = self.split_heads(k, batch_size)
        value = self.split_heads(v, batch_size)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)
        present = (key, value) if use_cache else None

        attn_scores = torch.matmul(query, key.transpose(-1, -2)) * self.scale

        # Apply attention mask if provided (version-agnostic handling)
        if attention_mask is not None:
            # Handle different mask formats from different transformers versions
            if attention_mask.dim() == 2:
                # [batch, key_len] format - expand to 4D
                attention_mask = attention_mask[:, None, None, :]
            elif attention_mask.dim() == 3:
                # [batch, 1, key_len] or [batch, query_len, key_len] format
                if attention_mask.shape[1] == 1:
                    attention_mask = attention_mask[:, :, None, :]
                else:
                    attention_mask = attention_mask[:, None, :, :]
            # attention_mask is now [batch, 1, query_len, key_len] or broadcastable to it
            attn_scores = attn_scores + attention_mask

        attn_probs = nn.functional.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)
        if head_mask is not None:
            attn_probs = attn_probs * head_mask

        attn_output = torch.matmul(attn_probs, value)
        attn_output = self.merge_heads(attn_output, batch_size)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        if use_cache:
            return attn_output, present, (attn_probs if output_attentions else None)
        else:
            return (attn_output, attn_probs) if output_attentions else (attn_output,)


class GPT3DevMLP(GPT2MLP):
    def __init__(self, intermediate_size, config):
        super().__init__(intermediate_size, config)
        self.c_fc = nn.Linear(config.n_embd, intermediate_size, bias=True)
        self.c_proj = nn.Linear(intermediate_size, config.n_embd, bias=True)
        self.act = nn.GELU()  # Use standard GeLU


class GPT3DevBlock(nn.Module):
    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.use_pre_layernorm = config.use_pre_layernorm
        self.layer_idx = layer_idx
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = GPT3DevAttention(config)
        self.mlp = GPT3DevMLP(4 * config.n_embd, config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

    def forward(self, hidden_states, *args, **kwargs):
        """Align the block forward signature with upstream GPT-2.

        Hugging Face frequently renames GPT-2 block parameters and adds new
        optional values (``cache_position`` etc.).  Accepting both positional
        and keyword arguments, normalising their names, and discarding any
        extras keeps the custom GPT-3 dev implementation compatible across
        releases.
        """

        param_order = [
            "past_key_values",
            "cache_position",
            "attention_mask",
            "head_mask",
            "encoder_hidden_states",
            "encoder_attention_mask",
            "use_cache",
            "output_attentions",
        ]

        params = {}
        for name, value in zip(param_order, args):
            if name not in params:
                params[name] = value
        params.update(kwargs)

        if "layer_past" in params and "past_key_values" not in params:
            params["past_key_values"] = params.pop("layer_past")
        if "past_key_value" in params and "past_key_values" not in params:
            params["past_key_values"] = params.pop("past_key_value")

        past_key_values = params.pop("past_key_values", None)
        cache_position = params.pop("cache_position", None)
        attention_mask = params.pop("attention_mask", None)
        head_mask = params.pop("head_mask", None)
        encoder_hidden_states = params.pop("encoder_hidden_states", None)
        encoder_attention_mask = params.pop("encoder_attention_mask", None)
        use_cache = params.pop("use_cache", False)
        output_attentions = params.pop("output_attentions", False)

        layer_past = None
        if past_key_values is not None:
            if hasattr(past_key_values, "to_legacy_cache"):
                legacy_cache = past_key_values.to_legacy_cache()
                try:
                    layer_past = legacy_cache[self.layer_idx]
                except (TypeError, IndexError):
                    layer_past = None
            else:
                layer_past = past_key_values

        _ = cache_position

        if self.use_pre_layernorm:
            # Pre-LayerNorm
            residual = hidden_states
            hidden_states = self.ln_1(hidden_states)
            attn_outputs = self.attn(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            attn_output = attn_outputs[0]
            outputs = attn_outputs[1:]  # present, (attentions)

            hidden_states = residual + attn_output

            residual = hidden_states
            hidden_states = self.ln_2(hidden_states)
            feed_forward_hidden_states = self.mlp(hidden_states)
            hidden_states = residual + feed_forward_hidden_states
        else:
            # Original GPT-2 Post-LayerNorm
            residual = hidden_states
            attn_outputs = self.attn(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            attn_output = attn_outputs[0]
            outputs = attn_outputs[1:]  # present, (attentions)

            hidden_states = residual + attn_output
            hidden_states = self.ln_1(hidden_states)

            residual = hidden_states
            feed_forward_hidden_states = self.mlp(hidden_states)
            hidden_states = residual + feed_forward_hidden_states
            hidden_states = self.ln_2(hidden_states)

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions)


class GPT3DevModel(nn.Module):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList(
            [GPT3DevBlock(config, layer_idx=i) for i in range(config.n_layer)]
        )
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

    def forward(self, input_ids=None, past_key_values=None, attention_mask=None,
                token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, use_cache=False, output_attentions=False,
                output_hidden_states=False, return_dict=None):
        batch_size, seq_len = input_ids.size(0), input_ids.size(1)
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.drop(hidden_states)

        # Create causal attention mask (version-agnostic)
        # This prevents attending to future positions
        causal_mask = _make_causal_mask(seq_len, hidden_states.dtype, input_ids.device)

        # Combine with padding mask if provided
        if attention_mask is not None:
            # attention_mask is typically [batch, seq_len] with 1s for valid positions, 0s for padding
            # Convert to 4D additive mask format
            padding_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype, tgt_len=seq_len)
            # Combine causal and padding masks
            combined_mask = causal_mask + padding_mask
        else:
            combined_mask = causal_mask

        for block in self.h:
            hidden_states = block(
                hidden_states,
                attention_mask=combined_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )[0]

        hidden_states = self.ln_f(hidden_states)
        return (hidden_states,)


class GPT3DevLMHeadModel(PreTrainedModel):
    config_class = GPT3DevConfig

    def __init__(self, config):
        super().__init__(config)  # calls PreTrainedModel.__init__
        self.transformer = GPT3DevModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.post_init()  # Initializes weights and registers modules for pretrain saving

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
    ):
        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
        )
        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        return (lm_logits,) if loss is None else (loss, lm_logits)


AutoConfig.register("gpt3dev", GPT3DevConfig)
AutoModel.register(GPT3DevConfig, GPT3DevModel)
AutoModelForCausalLM.register(GPT3DevConfig, GPT3DevLMHeadModel)
