import math
import torch
import torch.nn as nn
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2MLP

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM
)

from transformers.modeling_outputs import (
    CausalLMOutputWithCrossAttentions,
)
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import (
    GPT2LMHeadModel,
    GPT2Model,
    GPT2Block,
    GPT2Attention,
    GPT2MLP,
    CausalLMOutputWithCrossAttentions
)


class GPT3DevConfig(GPT2Config):
    model_type = "gpt3dev"

    def __init__(self, use_pre_layernorm=True, window_size=256, stride=128, **kwargs):
        super().__init__(**kwargs)
        self.use_pre_layernorm = use_pre_layernorm
        self.window_size = window_size
        self.stride = stride


class GPT3DevAttention(GPT2Attention):  # dense
    """GPT-3 style dense attention: nn.Linear instead of Conv1D."""
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__(config, is_cross_attention, layer_idx=layer_idx)
        # GPT-3 uses nn.Linear instead of Conv1D
        self.c_attn = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=True)
        self.c_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
    # forward() inherited from GPT2Attention — no override needed


class GPT3DevSparseAttention(GPT3DevAttention):  # local sparse
    """GPT-3 style locally banded sparse attention."""
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__(config, is_cross_attention, layer_idx=layer_idx)
        self.window_size = getattr(config, "window_size", 256)

    def forward(
        self,
        hidden_states,
        past_key_value=None,
        cache_position=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        **kwargs,
    ):
        bsz, tgt_len, _ = hidden_states.size()
        device = hidden_states.device
        dtype = hidden_states.dtype

        # Determine query/key positions using cache_position (new API)
        if cache_position is not None:
            q_pos = cache_position  # shape: (tgt_len,)
            seq_len = int(q_pos[-1].item()) + 1
        else:
            q_pos = torch.arange(tgt_len, device=device)
            seq_len = tgt_len
        k_pos = torch.arange(seq_len, device=device)

        diff = q_pos[:, None] - k_pos[None, :]  # (tgt_len, seq_len)
        is_causal = diff >= 0
        within_window = diff.abs() <= self.window_size
        allow_attention = is_causal & within_window
        del is_causal, within_window, diff

        sparse_mask = torch.zeros((1, 1, tgt_len, seq_len), dtype=dtype, device=device)
        sparse_mask.masked_fill_(~allow_attention, torch.finfo(dtype).min)
        del allow_attention

        # Combine with parent's causal mask
        if attention_mask is not None:
            # Parent may create mask with extra KV positions — trim to match
            if attention_mask.size(-1) != sparse_mask.size(-1):
                attention_mask = attention_mask[..., :sparse_mask.size(-1)]
            if attention_mask.size(-2) != sparse_mask.size(-2):
                attention_mask = attention_mask[..., :sparse_mask.size(-2), :]
            attention_mask = torch.minimum(attention_mask, sparse_mask)
        else:
            attention_mask = sparse_mask
        del sparse_mask

        return super().forward(
            hidden_states,
            past_key_value=past_key_value,
            cache_position=cache_position,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            **kwargs,
        )


class GPT3DevMLP(GPT2MLP):
    def __init__(self, intermediate_size, config):
        super().__init__(intermediate_size, config)
        self.c_fc = nn.Linear(config.hidden_size, intermediate_size, bias=True)
        self.c_proj = nn.Linear(intermediate_size, config.hidden_size, bias=True)
        self.act = nn.GELU()  # standard GeLU


class GPT3DevBlock(GPT2Block):
    """GPT-3 block with pre-LayerNorm and alternating dense/sparse attention."""
    def __init__(self, config, is_sparse: bool = False, layer_idx=None):
        super().__init__(config, layer_idx=layer_idx)
        self.use_pre_layernorm = config.use_pre_layernorm
        self.ln_1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.ln_2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        if is_sparse:
            self.attn = GPT3DevSparseAttention(config, layer_idx=layer_idx)
        else:
            self.attn = GPT3DevAttention(config, layer_idx=layer_idx)

        self.mlp = GPT3DevMLP(4 * config.hidden_size, config)

    def forward(
        self,
        hidden_states,
        past_key_value=None,
        cache_position=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=False,
        output_attentions=False,
        **kwargs,
    ):
        if self.use_pre_layernorm:
            # Pre-LayerNorm (GPT-3)
            residual = hidden_states
            hidden_states = self.ln_1(hidden_states)
            attn_output, attn_weights = self.attn(
                hidden_states,
                past_key_value=past_key_value,
                cache_position=cache_position,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
                **kwargs,
            )
            hidden_states = residual + attn_output

            residual = hidden_states
            hidden_states = self.ln_2(hidden_states)
            feed_forward_hidden_states = self.mlp(hidden_states)
            hidden_states = residual + feed_forward_hidden_states
        else:
            # Post-LayerNorm (GPT-2)
            residual = hidden_states
            attn_output, attn_weights = self.attn(
                hidden_states,
                past_key_value=past_key_value,
                cache_position=cache_position,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
                **kwargs,
            )
            hidden_states = residual + attn_output
            hidden_states = self.ln_1(hidden_states)

            residual = hidden_states
            feed_forward_hidden_states = self.mlp(hidden_states)
            hidden_states = residual + feed_forward_hidden_states
            hidden_states = self.ln_2(hidden_states)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs


class GPT3DevModel(GPT2Model):
    config_class = GPT3DevConfig

    def __init__(self, config):
        super().__init__(config)

        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        self.wpe = nn.Embedding(config.n_positions, config.hidden_size)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            self.h.append(GPT3DevBlock(config, is_sparse=(i % 2 == 1), layer_idx=i))

        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.post_init()
        # NOTE: _apply_residual_scaling is called from GPT3DevLMHeadModel.__init__
        # AFTER the final post_init(), so it is NOT undone by re-initialization.

    def _apply_residual_scaling(self):
        # GPT-3/GPT-2 modified init: scale residuals by 1 / sqrt(2 * num_layers)
        scale = 1 / math.sqrt(2 * self.config.num_hidden_layers)
        for block in self.h:
            block.attn.c_proj.weight.data.mul_(scale)
            block.mlp.c_proj.weight.data.mul_(scale)




class GPT3DevLMHeadModel(GPT2LMHeadModel):
    config_class = GPT3DevConfig

    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPT3DevModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()
        # GPT-3 modified init: scale residual projections by 1/sqrt(2*num_layers)
        # MUST be AFTER the final post_init() which re-initializes all weights
        self.transformer._apply_residual_scaling()

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
        output_logits=None,  # Force returning full logits even with labels (for debugging/distillation)
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=None,  # disabled for compatibility with newer transformers
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            # cache_position can be omitted; GPT2Model does not require it
        )

        hidden_states = transformer_outputs[0]

        # Set up for loss computation if labels are provided
        lm_logits = self.lm_head(hidden_states.contiguous())
        
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            return ((loss,) if loss is not None else ()) + (lm_logits,) + transformer_outputs[1:]

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )


AutoConfig.register("gpt3dev", GPT3DevConfig)
AutoModel.register(GPT3DevConfig, GPT3DevModel)
AutoModelForCausalLM.register(GPT3DevConfig, GPT3DevLMHeadModel)
