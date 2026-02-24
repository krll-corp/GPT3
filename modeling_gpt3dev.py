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
    def __init__(self, config, is_cross_attention=False):
        super().__init__(config, is_cross_attention)
        # GPT-3 uses biases
        self.c_attn = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=True)
        self.c_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=None,
        output_attentions=False,
        **kwargs,
    ):
        #normalize attention_mask shape to 4D if present
        if attention_mask is not None:
            if attention_mask.dim() == 1:
                attention_mask = attention_mask.view(1, 1, 1, -1)
            elif attention_mask.dim() == 2:
                attention_mask = attention_mask[:, None, None, :]
            elif attention_mask.dim() == 3:
                attention_mask = attention_mask[:, None, :, :]

        # ensure contiguous last dimension for linear with bias
        hidden_states = hidden_states.contiguous()
        # head_mask=None to avoid shape incompatibilities with newer transformers
        # (head causes issues with sparse attention)
        return super().forward(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=None,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs,
        )


class GPT3DevSparseAttention(GPT3DevAttention):  # local sparse
    def __init__(self, config, is_cross_attention=False):
        super().__init__(config, is_cross_attention)
        self.window_size = getattr(config, "window_size", 256)
        self.stride = getattr(config, "stride", 128)

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=None,
        output_attentions=False,
    ):
        hidden_states = hidden_states.contiguous()
        # here we no longer touch the structure of layer_past — only the fact that it exists or not
        bsz, tgt_len, _ = hidden_states.size()
        device = hidden_states.device
        dtype = hidden_states.dtype

        if layer_past is None:
            # no cache: regular train / eval without generate
            seq_len = tgt_len
            q_pos = torch.arange(0, tgt_len, device=device)
            k_pos = torch.arange(seq_len, device=device)
        else:
            # cache: infer past length from cached keys
            past_len = 0
            try:
                k = layer_past[0]
                if torch.is_tensor(k):
                    past_len = k.size(-2)
            except Exception:
                past_len = 0
            seq_len = past_len + tgt_len
            q_pos = torch.arange(past_len, seq_len, device=device)
            k_pos = torch.arange(seq_len, device=device)

        #dist[i,j] = pos_q_i - pos_k_j #positive means q is after k
        diff = q_pos[:, None] - k_pos[None, :]  # (tgt_len, seq_len),signed
        
        # CRITICAL: Enforce causal mask first - only attend to past or present
        is_causal = diff >= 0  # q_pos >= k_pos
        
        # local window
        within_window = diff.abs() <= self.window_size
        
        # strided: attend to tokens at stride intervals, BUT only in the past
        # only check stride for past tokens (where diff > 0)
        # for diff > 0: allow if diff is divisible by stride
        is_strided = (diff > 0) & (diff % self.stride == 0)
        

        allow_attention = is_causal & (within_window | is_strided)
        del is_causal, within_window, is_strided, diff
        
        # convert to additive mask format that transformers expects
        # True in allow_attention means we CAN attend, so we need to invert for masking
        sparse_mask = torch.zeros((1, 1, tgt_len, seq_len), dtype=dtype, device=device)
        sparse_mask.masked_fill_(~allow_attention, torch.finfo(dtype).min)
        del allow_attention
        
        # combine with existing attention mask if present
        if attention_mask is not None:
            # both masks are additive (0 allow, -inf block); take the intersection
            attention_mask = torch.minimum(attention_mask, sparse_mask)
        else:
            attention_mask = sparse_mask
        del sparse_mask

        return super().forward(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )


class GPT3DevMLP(GPT2MLP):
    def __init__(self, intermediate_size, config):
        super().__init__(intermediate_size, config)
        self.c_fc = nn.Linear(config.hidden_size, intermediate_size, bias=True)
        self.c_proj = nn.Linear(intermediate_size, config.hidden_size, bias=True)
        self.act = nn.GELU()  # standard GeLU


class GPT3DevBlock(GPT2Block):
    def __init__(self, config, is_sparse: bool = False):
        super().__init__(config)
        self.use_pre_layernorm = config.use_pre_layernorm
        self.ln_1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        
        if is_sparse:
            self.attn = GPT3DevSparseAttention(config)  # local sparse
        else:
            self.attn = GPT3DevAttention(config)  # dense
        
        self.mlp = GPT3DevMLP(4 * config.hidden_size, config)
        self.ln_2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        
        #residual projection scaling is applied after post_init in GPT3DevModel

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=None,
        output_attentions=False,
        cache_position=None,
        **kwargs,
    ):
        # transformers >= 4.5x: cache is given through past_key_value
        past_key_value = kwargs.pop("past_key_value", None)
        if past_key_value is not None:
            # if layer_past and past_key_value are both provided — this is a bug above
            if layer_past is not None:
                raise ValueError("Both layer_past and past_key_value were provided")
            layer_past = past_key_value

        # transformers GPT2Model passes head mask as layer_head_mask
        layer_head_mask = kwargs.pop("layer_head_mask", None)
        if layer_head_mask is not None:
            if head_mask is not None:
                raise ValueError("Both head_mask and layer_head_mask were provided")
            head_mask = layer_head_mask


        if attention_mask is not None: #we have to do this or transformers will complain
            if attention_mask.dim() == 1:
                attention_mask = attention_mask.view(1, 1, 1, -1)
            elif attention_mask.dim() == 2:
                attention_mask = attention_mask[:, None, None, :]
            elif attention_mask.dim() == 3:
                attention_mask = attention_mask[:, None, :, :]

        if attention_mask is not None and attention_mask.dtype not in (torch.bool, hidden_states.dtype):
            # SDPA accepts either a bool mask or a float of the same dtype as query/key/value
            attention_mask = attention_mask.to(dtype=hidden_states.dtype)


        if self.use_pre_layernorm:
            # Pre-LayerNorm, GPT-3
            residual = hidden_states
            hidden_states = self.ln_1(hidden_states)
            hidden_states = hidden_states.contiguous()
            attn_outputs = self.attn(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=None,  #disabled for compatibility
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            attn_output = attn_outputs[0]
            outputs = attn_outputs[1:]  #present,(attentions)

            hidden_states = residual + attn_output

            residual = hidden_states
            hidden_states = self.ln_2(hidden_states)
            hidden_states = hidden_states.contiguous()
            feed_forward_hidden_states = self.mlp(hidden_states)
            hidden_states = residual + feed_forward_hidden_states
        else:
            # Post-LayerNorm, GPT-2
            residual = hidden_states
            hidden_states = hidden_states.contiguous()
            attn_outputs = self.attn(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=None,  #disabled for compatibility
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            attn_output = attn_outputs[0]
            outputs = attn_outputs[1:]  #present,(attentions)

            hidden_states = residual + attn_output
            hidden_states = self.ln_1(hidden_states)

            residual = hidden_states
            residual = residual.contiguous()
            feed_forward_hidden_states = self.mlp(hidden_states)
            hidden_states = residual + feed_forward_hidden_states
            hidden_states = self.ln_2(hidden_states)

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions)


class GPT3DevModel(GPT2Model):
    config_class = GPT3DevConfig

    def __init__(self, config):
        super().__init__(config)

        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        self.wpe = nn.Embedding(config.n_positions, config.hidden_size)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            block = GPT3DevBlock(config, is_sparse=(i % 2 == 1))

            if hasattr(block.attn, "layer_idx"):
                block.attn.layer_idx = i

            self.h.append(block)

        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.post_init()
        self._apply_residual_scaling()

    def _apply_residual_scaling(self):
        # GPT-3/GPT-2 modified init
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
            head_mask=None,  #compatibility with newer transformers
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            # cache_position can be omitted; GPT2Model does not require it
        )

        hidden_states = transformer_outputs[0]

        # set up for loss computation if labels are provided
        lm_logits = self.lm_head(hidden_states.contiguous())
        
        loss = None
        if labels is not None:
            #shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            #flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            #model parallelism. why not?
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
