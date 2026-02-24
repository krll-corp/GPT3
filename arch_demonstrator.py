import torch
from transformers import GPT2TokenizerFast
from modeling_gpt3dev import GPT3DevConfig, GPT3DevLMHeadModel

torch.set_default_dtype(torch.bfloat16)

tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# d_head=n_embd/n_head and calculated automatically
"""config = GPT3DevConfig(
    vocab_size=tokenizer.vocab_size,
    n_positions=2048, # Max position embeddings
    n_ctx=2048,       # Max context length for generation
    n_embd=5140,       # Embedding dimension
    n_layer=40,       # Number of transformer blocks
    n_head=40,        # Number of attention heads
    n_inner=5140,     # Dimension of the feedforward layer
    d_head=128,
    activation_function='gelu',
    resid_pdrop=0.0,
    embd_pdrop=0.0,
    attn_pdrop=0.0,
    use_pre_layernorm=True,
)"""

"""config = GPT3DevConfig(
    vocab_size=tokenizer.vocab_size,
    n_positions=2048,     # Maximum position embeddings
    n_ctx=2048,           # Maximum context length for generation
    n_embd=12288,         # Embedding dimension set for 175B
    n_layer=96,           # Number of transformer blocks
    n_head=96,            # Number of attention heads
    n_inner=12288,        # Feedforward dimension (kept equal to n_embd as in your 13B config)
    d_head=128,           # Attention head size (to ensure 96 * 128 = 12288)
    activation_function='gelu',
    resid_pdrop=0.0,
    embd_pdrop=0.0,
    attn_pdrop=0.0,
    use_pre_layernorm=True,
)"""


"""config = GPT3DevConfig(
    vocab_size=tokenizer.vocab_size,
    n_positions=2048,  # Max position embeddings
    n_ctx=2048,        # Max context length for generation
    n_embd=5120,       # Embedding dimension
    n_layer=40,        # Number of transformer blocks
    n_head=40,         # Number of attention heads
    n_inner=5120,      # Dimension of the feedforward layer
    d_head=128,
    activation_function='gelu',
    resid_pdrop=0.0,
    embd_pdrop=0.0,
    attn_pdrop=0.0,
    use_pre_layernorm=True,
)"""

#125m
config = GPT3DevConfig(
    vocab_size=tokenizer.vocab_size,
    n_positions=2048,  # Max position embeddings
    n_ctx=2048,        # Max context length for generation
    n_embd=768,        # Embedding dimension
    n_layer=12,        # Number of transformer blocks
    n_head=12,         # Number of attention heads
    n_inner=768,       # Dimension of the feedforward layer
    d_head=64,
    activation_function='gelu',
    resid_pdrop=0.0,
    embd_pdrop=0.0,
    attn_pdrop=0.0,
    use_pre_layernorm=True,
)

model = GPT3DevLMHeadModel(config)

"""device = torch.device(
    'cuda' if torch.cuda.is_available() else
    'mps' if torch.backends.mps.is_available() else
    'cpu'
)"""
device=torch.device("cpu")

#model=torch.compile(model, backend="aot_eager", mode="max-autotune") # if you train with compile, you need to initiate and compile a model before loading the pretrained weights, otherwise you will get an error about missing keys in the state dict.
model.to(device)

print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")

final_save_path = f'gpt3_arch_demonstrator_{sum(p.numel() for p in model.parameters()) // 1000000000}B'
model.save_pretrained(final_save_path)
tokenizer.save_pretrained(final_save_path)
print(f"\nmodel saved to {final_save_path}")