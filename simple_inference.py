"""
Simple inference script for testing model coherence.

This script performs a basic generation test with a simple prompt to verify
that the model produces coherent text. Uses greedy decoding (no sampling).
"""
import argparse
import torch
from transformers import AutoTokenizer
import legacy.modeling_gpt3dev
from legacy.modeling_gpt3dev import GPT3DevForCalsualLM


def simple_inference(model_path, prompt="He is a doctor. His main goal is", max_length=50):
    """
    Perform simple inference with greedy decoding.

    Args:
        model_path: Path or HuggingFace model ID
        prompt: Input prompt for generation
        max_length: Maximum length of generated text

    Returns:
        Generated text string
    """
    print(f"Loading model from: {model_path}")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    model = GPT3DevLMHeadModel.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    # Set pad token if not set
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # Move model to appropriate device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    print(f"Using device: {device}")
    print(f"Prompt: '{prompt}'")
    print(f"Max length: {max_length}")
    print()

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate with greedy decoding (no sampling)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            do_sample=False,  # Greedy decoding
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode and return
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return generated_text


def main():
    parser = argparse.ArgumentParser(description="Simple inference for model coherence testing")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model path or HuggingFace model ID"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="He is a doctor. His main goal is",
        help="Input prompt for generation"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=50,
        help="Maximum length of generated text"
    )

    args = parser.parse_args()

    # Run inference
    generated_text = simple_inference(
        model_path=args.model,
        prompt=args.prompt,
        max_length=args.max_length
    )

    print("="*80)
    print("GENERATED TEXT:")
    print("="*80)
    print(generated_text)
    print("="*80)
    print()
    print("✓ Inference completed successfully")


if __name__ == "__main__":
    main()
