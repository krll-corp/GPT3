# evaluate.py

import argparse
import gc
import math
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm import tqdm
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast

# ----------------------------
# Model Definitions
# ----------------------------

class CustomGPT2Config(GPT2Config):
    def __init__(self, use_pre_layernorm=True, **kwargs):
        super().__init__(**kwargs)
        self.use_pre_layernorm = use_pre_layernorm

from transformers.models.gpt2.modeling_gpt2 import (
    GPT2LMHeadModel,
    GPT2Model,
    GPT2Block,
    GPT2Attention,
    GPT2MLP,
    CausalLMOutputWithCrossAttentions,
)

class CustomGPT2Attention(GPT2Attention):
    def __init__(self, config, is_cross_attention=False):
        super().__init__(config, is_cross_attention)
        # Ensure biases are included
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=True)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=True)

class CustomGPT2MLP(GPT2MLP):
    def __init__(self, intermediate_size, config):
        super().__init__(intermediate_size, config)
        self.c_fc = nn.Linear(config.n_embd, intermediate_size, bias=True)
        self.c_proj = nn.Linear(intermediate_size, config.n_embd, bias=True)
        self.act = nn.GELU()  # Use standard GeLU

class CustomGPT2Block(GPT2Block):
    def __init__(self, config):
        super().__init__(config)
        self.use_pre_layernorm = config.use_pre_layernorm
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = CustomGPT2Attention(config)
        self.mlp = CustomGPT2MLP(4 * config.n_embd, config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
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

class CustomGPT2Model(GPT2Model):
    def __init__(self, config):
        super().__init__(config)
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([CustomGPT2Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        self.init_weights()

class CustomGPT2LMHeadModel(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = CustomGPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.init_weights()

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
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

# ----------------------------
# Model and Tokenizer Loading
# ----------------------------

def load_model_and_tokenizer(model_path, device):
    tokenizer = GPT2TokenizerFast.from_pretrained(model_path)
    model = CustomGPT2LMHeadModel.from_pretrained(model_path)

    print(f"Using device: {device}")
    model.to(device)
    model.eval()

    return model, tokenizer

# ----------------------------
# Evaluation Functions
# ----------------------------

def _resolve_max_examples(requested: Optional[int], total: int) -> int:
    """Return the effective number of examples that should be processed."""

    if requested is None:
        return total
    return max(0, min(total, requested))


def evaluate_on_hellaswag(
    model,
    tokenizer,
    device,
    *,
    temperature=None,
    top_k=None,
    top_p=None,
    min_p=None,
    batch_size=8,
    block_size=512,
    max_examples: Optional[int] = None,
):
    print("\nStarting HellaSwag Evaluation...\n")
    # Load HellaSwag validation dataset
    hellaswag_dataset = load_dataset("Rowan/hellaswag", split="validation")

    total = 0
    correct = 0
    num_examples = len(hellaswag_dataset)
    num_choices = 4  # HellaSwag has 4 choices per question
    hellaswag_loss_sum = 0.0
    hellaswag_token_count = 0

    limit = _resolve_max_examples(max_examples, num_examples)
    if limit == 0:
        print("No HellaSwag examples selected for evaluation.")
        return float("nan")

    for i in tqdm(range(0, limit, batch_size), desc="Evaluating HellaSwag"):
        batch = hellaswag_dataset[i:i+batch_size]

        # Extract data from the batch
        contexts = batch['ctx']  # Updated from 'context' to 'ctx'
        endings_list = batch['endings']
        labels = batch['label']
        labels = [int(l) for l in labels]  # Convert labels from strings to integers

        inputs = []
        for context, endings in zip(contexts, endings_list):
            # For each ending, concatenate it with the context
            for ending in endings:
                input_text = context + ' ' + ending
                inputs.append(input_text)

        # Tokenize the concatenated inputs
        tokenized_inputs = tokenizer(
            inputs,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=block_size
        )

        input_ids = tokenized_inputs['input_ids'].to(device)
        attention_mask = tokenized_inputs['attention_mask'].to(device)

        # Compute loss for each input
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )
            logits = outputs.logits
            if temperature is not None:
                logits = logits / temperature

            # Shift logits and labels to align them
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()

            # Compute loss per token and then sum to get loss per example
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            loss_per_token = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            loss_per_token = loss_per_token.view(shift_labels.size())
            loss_per_example = loss_per_token.sum(dim=1)
            hellaswag_loss_sum += loss_per_token.sum().item()
            hellaswag_token_count += shift_labels.numel()

            # Reshape losses to [batch_size, num_choices]
            actual_batch_size = len(batch['ctx'])
            loss_per_example = loss_per_example.view(actual_batch_size, num_choices)

            # Choose the option with the lowest loss (highest likelihood)
            predicted_labels = torch.argmin(loss_per_example, dim=1)
            labels_tensor = torch.tensor(labels, device=device)

            total += actual_batch_size
            correct += (predicted_labels == labels_tensor).sum().item()

    accuracy = correct / total if total else float("nan")
    hellaswag_perplexity = math.exp(hellaswag_loss_sum / hellaswag_token_count) if hellaswag_token_count > 0 else float('inf')
    print(f"\nHellaSwag Accuracy: {accuracy:.4f}")
    print(f"HellaSwag Perplexity: {hellaswag_perplexity:.4f}")
    return accuracy

def evaluate_on_mmlu(
    model,
    tokenizer,
    device,
    *,
    temperature=None,
    top_k=None,
    top_p=None,
    min_p=None,
    batch_size=8,
    block_size=512,
    task='abstract_algebra',
    max_examples: Optional[int] = None,
):
    print(f"\nStarting MMLU Evaluation on task: {task}\n")
    # Load MMLU dataset with the correct subtask name
    try:
        mmlu_dataset = load_dataset(
            "cais/mmlu",
            task,  # Use the task parameter
            split="test",
            trust_remote_code=True
        )
        print(f"Loaded dataset for task: {task} with {len(mmlu_dataset)} examples.")
    except Exception as e:
        print(f"Error loading MMLU task '{task}': {e}")
        return None

    total = 0
    correct = 0
    num_examples = len(mmlu_dataset)
    num_choices = 4  # MMLU has 4 choices per question
    task_loss_sum = 0.0
    task_token_count = 0

    limit = _resolve_max_examples(max_examples, num_examples)
    if limit == 0:
        print(f"No MMLU examples selected for task '{task}'.")
        return float("nan"), float("inf")

    for i in tqdm(range(0, limit, batch_size), desc=f"Evaluating MMLU ({task})"):
        batch = mmlu_dataset[i:i + batch_size]

        # Extract data from the batch
        questions = batch['question']
        choices_list = batch['choices']
        answers = batch['answer']  # This should be integer indices (0,1,2,3)

        inputs = []
        for question, choices in zip(questions, choices_list):
            for choice in choices:
                input_text = f"{question} {choice}"
                inputs.append(input_text)

        # Tokenize the concatenated inputs
        tokenized_inputs = tokenizer(
            inputs,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=block_size
        )

        input_ids = tokenized_inputs['input_ids'].to(device)
        attention_mask = tokenized_inputs['attention_mask'].to(device)

        # Compute loss for each input
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )
            logits = outputs.logits
            if temperature is not None:
                logits = logits / temperature

            # Shift logits and labels to align them
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()

            # Compute loss per token and then sum to get loss per example
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            loss_per_token = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            loss_per_token = loss_per_token.view(shift_labels.size())
            loss_per_example = loss_per_token.sum(dim=1)
            task_loss_sum += loss_per_token.sum().item()
            task_token_count += shift_labels.numel()

            # Reshape losses to [batch_size, num_choices]
            actual_batch_size = len(batch['question'])
            loss_per_example = loss_per_example.view(actual_batch_size, num_choices)

            # Choose the option with the lowest loss (highest likelihood)
            predicted_labels = torch.argmin(loss_per_example, dim=1)
            labels_tensor = torch.tensor(answers, device=device)

            total += actual_batch_size
            correct += (predicted_labels == labels_tensor).sum().item()

    accuracy = correct / total if total else float("nan")
    task_perplexity = math.exp(task_loss_sum / task_token_count) if task_token_count > 0 else float('inf')
    print(f"\nMMLU Accuracy on '{task}': {accuracy:.4f}")
    print(f"MMLU Perplexity on '{task}': {task_perplexity:.4f}")
    return accuracy, task_perplexity

def evaluate_on_lambada(
    model,
    tokenizer,
    device,
    *,
    temperature=None,
    top_k=None,
    top_p=None,
    min_p=None,
    batch_size=8,
    block_size=512,
    max_examples: Optional[int] = None,
):
    print("\nStarting LAMBADA Evaluation...\n")
    # Load the LAMBADA test split
    lambada_dataset = load_dataset("lambada", split="test")
    
    total = 0
    correct = 0
    num_examples = len(lambada_dataset)
    lambada_loss_sum = 0.0
    lambada_token_count = 0
    final_token_correct = 0
    final_token_total = 0
    
    limit = _resolve_max_examples(max_examples, num_examples)
    if limit == 0:
        print("No LAMBADA examples selected for evaluation.")
        return float("nan")

    for i in tqdm(range(0, limit, batch_size), desc="Evaluating LAMBADA"):
        batch = lambada_dataset[i:i+batch_size]
    
        sentences = batch['text']
        # Initialize lists for contexts and target words
        contexts = []
        targets = []
    
        for sentence in sentences:
            # Split the sentence into words
            tokens = sentence.strip().split()
            if len(tokens) < 2:
                # If the sentence is too short, skip this example
                continue
            # The target is the last word
            target = tokens[-1]
            # The context is the sentence without the last word
            context = ' '.join(tokens[:-1])
            contexts.append(context)
            targets.append(target)
    
        # Tokenize the contexts
        tokenized_inputs = tokenizer(
            contexts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=block_size
        )
    
        input_ids = tokenized_inputs['input_ids'].to(device)
        attention_mask = tokenized_inputs['attention_mask'].to(device)
    
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )
            logits = outputs.logits
            if temperature is not None:
                logits = logits / temperature

            if top_k is not None or top_p is not None or min_p is not None:
                probs = torch.nn.functional.softmax(logits, dim=-1)
                if top_k:
                    topk_vals, _ = torch.topk(probs, top_k, dim=-1)
                    thresh = topk_vals[..., -1, None]
                    probs = torch.where(probs < thresh, torch.zeros_like(probs), probs)

                if top_p:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
                    cutoff = (cumsum_probs > top_p).float().cumsum(dim=-1)
                    sorted_probs = torch.where(cutoff > 0, torch.zeros_like(sorted_probs), sorted_probs)
                    probs = torch.zeros_like(probs).scatter_(-1, sorted_indices, sorted_probs)

                if min_p:
                    probs = torch.where(probs < min_p, torch.zeros_like(probs), probs)

                probs = probs / probs.sum(dim=-1, keepdim=True)
                predicted_ids = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(probs.size(0), -1)
            else:
                predicted_ids = torch.argmax(logits, dim=-1)

            correct += (predicted_ids == input_ids).sum().item()
            total += torch.numel(input_ids)

            # Shift logits and labels for loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            loss_per_token = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            lambada_loss_sum += loss_per_token.sum().item()
            lambada_token_count += shift_labels.numel()

            # Final-token accuracy fix
            final_token_ids = predicted_ids[:, -1]
            gold_token_ids = input_ids[:, -1]
            final_token_correct += (final_token_ids == gold_token_ids).sum().item()
            final_token_total += gold_token_ids.size(0)
    
    lambada_accuracy = correct / total if total > 0 else float("nan")
    lambada_perplexity = math.exp(lambada_loss_sum / lambada_token_count) if lambada_token_count > 0 else float('inf')
    final_token_accuracy = final_token_correct / final_token_total if final_token_total > 0 else 0

    print(f"\nLAMBADA Token-by-Token Accuracy: {lambada_accuracy:.4f}")
    print(f"LAMBADA Perplexity: {lambada_perplexity:.4f}")
    print(f"Final Token Accuracy: {final_token_accuracy:.4f}")
    return lambada_accuracy



# ----------------------------
# Main Execution
# ----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate GPT-3 style checkpoints on public benchmarks.")
    parser.add_argument("--model", required=True, help="Path or Hugging Face repo id of the model to evaluate.")
    parser.add_argument("--device", default=None, help="Device identifier, defaults to CUDA if available otherwise CPU.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for evaluation batches.")
    parser.add_argument("--block-size", type=int, default=256, help="Maximum sequence length in tokens.")
    parser.add_argument("--temperature", type=float, default=None, help="Optional temperature applied to logits.")
    parser.add_argument("--top-k", type=int, default=None, help="Optional top-k sampling filter.")
    parser.add_argument("--top-p", type=float, default=None, help="Optional top-p sampling filter.")
    parser.add_argument("--min-p", type=float, default=None, help="Optional minimum probability threshold.")
    parser.add_argument("--max-examples", type=int, default=64, help="Maximum number of evaluation examples per dataset.")
    parser.add_argument(
        "--mmlu-tasks",
        nargs="*",
        default=["abstract_algebra"],
        help="List of MMLU tasks to evaluate. Defaults to a small representative subset.",
    )
    parser.add_argument("--run-hellaswag", action="store_true", help="Evaluate on the HellaSwag benchmark.")
    parser.add_argument("--run-lambada", action="store_true", help="Evaluate on the LAMBADA benchmark.")
    parser.add_argument("--run-mmlu", action="store_true", help="Evaluate on the specified MMLU tasks.")
    return parser.parse_args()


def main():
    args = parse_args()

    device = (
        torch.device(args.device)
        if args.device
        else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )

    model, tokenizer = load_model_and_tokenizer(args.model, device)

    results = {}

    if args.run_lambada:
        lambada_accuracy = evaluate_on_lambada(
            model=model,
            tokenizer=tokenizer,
            device=device,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            min_p=args.min_p,
            batch_size=args.batch_size,
            block_size=args.block_size,
            max_examples=args.max_examples,
        )
        results["lambada_accuracy"] = lambada_accuracy
        print(f"LAMBADA Accuracy: {lambada_accuracy:.4f}")
        gc.collect()

    if args.run_hellaswag:
        hellaswag_accuracy = evaluate_on_hellaswag(
            model=model,
            tokenizer=tokenizer,
            device=device,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            min_p=args.min_p,
            batch_size=args.batch_size,
            block_size=args.block_size,
            max_examples=args.max_examples,
        )
        results["hellaswag_accuracy"] = hellaswag_accuracy
        gc.collect()

    if args.run_mmlu:
        mmlu_scores: List[float] = []
        mmlu_perplexities: List[float] = []
        for task in args.mmlu_tasks:
            mmlu_accuracy, mmlu_perplexity = evaluate_on_mmlu(
                model=model,
                tokenizer=tokenizer,
                device=device,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                min_p=args.min_p,
                batch_size=args.batch_size,
                block_size=args.block_size,
                task=task,
                max_examples=args.max_examples,
            )
            results[f"mmlu_accuracy_{task}"] = mmlu_accuracy
            results[f"mmlu_perplexity_{task}"] = mmlu_perplexity
            mmlu_scores.append(mmlu_accuracy)
            mmlu_perplexities.append(mmlu_perplexity)
            gc.collect()

        if mmlu_scores:
            results["mmlu_average_accuracy"] = float(np.nanmean(mmlu_scores))
            results["mmlu_average_perplexity"] = float(np.nanmean(mmlu_perplexities))
            print(f"MMLU Average Accuracy: {results['mmlu_average_accuracy']:.4f}")
            print(f"MMLU Average Perplexity: {results['mmlu_average_perplexity']:.4f}")

    if not results:
        print("No evaluation tasks were executed. Use --run-hellaswag, --run-lambada or --run-mmlu.")
        return

    print("\n===== Evaluation Summary =====")
    for key, value in results.items():
        print(f"{key}: {value}")


if __name__ == '__main__':
    main()
