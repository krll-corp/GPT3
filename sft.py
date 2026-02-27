# Datasets:
#   - HuggingFaceH4/no_robots (~10k)
#   - HuggingFaceH4/ultrachat_200k (sample)
#   - teknium/OpenHermes-2.5 (sample)
#   - Open-Orca/SlimOrca (sample)
#   - mlabonne/guanaco-llama2-1k
#   - OpenAssistant/oasst2 (filtered)
#   - Anthropic/hh-rlhf (sample)
#   - lmsys/lmsys-chat-1m (sample, filtered)

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid fork warnings

import gc
import time as time_module
import random
import torch
from transformers import (
    GPT2TokenizerFast,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    AutoModelForCausalLM,
)
from transformers.trainer_callback import EarlyStoppingCallback
from datasets import load_dataset, concatenate_datasets, Dataset
from torch.optim import AdamW
import wandb
from tqdm import tqdm

DATASET_CONFIG = {
    # minimal/no CoT reasoning
    "no_robots": {"enabled": True, "max_samples": None},  # ~10k - CLEANEST, human-written, direct answers
    "alpaca_cleaned": {"enabled": True, "max_samples": None},  # ~52k - Cleaned version, mostly direct
    "dolly": {"enabled": True, "max_samples": None},  # ~15k - Databricks human-written, clean
    "lima": {"enabled": True, "max_samples": None},  # ~1k - very high quality, curated
    "gpt_teacher": {"enabled": True, "max_samples": None},  # ~55k - GPT4 cleaned instruction pairs
    "sharegpt_vicuna": {"enabled": True, "max_samples": 30000},  # ~94k - Cleaned ShareGPT conversations
    "oig_h2o": {"enabled": True, "max_samples": 30000},  # ~195k - H2O cleaned OIG (human/bot tags)
    "oig_chip2": {"enabled": True, "max_samples": 30000},  # ~210k - OIG-small-chip2 (user/chip2 columns)
    
    # Heavy CoT/Reasoning. spare for future
    "ultrachat": {"enabled": False, "max_samples": 0},  # Contains reasoning patterns
    "openhermes": {"enabled": False, "max_samples": 0},  # HEAVY CoT - "Here's the logic", "Step 1:", etc
    "slimorca": {"enabled": False, "max_samples": 0},   # HEAVY CoT - based on FLAN reasoning
    "guanaco": {"enabled": False, "max_samples": 0},    # Contains reasoning patterns
    "oasst2": {"enabled": False, "max_samples": 0},     # Mixed quality, some CoT
    "anthropic_hh": {"enabled": False, "max_samples": 0}, # Preference data, not for SFT
    "lmsys_chat": {"enabled": False, "max_samples": 0}, # Real user convos with various models (reasoning)
    "dolphin": {"enabled": False, "max_samples": 0},    # Has CoT patterns
    "capybara": {"enabled": False, "max_samples": 0},   # DPO data, mixed
}

SEED = 42
random.seed(SEED)

wandb.init(project='gpt3-small-fineweb-v2', name=f'sft-{time_module.strftime("%Y%m%d-%H%M%S")}')

def build_opt(model, lr=1e-4, weight_decay=0.1):
    return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

tokenizer = GPT2TokenizerFast.from_pretrained('./gpt3-small-jetson-v2') ### TODO: change to tiktoken

HARMONY_SPECIAL_TOKENS = [
    "<|start|>", "<|end|>", "<|message|>", "<|channel|>",
    "<|return|>", "<|call|>", "<|constrain|>"
]

added = set(tokenizer.get_added_vocab().keys())
to_add = [t for t in HARMONY_SPECIAL_TOKENS if t not in added]
added_count = 0
if to_add:
    tokenizer.add_special_tokens({"additional_special_tokens": to_add})
    added_count = len(to_add)

tokenizer.add_special_tokens({"eos_token": "<|end|>"})
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

if not getattr(tokenizer, "chat_template", None) or "<|channel|>" not in tokenizer.chat_template:
    tokenizer.chat_template = (
        "{% for m in messages %}"
        "{% if m['role'] == 'assistant' %}"
        "<|start|>assistant<|channel|>final<|message|>{{ m['content'] }}<|end|>"
        "{% elif m['role'] == 'developer' %}"
        "<|start|>developer<|message|>{{ m['content'] }}<|end|>"
        "{% else %}"
        "<|start|>{{ m['role'] }}<|message|>{{ m['content'] }}<|end|>"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}<|start|>assistant<|channel|>final<|message|>{% endif %}"
    )

H_START_SYS = "<|start|>system<|message|>"
H_START_DEV = "<|start|>developer<|message|>"
H_START_USER = "<|start|>user<|message|>"
H_START_ASS_F = "<|start|>assistant<|channel|>final<|message|>"
H_END = "<|end|>"

TOK_START_SYS = tokenizer(H_START_SYS, add_special_tokens=False)["input_ids"]
TOK_START_DEV = tokenizer(H_START_DEV, add_special_tokens=False)["input_ids"]
TOK_START_USER = tokenizer(H_START_USER, add_special_tokens=False)["input_ids"]
TOK_START_ASS_F = tokenizer(H_START_ASS_F, add_special_tokens=False)["input_ids"]
TOK_END = tokenizer(H_END, add_special_tokens=False)["input_ids"]


model = AutoModelForCausalLM.from_pretrained('./gpt3-small-jetson-v2', trust_remote_code=True)

if added_count > 0:
    model.resize_token_embeddings(len(tokenizer))
    # we initialize new embeddings with smaller values to avoid instability
    # and position tokens are at the end of the embedding matrix
    with torch.no_grad():
        old_num = len(tokenizer) - added_count
        # Get mean and std of existing embeddings for better init
        embed_weight = model.get_input_embeddings().weight
        existing_mean = embed_weight[:old_num].mean()
        existing_std = embed_weight[:old_num].std() * 0.1
        # Reinitialize new embeddings with smaller scale
        embed_weight[old_num:] = torch.randn_like(embed_weight[old_num:]) * existing_std + existing_mean
        # Do the same for lm_head if it exists and is separate
        if hasattr(model, 'lm_head') and model.lm_head.weight is not embed_weight:
            lm_weight = model.lm_head.weight
            lm_weight[old_num:] = torch.randn_like(lm_weight[old_num:]) * existing_std + existing_mean
    print(f"  Reinitialized {added_count} new token embeddings with smaller scale")

print("Fine-tuning model with", model.num_parameters(), "parameters")


use_mps = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
device = torch.device('mps' if use_mps else 'cpu')
use_mps=False
device = "cpu" ###
print("Using device:", device)
model.to(device)

#=== DATA ===

def normalize_role(role: str) -> str:
    """Normalize role names across different datasets."""
    role = role.lower().strip()
    if role in ["human", "user", "prompter"]:
        return "user"
    elif role in ["gpt", "assistant", "bot", "chatgpt", "claude"]:
        return "assistant"
    elif role in ["system"]:
        return "system"
    else:
        return "user"  # Default to user for unknown roles


# Chain-of-thought / reasoning patterns to filter out
COT_PATTERNS = [
    "step by step",
    "step-by-step", 
    "let's think",
    "let me think",
    "think step",
    "reasoning:",
    "chain of thought",
    "chain-of-thought",
    "## step",
    "step 1:",
    "step 2:",
    "step 3:",
    "first, ",
    "second, ",
    "third, ",
    "finally, ",
    "therefore,",
    "in conclusion,",
    "to solve this",
    "let's break",
    "breaking down",
    "we need to",
    "we can see",
    "this means",
    "here's the logic",
    "here's the reasoning",
    "let me break this down",
    "to answer this",
    "let's analyze",
    "analyzing this",
]

COT_SYSTEM_PATTERNS = [
    "think step",
    "step-by-step",
    "step by step",
    "chain of thought",
    "reasoning process",
    "show your work",
    "explain your reasoning",
    "justify your answer",
    "justify your steps",
    "think like you are answering",
]

# Refusal patterns to filter out
REFUSAL_PATTERNS = [
    "i'm sorry",
    "i am sorry",
    "i apologize",
    "i cannot",
    "i can't",
    "i can not",
    "i'm not able",
    "i am not able",
    "i'm unable",
    "i am unable",
    "i don't have the ability",
    "i do not have the ability",
    "i'm not capable",
    "i am not capable",
    "i cannot provide",
    "i can't provide",
    "i cannot help",
    "i can't help",
    "i cannot assist",
    "i can't assist",
    "i'm afraid i",
    "i am afraid i",
    "unfortunately, i",
    "regrettably, i",
    "i must decline",
    "i have to decline",
    "i won't be able",
    "i will not be able",
    "not able to help",
    "not able to assist",
    "not able to provide",
    "beyond my capabilities",
    "outside my capabilities",
    "not within my capabilities",
    "i don't have access",
    "i do not have access",
    "i lack the ability",
    "it's not appropriate",
    "it is not appropriate",
    "not appropriate for me",
    "i shouldn't",
    "i should not",
    "i can't do that",
    "i cannot do that",
]

# AI identity patterns to filter out
AI_IDENTITY_PATTERNS = [
    "as an ai",
    "as a language model",
    "as an artificial",
    "i am an ai",
    "i'm an ai",
    "i am a language model",
    "i'm a language model",
    "i am a virtual assistant",
    "i'm a virtual assistant",
    "i am just an ai",
    "i'm just an ai",
    "i am a chatbot",
    "i'm a chatbot",
    "i am a bot",
    "i'm a bot",
    "i was trained",
    "i was created",
    "my training data",
    "my training",
    "i don't have emotions",
    "i do not have emotions",
    "i don't have feelings",
    "i do not have feelings",
    "i don't have opinions",
    "i do not have opinions",
    "i don't have personal",
    "i do not have personal",
    "i am not human",
    "i'm not human",
    "as a machine",
    "being an ai",
    "since i'm an ai",
    "since i am an ai",
    "i exist as",
    "openai",
    "anthropic",
    "trained by",
    "developed by",
]

def has_cot_patterns(messages: list) -> bool:
    """Check if messages contain chain-of-thought patterns we want to filter."""
    for msg in messages:
        content = msg.get("content", "").lower()
        role = msg.get("role", "")
        
        # Check system/user messages for CoT instructions
        if role in ["system", "user"]:
            for pattern in COT_SYSTEM_PATTERNS:
                if pattern in content:
                    return True
        
        # Check assistant messages for CoT reasoning patterns
        if role == "assistant":
            # Count how many CoT patterns appear
            cot_count = sum(1 for pattern in COT_PATTERNS if pattern in content)
            # If multiple CoT patterns, likely a reasoning response
            if cot_count >= 2:
                return True
            # Also check for numbered lists that look like steps
            import re
            step_pattern = r'^\s*\d+[\.\)]\s+'
            lines = content.split('\n')
            numbered_lines = sum(1 for line in lines if re.match(step_pattern, line))
            if numbered_lines >= 3:
                return True
    
    return False


def has_refusal_patterns(messages: list) -> bool:
    """Check if assistant messages contain refusal patterns."""
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        
        content = msg.get("content", "").lower()
        
        # Check for refusal patterns
        for pattern in REFUSAL_PATTERNS:
            if pattern in content:
                return True
    
    return False


def has_ai_identity_patterns(messages: list) -> bool:
    """Check if assistant messages leak AI identity."""
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        
        content = msg.get("content", "").lower()
        
        # Check for AI identity patterns
        for pattern in AI_IDENTITY_PATTERNS:
            if pattern in content:
                return True
    
    return False


def should_filter_sample(messages: list) -> tuple:
    """
    Check all filtering criteria. Returns (should_filter, reason).
    """
    if has_cot_patterns(messages):
        return True, "cot"
    if has_refusal_patterns(messages):
        return True, "refusal"
    if has_ai_identity_patterns(messages):
        return True, "ai_identity"
    return False, None


def convert_no_robots(sample) -> list:
    """HuggingFaceH4/no_robots format."""
    return sample["messages"]


def convert_ultrachat(sample) -> list:
    """HuggingFaceH4/ultrachat_200k format."""
    # Format: {"messages": [{"role": ..., "content": ...}]}
    return sample["messages"]


def convert_openhermes(sample) -> list:
    """teknium/OpenHermes-2.5 format."""
    # Format: {"conversations": [{"from": "human/gpt", "value": ...}]}
    messages = []
    for turn in sample.get("conversations", []):
        role = normalize_role(turn.get("from", "user"))
        content = turn.get("value", "")
        if content:
            messages.append({"role": role, "content": content})
    return messages


def convert_slimorca(sample) -> list:
    """Open-Orca/SlimOrca format."""
    # Format: {"conversations": [{"from": "system/human/gpt", "value": ...}]}
    messages = []
    for turn in sample.get("conversations", []):
        role = normalize_role(turn.get("from", "user"))
        content = turn.get("value", "")
        if content:
            messages.append({"role": role, "content": content})
    return messages


def convert_guanaco(sample) -> list:
    """mlabonne/guanaco-llama2-1k format - Llama-2 INST style."""
    # Format: "<s>[INST] user message [/INST] assistant response </s>"
    import re
    text = sample.get("text", "")
    messages = []
    
    # Pattern to match [INST]...[/INST]... pairs
    # Handle both single and multi-turn conversations
    pattern = r'\[INST\]\s*(.*?)\s*\[/INST\]\s*(.*?)(?=\[INST\]|</s>|$)'
    matches = re.findall(pattern, text, re.DOTALL)
    
    for user_msg, assistant_msg in matches:
        user_msg = user_msg.strip()
        assistant_msg = assistant_msg.strip().rstrip('</s>').strip()
        if user_msg:
            messages.append({"role": "user", "content": user_msg})
        if assistant_msg:
            messages.append({"role": "assistant", "content": assistant_msg})
    
    return messages


def convert_oasst2(sample) -> list:
    """OpenAssistant/oasst2 format - individual messages with role field.
    
    OASST2 is a tree-structured dataset where each row is a single message.
    We return single-turn pairs: prompter->assistant.
    """
    messages = []
    role = sample.get("role", "")
    text = sample.get("text", "")
    
    if not text:
        return messages
    
    # Map OASST2 roles to our format
    if role == "prompter":
        messages.append({"role": "user", "content": text})
    elif role == "assistant":
        messages.append({"role": "assistant", "content": text})
    
    # Single messages won't form valid pairs, but we handle this in loading
    return messages


def convert_anthropic_hh(sample) -> list:
    """Anthropic/hh-rlhf format."""
    # Format: {"chosen": "Human: ... Assistant: ...", "rejected": ...}
    # We use the 'chosen' response
    text = sample.get("chosen", "")
    messages = []
    
    # Split by Human: and Assistant:
    parts = text.split("\n\nHuman: ")
    for i, part in enumerate(parts):
        if i == 0 and not part.strip():
            continue
        
        # Check if this part has an assistant response
        if "\n\nAssistant: " in part:
            human_part, assistant_part = part.split("\n\nAssistant: ", 1)
            if human_part.strip():
                messages.append({"role": "user", "content": human_part.strip()})
            if assistant_part.strip():
                messages.append({"role": "assistant", "content": assistant_part.strip()})
        else:
            if part.strip():
                messages.append({"role": "user", "content": part.strip()})
    
    return messages


def convert_lmsys_chat(sample) -> list:
    """lmsys/lmsys-chat-1m format."""
    # Format: {"conversation": [{"role": ..., "content": ...}]}
    conv = sample.get("conversation", [])
    messages = []
    for turn in conv:
        role = normalize_role(turn.get("role", "user"))
        content = turn.get("content", "")
        if content:
            messages.append({"role": role, "content": content})
    return messages


def convert_dolphin(sample) -> list:
    """cognitivecomputations/dolphin format."""
    # Usually has 'instruction', 'input', 'output' or 'conversations'
    if "conversations" in sample:
        messages = []
        for turn in sample["conversations"]:
            role = normalize_role(turn.get("from", turn.get("role", "user")))
            content = turn.get("value", turn.get("content", ""))
            if content:
                messages.append({"role": role, "content": content})
        return messages
    
    messages = []
    instruction = sample.get("instruction", "")
    input_text = sample.get("input", "")
    output = sample.get("output", "")
    
    if instruction:
        user_content = instruction
        if input_text:
            user_content += f"\n\n{input_text}"
        messages.append({"role": "user", "content": user_content})
    if output:
        messages.append({"role": "assistant", "content": output})
    
    return messages


def convert_capybara(sample) -> list:
    """argilla/distilabel-capybara-dpo-7k-binarized format."""
    # Format: {"conversation": [{"input": ..., "output": ...}, ...]}
    messages = []
    conversation = sample.get("conversation", [])
    
    for turn in conversation:
        user_input = turn.get("input", "")
        assistant_output = turn.get("output", "")
        
        if user_input:
            messages.append({"role": "user", "content": user_input})
        if assistant_output:
            messages.append({"role": "assistant", "content": assistant_output})
    
    return messages


def convert_alpaca_cleaned(sample) -> list:
    """yahma/alpaca-cleaned format - instruction/input/output."""
    messages = []
    instruction = sample.get("instruction", "")
    input_text = sample.get("input", "")
    output = sample.get("output", "")
    
    if instruction:
        user_content = instruction
        if input_text and input_text.strip():
            user_content += f"\n\n{input_text}"
        messages.append({"role": "user", "content": user_content})
    if output:
        messages.append({"role": "assistant", "content": output})
    
    return messages


def convert_dolly(sample) -> list:
    """databricks/databricks-dolly-15k format - human-written, clean."""
    messages = []
    instruction = sample.get("instruction", "")
    context = sample.get("context", "")
    response = sample.get("response", "")
    
    if instruction:
        user_content = instruction
        if context and context.strip():
            user_content += f"\n\nContext: {context}"
        messages.append({"role": "user", "content": user_content})
    if response:
        messages.append({"role": "assistant", "content": response})
    
    return messages


def convert_lima(sample) -> list:
    """GAIR/lima format - very high quality curated conversations."""
    messages = []
    conversations = sample.get("conversations", [])
    
    for i, turn in enumerate(conversations):
        # LIMA alternates: user, assistant, user, assistant...
        role = "user" if i % 2 == 0 else "assistant"
        if turn:
            messages.append({"role": role, "content": turn})
    
    return messages


# Note: yizhongw/self_instruct uses deprecated loading script - use alpaca_cleaned instead


def convert_gpt_teacher(sample) -> list:
    """teknium/GPT4-LLM-Cleaned or GPTeacher format - creative tasks."""
    messages = []
    # Format: {"instruction": ..., "input": ..., "output": ...}
    instruction = sample.get("instruction", "")
    input_text = sample.get("input", "")
    output = sample.get("output", sample.get("response", ""))
    
    if instruction:
        user_content = instruction
        if input_text and input_text.strip():
            user_content += f"\n\n{input_text}"
        messages.append({"role": "user", "content": user_content})
    if output:
        messages.append({"role": "assistant", "content": output})
    
    return messages


def convert_sharegpt_vicuna(sample) -> list:
    """anon8231489123/ShareGPT_Vicuna_unfiltered format - cleaned ShareGPT."""
    messages = []
    conversations = sample.get("conversations", [])
    
    for turn in conversations:
        role = normalize_role(turn.get("from", turn.get("role", "user")))
        content = turn.get("value", turn.get("content", ""))
        if content and content.strip():
            messages.append({"role": role, "content": content.strip()})
    
    return messages


def convert_oig_h2o(sample) -> list:
    """h2oai/h2ogpt-oig-instruct-cleaned format - uses <human>:/<bot>: tags in 'input' field."""
    messages = []
    # Format: {"input": "<human>: ... <bot>: ..."}
    text = sample.get("input", sample.get("text", ""))
    
    if not text:
        return messages
    
    # Parse the <human>: and <bot>: format
    import re
    parts = re.split(r'<human>:|<bot>:', text)
    
    for i, part in enumerate(parts):
        part = part.strip()
        if not part:
            continue
        # First part after split is before first tag, skip if empty
        # Then alternates: human, bot, human, bot...
        if i == 0:
            continue  # Skip text before first tag
        role = "user" if i % 2 == 1 else "assistant"
        messages.append({"role": role, "content": part})
    
    return messages


def convert_oig_chip2(sample) -> list:
    """0-hero/OIG-small-chip2 format - uses 'user' and 'chip2' columns."""
    messages = []
    user = sample.get("user", "").strip()
    chip2 = sample.get("chip2", "").strip()
    
    if user:
        messages.append({"role": "user", "content": user})
    if chip2:
        messages.append({"role": "assistant", "content": chip2})
    
    return messages



def load_and_sample(load_fn, max_samples, seed=SEED):
    """Load a dataset and optionally sample from it."""
    try:
        ds = load_fn()
        if max_samples and len(ds) > max_samples:
            ds = ds.shuffle(seed=seed).select(range(max_samples))
        return ds
    except Exception as e:
        print(f"Warning: Failed to load dataset: {e}")
        return None


def load_all_datasets():
    """Load and combine all configured datasets."""
    all_messages = []
    dataset_sources = []
    
    # Track filtering statistics for visualization (now with breakdown by reason)
    filter_stats = {}
    
    # Helper function for comprehensive filtering
    def add_sample_with_filtering(msgs, dataset_name, stats):
        """Add sample if it passes all filters, update stats."""
        if not msgs or len(msgs) < 2:
            return False
        
        stats["total"] += 1
        should_filter, reason = should_filter_sample(msgs)
        
        if should_filter:
            stats["filtered"] += 1
            stats["by_reason"][reason] = stats["by_reason"].get(reason, 0) + 1
            return False
        
        all_messages.append(msgs)
        dataset_sources.append(dataset_name)
        stats["kept"] += 1
        return True
    
    def init_stats():
        return {"kept": 0, "filtered": 0, "total": 0, "by_reason": {}}
    
    # 1. No Robots (cleanest dataset - still filter for safety)
    if DATASET_CONFIG["no_robots"]["enabled"]:
        print("Loading no_robots...")
        try:
            ds = load_dataset("HuggingFaceH4/no_robots", split="train")
            if DATASET_CONFIG["no_robots"]["max_samples"]:
                ds = ds.shuffle(seed=SEED).select(range(min(len(ds), DATASET_CONFIG["no_robots"]["max_samples"])))
            stats = init_stats()
            for sample in ds:
                msgs = convert_no_robots(sample)
                add_sample_with_filtering(msgs, "no_robots", stats)
            filter_stats["no_robots"] = stats
            print(f"  Loaded {stats['kept']}/{stats['total']} (filtered: {stats['by_reason']})")
        except Exception as e:
            print(f"  Failed to load no_robots: {e}")
    
    # 2. UltraChat (pre-filtered but double-check)
    if DATASET_CONFIG["ultrachat"]["enabled"]:
        print("Loading ultrachat...")
        try:
            ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
            max_s = DATASET_CONFIG["ultrachat"]["max_samples"]
            if max_s:
                ds = ds.shuffle(seed=SEED).select(range(min(len(ds), max_s)))
            stats = init_stats()
            for sample in ds:
                msgs = convert_ultrachat(sample)
                add_sample_with_filtering(msgs, "ultrachat", stats)
            filter_stats["ultrachat"] = stats
            print(f"  Loaded {stats['kept']}/{stats['total']} (filtered: {stats['by_reason']})")
        except Exception as e:
            print(f"  Failed to load ultrachat: {e}")
    
    # 3. OpenHermes (known to have CoT)
    if DATASET_CONFIG["openhermes"]["enabled"]:
        print("Loading OpenHermes-2.5 (comprehensive filtering)...")
        try:
            ds = load_dataset("teknium/OpenHermes-2.5", split="train")
            max_s = DATASET_CONFIG["openhermes"]["max_samples"]
            if max_s:
                ds = ds.shuffle(seed=SEED).select(range(min(len(ds), max_s)))
            stats = init_stats()
            for sample in ds:
                msgs = convert_openhermes(sample)
                add_sample_with_filtering(msgs, "openhermes", stats)
            filter_stats["openhermes"] = stats
            print(f"  Loaded {stats['kept']}/{stats['total']} (filtered: {stats['by_reason']})")
        except Exception as e:
            print(f"  Failed to load OpenHermes: {e}")
    
    # 4. SlimOrca (heavy CoT - needs comprehensive filtering)
    if DATASET_CONFIG["slimorca"]["enabled"]:
        print("Loading SlimOrca (comprehensive filtering)...")
        try:
            ds = load_dataset("Open-Orca/SlimOrca", split="train")
            max_s = DATASET_CONFIG["slimorca"]["max_samples"]
            if max_s:
                ds = ds.shuffle(seed=SEED).select(range(min(len(ds), max_s)))
            stats = init_stats()
            for sample in ds:
                msgs = convert_slimorca(sample)
                add_sample_with_filtering(msgs, "slimorca", stats)
            filter_stats["slimorca"] = stats
            print(f"  Loaded {stats['kept']}/{stats['total']} (filtered: {stats['by_reason']})")
        except Exception as e:
            print(f"  Failed to load SlimOrca: {e}")
    
    # 5. Guanaco
    if DATASET_CONFIG["guanaco"]["enabled"]:
        print("Loading Guanaco (comprehensive filtering)...")
        try:
            ds = load_dataset("mlabonne/guanaco-llama2-1k", split="train")
            stats = init_stats()
            for sample in ds:
                msgs = convert_guanaco(sample)
                add_sample_with_filtering(msgs, "guanaco", stats)
            filter_stats["guanaco"] = stats
            print(f"  Loaded {stats['kept']}/{stats['total']} (filtered: {stats['by_reason']})")
        except Exception as e:
            print(f"  Failed to load Guanaco: {e}")
    
    # 6. OASST2 - Use the pre-processed conversation version
    if DATASET_CONFIG["oasst2"]["enabled"]:
        print("Loading OASST2 (comprehensive filtering)...")
        try:
            ds = load_dataset("timdettmers/openassistant-guanaco", split="train")
            max_s = DATASET_CONFIG["oasst2"]["max_samples"]
            if max_s:
                ds = ds.shuffle(seed=SEED).select(range(min(len(ds), max_s)))
            stats = init_stats()
            for sample in ds:
                text = sample.get("text", "")
                messages = []
                parts = text.split("### ")
                for part in parts:
                    part = part.strip()
                    if part.startswith("Human:"):
                        content = part[6:].strip()
                        if content:
                            messages.append({"role": "user", "content": content})
                    elif part.startswith("Assistant:"):
                        content = part[10:].strip()
                        if content:
                            messages.append({"role": "assistant", "content": content})
                add_sample_with_filtering(messages, "oasst2", stats)
            filter_stats["oasst2"] = stats
            print(f"  Loaded {stats['kept']}/{stats['total']} (filtered: {stats['by_reason']})")
        except Exception as e:
            print(f"  Failed to load OASST2: {e}")
    
    # 7. Anthropic HH-RLHF - DISABLED (preference data, not suitable for SFT)
    if DATASET_CONFIG["anthropic_hh"]["enabled"]:
        print("Loading Anthropic HH-RLHF (comprehensive filtering)...")
        print("  WARNING: This dataset is designed for RLHF, not SFT. Consider disabling.")
        try:
            ds = load_dataset("Anthropic/hh-rlhf", split="train")
            max_s = DATASET_CONFIG["anthropic_hh"]["max_samples"]
            if max_s:
                ds = ds.shuffle(seed=SEED).select(range(min(len(ds), max_s)))
            kept, cot_filtered, total = 0, 0, 0
            for sample in ds:
                msgs = convert_anthropic_hh(sample)
                if msgs and len(msgs) >= 2:
                    total += 1
                    if has_cot_patterns(msgs):
                        cot_filtered += 1
                        continue
                    all_messages.append(msgs)
            stats = init_stats()
            for sample in ds:
                msgs = convert_anthropic_hh(sample)
                add_sample_with_filtering(msgs, "anthropic_hh", stats)
            filter_stats["anthropic_hh"] = stats
            print(f"  Loaded {stats['kept']}/{stats['total']} (filtered: {stats['by_reason']})")
        except Exception as e:
            print(f"  Failed to load Anthropic HH: {e}")
    
    # 8. LMSYS Chat (needs heavy filtering - real user conversations)
    if DATASET_CONFIG["lmsys_chat"]["enabled"]:
        print("Loading LMSYS Chat (comprehensive filtering)...")
        try:
            ds = load_dataset("lmsys/lmsys-chat-1m", split="train")
            max_s = DATASET_CONFIG["lmsys_chat"]["max_samples"]
            if max_s:
                ds = ds.shuffle(seed=SEED).select(range(min(len(ds), max_s)))
            stats = init_stats()
            for sample in ds:
                msgs = convert_lmsys_chat(sample)
                add_sample_with_filtering(msgs, "lmsys_chat", stats)
            filter_stats["lmsys_chat"] = stats
            print(f"  Loaded {stats['kept']}/{stats['total']} (filtered: {stats['by_reason']})")
        except Exception as e:
            print(f"  Failed to load LMSYS Chat: {e}")
    
    # 9. Dolphin (uncensored but has CoT patterns)
    if DATASET_CONFIG["dolphin"]["enabled"]:
        print("Loading Dolphin (comprehensive filtering)...")
        try:
            ds = load_dataset("cognitivecomputations/dolphin", split="train", name="flan1m-alpaca-uncensored")
            max_s = DATASET_CONFIG["dolphin"]["max_samples"]
            if max_s:
                ds = ds.shuffle(seed=SEED).select(range(min(len(ds), max_s)))
            stats = init_stats()
            for sample in ds:
                msgs = convert_dolphin(sample)
                add_sample_with_filtering(msgs, "dolphin", stats)
            filter_stats["dolphin"] = stats
            print(f"  Loaded {stats['kept']}/{stats['total']} (filtered: {stats['by_reason']})")
        except Exception as e:
            print(f"  Failed to load Dolphin: {e}")
    
    # 10. Capybara (DPO data - extract chosen, filter comprehensively)
    if DATASET_CONFIG["capybara"]["enabled"]:
        print("Loading Capybara (comprehensive filtering)...")
        try:
            ds = load_dataset("argilla/distilabel-capybara-dpo-7k-binarized", split="train")
            max_s = DATASET_CONFIG["capybara"]["max_samples"]
            if max_s:
                ds = ds.shuffle(seed=SEED).select(range(min(len(ds), max_s)))
            stats = init_stats()
            for sample in ds:
                msgs = convert_capybara(sample)
                add_sample_with_filtering(msgs, "capybara", stats)
            filter_stats["capybara"] = stats
            print(f"  Loaded {stats['kept']}/{stats['total']} (filtered: {stats['by_reason']})")
        except Exception as e:
            print(f"  Failed to load Capybara: {e}")
    
    # 11. Alpaca Cleaned (cleaned version of Stanford Alpaca, mostly direct answers)
    if DATASET_CONFIG.get("alpaca_cleaned", {}).get("enabled", False):
        print("Loading Alpaca Cleaned (comprehensive filtering)...")
        try:
            ds = load_dataset("yahma/alpaca-cleaned", split="train")
            max_s = DATASET_CONFIG["alpaca_cleaned"]["max_samples"]
            if max_s:
                ds = ds.shuffle(seed=SEED).select(range(min(len(ds), max_s)))
            stats = init_stats()
            for sample in ds:
                msgs = convert_alpaca_cleaned(sample)
                add_sample_with_filtering(msgs, "alpaca_cleaned", stats)
            filter_stats["alpaca_cleaned"] = stats
            print(f"  Loaded {stats['kept']}/{stats['total']} (filtered: {stats['by_reason']})")
        except Exception as e:
            print(f"  Failed to load Alpaca Cleaned: {e}")
    
    # 12. Dolly (Databricks human-written, very clean - no GPT generation)
    if DATASET_CONFIG.get("dolly", {}).get("enabled", False):
        print("Loading Dolly 15k (human-written, clean)...")
        try:
            ds = load_dataset("databricks/databricks-dolly-15k", split="train")
            max_s = DATASET_CONFIG["dolly"]["max_samples"]
            if max_s:
                ds = ds.shuffle(seed=SEED).select(range(min(len(ds), max_s)))
            stats = init_stats()
            for sample in ds:
                msgs = convert_dolly(sample)
                add_sample_with_filtering(msgs, "dolly", stats)
            filter_stats["dolly"] = stats
            print(f"  Loaded {stats['kept']}/{stats['total']} (filtered: {stats['by_reason']})")
        except Exception as e:
            print(f"  Failed to load Dolly: {e}")
    
    # 13. LIMA (very high quality curated dataset, ~1k samples)
    if DATASET_CONFIG.get("lima", {}).get("enabled", False):
        print("Loading LIMA (high quality curated)...")
        try:
            ds = load_dataset("GAIR/lima", split="train")
            max_s = DATASET_CONFIG["lima"]["max_samples"]
            if max_s:
                ds = ds.shuffle(seed=SEED).select(range(min(len(ds), max_s)))
            stats = init_stats()
            for sample in ds:
                msgs = convert_lima(sample)
                add_sample_with_filtering(msgs, "lima", stats)
            filter_stats["lima"] = stats
            print(f"  Loaded {stats['kept']}/{stats['total']} (filtered: {stats['by_reason']})")
        except Exception as e:
            print(f"  Failed to load LIMA: {e}")
    
    # 14. Self-Instruct - DISABLED (yizhongw/self_instruct uses deprecated loading script)
    # Use tatsu-lab/alpaca or yahma/alpaca-cleaned instead
    
    # 15. GPT-Teacher / GPT4-LLM-Cleaned (creative tasks, ~55k)
    if DATASET_CONFIG.get("gpt_teacher", {}).get("enabled", False):
        print("Loading GPT-Teacher / GPT4-LLM-Cleaned (creative tasks)...")
        try:
            # Try the cleaned GPT4-LLM dataset first
            ds = load_dataset("teknium/GPT4-LLM-Cleaned", split="train")
            max_s = DATASET_CONFIG["gpt_teacher"]["max_samples"]
            if max_s:
                ds = ds.shuffle(seed=SEED).select(range(min(len(ds), max_s)))
            stats = init_stats()
            for sample in ds:
                msgs = convert_gpt_teacher(sample)
                add_sample_with_filtering(msgs, "gpt_teacher", stats)
            filter_stats["gpt_teacher"] = stats
            print(f"  Loaded {stats['kept']}/{stats['total']} (filtered: {stats['by_reason']})")
        except Exception as e:
            print(f"  Failed to load GPT-Teacher: {e}")
    
    # 16. ShareGPT Vicuna Unfiltered (cleaned ShareGPT conversations, ~94k)
    if DATASET_CONFIG.get("sharegpt_vicuna", {}).get("enabled", False):
        print("Loading ShareGPT Vicuna (cleaned ShareGPT conversations)...")
        try:
            ds = load_dataset("anon8231489123/ShareGPT_Vicuna_unfiltered", split="train", 
                             data_files="ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json")
            max_s = DATASET_CONFIG["sharegpt_vicuna"]["max_samples"]
            if max_s:
                ds = ds.shuffle(seed=SEED).select(range(min(len(ds), max_s)))
            stats = init_stats()
            for sample in ds:
                msgs = convert_sharegpt_vicuna(sample)
                add_sample_with_filtering(msgs, "sharegpt_vicuna", stats)
            filter_stats["sharegpt_vicuna"] = stats
            print(f"  Loaded {stats['kept']}/{stats['total']} (filtered: {stats['by_reason']})")
        except Exception as e:
            print(f"  Failed to load ShareGPT Vicuna: {e}")
    
    # 17. OIG H2O Cleaned (h2oai cleaned OIG with <human>:/<bot>: tags, ~195k)
    if DATASET_CONFIG.get("oig_h2o", {}).get("enabled", False):
        print("Loading OIG H2O Cleaned (h2oai version)...")
        try:
            ds = load_dataset("h2oai/h2ogpt-oig-instruct-cleaned", split="train")
            max_s = DATASET_CONFIG["oig_h2o"]["max_samples"]
            if max_s:
                ds = ds.shuffle(seed=SEED).select(range(min(len(ds), max_s)))
            stats = init_stats()
            for sample in ds:
                msgs = convert_oig_h2o(sample)
                add_sample_with_filtering(msgs, "oig_h2o", stats)
            filter_stats["oig_h2o"] = stats
            print(f"  Loaded {stats['kept']}/{stats['total']} (filtered: {stats['by_reason']})")
        except Exception as e:
            print(f"  Failed to load OIG H2O: {e}")
    
    # 18. OIG Chip2 (0-hero version with user/chip2 columns, ~210k)
    if DATASET_CONFIG.get("oig_chip2", {}).get("enabled", False):
        print("Loading OIG Chip2 (0-hero version)...")
        try:
            ds = load_dataset("0-hero/OIG-small-chip2", split="train")
            max_s = DATASET_CONFIG["oig_chip2"]["max_samples"]
            if max_s:
                ds = ds.shuffle(seed=SEED).select(range(min(len(ds), max_s)))
            stats = init_stats()
            for sample in ds:
                msgs = convert_oig_chip2(sample)
                add_sample_with_filtering(msgs, "oig_chip2", stats)
            filter_stats["oig_chip2"] = stats
            print(f"  Loaded {stats['kept']}/{stats['total']} (filtered: {stats['by_reason']})")
        except Exception as e:
            print(f"  Failed to load OIG Chip2: {e}")
    
    print(f"\nTotal samples loaded: {len(all_messages)}")
    
    # Log dataset composition to wandb with pie charts
    from collections import Counter
    import matplotlib.pyplot as plt
    
    source_counts = Counter(dataset_sources)
    
    # ==== 1. Per-dataset filtering pie charts (kept vs filtered) ====
    print("\n=== Comprehensive Filtering Statistics ===")
    datasets_with_filtering = [k for k, v in filter_stats.items() if v["total"] > 0]
    
    if datasets_with_filtering:
        # Calculate grid size for subplots
        n_datasets = len(datasets_with_filtering)
        n_cols = min(3, n_datasets)
        n_rows = (n_datasets + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
        if n_datasets == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_datasets > 1 else [axes]
        
        for idx, dataset_name in enumerate(datasets_with_filtering):
            stats = filter_stats[dataset_name]
            kept = stats["kept"]
            filtered = stats["filtered"]
            total = stats["total"]
            by_reason = stats.get("by_reason", {})
            filter_pct = (filtered / total * 100) if total > 0 else 0
            
            reason_str = ", ".join([f"{k}:{v}" for k, v in by_reason.items()]) if by_reason else "none"
            print(f"  {dataset_name}: {kept}/{total} kept ({filter_pct:.1f}% filtered) - reasons: {reason_str}")
            
            ax = axes[idx]
            sizes = [kept, filtered]
            labels = [f'Kept ({kept})', f'Filtered ({filtered})']
            colors = ['#4CAF50', '#FF5722']  # Green for kept, Red for filtered
            explode = (0, 0.05)  # Slightly explode the filtered slice
            
            if total > 0 and (kept > 0 or filtered > 0):
                ax.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                       colors=colors, startangle=90, shadow=True)
                ax.set_title(f'{dataset_name}\n(Total: {total})')
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                ax.set_title(dataset_name)
        
        # Hide unused subplots
        for idx in range(len(datasets_with_filtering), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Filtering Per Dataset: Kept vs Filtered\n(CoT + Refusals + AI Identity)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        wandb.log({"filtering_per_dataset": wandb.Image(fig)})
        plt.savefig("filtering_per_dataset.png", dpi=150)
        plt.close(fig)
        print("  Saved: filtering_per_dataset.png")
    
    # ==== 2. Filter reason breakdown (stacked bar chart) ====
    if filter_stats:
        fig, ax = plt.subplots(figsize=(14, 7))
        datasets = list(filter_stats.keys())
        
        # Collect reasons across all datasets
        all_reasons = set()
        for d in datasets:
            all_reasons.update(filter_stats[d].get("by_reason", {}).keys())
        all_reasons = sorted(all_reasons)
        
        # Prepare data for stacked bars
        kept_vals = [filter_stats[d]["kept"] for d in datasets]
        reason_vals = {reason: [filter_stats[d].get("by_reason", {}).get(reason, 0) for d in datasets] 
                       for reason in all_reasons}
        
        x = range(len(datasets))
        width = 0.6
        
        # Color scheme
        colors = {
            'cot': '#FF9800',      # Orange for CoT
            'refusal': '#F44336', # Red for refusals
            'ai_identity': '#9C27B0',  # Purple for AI identity
        }
        
        # Plot kept first
        bars_kept = ax.bar(x, kept_vals, width, label='Kept', color='#4CAF50')
        
        # Stack filtered reasons on top
        bottom = kept_vals.copy()
        for reason in all_reasons:
            vals = reason_vals[reason]
            color = colors.get(reason, '#607D8B')
            ax.bar(x, vals, width, bottom=bottom, label=f'Filtered ({reason})', color=color)
            bottom = [b + v for b, v in zip(bottom, vals)]
        
        ax.set_xlabel('Dataset')
        ax.set_ylabel('Number of Samples')
        ax.set_title('Filtering Breakdown by Reason: Kept vs Filtered (CoT/Refusal/AI Identity)')
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=45, ha='right')
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        wandb.log({"filtering_breakdown_stacked": wandb.Image(fig)})
        plt.savefig("filtering_breakdown_stacked.png", dpi=150)
        plt.close(fig)
        print("  Saved: filtering_breakdown_stacked.png")
    
    # ==== 3. Overall filtering by reason pie chart ====
    if filter_stats:
        # Aggregate all reasons
        all_reason_counts = {}
        for stats in filter_stats.values():
            for reason, count in stats.get("by_reason", {}).items():
                all_reason_counts[reason] = all_reason_counts.get(reason, 0) + count
        
        total_kept = sum(v["kept"] for v in filter_stats.values())
        total_filtered = sum(v["filtered"] for v in filter_stats.values())
        
        if all_reason_counts:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Prepare data: kept + each filter reason
            labels = ['Kept'] + [f'Filtered ({r})' for r in all_reason_counts.keys()]
            sizes = [total_kept] + list(all_reason_counts.values())
            colors_list = ['#4CAF50'] + [colors.get(r, '#607D8B') for r in all_reason_counts.keys()]
            
            explode = [0] + [0.03] * len(all_reason_counts)
            
            ax.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                   colors=colors_list, startangle=90, shadow=True)
            ax.set_title(f'Overall Filtering Summary\nTotal: {total_kept + total_filtered} samples')
            
            plt.tight_layout()
            wandb.log({"filtering_overall_by_reason": wandb.Image(fig)})
            plt.savefig("filtering_overall_by_reason.png", dpi=150)
            plt.close(fig)
            print("  Saved: filtering_overall_by_reason.png")
    
    # ==== 4. Dataset composition pie chart (after filtering) ====
    if source_counts:
        fig, ax = plt.subplots(figsize=(10, 8))
        labels = list(source_counts.keys())
        sizes = list(source_counts.values())
        chart_colors = plt.cm.Set3(range(len(labels)))
        
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=chart_colors, startangle=90)
        ax.set_title('Dataset Composition (After Comprehensive Filtering)')
        plt.tight_layout()
        wandb.log({"dataset_composition_pie": wandb.Image(fig)})
        plt.savefig("dataset_composition.png", dpi=150)
        plt.close(fig)
        print("  Saved: dataset_composition.png")
    
    # Log the raw counts and filtering stats
    total_kept = sum(v["kept"] for v in filter_stats.values())
    total_filtered = sum(v["filtered"] for v in filter_stats.values())
    
    # Aggregate filter reasons
    all_reason_totals = {}
    for stats in filter_stats.values():
        for reason, count in stats.get("by_reason", {}).items():
            all_reason_totals[reason] = all_reason_totals.get(reason, 0) + count
    
    wandb.log({
        "dataset_counts": dict(source_counts),
        "filter_stats": {k: {"kept": v["kept"], "filtered": v["filtered"], "total": v["total"]} 
                         for k, v in filter_stats.items()},
        "total_kept": total_kept,
        "total_filtered": total_filtered,
        "filter_by_reason": all_reason_totals,
        "filter_rate_pct": (total_filtered / (total_kept + total_filtered) * 100) if (total_kept + total_filtered) > 0 else 0
    })
    
    print(f"\n=== Summary ===")
    print(f"Total kept: {total_kept}")
    print(f"Total filtered: {total_filtered}")
    print(f"  - By CoT: {all_reason_totals.get('cot', 0)}")
    print(f"  - By Refusal: {all_reason_totals.get('refusal', 0)}")
    print(f"  - By AI Identity: {all_reason_totals.get('ai_identity', 0)}")
    print(f"Filter rate: {(total_filtered / (total_kept + total_filtered) * 100):.1f}%")
    
    return all_messages, dataset_sources


#======


def render_harmony_conversation(messages):
    parts = []
    for m in messages:
        role = m["role"]
        content = m["content"]
        if role == "assistant":
            parts.append(f"{H_START_ASS_F}{content}{H_END}")
        elif role == "system":
            parts.append(f"{H_START_SYS}{content}{H_END}")
        elif role == "developer":
            parts.append(f"{H_START_DEV}{content}{H_END}")
        else:
            parts.append(f"{H_START_USER}{content}{H_END}")
    return "".join(parts)


def preprocess_harmony(messages, max_length=1024):
    conversation = render_harmony_conversation(messages)
    
    toks = tokenizer(
        conversation, 
        truncation=True, 
        max_length=max_length, 
        padding=False,
        return_offsets_mapping=True
    )
    input_ids = toks["input_ids"]
    attention_mask = toks["attention_mask"]
    offset_mapping = toks["offset_mapping"]
    
    labels = [-100] * len(input_ids)
    
    char_cursor = 0
    assistant_char_ranges = []
    
    for m in messages:
        role = m["role"]
        content = m["content"]
        
        if role == "assistant":
            header = H_START_ASS_F
        elif role == "developer":
            header = H_START_DEV
        elif role == "system":
            header = H_START_SYS
        else:
            header = H_START_USER
        
        # This message spans from char_cursor to char_cursor + len(header + content + H_END)
        msg_start = char_cursor
        header_end = char_cursor + len(header)
        content_end = header_end + len(content)
        msg_end = content_end + len(H_END)
        
        if role == "assistant":
            # We want to label from after the header to the end of the message (including <|end|>)
            assistant_char_ranges.append((header_end, msg_end))
        
        char_cursor = msg_end
    
    # Now map character ranges to token indices using offset_mapping
    for start_char, end_char in assistant_char_ranges:
        for tok_idx, (tok_start, tok_end) in enumerate(offset_mapping):
            # If this token overlaps with the assistant range, label it
            if tok_end > start_char and tok_start < end_char:
                labels[tok_idx] = input_ids[tok_idx]
    
    # Validate: check for any tokens that are labeled but shouldn't be (quick sanity check)
    # Count labels to ensure we're not labeling everything
    labeled_count = sum(1 for l in labels if l != -100)
    total_tokens = len(input_ids)
    
    # If almost all tokens are labeled, something went wrong
    if labeled_count > total_tokens * 0.95:
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": [-100] * len(input_ids),
        }
    
    # Remove offset_mapping from output (not needed downstream)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


print("=" * 60)
print("Loading all datasets...")
print("=" * 60)

all_messages, dataset_sources = load_all_datasets()

combined = list(zip(all_messages, dataset_sources))
random.shuffle(combined)
all_messages, dataset_sources = zip(*combined) if combined else ([], [])


split_idx = int(len(all_messages) * 0.95)
train_messages = all_messages[:split_idx]
eval_messages = all_messages[split_idx:]

print(f"\nTrain samples: {len(train_messages)}")
print(f"Eval samples: {len(eval_messages)}")

print("\nPreprocessing train data...")
train_data = []
for msgs in tqdm(train_messages, desc="Training samples", unit="sample"):
    try:
        processed = preprocess_harmony(msgs)
        # Skip very short samples
        if len(processed["input_ids"]) > 10:
            train_data.append(processed)
    except Exception as e:
        continue

print(f"Processed {len(train_data)} train samples")

print("Preprocessing eval data...")
eval_data = []
for msgs in tqdm(eval_messages, desc="Eval samples", unit="sample"):
    try:
        processed = preprocess_harmony(msgs)
        if len(processed["input_ids"]) > 10:
            eval_data.append(processed)
    except Exception as e:
        continue

print(f"Processed {len(eval_data)} eval samples")


del all_messages, train_messages, eval_messages, dataset_sources, combined
gc.collect()

# I am not sure but I think HF can handle datasets better than I do
print("Creating memory-mapped datasets...")
train_dataset = Dataset.from_list(train_data)
eval_dataset = Dataset.from_list(eval_data)

import tempfile
import shutil

cache_dir = "./sft_v4_cache"
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)
os.makedirs(cache_dir, exist_ok=True)

train_dataset.save_to_disk(f"{cache_dir}/train")
eval_dataset.save_to_disk(f"{cache_dir}/eval")

del train_dataset, eval_dataset, train_data, eval_data
gc.collect()


from datasets import load_from_disk
train_dataset = load_from_disk(f"{cache_dir}/train")
eval_dataset = load_from_disk(f"{cache_dir}/eval")
print(f"Datasets cached to {cache_dir} (memory-mapped)")
gc.collect()


def custom_collate(examples): # Filter out any potentially bad examples
    valid_examples = []
    for ex in examples:
        try:
            if ("input_ids" in ex and "attention_mask" in ex and "labels" in ex and
                len(ex["input_ids"]) > 10 and len(ex["input_ids"]) == len(ex["labels"])):
                # Also check that not all labels are -100 (no learning signal)
                if not all(l == -100 for l in ex["labels"]):
                    valid_examples.append(ex)
        except:
            continue
    
    if not valid_examples or len(valid_examples) == 0:
        # Return a dummy batch with all -100 labels to skip
        return {
            "input_ids": torch.tensor([[tokenizer.pad_token_id] * 50], dtype=torch.long),
            "attention_mask": torch.tensor([[0] * 50], dtype=torch.long),
            "labels": torch.tensor([[-100] * 50], dtype=torch.long),
        }
    
    max_length = max(len(ex["input_ids"]) for ex in valid_examples)
    input_ids, attention_masks, labels = [], [], []
    for ex in valid_examples:
        seq_len = len(ex["input_ids"])
        pad_len = max_length - seq_len
        input_ids.append(ex["input_ids"] + [tokenizer.pad_token_id] * pad_len)
        attention_masks.append(ex["attention_mask"] + [0] * pad_len)
        labels.append(ex["labels"] + [-100] * pad_len)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }



training_args = TrainingArguments(
    output_dir='./gpt4-small-jetson-sft-v4-multi',
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    eval_strategy="steps",
    use_mps_device=use_mps,
    use_cpu=not use_mps,
    gradient_accumulation_steps=24,
    max_grad_norm=1.0,  #0.5
    lr_scheduler_type="cosine",
    optim="adamw_torch",
    warmup_ratio=0.05,
    remove_unused_columns=True,
    eval_on_start=False,
    eval_steps=500,
    save_steps=500,
    logging_steps=1,
    learning_rate=5e-5,  #5e-06 
    warmup_steps=1000,
    save_total_limit=3,
    report_to=['wandb'],
    run_name='sft-v4-multi-dataset',
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
    greater_is_better=False,
    prediction_loss_only=True,
    dataloader_num_workers=0,
    dataloader_pin_memory=False,  # False for MPS
)


class GenerateTextCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 100 == 0:
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        
        model = kwargs["model"]
        tok = kwargs.get("tokenizer") or tokenizer
        
        if state.global_step % 100 == 0 and state.global_step > 0:
            messages = [
                {"role": "user", "content": "Write a brief thank you email to a colleague for their help."}
            ]
            inputs = tok.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            ).to(model.device)
            
            try:
                eos_ids = tok.convert_tokens_to_ids(H_END)
                with torch.no_grad():
                    outputs = model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_new_tokens=200,
                        num_return_sequences=1,
                        do_sample=True,
                        top_p=0.95,
                        repetition_penalty=1.2,
                        pad_token_id=tok.pad_token_id,
                        eos_ids=eos_ids,
                    )
                text = tok.decode(outputs[0], skip_special_tokens=False)
                wandb.log({
                    "sample_text": wandb.Html(f"<pre>{text}</pre>"),
                    "step": state.global_step
                })
                del outputs, inputs
            except Exception as e:
                wandb.log({
                    "generation_error": str(e),
                    "step": state.global_step
                })

class CustomTrainer(Trainer):
    """ Applies manual gradient clipping after backward pass
    and zeros any NaN/Inf gradients to avoid exploding updates.
    Also handles NaN losses during evaluation.
    """
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Override compute_loss to handle NaN losses in both train and eval."""
        input_ids = inputs.get("input_ids")
        labels = inputs.get("labels")
        
        # Check if labels are all -100 (skip token) - but only short-circuit for training
        # During eval (return_outputs=True), we need actual outputs for metrics
        if labels is not None and (labels == -100).all() and not return_outputs:
            # No labels to predict - return zero loss to skip this batch
            loss = torch.tensor(0.0, device=input_ids.device, requires_grad=True)
            return loss
        
        # Call parent's compute_loss
        if return_outputs:
            loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        else:
            loss = super().compute_loss(model, inputs, return_outputs=False)
        
        # Check for NaN/Inf loss and replace with zero
        if torch.isnan(loss) or torch.isinf(loss):
            loss = torch.tensor(0.0, device=loss.device, dtype=loss.dtype, requires_grad=loss.requires_grad)
        
        if return_outputs:
            return loss, outputs
        return loss
    
    def training_step(self, model, inputs, *args, **kwargs):
        model.train()
        inputs = self._prepare_inputs(inputs)

        input_ids = inputs.get("input_ids")
        labels = inputs.get("labels")
        
        if labels is not None and (labels == -100).all():
            return torch.tensor(0.0, device=input_ids.device, requires_grad=False)

        loss = self.compute_loss(model, inputs)
        
        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(0.0, device=loss.device, requires_grad=False)

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        loss.backward()

        # Detect NaNs/Infs in gradients BEFORE clipping
        bad_grad = False
        for p in model.parameters():
            if p.grad is not None:
                if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                    bad_grad = True
                    break

        if bad_grad:
            # If something went wrong and we have bad gradients, zero them
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.zero_()
            return loss.detach()


        max_norm = getattr(self.args, "max_grad_norm", 0.5) or 0.5
        try:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        except Exception:
            pass

        return loss.detach()



trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=custom_collate,
    callbacks=[
        GenerateTextCallback(),
        EarlyStoppingCallback(early_stopping_patience=10, early_stopping_threshold=0.0)
    ],
    optimizers=(build_opt(model, lr=5e-6), None),
)


print("\n" + "=" * 60)
print("Starting training...")
print("=" * 60)

gc.collect()
if torch.backends.mps.is_available():
    torch.mps.empty_cache()

resume_from_checkpoint = None #"./gpt3-small-jetson-v2-sft"
output_dir = './gpt3-small-jetson-v2-sft'
if os.path.exists(output_dir):
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith('checkpoint-')]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('-')[1]))
        resume_from_checkpoint = os.path.join(output_dir, latest_checkpoint)
        print(f"Resuming from checkpoint: {resume_from_checkpoint}")

try:
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
except KeyboardInterrupt:
    print(f"\nTraining interrupted by user, saving...")
except Exception as e:
    print(f"Training error: {e}, saving...")
finally:
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    wandb.finish()
gc.collect()
torch.cuda.empty_cache() if torch.cuda.is_available() else None

print(f"\nTraining complete. Saved to {output_dir}")
