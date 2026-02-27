# GPT3

Welcome to the GPT3 repository! This project is an attempt to recreate the architecture and approach from the original OpenAI GPT-3 paper. The repository includes scripts for training, fine-tuning, and inference of a GPT-3-like model using PyTorch and the Hugging Face Transformers library.

This repository contains the v2 rewrite of my GPT-3-style research project: custom architecture, foundation training, benchmark evaluation, and a new SFT workflow.

If you want the old "stable" code snapshot, use the `v1-stable` tag/release archive.

## What Changed In v2

- I finally came up with a semi-working strided attention implementation, so the architecture matches (as GPTs told me) the original GPT-3 paper (with sparse+dense attention blocks).
- Replaced multiple legacy training scripts with a single `train.py` pipeline.
- Reworked model implementation in `modeling_gpt3dev.py` for newer `transformers` compatibility.
- Upgraded evaluation in `eval.py` with CLI flags for HellaSwag, LAMBADA, and MMLU tasks.
- Added a new full SFT workflow in `sft.py` with multi-dataset loading, filtering, and chat-template training.
- Removed old scripts (`train-17m.py`, `train-125m.py`, `fine-tune-SFT.py`, `inference.py`, etc.) to keep the repo focused.

## Repository Layout

- `train.py`: Foundation-model training (streaming FineWeb, periodic validation/checkpointing, W&B logging).
- `modeling_gpt3dev.py`: Custom GPT-3-like architecture (`GPT3DevConfig`, sparse+dense attention blocks, HF auto-registration).
- `eval.py`: Benchmark runner for HellaSwag / LAMBADA / MMLU.
- `sft.py`: v2 supervised fine-tuning workflow with Harmony-style chat formatting and dataset filtering.
- `inference_simple.py`: Minimal local generation script.
- `arch_demonstrator.py`: Build/save architecture-only checkpoint for experiments.

## Install

```bash
git clone https://github.com/krll-corp/GPT3.git
cd GPT3
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Optional:

- Login for experiment tracking: `wandb login`

## Quickstart

### 1) Train Foundation Model (v2)

```bash
python3 train.py \
  --acc 128 \
  --log-steps 1 \
  --eval-steps 500 \
  --ckpt-steps 100 \
  --hf-save-steps 1000
```

Useful flags:

- `--resume`: Resume from `./emergency_ckpt_jetson_gpt3_v2`.
- `--skip-val`: Disable validation.
- `--skip-gen`: Disable periodic generation previews.
- `--opt-state-on-cpu`: Move optimizer state to CPU (lower VRAM, slower).
- `--total-tokens` and `--warmup-tokens`: Token-based run length control.

By default, the script trains a ~125M model (`12 layers`, `768 hidden`, context block `512`) on streaming FineWeb.

### 2) Evaluate A Checkpoint

```bash
python3 eval.py \
  --model k050506koch/GPT3-dev-125m-1202 \
  --run-hellaswag \
  --run-lambada \
  --run-mmlu \
  --mmlu-tasks abstract_algebra anatomy \
  --batch-size 2 \
  --block-size 128 \
  --trust-remote-code
```

You can pass either a local checkpoint path or a Hugging Face repo id.

### 3) Run SFT Workflow

```bash
python3 sft.py
```

Current defaults in `sft.py`:

- Loads base model/tokenizer from `./gpt3-small-jetson-v2`.
- Adds Harmony-style special tokens and chat template.
- Builds a filtered multi-dataset instruction corpus.
- Saves SFT outputs to `./gpt3-small-jetson-v2-sft`.

If your base checkpoint path is different, edit it near the top of `sft.py`.

### 4) Minimal Inference

Edit `local_path` in `inference_simple.py`, then run:

```bash
python3 inference_simple.py
```

## Pretrained Checkpoints

Public checkpoints (all of them use v1 architecture):

- `k050506koch/GPT3-dev` (17M)
- `k050506koch/GPT3-dev-125m`
- `k050506koch/GPT3-dev-125m-0612`
- `k050506koch/GPT3-dev-125m-1202`

Because this is a custom architecture, load with `trust_remote_code=True` when using Auto classes.

## Notes And Caveats

- This is an actively evolving v2 codebase, so errors / inconsistencies may occur, but I will be happy if you open an error or create a PR.
- `train.py` is tuned for memory-constrained CUDA setups (i use Nvidia Jetson Orin Nano Super, it has 8GB of unified memory. 

> “And in those days many a clever programmer derived an immense intellectual satisfaction from the cunning tricks by which he contrived to squeeze the impossible into the constraints of his equipment.”
).
- `sft.py` currently has hardcoded paths/settings intended for local experimentation; adjust before large runs.
- `requirements.txt` is pinned to my working environment and includes more than strictly minimal runtime dependencies.

## License

MIT because it's true open-source. See `LICENSE`.

## Acknowledgements

Thanks OpenAI, HuggingFace and Pytorch for making this project possible!

- [OpenAI GPT-3 paper](https://arxiv.org/abs/2005.14165)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [PyTorch](https://pytorch.org/)
