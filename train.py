import os, math, argparse, shutil, gc
from contextlib import nullcontext
import time as time_module  # avoid conflict with time.sleep import

# === Memory allocator settings for Jetson unified memory ===
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
# Reduce Python's memory fragmentation
os.environ["MALLOC_TRIM_THRESHOLD_"] = "65536"

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from transformers import GPT2TokenizerFast, get_cosine_schedule_with_warmup
from datasets import load_dataset, interleave_datasets
from lion_pytorch import Lion
import modeling_gpt3dev
import wandb
import datetime

P = argparse.ArgumentParser()
P.add_argument('--resume', action='store_true')
P.add_argument('--compile', action='store_true')
P.add_argument('--shuffle-buffer-size', type=int, default=1024)
P.add_argument('--log-steps', type=int, default=1)
P.add_argument('--eval-steps', type=int, default=500)
P.add_argument('--ckpt-steps', type=int, default=100)
P.add_argument('--hf-save-steps', type=int, default=1000)
P.add_argument('--gen-steps', type=int, default=500)
P.add_argument('--local_rank', type=int, default=0) # for compatibility with torch.distributed.launch
P.add_argument('--node_rank', type=int, default=0)  # for compatibility with torch.distributed.launch
P.add_argument('--world_size', type=int, default=1)  # for compatibility with torch.distributed.launch
P.add_argument('--master_addr', type=str, default='localhost')  # for compatibility with torch.distributed.launch
P.add_argument('--master_port', type=int, default=12345)  # for compatibility with torch.distributed.launch
P.add_argument('--dist_backend', type=str, default='nccl')  # for compatibility with torch.distributed.launch
P.add_argument('--empty_cache_steps', type=int, default=0)
P.add_argument('--acc', type=int, default=256, help='Gradient accumulation steps (lower = less RAM)')
P.add_argument('--skip-val', action='store_true', help='Skip validation to save RAM')
P.add_argument('--skip-gen', action='store_true', help='Skip generation to save RAM')
P.add_argument('--opt-state-on-cpu', action='store_true', help='Move optimizer state to CPU to reduce VRAM (slower)')
P.add_argument('--total-tokens', type=int, default=None, help='Total training tokens (overrides step-based total)')
P.add_argument('--warmup-tokens', type=int, default=None, help='Warmup tokens (overrides step-based warmup)')
P.add_argument('--fw-val-steps', type=int, default=20, help='FineWeb validation batches per eval')
P.add_argument('--wt-val-steps', type=int, default=20, help='WikiText validation batches per eval')
args = P.parse_args()

CKPT_DIR = './emergency_ckpt_jetson_gpt3_v2'; os.makedirs(CKPT_DIR, exist_ok=True)
RUN_ID_FILE = f'{CKPT_DIR}/run_id.txt'

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True  # Optimize cuDNN kernels for fixed input sizes

torch.cuda.set_per_process_memory_fraction(0.95)  # Leave some headroom

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
is_cuda = device.type == 'cuda'

bf16_ok = is_cuda and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
AMP_ENABLED = is_cuda
AMP_DTYPE = torch.bfloat16 if bf16_ok else torch.float16
print(f"AMP enabled={AMP_ENABLED} dtype={AMP_DTYPE} (bf16_ok={bf16_ok})")

def amp_autocast():
    if not AMP_ENABLED:
        return nullcontext()
    return torch.amp.autocast(device_type=device.type, dtype=AMP_DTYPE, enabled=True)


if not is_cuda:
    torch.set_default_dtype(torch.bfloat16)

wb_settings = wandb.Settings(_disable_stats=True, code_dir='.', start_method='thread')
if args.resume and os.path.exists(RUN_ID_FILE):
    run_id = open(RUN_ID_FILE).read().strip()
    run = wandb.init(project='gpt3-small-fineweb-v2', id=run_id, resume='allow', settings=wb_settings)
else:
    run = wandb.init(project='gpt3-small-fineweb-v2', name=str(datetime.date.today()), settings=wb_settings)
    with open(RUN_ID_FILE, 'w') as f: f.write(run.id)
wandb.define_metric('step')

BLOCK = 512

_tok = GPT2TokenizerFast.from_pretrained('gpt2')
_tok.pad_token = _tok.eos_token
_tok.model_max_length = 2048

stream = interleave_datasets([
    load_dataset('HuggingFaceFW/fineweb', name='CC-MAIN-2024-51', split='train', streaming=True)
]).shuffle(buffer_size=args.shuffle_buffer_size, seed=42)

class Packed(IterableDataset):
    def __init__(self, src, tok, blk): self.src, self.tok, self.blk = src, tok, blk
    def __iter__(self):
        buf = []
        gc_counter = 0
        for ex in self.src:
            # Tokenize and immediately free text
            text = ex['text']
            ids = self.tok(text)['input_ids']
            del text, ex  # Free source text immediately
            buf.extend(ids)
            del ids
            while len(buf) >= self.blk:
                chunk = buf[:self.blk]
                buf = buf[self.blk:]
                t = torch.tensor(chunk, dtype=torch.long)
                del chunk
                yield {'input_ids': t, 'labels': t.clone()}
            # Periodic garbage collection in data loading
            gc_counter += 1
            if gc_counter >= 100:
                gc.collect()
                gc_counter = 0

train_loader = DataLoader(Packed(stream, _tok, BLOCK), batch_size=1, num_workers=0, pin_memory=False)

def make_val_loader():
    val_stream = load_dataset('wikitext', 'wikitext-2-raw-v1', split='validation', streaming=True) #because i had too many OOMs
    return DataLoader(Packed(val_stream, _tok, BLOCK), batch_size=1, num_workers=0, pin_memory=False)

def make_fw_val_loader():
    fw_stream = interleave_datasets([
        load_dataset('HuggingFaceFW/fineweb', name='CC-MAIN-2024-51', split='train', streaming=True)
    ]).shuffle(buffer_size=args.shuffle_buffer_size, seed=43)
    return DataLoader(Packed(fw_stream, _tok, BLOCK), batch_size=1, num_workers=0, pin_memory=False)

def make_model(vocab):
    cfg = modeling_gpt3dev.GPT3DevConfig( #125m params
        vocab_size=vocab,
        window_size=BLOCK,
        stride=128,
        n_positions=2048,
        n_ctx=2048,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        use_pre_layernorm=True,
        resid_pdrop=0.1,
        attn_pdrop=0.1,
        embd_pdrop=0.1,
    )
    cfg.auto_map = {
        "AutoConfig": "modeling_gpt3dev.GPT3DevConfig",
        "AutoModel": "modeling_gpt3dev.GPT3DevLMHeadModel",
        "AutoModelForCausalLM": "modeling_gpt3dev.GPT3DevLMHeadModel",
    }
    try: cfg.attn_implementation = "eager"
    except: pass
    cfg._attn_implementation = "eager"

    # Enable checkpointing with non-reentrant mode to reduce CUDA memory fragmentation
    model = modeling_gpt3dev.GPT3DevLMHeadModel(cfg)

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = model.to(device)
    print(f"Model has: {sum(p.numel() for p in model.parameters())} parameters")
    if args.compile: model = torch.compile(model, mode='reduce-overhead', backend='aot_eager')
    return model

print("getting model ready (v2)")
model = make_model(_tok.vocab_size)
print("ready")

# Gradient accumulation steps from args
# Effective tokens per optimizer step = ACC * batch_size * BLOCK

#TODO: move all flags and global vars together or read them from argparse
TOTAL_STEPS_DEFAULT, WARMUP_STEPS_DEFAULT = 600_000, 1000
ACC = args.acc
TOKENS_PER_STEP = ACC * 1 * BLOCK
if args.total_tokens is None:
    total_tokens = TOTAL_STEPS_DEFAULT * TOKENS_PER_STEP
else:
    total_tokens = int(args.total_tokens)
if args.warmup_tokens is None:
    warmup_tokens = WARMUP_STEPS_DEFAULT * TOKENS_PER_STEP
else:
    warmup_tokens = int(args.warmup_tokens)
TOTAL = max(1, math.ceil(total_tokens / TOKENS_PER_STEP))
WARMUP = max(1, math.ceil(warmup_tokens / TOKENS_PER_STEP))
print(f"Using gradient accumulation steps: {ACC}")
print(f"Tokens/step: {TOKENS_PER_STEP} | total_tokens: {total_tokens} | warmup_tokens: {warmup_tokens}")
print(f"Derived steps: total_steps={TOTAL} warmup_steps={WARMUP}")
opt = Lion(model.parameters(), lr=3e-4, betas=(0.9, 0.99), weight_decay=0.01)
sched = get_cosine_schedule_with_warmup(opt, WARMUP, TOTAL)

# FP16 needs GradScaler. BF16 does not.
scaler = torch.amp.GradScaler(
    enabled=is_cuda and (AMP_DTYPE == torch.float16),
    init_scale=2**16,
    growth_factor=2.0,
    backoff_factor=0.5,
    #growth_interval=2**30,  # effectively constant unless overflow -> downscale ### DEBUG ONLY
)

step = 0
def load_state():
    global step
    if args.resume and os.path.exists(f'{CKPT_DIR}/model.pt'):
        model.load_state_dict(torch.load(f'{CKPT_DIR}/model.pt'))
        opt.load_state_dict(torch.load(f'{CKPT_DIR}/opt.pt'))
        sched.load_state_dict(torch.load(f'{CKPT_DIR}/sched.pt'))
        if os.path.exists(f'{CKPT_DIR}/scaler.pt'):
            scaler.load_state_dict(torch.load(f'{CKPT_DIR}/scaler.pt'))
        step = int(open(f'{CKPT_DIR}/step.txt').read())
load_state()
if is_cuda: torch.cuda.empty_cache()

train_it = iter(train_loader); t0 = time_module.time()
opt_state_moved = False
try:
    while step < TOTAL:
        run_loss = 0.0; opt.zero_grad(set_to_none=True)
        try:
            for _ in range(ACC):
                batch = next(train_it); batch = {k:v.to(device) for k,v in batch.items()}
                with amp_autocast():
                    out = model(**batch, use_cache=False)
                    loss = out.loss / ACC
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                run_loss += loss.detach().float().item()
                del batch, out, loss  # Explicitly free memory
            # Only do optimizer step if no OOM occurred
            if scaler.is_enabled():
                scaler.unscale_(opt)
            grad_norm = float(nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            if scaler.is_enabled():
                prev_scale = scaler.get_scale()
                scaler.step(opt)
                scaler.update()
                # With huge growth_interval, scale only changes on overflow (downscale).
                if scaler.get_scale() < prev_scale:
                    opt.zero_grad(set_to_none=True)
                    if is_cuda: torch.cuda.empty_cache()
                    continue
            else:
                opt.step()
            sched.step(); step += 1
        except RuntimeError as e:
            if 'NVML_SUCCESS' in str(e) or 'out of memory' in str(e).lower():
                print(f"\n[OOM during training step {step}, attempting recovery...]")
                gc.collect()
                if is_cuda:
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                time_module.sleep(3.0) # sorry
                opt.zero_grad(set_to_none=True)
                continue  # Skip this step and try again
            else:
                raise

        # After first optimizer step, optionally move optimizer state to CPU to free VRAM
        if args.opt_state_on_cpu and not opt_state_moved:
            for state in opt.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to('cpu')
            opt_state_moved = True
            if is_cuda:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

        # Aggressive memory cleanup every step on memory-constrained systems
        if args.empty_cache_steps and step % args.empty_cache_steps == 0:
            if is_cuda: torch.cuda.empty_cache()
            gc.collect()

        if step % args.log_steps == 0:
            sps = (time_module.time() - t0) / args.log_steps; t0 = time_module.time(); lr = sched.get_last_lr()[0]
            ppl = math.exp(run_loss) if run_loss < 20 else float('inf')
            print(f"\rstep {step:>6} loss {run_loss:.4f} ppl {ppl:.1f} gn {grad_norm:.2f} {sps:.2f}s", end='', flush=True)
            wandb.log({'step':step,'train/loss':run_loss,'train/perplexity':ppl,'train/grad_norm':grad_norm,'train/lr':lr,'train/sec_per_step':sps, 'train/scaler_scale': scaler.get_scale()})
            if is_cuda: torch.cuda.empty_cache()

        if args.gen_steps and step % args.gen_steps == 0 and not args.skip_gen:
            model.eval()
            prompt = "He is a doctor. His main goal is"
            inputs = _tok(prompt, return_tensors="pt").to(device)
            try:
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=32,
                        do_sample=True,
                        temperature=0.7,
                        top_k=50,
                        pad_token_id=_tok.eos_token_id,
                        use_cache=True,
                    )
                print(f"\n[Gen step {step}] { _tok.decode(outputs[0], skip_special_tokens=True) }\n")
                wandb.log({'step':step,'train/text': wandb.Html(f"<p>{_tok.decode(outputs[0], skip_special_tokens=False)}</p>")})
                del outputs
            except Exception as e:
                print(f"\n[Gen step {step}] Generation failed: {e}")
            del inputs
            model.train()
            gc.collect()
            if is_cuda:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            time_module.sleep(1.0)  # let unified memory settle

        if step % args.eval_steps == 0 and not args.skip_val:
            # Force memory cleanup before validation
            gc.collect()
            if is_cuda:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            time_module.sleep(1.0)  # let unified memory settle
            
            model.eval(); v_loss=0.0; v_n=0
            try:
                val_loader = make_val_loader()
                with torch.no_grad():
                    for i, b in enumerate(val_loader):
                        if i >= args.wt_val_steps: break
                        b = {k: v.to(device) for k, v in b.items()}
                        with amp_autocast():
                            loss_t = model(**b, use_cache=False).loss
                        if not torch.isfinite(loss_t).item():
                            with torch.amp.autocast(device_type=device.type, enabled=False):
                                loss_t = model(**b, use_cache=False).loss
                        v_loss += loss_t.item(); v_n += 1
                        del b
                        if is_cuda: torch.cuda.empty_cache()
                del val_loader
                v_loss /= max(v_n, 1); v_ppl = math.exp(v_loss) if v_loss < 20 else float('inf')
                print(f" val loss {v_loss:.4f} ppl {v_ppl:.1f}")
                wandb.log({'step':step,'val/loss':v_loss,'val/perplexity':v_ppl})
            except Exception as e:
                print(f" val SKIPPED (OOM): {e}")
            model.train()
            gc.collect()
            if is_cuda:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            # FineWeb validation (train-like) to detect domain shift vs true overfit
            if args.fw_val_steps:
                fw_loss = 0.0; fw_n = 0
                try:
                    fw_loader = make_fw_val_loader()
                    with torch.no_grad():
                        for i, b in enumerate(fw_loader):
                            if i >= args.fw_val_steps: break
                            b = {k: v.to(device) for k, v in b.items()}
                            with amp_autocast():
                                loss_t = model(**b, use_cache=False).loss
                            if not torch.isfinite(loss_t).item():
                                with torch.amp.autocast(device_type=device.type, enabled=False):
                                    loss_t = model(**b, use_cache=False).loss
                            fw_loss += loss_t.item(); fw_n += 1
                            del b
                            if is_cuda: torch.cuda.empty_cache()
                    del fw_loader
                    fw_loss /= max(fw_n, 1); fw_ppl = math.exp(fw_loss) if fw_loss < 20 else float('inf')
                    print(f" fw val loss {fw_loss:.4f} ppl {fw_ppl:.1f}")
                    wandb.log({'step':step,'fw_val/loss':fw_loss,'fw_val/perplexity':fw_ppl})
                except Exception as e:
                    print(f" fw val SKIPPED (OOM): {e}")
                gc.collect()
                if is_cuda:
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()

        if step % args.ckpt_steps == 0:
            gc.collect()
            if is_cuda: torch.cuda.empty_cache()
            # Save to temp files first, then rename to avoid corruption
            torch.save(model.state_dict(), f'{CKPT_DIR}/model.pt.tmp')
            os.replace(f'{CKPT_DIR}/model.pt.tmp', f'{CKPT_DIR}/model.pt')
            torch.save(opt.state_dict(), f'{CKPT_DIR}/opt.pt.tmp')
            os.replace(f'{CKPT_DIR}/opt.pt.tmp', f'{CKPT_DIR}/opt.pt')
            torch.save(sched.state_dict(), f'{CKPT_DIR}/sched.pt.tmp')
            os.replace(f'{CKPT_DIR}/sched.pt.tmp', f'{CKPT_DIR}/sched.pt')
            torch.save(scaler.state_dict(), f'{CKPT_DIR}/scaler.pt.tmp')
            os.replace(f'{CKPT_DIR}/scaler.pt.tmp', f'{CKPT_DIR}/scaler.pt')
            with open(f'{CKPT_DIR}/step.txt','w') as f: f.write(str(step))
            with open(RUN_ID_FILE,'w') as f: f.write(run.id)
            print(f"\n[Checkpoint saved @{step}]")
            gc.collect()
            if is_cuda: torch.cuda.empty_cache()

        if step % args.hf_save_steps == 0 and step:
            # Aggressive cleanup before HF save
            gc.collect()
            if is_cuda:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            
            out = f'./gpt3-small-jetson-v2-step-{step}'
            model.save_pretrained(out); _tok.save_pretrained(out) # I also had OOMs here
            # Save cache-compatible modeling into the HF folder
            shutil.copy('modeling_gpt3dev.py', f'{out}/modeling_gpt3dev.py')
            print(f"\n[HF checkpoint @ step {step}] (v2 modeling)")
            
            gc.collect()
            if is_cuda:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                # Reset memory stats to help with fragmentation
                torch.cuda.reset_peak_memory_stats()
            time_module.sleep(2.0)  # i am also sorry for this, but it seems to help with memory fragmentation

    model.save_pretrained('./gpt3-small-jetson-v2'); _tok.save_pretrained('./gpt3-small-jetson-v2')
    shutil.copy('modeling_gpt3dev.py', './gpt3-small-jetson-modeling_gpt3dev.py') # BACKUP
except KeyboardInterrupt:
    print("\nInterrupted, saving …")
    torch.save(model.state_dict(),f'{CKPT_DIR}/model.pt')
    opt.zero_grad(set_to_none=True)
    torch.save(opt.state_dict(),f'{CKPT_DIR}/opt.pt')
    torch.save(sched.state_dict(),f'{CKPT_DIR}/sched.pt')
    torch.save(scaler.state_dict(), f'{CKPT_DIR}/scaler.pt')
    with open(f'{CKPT_DIR}/step.txt','w') as f: f.write(str(step))
    with open(RUN_ID_FILE,'w') as f: f.write(run.id)
    model.save_pretrained('./gpt3-small-jetson-v2'); _tok.save_pretrained('./gpt3-small-jetson-v2')
    shutil.copy('modeling_gpt3dev.py', './gpt3-small-jetson-modeling_gpt3dev.py')

wandb.finish()
exit(0)
