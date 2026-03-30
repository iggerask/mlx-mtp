#!/usr/bin/env python3
"""Profile where Python overhead goes in a single MTP step."""

import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.utils
from huggingface_hub import snapshot_download
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

from vllm_mlx_mtp.hidden_capture import HiddenStateCapture
from vllm_mlx_mtp.mtp_head import build_mtp_head, load_mtp_weights_from_file
from vllm_mlx_mtp.optimizations import quantize_mtp_head
from vllm_mlx_mtp.gdn_capture import GDNStateCapture
from mlx_fused_moe.patch_moe_full import patch_moe_full

MODEL_NAME = "mlx-community/Qwen3.5-35B-A3B-4bit"
BF16_SOURCE = "Qwen/Qwen3.5-35B-A3B"
MTP_WEIGHTS = Path("mtp_weights/Qwen_Qwen3.5-35B-A3B.safetensors")

model, tokenizer = load(MODEL_NAME)
patch_moe_full(model, verbose=False)

model_path = Path(snapshot_download(BF16_SOURCE, allow_patterns=["config.json"]))
with open(model_path / "config.json") as f:
    config = json.load(f)
weights = load_mtp_weights_from_file(MTP_WEIGHTS)
mtp_head = build_mtp_head(weights, config, norm_shift=True)
quantize_mtp_head(mtp_head, bits=4, group_size=64)

capture = HiddenStateCapture(model)
gdn_cap = GDNStateCapture(model)
gdn_cap.patch()

# Get LM head refs
lm_head = model.language_model.lm_head
embed_fn = model.language_model.model.embed_tokens

prompt = "Write a Python function that implements merge sort:"
prompt_arr = mx.array(tokenizer.encode(prompt))
cache = make_prompt_cache(model)
logits = model(prompt_arr[None], cache=cache)
hidden = capture.get_hidden_state()
mx.eval(logits, hidden, *[c.state for c in cache if hasattr(c, "state")])

token_0 = mx.argmax(logits[:, -1, :], axis=-1)
mx.eval(token_0)
last_hidden = hidden[:, -1:, :]

def sample(logits, temp=0.0):
    return mx.argmax(logits, axis=-1)

# Warmup
for _ in range(5):
    tok_embed = embed_fn(token_0[None])
    if tok_embed.ndim == 2:
        tok_embed = tok_embed[:, None, :]
    mtp_h = mtp_head(last_hidden, tok_embed)
    mtp_logits = lm_head(mtp_h)
    draft = sample(mtp_logits[:, -1, :])
    gdn_cap.prepare(cache)
    verify_input = mx.concatenate([token_0.reshape(1, 1), draft.reshape(1, 1)], axis=1)
    verify_logits = model(verify_input, cache=cache)
    verify_hidden = capture.get_hidden_state()
    verified = sample(verify_logits[:, 0, :])
    bonus = sample(verify_logits[:, 1, :])
    ims = gdn_cap.get_intermediates()
    gdn_cap.disable()
    mx.eval(draft, verified, bonus, verify_hidden, *ims)
    # Restore cache
    gdn_cap.restore(cache, position=0, n_kv_trim=1)
    last_hidden = verify_hidden[:, 0:1, :]

# Profile individual operations
N = 50
timings = {}

def time_op(name, fn, n=N):
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    avg = sum(sorted(times)[:n//2]) / (n//2) * 1000  # best half average, ms
    timings[name] = avg

# 1. Graph construction (Python overhead only, no eval)
def build_graph():
    tok_embed = embed_fn(token_0[None])
    if tok_embed.ndim == 2:
        tok_embed = tok_embed[:, None, :]
    mtp_h = mtp_head(last_hidden, tok_embed)
    mtp_logits = lm_head(mtp_h)
    draft = sample(mtp_logits[:, -1, :])
    verify_input = mx.concatenate([token_0.reshape(1, 1), draft.reshape(1, 1)], axis=1)
    verify_logits = model(verify_input, cache=cache)
    verify_hidden = capture.get_hidden_state()
    verified = sample(verify_logits[:, 0, :])
    bonus = sample(verify_logits[:, 1, :])
    return draft, verified, bonus, verify_hidden

# Note: can't measure graph construction without accumulating dead graph nodes
# Instead, measure the FULL step including eval

# Full step (no ZR)
def full_step_no_zr():
    tok_embed = embed_fn(token_0[None])
    if tok_embed.ndim == 2:
        tok_embed = tok_embed[:, None, :]
    mtp_h = mtp_head(last_hidden, tok_embed)
    mtp_logits = lm_head(mtp_h)
    draft = sample(mtp_logits[:, -1, :])
    verify_input = mx.concatenate([token_0.reshape(1, 1), draft.reshape(1, 1)], axis=1)
    verify_logits = model(verify_input, cache=cache)
    verify_hidden = capture.get_hidden_state()
    verified = sample(verify_logits[:, 0, :])
    bonus = sample(verify_logits[:, 1, :])
    mx.eval(draft, verified, bonus, verify_hidden)
    d = draft.item()
    v = verified.item()
    b = bonus.item()
    gdn_cap.restore(cache, position=0, n_kv_trim=1)

time_op("full_step_no_zr", full_step_no_zr)

# Full step with ZR
def full_step_zr():
    gdn_cap.prepare(cache)
    tok_embed = embed_fn(token_0[None])
    if tok_embed.ndim == 2:
        tok_embed = tok_embed[:, None, :]
    mtp_h = mtp_head(last_hidden, tok_embed)
    mtp_logits = lm_head(mtp_h)
    draft = sample(mtp_logits[:, -1, :])
    verify_input = mx.concatenate([token_0.reshape(1, 1), draft.reshape(1, 1)], axis=1)
    verify_logits = model(verify_input, cache=cache)
    verify_hidden = capture.get_hidden_state()
    verified = sample(verify_logits[:, 0, :])
    bonus = sample(verify_logits[:, 1, :])
    ims = gdn_cap.get_intermediates()
    gdn_cap.disable()
    mx.eval(draft, verified, bonus, verify_hidden, *ims)
    d = draft.item()
    v = verified.item()
    b = bonus.item()
    gdn_cap.restore(cache, position=0, n_kv_trim=1)

time_op("full_step_zr", full_step_zr)

# Just eval (graph already built)
draft_g, verified_g, bonus_g, hidden_g = build_graph()
def just_eval():
    mx.eval(draft_g, verified_g, bonus_g, hidden_g)
# Can only run once since eval consumes the graph
t0 = time.perf_counter()
just_eval()
timings["single_eval"] = (time.perf_counter() - t0) * 1000

# .item() cost
mx.eval(token_0)
def item_calls():
    token_0.item()
    token_0.item()
    token_0.item()
time_op("3x_item", item_calls)

# Cache restore cost
def restore_only():
    gdn_cap.restore(cache, position=0, n_kv_trim=1)
time_op("cache_restore", restore_only)

# GDN prepare cost (no forward)
def prepare_only():
    gdn_cap.prepare(cache)
    gdn_cap.disable()
time_op("gdn_prepare", prepare_only)

print("\n=== Python Overhead Profile ===\n")
for name, ms in sorted(timings.items(), key=lambda x: -x[1]):
    print(f"  {name:<25} {ms:>7.3f} ms")

print(f"\n  Estimated Python overhead = full_step - GPU_work")
gpu_work = timings.get("single_eval", 0)
full = timings.get("full_step_zr", 0)
print(f"  GPU work (single eval):    {gpu_work:.3f} ms")
print(f"  Full step (ZR):            {full:.3f} ms")
print(f"  Python overhead:           {full - gpu_work:.3f} ms")
