#!/usr/bin/env python3
"""Measure exact zero-replay capture overhead on 2-token forward."""

import time
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from vllm_mlx_mtp.hidden_capture import HiddenStateCapture
from vllm_mlx_mtp.gdn_capture import GDNStateCapture
from mlx_fused_moe.patch_moe_full import patch_moe_full

MODEL_NAME = "mlx-community/Qwen3.5-35B-A3B-4bit"


def time_it(fn, warmup=5, iters=30):
    for _ in range(warmup):
        fn()
    mx.synchronize()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        mx.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return min(times), sorted(times)


print("Loading model...")
model, tokenizer = load(MODEL_NAME)
patch_moe_full(model, verbose=False)

capture = HiddenStateCapture(model)
gdn_cap = GDNStateCapture(model)
gdn_cap.patch()

prompt = "Hello world, this is a test of the system."
prompt_arr = mx.array(tokenizer.encode(prompt))

# Setup caches and get initial state
cache = make_prompt_cache(model)
logits = model(prompt_arr[None], cache=cache)
hidden = capture.get_hidden_state()
mx.eval(logits, hidden, *[c.state for c in cache if hasattr(c, "state")])
tok = mx.argmax(logits[:, -1, :], axis=-1)
mx.eval(tok)

input_2 = mx.array([[tok.item(), tok.item()]])

# 1. Raw 2-token forward (no capture)
def run_no_capture():
    out = model(input_2, cache=cache)
    h = capture.get_hidden_state()
    mx.eval(out, h)

ms_nc, _ = time_it(run_no_capture)
print(f"2-tok forward (no ZR capture): {ms_nc:.2f}ms")

# 2. 2-token forward WITH capture (GDN split)
def run_with_capture():
    gdn_cap.prepare(cache)
    out = model(input_2, cache=cache)
    h = capture.get_hidden_state()
    intermediates = gdn_cap.get_intermediates()
    mx.eval(out, h, *intermediates)
    gdn_cap.disable()

ms_wc, _ = time_it(run_with_capture)
print(f"2-tok forward (with ZR capture): {ms_wc:.2f}ms")
print(f"ZR capture overhead: {ms_wc - ms_nc:.2f}ms")

# 3. Measure restore cost
from mlx_lm.models.cache import ArraysCache, KVCache
gdn_cap.prepare(cache)
out = model(input_2, cache=cache)
intermediates = gdn_cap.get_intermediates()
mx.eval(out, *intermediates)
gdn_cap.disable()

times_restore = []
for _ in range(1000):
    t0 = time.perf_counter()
    gdn_cap.restore(cache, position=0, n_kv_trim=1)
    times_restore.append((time.perf_counter() - t0) * 1e6)
print(f"ZR restore (Python only): {min(times_restore):.0f}μs")

# 4. Measure prepare cost (conv copy + eval)
times_prepare = []
for _ in range(100):
    t0 = time.perf_counter()
    gdn_cap.prepare(cache)
    gdn_cap.disable()
    times_prepare.append((time.perf_counter() - t0) * 1e6)
print(f"ZR prepare (no forward): {min(times_prepare):.0f}μs")

# 5. Full step simulation with timing
print("\n=== Full step simulation ===")
embed_fn = model.language_model.model.embed_tokens
lm_head = model.language_model.lm_head

from vllm_mlx_mtp.mtp_head import build_mtp_head, load_mtp_weights_from_file
from vllm_mlx_mtp.optimizations import quantize_mtp_head
from huggingface_hub import snapshot_download
from pathlib import Path
import json

bf16_dir = Path(snapshot_download("Qwen/Qwen3.5-35B-A3B", allow_patterns=["config.json"]))
cfg = json.loads((bf16_dir / "config.json").read_text())
weights = load_mtp_weights_from_file(Path("mtp_weights/Qwen_Qwen3.5-35B-A3B.safetensors"))
mtp_head = build_mtp_head(weights, cfg, norm_shift=True)
quantize_mtp_head(mtp_head, bits=4, group_size=64)
mx.eval(mtp_head.parameters())

last_h = hidden[:, -1:, :]
tok_arr = tok

# Full step: prepare + draft + verify + eval + restore
def run_full_step():
    # Draft (lazy)
    tok_embed = embed_fn(tok_arr[None])
    if tok_embed.ndim == 2:
        tok_embed = tok_embed[:, None, :]
    mtp_h = mtp_head(last_h, tok_embed)
    draft_logits = lm_head(mtp_h)
    draft = mx.argmax(draft_logits[:, -1, :], axis=-1)

    # Prepare ZR
    gdn_cap.prepare(cache)

    # Verify
    verify_input = mx.concatenate([tok_arr.reshape(1, 1), draft.reshape(1, 1)], axis=1)
    vlogits = model(verify_input, cache=cache)
    vhidden = capture.get_hidden_state()
    verified = mx.argmax(vlogits[:, 0, :], axis=-1)
    bonus = mx.argmax(vlogits[:, 1, :], axis=-1)

    # Eval
    intermediates = gdn_cap.get_intermediates()
    mx.eval(draft, verified, bonus, vlogits, vhidden, *intermediates)
    gdn_cap.disable()

    # Restore (simulate reject)
    d = draft.item()
    v = verified.item()
    gdn_cap.restore(cache, position=0, n_kv_trim=1)

    return d, v, bonus.item()

ms_full, sorted_full = time_it(run_full_step, warmup=5, iters=30)
print(f"Full step (draft+verify+restore): {ms_full:.2f}ms")
print(f"  Median: {sorted_full[len(sorted_full)//2]:.2f}ms")
print(f"  vs raw 2-tok: +{ms_full - ms_nc:.2f}ms overhead")
