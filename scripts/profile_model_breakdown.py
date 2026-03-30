#!/usr/bin/env python3
"""Profile model forward: backbone vs LM head, and per-layer costs."""

import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.utils
from huggingface_hub import snapshot_download
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from mlx_fused_moe.patch_moe_full import patch_moe_full

MODEL_NAME = "mlx-community/Qwen3.5-35B-A3B-4bit"

model, tokenizer = load(MODEL_NAME)
patch_moe_full(model, verbose=False)

text_model = model.language_model
backbone = text_model.model
lm_head = text_model.lm_head if not text_model.args.tie_word_embeddings else backbone.embed_tokens.as_linear

prompt = "Hello world, this is a test"
prompt_arr = mx.array(tokenizer.encode(prompt))
cache = make_prompt_cache(model)

# Prefill
logits = model(prompt_arr[None], cache=cache)
mx.eval(logits, *[c.state for c in cache if hasattr(c, "state")])

# Create 2-token input (like MTP verify)
tok1 = mx.array([[1, 2]])  # dummy tokens

N = 30

# 1. Full model (backbone + LM head)
def full_model():
    return model(tok1, cache=cache)

times = []
for _ in range(N):
    t0 = time.perf_counter()
    out = full_model()
    mx.eval(out)
    times.append(time.perf_counter() - t0)
    # Trim cache to keep it stable
    for c in cache:
        if hasattr(c, 'offset'):
            c.offset = max(0, c.offset - 2)
full_ms = sum(sorted(times)[:N//2]) / (N//2) * 1000

# 2. Backbone only (no LM head)
def backbone_only():
    return backbone(tok1, cache)

times = []
for _ in range(N):
    t0 = time.perf_counter()
    h = backbone_only()
    mx.eval(h)
    times.append(time.perf_counter() - t0)
    for c in cache:
        if hasattr(c, 'offset'):
            c.offset = max(0, c.offset - 2)
backbone_ms = sum(sorted(times)[:N//2]) / (N//2) * 1000

# 3. LM head on 2 positions
hidden_2tok = mx.random.normal((1, 2, backbone.layers[0].input_layernorm.weight.shape[0]))
mx.eval(hidden_2tok)

times = []
for _ in range(N):
    t0 = time.perf_counter()
    logits = lm_head(hidden_2tok)
    mx.eval(logits)
    times.append(time.perf_counter() - t0)
lm_head_2_ms = sum(sorted(times)[:N//2]) / (N//2) * 1000

# 4. LM head on 1 position
hidden_1tok = hidden_2tok[:, :1, :]

times = []
for _ in range(N):
    t0 = time.perf_counter()
    logits = lm_head(hidden_1tok)
    mx.eval(logits)
    times.append(time.perf_counter() - t0)
lm_head_1_ms = sum(sorted(times)[:N//2]) / (N//2) * 1000

# 5. 1-token full model
tok1_single = mx.array([[1]])

times = []
for _ in range(N):
    t0 = time.perf_counter()
    out = model(tok1_single, cache=cache)
    mx.eval(out)
    times.append(time.perf_counter() - t0)
    for c in cache:
        if hasattr(c, 'offset'):
            c.offset = max(0, c.offset - 1)
single_ms = sum(sorted(times)[:N//2]) / (N//2) * 1000

# 6. Count dispatches: measure overhead of mx.eval on a trivial graph
x = mx.ones((1,))
times = []
for _ in range(N):
    y = x + 1
    t0 = time.perf_counter()
    mx.eval(y)
    times.append(time.perf_counter() - t0)
trivial_eval_us = sum(sorted(times)[:N//2]) / (N//2) * 1e6

print(f"\n=== Model Forward Breakdown (2-token verify) ===\n")
print(f"  Full model (2 tok):      {full_ms:>7.2f} ms")
print(f"  Backbone only (2 tok):   {backbone_ms:>7.2f} ms")
print(f"  LM head (2 positions):   {lm_head_2_ms:>7.2f} ms")
print(f"  LM head (1 position):    {lm_head_1_ms:>7.2f} ms")
print(f"  Sum (backbone + lm×2):   {backbone_ms + lm_head_2_ms:>7.2f} ms")
print(f"  Full model (1 tok):      {single_ms:>7.2f} ms")
print(f"  Incremental 2nd token:   {full_ms - single_ms:>7.2f} ms")
print(f"  Trivial eval overhead:   {trivial_eval_us:>7.1f} μs")
print(f"\n  LM head cost (2tok):     {full_ms - backbone_ms:>7.2f} ms")
print(f"  LM head fraction:        {(full_ms - backbone_ms) / full_ms * 100:>6.1f}%")
