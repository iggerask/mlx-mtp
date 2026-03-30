#!/usr/bin/env python3
"""Profile per-call overhead in the MTP step to find what's eating the 2ms."""

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
cur_tok = token_0.reshape(1)

# Warmup
for _ in range(5):
    gdn_cap.prepare(cache)
    tok_embed = embed_fn(cur_tok[None])
    if tok_embed.ndim == 2:
        tok_embed = tok_embed[:, None, :]
    mtp_h = mtp_head(last_hidden, tok_embed)
    draft = mx.argmax(lm_head(mtp_h)[:, -1, :], axis=-1)
    verify_input = mx.concatenate([cur_tok.reshape(1, 1), draft.reshape(1, 1)], axis=1)
    verify_logits = model(verify_input, cache=cache)
    verify_hidden = capture.get_hidden_state()
    verified = mx.argmax(verify_logits[:, 0, :], axis=-1)
    bonus = mx.argmax(verify_logits[:, 1, :], axis=-1)
    ims = gdn_cap.get_intermediates()
    gdn_cap.disable()
    mx.eval(draft, verified, bonus, verify_hidden, *ims)
    gdn_cap.restore(cache, position=0, n_kv_trim=1)

# Now profile each call's Python overhead (time to BUILD the graph, not execute)
N = 100
cumulative = {}

for _ in range(N):
    t = {}

    t0 = time.perf_counter()
    gdn_cap.prepare(cache)
    t["prepare"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    tok_embed = embed_fn(cur_tok[None])
    if tok_embed.ndim == 2:
        tok_embed = tok_embed[:, None, :]
    t["embed"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    mtp_h = mtp_head(last_hidden, tok_embed)
    t["mtp_head"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    mtp_logits = lm_head(mtp_h)
    t["lm_head_draft"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    draft = mx.argmax(mtp_logits[:, -1, :], axis=-1)
    t["argmax_draft"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    verify_input = mx.concatenate([cur_tok.reshape(1, 1), draft.reshape(1, 1)], axis=1)
    t["concat"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    verify_logits = model(verify_input, cache=cache)
    t["model_forward"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    verify_hidden = capture.get_hidden_state()
    t["get_hidden"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    verified = mx.argmax(verify_logits[:, 0, :], axis=-1)
    bonus = mx.argmax(verify_logits[:, 1, :], axis=-1)
    t["argmax_verify"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    ims = gdn_cap.get_intermediates()
    gdn_cap.disable()
    t["get_ims"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    mx.eval(draft, verified, bonus, verify_hidden, *ims)
    t["eval"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    d = draft.item()
    v = verified.item()
    b = bonus.item()
    t["items"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    gdn_cap.restore(cache, position=0, n_kv_trim=1)
    t["restore"] = time.perf_counter() - t0

    for k, v in t.items():
        cumulative.setdefault(k, []).append(v)

print("\n=== Per-Call Overhead (graph construction only, ms) ===\n")
total = 0
for k in ["prepare", "embed", "mtp_head", "lm_head_draft", "argmax_draft",
           "concat", "model_forward", "get_hidden", "argmax_verify",
           "get_ims", "eval", "items", "restore"]:
    vals = cumulative[k]
    avg = sum(sorted(vals)[:N//2]) / (N//2) * 1000
    total += avg
    print(f"  {k:<20} {avg:>7.3f} ms")
print(f"  {'TOTAL':<20} {total:>7.3f} ms")
