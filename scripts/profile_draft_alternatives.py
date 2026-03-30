#!/usr/bin/env python3
"""Profile alternative draft strategies to find one cheaper than 1.9ms LM head."""

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
text_model = model.language_model
lm_head = text_model.lm_head if not text_model.args.tie_word_embeddings else text_model.model.embed_tokens.as_linear
embed_fn = text_model.model.embed_tokens

prompt = "Write a Python function that implements merge sort with type hints"
prompt_arr = mx.array(tokenizer.encode(prompt))
cache = make_prompt_cache(model)
logits = model(prompt_arr[None], cache=cache)
hidden = capture.get_hidden_state()
mx.eval(logits, hidden, *[c.state for c in cache if hasattr(c, "state")])

last_hidden = hidden[:, -1:, :]
cur_tok = mx.argmax(logits[:, -1, :], axis=-1).reshape(1)
mx.eval(cur_tok)

# Get MTP hidden state for draft
tok_embed = embed_fn(cur_tok[None])
if tok_embed.ndim == 2:
    tok_embed = tok_embed[:, None, :]
mtp_h = mtp_head(last_hidden, tok_embed)
mx.eval(mtp_h)

N = 50

def bench(name, fn):
    # Warmup
    for _ in range(5):
        r = fn()
        mx.eval(r) if not isinstance(r, (list, tuple)) else mx.eval(*r)
    times = []
    for _ in range(N):
        t0 = time.perf_counter()
        r = fn()
        mx.eval(r) if not isinstance(r, (list, tuple)) else mx.eval(*r)
        times.append(time.perf_counter() - t0)
    avg = sum(sorted(times)[:N//2]) / (N//2) * 1000
    print(f"  {name:<40} {avg:>7.3f} ms")
    return avg

print("\n=== Draft Strategy Alternatives ===\n")

# 1. Full LM head (baseline)
bench("Full LM head (248K)", lambda: mx.argmax(lm_head(mtp_h)[:, -1, :], axis=-1))

# 2. Top-K cached draft (K=512)
top_k_indices = mx.argpartition(-logits[:, -1, :].reshape(-1), kth=512)[:512]
mx.eval(top_k_indices)
W = lm_head["weight"]
S = lm_head["scales"]
B = lm_head["biases"]
gs = lm_head.group_size
bits = lm_head.bits

def topk_draft():
    sub_W = W[top_k_indices]
    sub_S = S[top_k_indices]
    sub_B = B[top_k_indices]
    h = mtp_h.reshape(-1, mtp_h.shape[-1])
    sub_logits = mx.quantized_matmul(h, sub_W, sub_S, sub_B, transpose=True, group_size=gs, bits=bits)
    return top_k_indices[mx.argmax(sub_logits, axis=-1)]

bench("Top-K 512 Q4 matmul", topk_draft)

# 3. Top-K with float16 weights (extract and use dense matmul)
# Dequantize the top-K weight slice
sub_W_f16 = mx.dequantize(W[top_k_indices], S[top_k_indices], B[top_k_indices], gs, bits)
mx.eval(sub_W_f16)

def topk_f16():
    h = mtp_h.reshape(-1, mtp_h.shape[-1])
    logits = h @ sub_W_f16.T
    return top_k_indices[mx.argmax(logits, axis=-1)]

bench("Top-K 512 f16 matmul", topk_f16)

# 4. Embedding similarity (cosine with embedding table)
# For each candidate, compute dot product of mtp_h with embedding vector
embed_W = text_model.model.embed_tokens.weight  # (248320, 4096) at Q4
mx.eval(embed_W) if hasattr(embed_W, 'shape') else None

def embed_similarity():
    sub_E = embed_W[top_k_indices]  # (512, ...) Q4
    # Dequantize
    sub_E_f = mx.dequantize(sub_E,
        text_model.model.embed_tokens.scales[top_k_indices],
        text_model.model.embed_tokens.biases[top_k_indices],
        text_model.model.embed_tokens.group_size,
        text_model.model.embed_tokens.bits)
    h = mtp_h.reshape(-1, mtp_h.shape[-1])
    sims = h @ sub_E_f.T
    return top_k_indices[mx.argmax(sims, axis=-1)]

bench("Embed similarity 512", embed_similarity)

# 5. Just argmax on last step's logits (no MTP head needed)
prev_logits = logits[:, -1, :]
def greedy_repeat():
    return mx.argmax(prev_logits, axis=-1)
bench("Repeat last argmax (free)", greedy_repeat)

# 6. Smaller K values
for K in [64, 128, 256]:
    top_k_K = mx.argpartition(-logits[:, -1, :].reshape(-1), kth=K)[:K]
    mx.eval(top_k_K)
    sub_W_K = mx.dequantize(W[top_k_K], S[top_k_K], B[top_k_K], gs, bits)
    mx.eval(sub_W_K)

    def topk_f16_K(indices=top_k_K, weights=sub_W_K):
        h = mtp_h.reshape(-1, mtp_h.shape[-1])
        logits = h @ weights.T
        return indices[mx.argmax(logits, axis=-1)]

    bench(f"Top-K {K} f16 matmul", topk_f16_K)

# 7. Pre-computed f16 LM head slice (512 candidates, weights cached)
# This is the ideal case: weights already dequantized and cached
sub_W_cached = mx.dequantize(W[top_k_indices], S[top_k_indices], B[top_k_indices], gs, bits)
mx.eval(sub_W_cached)

def topk_cached_f16():
    h = mtp_h.reshape(-1, mtp_h.shape[-1])
    logits = h @ sub_W_cached.T
    return top_k_indices[mx.argmax(logits, axis=-1)]

bench("Top-K 512 CACHED f16 matmul", topk_cached_f16)

# Profile the indexing cost separately
def just_index():
    return W[top_k_indices], S[top_k_indices], B[top_k_indices]
bench("Weight indexing only (512)", just_index)
