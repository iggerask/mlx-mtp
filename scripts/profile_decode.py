#!/usr/bin/env python3
"""Profile a single decode step to identify where time is spent."""

import time
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

MODEL_NAME = "mlx-community/Qwen3.5-35B-A3B-4bit"

def time_it(fn, name, warmup=3, iters=20):
    for _ in range(warmup):
        fn()
    mx.synchronize()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        mx.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    avg = sum(times) / len(times)
    mn = min(times)
    print(f"  {name:<40} avg={avg:.2f}ms  min={mn:.2f}ms")
    return mn

def main():
    print("Loading model...")
    model, tokenizer = load(MODEL_NAME)

    prompt = "Explain how transformers work in machine learning:"
    prompt_arr = mx.array(tokenizer.encode(prompt))
    cache = make_prompt_cache(model)
    logits = model(prompt_arr[None], cache=cache)
    mx.eval(logits)
    tok = mx.argmax(logits[:, -1, :], axis=-1)
    mx.eval(tok)

    print("\n=== Single token decode step breakdown ===\n")

    lm = model.language_model if hasattr(model, "language_model") else model
    layers = lm.model.layers if hasattr(lm, "model") else lm.layers
    input_tok = mx.array([[tok.item()]])

    def full_step():
        out = model(input_tok, cache=cache)
        mx.eval(out)

    time_it(full_step, "Full decode step", warmup=5, iters=30)

    embed = lm.model.embed_tokens if hasattr(lm, "model") else lm.embed_tokens
    h = embed(input_tok)
    mx.eval(h)

    # GDN layer 0
    gdn_layer = layers[0]
    time_it(lambda: (mx.eval(gdn_layer(h, mask=None, cache=cache[0]))), "Single GDN layer (layer 0)")

    # Attention layer 3
    attn_layer = layers[3]
    time_it(lambda: (mx.eval(attn_layer(h, mask=None, cache=cache[3]))), "Single Attention layer (layer 3)")

    # GDN sub-components
    gdn = gdn_layer.linear_attn
    time_it(lambda: mx.eval(gdn.in_proj_qkv(h), gdn.in_proj_z(h), gdn.in_proj_a(h), gdn.in_proj_b(h)),
            "  GDN: 4 input projections")
    time_it(lambda: mx.eval(gdn.out_proj(mx.zeros((1,1,gdn.num_v_heads*gdn.head_v_dim), dtype=h.dtype))),
            "  GDN: output projection")

    # MoE in GDN layer
    moe = getattr(gdn_layer, "mlp", None) or getattr(gdn_layer, "feed_forward", None)
    if moe:
        time_it(lambda: mx.eval(moe(h)), "  MoE/FFN in GDN layer")
        if hasattr(moe, "shared_expert"):
            time_it(lambda: mx.eval(moe.shared_expert(h.reshape(1,-1))), "  Shared expert only")

    # Attention sub-components
    attn = getattr(attn_layer, "self_attn", None) or getattr(attn_layer, "attn", None)
    if attn:
        time_it(lambda: mx.eval(attn.q_proj(h), attn.k_proj(h), attn.v_proj(h)),
                "  Attn: Q/K/V projections")
        n_dim = attn.num_attention_heads * attn.head_dim
        time_it(lambda: mx.eval(attn.o_proj(mx.zeros((1,1,n_dim), dtype=h.dtype))),
                "  Attn: output projection")

    # LM head
    head = getattr(lm, "lm_head", None) or getattr(lm, "head", None)
    if head:
        hdim = 2048  # Qwen3.5-35B-A3B hidden_size
        time_it(lambda: mx.eval(head(mx.zeros((1,1,hdim), dtype=h.dtype))),
                "LM head (vocab projection)")

    # Layer counts
    print("\n=== Layer counts ===")
    n_gdn = sum(1 for l in layers if hasattr(l, "linear_attn") and l.linear_attn is not None)
    n_attn = len(layers) - n_gdn
    print(f"  GDN layers: {n_gdn}, Attention layers: {n_attn}, Total: {len(layers)}")

    # Estimated breakdown
    print("\n=== Estimated time breakdown ===")
    gdn_t = time_it(lambda: mx.eval(gdn_layer(h, mask=None, cache=cache[0])), "GDN layer (final)", warmup=3, iters=20)
    attn_t = time_it(lambda: mx.eval(attn_layer(h, mask=None, cache=cache[3])), "Attn layer (final)", warmup=3, iters=20)
    full_t = time_it(full_step, "Full step (final)", warmup=3, iters=20)

    est_gdn = gdn_t * n_gdn
    est_attn = attn_t * n_attn
    overhead = full_t - est_gdn - est_attn

    print(f"\n  GDN:  {n_gdn} × {gdn_t:.2f}ms = {est_gdn:.2f}ms ({est_gdn/full_t*100:.0f}%)")
    print(f"  Attn: {n_attn} × {attn_t:.2f}ms = {est_attn:.2f}ms ({est_attn/full_t*100:.0f}%)")
    print(f"  Overhead: {overhead:.2f}ms ({overhead/full_t*100:.0f}%)")
    print(f"  Full step: {full_t:.2f}ms ({1000/full_t:.1f} tok/s)")

if __name__ == "__main__":
    main()
