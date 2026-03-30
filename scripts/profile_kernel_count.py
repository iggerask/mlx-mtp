#!/usr/bin/env python3
"""Count operations in a single decode step by exporting the computation graph.

Uses mx.export_to_dot to visualize and count the number of operations
(proxy for kernel dispatches) in one decode step.
"""

import json
import re
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

MODEL_NAME = "mlx-community/Qwen3.5-35B-A3B-4bit"


def count_graph_ops(model, tokenizer):
    """Export computation graph and count operations."""
    cache = make_prompt_cache(model)
    prompt_arr = mx.array(tokenizer.encode("Hello world"))

    # Prefill
    logits = model(prompt_arr[None], cache=cache)
    mx.eval(logits, *[c.state for c in cache if hasattr(c, "state")])

    token = mx.argmax(logits[:, -1, :], axis=-1)
    mx.eval(token)

    # Build one decode step graph without eval
    next_logits = model(mx.array([[token.item()]]), cache=cache)
    next_token = mx.argmax(next_logits[:, -1, :], axis=-1)

    # Export to DOT
    dot_path = "/tmp/decode_step.dot"
    mx.export_to_dot(dot_path, next_token)

    # Parse DOT file to count operations
    dot_content = Path(dot_path).read_text()

    # Count operation types from node labels
    op_counts = {}
    for match in re.finditer(r'label="([^"]+)"', dot_content):
        label = match.group(1)
        # Strip shape info
        op_name = label.split("\\n")[0] if "\\n" in label else label
        op_counts[op_name] = op_counts.get(op_name, 0) + 1

    return op_counts, dot_content


def profile_per_layer_timing(model, tokenizer):
    """Try to measure per-layer timing by running partial forward passes."""
    if hasattr(model, "language_model"):
        text_model = model.language_model
    else:
        text_model = model

    cache = make_prompt_cache(model)
    prompt_arr = mx.array(tokenizer.encode("Hello world"))
    logits = model(prompt_arr[None], cache=cache)
    mx.eval(logits, *[c.state for c in cache if hasattr(c, "state")])

    token = mx.array([[1]])

    # Time individual components
    # 1. Embedding lookup
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        emb = text_model.model.embed_tokens(token)
        mx.eval(emb)
    t1 = time.perf_counter()
    embed_ms = (t1 - t0) / 10 * 1000

    # 2. Full forward (for reference)
    cache2 = make_prompt_cache(model)
    logits2 = model(prompt_arr[None], cache=cache2)
    mx.eval(logits2, *[c.state for c in cache2 if hasattr(c, "state")])

    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        logits2 = model(mx.array([[1]]), cache=cache2)
        mx.eval(logits2)
    t1 = time.perf_counter()
    full_ms = (t1 - t0) / 10 * 1000

    # 3. Try to time the MoE block in isolation
    layers = text_model.model.layers
    moe_block = layers[0].block_sparse_moe if hasattr(layers[0], "block_sparse_moe") else None

    moe_ms = None
    if moe_block is not None:
        # Create a dummy hidden state
        hidden_size = text_model.args.hidden_size
        dummy_h = mx.random.normal((1, 1, hidden_size))
        mx.eval(dummy_h)

        mx.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            out = moe_block(dummy_h)
            mx.eval(out)
        t1 = time.perf_counter()
        moe_ms = (t1 - t0) / 10 * 1000

    # 4. Time the shared expert alone
    shared_expert_ms = None
    if moe_block is not None and hasattr(moe_block, "shared_expert"):
        dummy_h = mx.random.normal((1, 1, hidden_size))
        mx.eval(dummy_h)

        mx.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            out = moe_block.shared_expert(dummy_h)
            mx.eval(out)
        t1 = time.perf_counter()
        shared_expert_ms = (t1 - t0) / 10 * 1000

    # 5. Time just the router
    router_ms = None
    if moe_block is not None:
        dummy_h = mx.random.normal((1, 1, hidden_size))
        mx.eval(dummy_h)

        mx.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            gates = moe_block.gate(dummy_h)
            gates = mx.softmax(gates, axis=-1, precise=True)
            k = moe_block.top_k
            inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
            scores = mx.take_along_axis(gates, inds, axis=-1)
            mx.eval(inds, scores)
        t1 = time.perf_counter()
        router_ms = (t1 - t0) / 10 * 1000

    # 6. Time gather_mm (SwitchGLU) alone
    switch_ms = None
    if moe_block is not None:
        dummy_h = mx.random.normal((1, 1, hidden_size))
        mx.eval(dummy_h)

        # Get real indices from router
        gates = moe_block.gate(dummy_h)
        gates = mx.softmax(gates, axis=-1, precise=True)
        k = moe_block.top_k
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        mx.eval(inds)

        mx.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            out = moe_block.switch_mlp(dummy_h, inds)
            mx.eval(out)
        t1 = time.perf_counter()
        switch_ms = (t1 - t0) / 10 * 1000

    return {
        "embed_ms": embed_ms,
        "full_step_ms": full_ms,
        "moe_block_ms": moe_ms,
        "shared_expert_ms": shared_expert_ms,
        "router_ms": router_ms,
        "switch_mlp_ms": switch_ms,
    }


def main():
    print("Loading model...")
    model, tokenizer = load(MODEL_NAME)

    # Warmup
    cache = make_prompt_cache(model)
    logits = model(mx.array(tokenizer.encode("Hi"))[None], cache=cache)
    mx.eval(logits)

    # Count operations
    print("\n" + "=" * 60)
    print("Computation Graph Analysis")
    print("=" * 60)

    op_counts, dot_content = count_graph_ops(model, tokenizer)

    total_ops = sum(op_counts.values())
    print(f"\nTotal operations in one decode step: {total_ops}")
    print(f"\nOperation breakdown (top 20):")
    for op, count in sorted(op_counts.items(), key=lambda x: -x[1])[:20]:
        print(f"  {op:40s}: {count:4d}")

    # Save DOT for analysis
    Path("/tmp/decode_step.dot").write_text(dot_content)
    print(f"\nFull graph saved to /tmp/decode_step.dot ({len(dot_content)} bytes)")

    # Per-component timing
    print("\n" + "=" * 60)
    print("Per-Component Timing (10-run average)")
    print("=" * 60)

    timings = profile_per_layer_timing(model, tokenizer)
    for name, ms in timings.items():
        if ms is not None:
            print(f"  {name:25s}: {ms:.2f}ms")

    # Derived metrics
    if timings["moe_block_ms"] and timings["full_step_ms"]:
        moe_pct = timings["moe_block_ms"] / timings["full_step_ms"] * 100
        print(f"\n  MoE block as % of full step: {moe_pct:.0f}%")
        print(f"  MoE blocks × 40 layers:      {timings['moe_block_ms'] * 40:.1f}ms (estimated)")
        print(f"  Non-MoE overhead:             {timings['full_step_ms'] - timings['moe_block_ms'] * 40:.1f}ms")

    if timings["router_ms"] and timings["switch_mlp_ms"] and timings["shared_expert_ms"]:
        total_moe = timings["router_ms"] + timings["switch_mlp_ms"] + timings["shared_expert_ms"]
        print(f"\n  MoE breakdown:")
        print(f"    Router (gate + topk):   {timings['router_ms']:.2f}ms ({timings['router_ms']/total_moe*100:.0f}%)")
        print(f"    SwitchGLU (gather_mm):  {timings['switch_mlp_ms']:.2f}ms ({timings['switch_mlp_ms']/total_moe*100:.0f}%)")
        print(f"    Shared expert:          {timings['shared_expert_ms']:.2f}ms ({timings['shared_expert_ms']/total_moe*100:.0f}%)")
        print(f"    Sum:                    {total_moe:.2f}ms vs measured MoE: {timings['moe_block_ms']:.2f}ms")


if __name__ == "__main__":
    main()
