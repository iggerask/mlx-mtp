#!/usr/bin/env python3
"""Benchmark mx.compile on the full decode step.

Previous tests showed mx.compile had no impact on GPU execution time.
This test specifically measures whether it reduces Python graph build time (1.6ms).
"""

import time
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

MODEL_NAME = "mlx-community/Qwen3.5-35B-A3B-4bit"
NUM_TOKENS = 40


def benchmark_decode(model, tokenizer, label, compile_fn=None):
    """Benchmark decode with optional compilation."""
    cache = make_prompt_cache(model)
    prompt = mx.array(tokenizer.encode("Hello world, this is a test of"))

    # Prefill
    logits = model(prompt[None], cache=cache)
    mx.eval(logits, *[c.state for c in cache if hasattr(c, "state")])
    token = mx.argmax(logits[:, -1, :], axis=-1)
    mx.eval(token)

    if compile_fn:
        step_fn = compile_fn
    else:
        def step_fn(tok, c):
            return model(tok, cache=c)
        step_fn = lambda tok: model(tok, cache=cache)

    # Warmup (important for compile to JIT)
    for _ in range(5):
        logits = model(mx.array([[token.item()]]), cache=cache)
        token = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(token)

    # Reset cache
    cache = make_prompt_cache(model)
    logits = model(prompt[None], cache=cache)
    mx.eval(logits, *[c.state for c in cache if hasattr(c, "state")])
    token = mx.argmax(logits[:, -1, :], axis=-1)
    mx.eval(token)

    # Measure with fine-grained timing
    step_times = []
    graph_times = []
    eval_times = []

    for i in range(NUM_TOKENS):
        mx.synchronize()
        t0 = time.perf_counter()

        t_graph = time.perf_counter()
        logits = model(mx.array([[token.item()]]), cache=cache)
        next_token = mx.argmax(logits[:, -1, :], axis=-1)
        t_graph_end = time.perf_counter()

        t_eval = time.perf_counter()
        mx.eval(next_token)
        t_eval_end = time.perf_counter()

        step_times.append((t_eval_end - t0) * 1000)
        graph_times.append((t_graph_end - t_graph) * 1000)
        eval_times.append((t_eval_end - t_eval) * 1000)

        token = next_token

    # Skip warmup
    skip = 3
    st = step_times[skip:]
    gt = graph_times[skip:]
    et = eval_times[skip:]

    avg_step = sum(st) / len(st)
    avg_graph = sum(gt) / len(gt)
    avg_eval = sum(et) / len(et)
    tps = len(st) / (sum(st) / 1000)

    print(f"\n{label}:")
    print(f"  Step:  {avg_step:.1f}ms  Graph: {avg_graph:.2f}ms  Eval: {avg_eval:.1f}ms  -> {tps:.1f} tok/s")
    return avg_step, avg_graph, avg_eval


def benchmark_compiled_decode(model, tokenizer, label):
    """Benchmark with mx.compile wrapping the forward pass."""
    cache = make_prompt_cache(model)
    prompt = mx.array(tokenizer.encode("Hello world, this is a test of"))

    # Prefill
    logits = model(prompt[None], cache=cache)
    mx.eval(logits, *[c.state for c in cache if hasattr(c, "state")])
    token = mx.argmax(logits[:, -1, :], axis=-1)
    mx.eval(token)

    # Compile the forward pass
    compiled_model = mx.compile(model)

    # Warmup compiled path (triggers JIT compilation)
    for _ in range(5):
        logits = compiled_model(mx.array([[token.item()]]), cache=cache)
        token = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(token)

    # Reset
    cache = make_prompt_cache(model)
    logits = compiled_model(prompt[None], cache=cache)
    mx.eval(logits, *[c.state for c in cache if hasattr(c, "state")])
    token = mx.argmax(logits[:, -1, :], axis=-1)
    mx.eval(token)

    step_times = []
    graph_times = []
    eval_times = []

    for i in range(NUM_TOKENS):
        mx.synchronize()
        t0 = time.perf_counter()

        t_graph = time.perf_counter()
        logits = compiled_model(mx.array([[token.item()]]), cache=cache)
        next_token = mx.argmax(logits[:, -1, :], axis=-1)
        t_graph_end = time.perf_counter()

        t_eval = time.perf_counter()
        mx.eval(next_token)
        t_eval_end = time.perf_counter()

        step_times.append((t_eval_end - t0) * 1000)
        graph_times.append((t_graph_end - t_graph) * 1000)
        eval_times.append((t_eval_end - t_eval) * 1000)

        token = next_token

    skip = 3
    st = step_times[skip:]
    gt = graph_times[skip:]
    et = eval_times[skip:]

    avg_step = sum(st) / len(st)
    avg_graph = sum(gt) / len(gt)
    avg_eval = sum(et) / len(et)
    tps = len(st) / (sum(st) / 1000)

    print(f"\n{label}:")
    print(f"  Step:  {avg_step:.1f}ms  Graph: {avg_graph:.2f}ms  Eval: {avg_eval:.1f}ms  -> {tps:.1f} tok/s")
    return avg_step, avg_graph, avg_eval


def benchmark_async_pipeline(model, tokenizer, label):
    """Benchmark with async_eval pipeline overlap."""
    cache = make_prompt_cache(model)
    prompt = mx.array(tokenizer.encode("Hello world, this is a test of"))

    logits = model(prompt[None], cache=cache)
    mx.eval(logits, *[c.state for c in cache if hasattr(c, "state")])
    token = mx.argmax(logits[:, -1, :], axis=-1)
    mx.eval(token)

    # Warmup
    for _ in range(5):
        logits = model(mx.array([[token.item()]]), cache=cache)
        token = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(token)

    # Reset
    cache = make_prompt_cache(model)
    logits = model(prompt[None], cache=cache)
    mx.eval(logits, *[c.state for c in cache if hasattr(c, "state")])
    token = mx.argmax(logits[:, -1, :], axis=-1)
    mx.eval(token)

    # Async pipeline: build next graph while GPU processes current
    mx.synchronize()
    t0 = time.perf_counter()

    # Prime the pipeline
    logits = model(mx.array([[token.item()]]), cache=cache)
    next_token = mx.argmax(logits[:, -1, :], axis=-1)
    mx.async_eval(next_token)

    tokens = []
    for _ in range(NUM_TOKENS - 1):
        # CPU: build next graph while GPU processes current
        prev = next_token
        logits = model(mx.array([[prev.item()]]), cache=cache)
        next_token = mx.argmax(logits[:, -1, :], axis=-1)
        mx.async_eval(next_token)
        tokens.append(prev.item())

    mx.eval(next_token)
    tokens.append(next_token.item())

    t1 = time.perf_counter()
    total_ms = (t1 - t0) * 1000
    tps = NUM_TOKENS / (total_ms / 1000)

    print(f"\n{label}:")
    print(f"  Total: {total_ms:.0f}ms for {NUM_TOKENS} tokens -> {tps:.1f} tok/s")
    return total_ms


def benchmark_generation_stream(model, tokenizer, label):
    """Use mlx-lm's generation_stream approach."""
    cache = make_prompt_cache(model)
    prompt = mx.array(tokenizer.encode("Hello world, this is a test of"))

    logits = model(prompt[None], cache=cache)
    mx.eval(logits, *[c.state for c in cache if hasattr(c, "state")])
    token = mx.argmax(logits[:, -1, :], axis=-1)
    mx.eval(token)

    # Warmup
    for _ in range(5):
        logits = model(mx.array([[token.item()]]), cache=cache)
        token = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(token)

    # Reset
    cache = make_prompt_cache(model)
    logits = model(prompt[None], cache=cache)
    mx.eval(logits, *[c.state for c in cache if hasattr(c, "state")])
    token = mx.argmax(logits[:, -1, :], axis=-1)
    mx.eval(token)

    # Use the generation stream (new_stream + async_eval)
    gen_stream = mx.new_stream(mx.default_device())

    mx.synchronize()
    t0 = time.perf_counter()

    with mx.stream(gen_stream):
        logits = model(mx.array([[token.item()]]), cache=cache)
        next_token = mx.argmax(logits[:, -1, :], axis=-1)
        mx.async_eval(next_token)

    tokens = []
    for _ in range(NUM_TOKENS - 1):
        prev = next_token
        with mx.stream(gen_stream):
            logits = model(mx.array([[prev.item()]]), cache=cache)
            next_token = mx.argmax(logits[:, -1, :], axis=-1)
            mx.async_eval(next_token)
        tokens.append(prev.item())

    mx.eval(next_token)
    tokens.append(next_token.item())

    t1 = time.perf_counter()
    total_ms = (t1 - t0) * 1000
    tps = NUM_TOKENS / (total_ms / 1000)

    print(f"\n{label}:")
    print(f"  Total: {total_ms:.0f}ms for {NUM_TOKENS} tokens -> {tps:.1f} tok/s")
    return total_ms


def main():
    print("Loading model...")
    model, tokenizer = load(MODEL_NAME)

    # Warmup
    cache = make_prompt_cache(model)
    logits = model(mx.array(tokenizer.encode("Hi"))[None], cache=cache)
    mx.eval(logits)
    for _ in range(3):
        logits = model(mx.array([[1]]), cache=cache)
        mx.eval(logits)

    print(f"\n{'='*60}")
    print(f"Benchmarking decode strategies ({NUM_TOKENS} tokens)")
    print(f"{'='*60}")

    # 1. Baseline (synchronous)
    s1, g1, e1 = benchmark_decode(model, tokenizer, "1. Baseline (sync eval)")

    # 2. mx.compile
    try:
        s2, g2, e2 = benchmark_compiled_decode(model, tokenizer, "2. mx.compile(model)")
    except Exception as ex:
        print(f"\n2. mx.compile(model): FAILED - {ex}")
        s2, g2, e2 = s1, g1, e1

    # 3. async_eval pipeline
    t3 = benchmark_async_pipeline(model, tokenizer, "3. async_eval pipeline")

    # 4. Generation stream (separate Metal stream)
    t4 = benchmark_generation_stream(model, tokenizer, "4. Dedicated generation stream + async")

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"Baseline:        {s1:.1f}ms/step, graph build: {g1:.2f}ms")
    print(f"mx.compile:      {s2:.1f}ms/step, graph build: {g2:.2f}ms")
    print(f"async_eval:      {t3:.0f}ms total ({NUM_TOKENS/(t3/1000):.1f} tok/s)")
    print(f"gen stream:      {t4:.0f}ms total ({NUM_TOKENS/(t4/1000):.1f} tok/s)")


if __name__ == "__main__":
    main()
