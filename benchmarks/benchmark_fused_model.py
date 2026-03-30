"""
Benchmark the fused gather_qmm_swiglu C++ extension in the full Qwen3.5-35B-A3B model.

Compares:
1. Baseline: Standard MLX inference
2. Fused: Patched SwitchGLU with fused gather_qmm_swiglu kernel
"""

import sys
import time
import mlx.core as mx
from mlx_lm import load, generate

MODEL = "mlx-community/Qwen3.5-35B-A3B-4bit"
PROMPT = "Explain the concept of quantum entanglement in simple terms."
MAX_TOKENS = 200
WARMUP_TOKENS = 50
N_RUNS = 3


def timed_generate(model, tokenizer, prompt, max_tokens):
    """Generate tokens and return (text, tokens_per_second, total_tokens)."""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Tokenize
    input_ids = mx.array(tokenizer.encode(text))
    prompt_len = input_ids.shape[0]

    # Warmup: generate a few tokens to fill caches
    result = generate(
        model, tokenizer, prompt=text, max_tokens=WARMUP_TOKENS, verbose=False
    )

    # Now benchmark with full generation
    mx.metal.clear_cache()

    t0 = time.perf_counter()
    result = generate(
        model, tokenizer, prompt=text, max_tokens=max_tokens, verbose=False
    )
    elapsed = time.perf_counter() - t0

    # Count output tokens
    output_ids = tokenizer.encode(result)
    gen_tokens = len(output_ids) - prompt_len
    tps = gen_tokens / elapsed if elapsed > 0 else 0

    return result, tps, gen_tokens, elapsed


def benchmark_decode_step(model, tokenizer, prompt, n_steps=100):
    """Benchmark individual decode steps for more precise measurement."""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = mx.array(tokenizer.encode(text))

    # Use generate to fill the KV cache with a few tokens
    _ = generate(model, tokenizer, prompt=text, max_tokens=20, verbose=False)

    # Now use the lower-level API for precise decode timing
    from mlx_lm.utils import stream_generate

    # Time individual token generation
    times = []
    token_count = 0
    for response in stream_generate(
        model, tokenizer, prompt=text, max_tokens=n_steps
    ):
        if token_count > 10:  # skip initial warmup tokens
            times.append(response.generation_time)
        token_count += 1
        if token_count >= n_steps:
            break

    if times:
        avg_ms = sum(times) / len(times) * 1000
        tps = 1000.0 / avg_ms if avg_ms > 0 else 0
        return avg_ms, tps, len(times)
    return 0, 0, 0


def main():
    print("=" * 70)
    print("Benchmark: Fused gather_qmm_swiglu in Qwen3.5-35B-A3B-4bit")
    print("=" * 70)

    # Load model
    print(f"\nLoading model: {MODEL}")
    model, tokenizer = load(MODEL)
    print("Model loaded.")

    # --- Baseline ---
    print("\n--- Baseline (standard MLX) ---")
    baseline_results = []
    for i in range(N_RUNS):
        _, tps, ntok, elapsed = timed_generate(
            model, tokenizer, PROMPT, MAX_TOKENS
        )
        baseline_results.append(tps)
        print(f"  Run {i+1}: {tps:.1f} t/s ({ntok} tokens in {elapsed:.2f}s)")

    baseline_avg = sum(baseline_results) / len(baseline_results)
    print(f"  Average: {baseline_avg:.1f} t/s")

    # --- Fused ---
    print("\n--- Fused (C++ gather_qmm_swiglu) ---")
    sys.path.insert(0, "/Users/ingemarrask/personal-dev/mlx-mtp")
    from mlx_fused_moe.patch import patch_model

    n_patched = patch_model(model)

    fused_results = []
    for i in range(N_RUNS):
        _, tps, ntok, elapsed = timed_generate(
            model, tokenizer, PROMPT, MAX_TOKENS
        )
        fused_results.append(tps)
        print(f"  Run {i+1}: {tps:.1f} t/s ({ntok} tokens in {elapsed:.2f}s)")

    fused_avg = sum(fused_results) / len(fused_results)
    print(f"  Average: {fused_avg:.1f} t/s")

    # --- Comparison ---
    speedup = fused_avg / baseline_avg if baseline_avg > 0 else 0
    print(f"\n{'='*70}")
    print(f"Results:")
    print(f"  Baseline:  {baseline_avg:.1f} t/s")
    print(f"  Fused:     {fused_avg:.1f} t/s")
    print(f"  Speedup:   {speedup:.3f}x")
    print(f"  Layers patched: {n_patched}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
