#!/usr/bin/env python3
"""Proof-of-concept: custom fused Metal kernels for MoE routing.

Tests whether fusing small operations into custom Metal kernels
can reduce kernel dispatch overhead in the decode step.

Target: fuse softmax + topk + score_normalize into single kernel.
Currently these are 4+ separate ops per layer × 40 layers = 160+ dispatches.
"""

import time
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

MODEL_NAME = "mlx-community/Qwen3.5-35B-A3B-4bit"
NUM_EXPERTS = 256
TOP_K = 8
HIDDEN_SIZE = 2048


def benchmark_current_router():
    """Benchmark the current MLX router implementation."""
    # Simulate router: gate output → softmax → topk → normalize
    gate_out = mx.random.normal((1, 1, NUM_EXPERTS))
    mx.eval(gate_out)

    # Warmup
    for _ in range(20):
        probs = mx.softmax(gate_out, axis=-1, precise=True)
        inds = mx.argpartition(probs, kth=-TOP_K, axis=-1)[..., -TOP_K:]
        scores = mx.take_along_axis(probs, inds, axis=-1)
        scores = scores / scores.sum(axis=-1, keepdims=True)
        mx.eval(inds, scores)

    # Benchmark
    N = 100
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        probs = mx.softmax(gate_out, axis=-1, precise=True)
        inds = mx.argpartition(probs, kth=-TOP_K, axis=-1)[..., -TOP_K:]
        scores = mx.take_along_axis(probs, inds, axis=-1)
        scores = scores / scores.sum(axis=-1, keepdims=True)
        mx.eval(inds, scores)
    mx.synchronize()
    t1 = time.perf_counter()

    return (t1 - t0) / N * 1000


def benchmark_fused_router():
    """Benchmark a fused softmax+topk kernel using mx.fast.metal_kernel."""

    # Fused kernel: softmax + partial sort (topk) + normalize
    # Input: raw gate logits (1, num_experts)
    # Output: top_k indices and normalized scores
    fused_topk_kernel = mx.fast.metal_kernel(
        name="fused_softmax_topk",
        input_names=["gate_logits"],
        output_names=["top_indices", "top_scores"],
        source=r"""
            // Thread processes one batch/seq position
            uint pos = thread_position_in_grid.x;
            uint N = gate_logits_shape[gate_logits_ndim - 1];  // num_experts

            // Step 1: Find max for numerical stability
            float max_val = -INFINITY;
            for (uint i = 0; i < N; i++) {
                float v = (float)gate_logits[pos * N + i];
                if (v > max_val) max_val = v;
            }

            // Step 2: Compute softmax probabilities
            float sum_exp = 0.0f;
            for (uint i = 0; i < N; i++) {
                sum_exp += exp((float)gate_logits[pos * N + i] - max_val);
            }

            // Step 3: Find top-K using partial selection
            // Use a simple min-heap approach for K=8
            const uint K = 8;
            float top_vals[8];
            int top_inds[8];
            for (uint k = 0; k < K; k++) {
                top_vals[k] = -INFINITY;
                top_inds[k] = -1;
            }

            for (uint i = 0; i < N; i++) {
                float prob = exp((float)gate_logits[pos * N + i] - max_val) / sum_exp;

                // Find minimum in current top-K
                uint min_idx = 0;
                float min_val = top_vals[0];
                for (uint k = 1; k < K; k++) {
                    if (top_vals[k] < min_val) {
                        min_val = top_vals[k];
                        min_idx = k;
                    }
                }

                if (prob > min_val) {
                    top_vals[min_idx] = prob;
                    top_inds[min_idx] = (int)i;
                }
            }

            // Step 4: Normalize top-K scores
            float score_sum = 0.0f;
            for (uint k = 0; k < K; k++) {
                score_sum += top_vals[k];
            }

            // Write outputs
            for (uint k = 0; k < K; k++) {
                top_indices[pos * K + k] = top_inds[k];
                top_scores[pos * K + k] = (T)(top_vals[k] / score_sum);
            }
        """,
        ensure_row_contiguous=True,
    )

    gate_out = mx.random.normal((1, 1, NUM_EXPERTS))
    mx.eval(gate_out)

    # Flatten for kernel
    flat_gate = gate_out.reshape(-1, NUM_EXPERTS)

    # Warmup
    for _ in range(20):
        results = fused_topk_kernel(
            inputs=[flat_gate],
            template=[("T", mx.float32)],
            grid=(1, 1, 1),
            threadgroup=(1, 1, 1),
            output_shapes=[(1, TOP_K), (1, TOP_K)],
            output_dtypes=[mx.int32, mx.float32],
        )
        mx.eval(*results)

    # Benchmark
    N = 100
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        results = fused_topk_kernel(
            inputs=[flat_gate],
            template=[("T", mx.float32)],
            grid=(1, 1, 1),
            threadgroup=(1, 1, 1),
            output_shapes=[(1, TOP_K), (1, TOP_K)],
            output_dtypes=[mx.int32, mx.float32],
        )
        mx.eval(*results)
    mx.synchronize()
    t1 = time.perf_counter()

    return (t1 - t0) / N * 1000, results


def verify_correctness():
    """Verify fused kernel produces same results as separate ops."""
    gate_out = mx.random.normal((1, 1, NUM_EXPERTS))
    mx.eval(gate_out)

    # Reference
    probs = mx.softmax(gate_out, axis=-1, precise=True)
    inds = mx.argpartition(probs, kth=-TOP_K, axis=-1)[..., -TOP_K:]
    scores = mx.take_along_axis(probs, inds, axis=-1)
    scores = scores / scores.sum(axis=-1, keepdims=True)
    mx.eval(inds, scores)

    ref_inds = set(inds.reshape(-1).tolist())
    ref_scores_dict = {inds.reshape(-1)[i].item(): scores.reshape(-1)[i].item()
                       for i in range(TOP_K)}

    # Fused
    fused_topk_kernel = mx.fast.metal_kernel(
        name="fused_softmax_topk_verify",
        input_names=["gate_logits"],
        output_names=["top_indices", "top_scores"],
        source=r"""
            uint pos = thread_position_in_grid.x;
            uint N = gate_logits_shape[gate_logits_ndim - 1];
            float max_val = -INFINITY;
            for (uint i = 0; i < N; i++) {
                float v = (float)gate_logits[pos * N + i];
                if (v > max_val) max_val = v;
            }
            float sum_exp = 0.0f;
            for (uint i = 0; i < N; i++) {
                sum_exp += exp((float)gate_logits[pos * N + i] - max_val);
            }
            const uint K = 8;
            float top_vals[8];
            int top_inds[8];
            for (uint k = 0; k < K; k++) { top_vals[k] = -INFINITY; top_inds[k] = -1; }
            for (uint i = 0; i < N; i++) {
                float prob = exp((float)gate_logits[pos * N + i] - max_val) / sum_exp;
                uint min_idx = 0;
                float min_val = top_vals[0];
                for (uint k = 1; k < K; k++) {
                    if (top_vals[k] < min_val) { min_val = top_vals[k]; min_idx = k; }
                }
                if (prob > min_val) { top_vals[min_idx] = prob; top_inds[min_idx] = (int)i; }
            }
            float score_sum = 0.0f;
            for (uint k = 0; k < K; k++) score_sum += top_vals[k];
            for (uint k = 0; k < K; k++) {
                top_indices[pos * K + k] = top_inds[k];
                top_scores[pos * K + k] = (T)(top_vals[k] / score_sum);
            }
        """,
        ensure_row_contiguous=True,
    )

    flat_gate = gate_out.reshape(-1, NUM_EXPERTS)
    results = fused_topk_kernel(
        inputs=[flat_gate],
        template=[("T", mx.float32)],
        grid=(1, 1, 1),
        threadgroup=(1, 1, 1),
        output_shapes=[(1, TOP_K), (1, TOP_K)],
        output_dtypes=[mx.int32, mx.float32],
    )
    mx.eval(*results)

    fused_inds = set(results[0].reshape(-1).tolist())
    fused_scores = results[1].reshape(-1).tolist()
    fused_inds_list = results[0].reshape(-1).tolist()

    match = ref_inds == fused_inds
    print(f"Top-K indices match: {match}")
    if match:
        for i in range(TOP_K):
            idx = fused_inds_list[i]
            ref_s = ref_scores_dict.get(idx, 0.0)
            fused_s = fused_scores[i]
            print(f"  Expert {idx}: ref={ref_s:.6f}, fused={fused_s:.6f}, diff={abs(ref_s-fused_s):.8f}")
    return match


def benchmark_dispatch_overhead():
    """Measure pure kernel dispatch overhead with trivial kernels."""
    # Minimal kernel: just copy input to output
    copy_kernel = mx.fast.metal_kernel(
        name="trivial_copy",
        input_names=["inp"],
        output_names=["out"],
        source=r"""
            uint i = thread_position_in_grid.x;
            out[i] = inp[i];
        """,
    )

    x = mx.ones((256,))
    mx.eval(x)

    # Measure single dispatch overhead
    N = 1000
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        result = copy_kernel(
            inputs=[x],
            template=[("T", mx.float32)],
            grid=(256, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(256,)],
            output_dtypes=[mx.float32],
        )
        mx.eval(result[0])
    mx.synchronize()
    t1 = time.perf_counter()
    single_us = (t1 - t0) / N * 1e6

    # Measure batched dispatches (10 kernels per eval)
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(N // 10):
        results = []
        for _ in range(10):
            r = copy_kernel(
                inputs=[x],
                template=[("T", mx.float32)],
                grid=(256, 1, 1),
                threadgroup=(256, 1, 1),
                output_shapes=[(256,)],
                output_dtypes=[mx.float32],
            )
            results.append(r[0])
        mx.eval(*results)
    mx.synchronize()
    t1 = time.perf_counter()
    batched_us = (t1 - t0) / (N // 10) * 1e6 / 10

    # Measure MLX native op dispatch overhead
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        y = x + 1
        mx.eval(y)
    mx.synchronize()
    t1 = time.perf_counter()
    native_us = (t1 - t0) / N * 1e6

    return single_us, batched_us, native_us


def main():
    print("=" * 60)
    print("Custom Metal Kernel Experiments")
    print("=" * 60)

    # 1. Kernel dispatch overhead
    print("\n1. Kernel Dispatch Overhead")
    print("-" * 40)
    single, batched, native = benchmark_dispatch_overhead()
    print(f"  Custom kernel (1 per eval):   {single:.1f}μs")
    print(f"  Custom kernel (10 per eval):  {batched:.1f}μs")
    print(f"  Native MLX op (1 per eval):   {native:.1f}μs")
    print(f"  Dispatch overhead: ~{single:.0f}μs per kernel")
    print(f"  With 2612 ops per step: ~{2612 * single / 1000:.1f}ms overhead")

    # 2. Correctness check
    print("\n2. Fused Router Correctness")
    print("-" * 40)
    verify_correctness()

    # 3. Router benchmark
    print("\n3. Router Benchmark (isolated)")
    print("-" * 40)
    current_ms = benchmark_current_router()
    fused_ms, _ = benchmark_fused_router()
    print(f"  Current (4 separate ops):  {current_ms:.3f}ms")
    print(f"  Fused (1 custom kernel):   {fused_ms:.3f}ms")
    print(f"  Speedup:                   {current_ms/fused_ms:.2f}x")
    print(f"  Savings × 40 layers:       {(current_ms - fused_ms) * 40:.2f}ms")

    # 4. Impact estimate
    print("\n4. Full Model Impact Estimate")
    print("-" * 40)
    print(f"  Current decode step: ~13.3ms")
    print(f"  Router savings (40L): {(current_ms - fused_ms) * 40:.2f}ms")

    # Estimate for more fusions
    savings_router = (current_ms - fused_ms) * 40
    # Other fusible patterns: RMSNorm+proj (191 norms), activation fusions
    # Estimate ~30% of non-matmul ops could be fused
    other_savings_ms = 0.5  # conservative estimate
    total_savings = savings_router + other_savings_ms

    new_step = 13.3 - total_savings
    print(f"  Estimated total savings:  {total_savings:.1f}ms")
    print(f"  New step time:            {new_step:.1f}ms")
    print(f"  Speedup:                  {13.3/new_step:.2f}x")
    print(f"  Combined with MTP 1.13x:  {13.3/new_step * 1.13:.2f}x")


if __name__ == "__main__":
    main()
