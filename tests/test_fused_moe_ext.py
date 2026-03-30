"""Test the C++ fused gather_qmm_swiglu extension against reference MLX ops."""
import mlx.core as mx
import numpy as np
import time

# Import the C++ extension
from mlx_fused_moe._ext import gather_qmm_swiglu


def reference_gather_qmm_swiglu(
    x, gate_weight, gate_scales, gate_biases,
    up_weight, up_scales, up_biases, expert_indices,
    group_size=64, bits=4
):
    """Reference implementation using standard MLX ops."""
    results = []
    for i in range(expert_indices.size):
        eidx = expert_indices.reshape(-1)[i]
        # Dequantized matmul for gate
        gate_out = mx.quantized_matmul(
            x.reshape(1, -1),
            gate_weight[eidx],
            scales=gate_scales[eidx],
            biases=gate_biases[eidx],
            group_size=group_size,
            bits=bits,
        )
        # Dequantized matmul for up
        up_out = mx.quantized_matmul(
            x.reshape(1, -1),
            up_weight[eidx],
            scales=up_scales[eidx],
            biases=up_biases[eidx],
            group_size=group_size,
            bits=bits,
        )
        # SwiGLU: silu(gate) * up
        result = mx.sigmoid(gate_out) * gate_out * up_out
        results.append(result.reshape(-1))

    return mx.stack(results)


def test_correctness():
    """Test C++ extension correctness against reference implementation."""
    print("=" * 60)
    print("Testing C++ fused gather_qmm_swiglu extension")
    print("=" * 60)

    # Qwen3.5 MoE dimensions
    n_experts = 64
    input_dim = 2048
    intermediate_size = 512
    group_size = 64
    bits = 4
    top_k = 8

    pack_factor = 32 // bits  # 8 for 4-bit
    packed_input_dim = input_dim // pack_factor
    n_groups = input_dim // group_size

    mx.random.seed(42)

    # Create test data
    x = mx.random.normal((input_dim,)).astype(mx.float16)

    # Quantized weights: [n_experts, intermediate_size, packed_input_dim]
    gate_weight = mx.random.randint(0, 255, (n_experts, intermediate_size, packed_input_dim)).astype(mx.uint32)
    up_weight = mx.random.randint(0, 255, (n_experts, intermediate_size, packed_input_dim)).astype(mx.uint32)

    # Scales and biases: [n_experts, intermediate_size, n_groups]
    gate_scales = (mx.random.normal((n_experts, intermediate_size, n_groups)) * 0.01).astype(mx.float16)
    gate_biases = (mx.random.normal((n_experts, intermediate_size, n_groups)) * 0.001).astype(mx.float16)
    up_scales = (mx.random.normal((n_experts, intermediate_size, n_groups)) * 0.01).astype(mx.float16)
    up_biases = (mx.random.normal((n_experts, intermediate_size, n_groups)) * 0.001).astype(mx.float16)

    # Expert indices
    expert_indices = mx.array([3, 7, 12, 25, 31, 42, 50, 63], dtype=mx.int32)

    mx.eval(x, gate_weight, gate_scales, gate_biases, up_weight, up_scales, up_biases, expert_indices)

    print(f"\nDimensions:")
    print(f"  input_dim={input_dim}, intermediate_size={intermediate_size}")
    print(f"  n_experts={n_experts}, top_k={top_k}")
    print(f"  group_size={group_size}, bits={bits}")
    print(f"  x shape: {x.shape}")
    print(f"  gate_weight shape: {gate_weight.shape}")
    print(f"  gate_scales shape: {gate_scales.shape}")
    print(f"  expert_indices: {expert_indices.tolist()}")

    # Reference output
    print("\nComputing reference output...")
    ref_out = reference_gather_qmm_swiglu(
        x, gate_weight, gate_scales, gate_biases,
        up_weight, up_scales, up_biases, expert_indices,
        group_size=group_size, bits=bits,
    )
    mx.eval(ref_out)
    print(f"  Reference output shape: {ref_out.shape}")

    # C++ extension output
    print("Computing C++ extension output...")
    ext_out = gather_qmm_swiglu(
        x, gate_weight, gate_scales, gate_biases,
        up_weight, up_scales, up_biases, expert_indices,
        top_k=top_k, group_size=group_size, bits=bits,
    )
    mx.eval(ext_out)
    print(f"  Extension output shape: {ext_out.shape}")

    # Compare
    ref_np = np.array(ref_out, dtype=np.float32)
    ext_np = np.array(ext_out, dtype=np.float32)

    max_diff = np.max(np.abs(ref_np - ext_np))
    mean_diff = np.mean(np.abs(ref_np - ext_np))
    ref_norm = np.linalg.norm(ref_np)
    rel_err = np.linalg.norm(ref_np - ext_np) / (ref_norm + 1e-8)

    print(f"\n--- Correctness ---")
    print(f"  Max absolute difference: {max_diff:.6f}")
    print(f"  Mean absolute difference: {mean_diff:.6f}")
    print(f"  Relative error: {rel_err:.6f}")
    print(f"  Reference norm: {ref_norm:.4f}")

    if max_diff < 0.1:
        print(f"  PASS ✓ (max diff < 0.1)")
    elif max_diff < 1.0:
        print(f"  MARGINAL (max diff < 1.0 but > 0.1)")
    else:
        print(f"  FAIL ✗ (max diff >= 1.0)")

    # Performance comparison
    print(f"\n--- Performance ---")
    n_iters = 100

    # Warmup
    for _ in range(10):
        _ = gather_qmm_swiglu(
            x, gate_weight, gate_scales, gate_biases,
            up_weight, up_scales, up_biases, expert_indices,
            top_k=top_k,
        )
        mx.eval(_)

    # C++ extension timing
    t0 = time.perf_counter()
    for _ in range(n_iters):
        out = gather_qmm_swiglu(
            x, gate_weight, gate_scales, gate_biases,
            up_weight, up_scales, up_biases, expert_indices,
            top_k=top_k,
        )
        mx.eval(out)
    ext_time = (time.perf_counter() - t0) / n_iters * 1000

    # Reference timing
    for _ in range(10):
        _ = reference_gather_qmm_swiglu(
            x, gate_weight, gate_scales, gate_biases,
            up_weight, up_scales, up_biases, expert_indices,
        )
        mx.eval(_)

    t0 = time.perf_counter()
    for _ in range(n_iters):
        out = reference_gather_qmm_swiglu(
            x, gate_weight, gate_scales, gate_biases,
            up_weight, up_scales, up_biases, expert_indices,
        )
        mx.eval(out)
    ref_time = (time.perf_counter() - t0) / n_iters * 1000

    print(f"  C++ extension: {ext_time:.3f} ms/call")
    print(f"  Reference ops: {ref_time:.3f} ms/call")
    print(f"  Speedup: {ref_time/ext_time:.2f}x")

    return max_diff < 1.0


if __name__ == "__main__":
    success = test_correctness()
    print(f"\n{'='*60}")
    print(f"Overall: {'PASS' if success else 'FAIL'}")
