"""
Correctness test for gather_qmm_down_reduce kernel.

Tests that the fused down_proj + score-weighted reduce produces the same
output as the separate MLX operations: gather_qmm(down_proj) → score * y → sum.
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_fused_moe._ext import gather_qmm_down_reduce


def make_quantized_switch_weights(n_experts, out_dim, in_dim, group_size=64, bits=4):
    """Create random quantized expert weights mimicking QuantizedSwitchLinear."""
    # Create random weights per expert, quantize
    weights = mx.random.normal((n_experts, out_dim, in_dim)).astype(mx.bfloat16)
    mx.eval(weights)
    qw, scales, biases = mx.quantize(weights, group_size=group_size, bits=bits)
    mx.eval(qw, scales, biases)
    return qw, scales, biases


def reference_down_reduce(x_intermediate, weight, scales, biases, expert_indices, scores_2d, top_k, group_size=64, bits=4):
    """Reference: MLX gather_qmm + score weighting + sum."""
    # x_intermediate: (n_tokens * top_k, intermediate_size) → need (n_tokens, top_k, 1, intermediate_size)
    n_total = x_intermediate.shape[0]
    n_tokens = n_total // top_k
    intermediate_size = x_intermediate.shape[1]

    x_exp = x_intermediate.reshape(n_tokens, top_k, 1, intermediate_size)

    # gather_qmm expects indices as (n_tokens, top_k)
    idx = expert_indices.reshape(n_tokens, top_k)

    # Run gather_qmm for down_proj
    out = mx.gather_qmm(
        x_exp, weight, scales, biases,
        rhs_indices=idx,
        transpose=True,
        group_size=group_size,
        bits=bits,
    )
    # out: (n_tokens, top_k, 1, hidden_size)
    out = out.squeeze(-2)  # (n_tokens, top_k, hidden_size)

    # Score weighting and reduction
    y = (out * scores_2d[..., None]).sum(axis=-2)  # (n_tokens, hidden_size)
    return y


def test_single_token():
    """Test with 1 token, 8 experts, intermediate=512, hidden=2048."""
    print("Test: single token (1 token, top_k=8, 512→2048)...")
    n_experts = 256
    top_k = 8
    intermediate_size = 512
    hidden_size = 2048

    qw, scales, biases = make_quantized_switch_weights(n_experts, hidden_size, intermediate_size)

    # Random intermediate activations and expert indices
    x_inter = mx.random.normal((top_k, intermediate_size)).astype(mx.bfloat16)
    expert_ids = mx.array([3, 17, 42, 100, 55, 200, 128, 77], dtype=mx.int32)
    scores = mx.random.uniform(shape=(1, top_k)).astype(mx.bfloat16)
    scores = scores / scores.sum(axis=-1, keepdims=True)
    mx.eval(x_inter, expert_ids, scores)

    # Reference
    ref = reference_down_reduce(x_inter, qw, scales, biases, expert_ids, scores, top_k)
    mx.eval(ref)

    # Fused kernel
    fused = gather_qmm_down_reduce(
        x_inter, qw, scales, biases, expert_ids, scores.reshape(-1),
        top_k=top_k, group_size=64, bits=4
    )
    mx.eval(fused)

    ref_np = np.array(ref.astype(mx.float32), dtype=np.float32)
    fused_np = np.array(fused.astype(mx.float32), dtype=np.float32)

    max_err = np.max(np.abs(ref_np - fused_np))
    rel_err = max_err / (np.max(np.abs(ref_np)) + 1e-8)
    print(f"  max abs error: {max_err:.6f}, rel error: {rel_err:.6f}")
    assert rel_err < 0.02, f"Relative error too high: {rel_err}"
    print("  PASSED")


def test_multi_token():
    """Test with 2 tokens (MTP batch verify case)."""
    print("Test: 2 tokens (top_k=8, 512→2048)...")
    n_experts = 256
    top_k = 8
    intermediate_size = 512
    hidden_size = 2048

    qw, scales, biases = make_quantized_switch_weights(n_experts, hidden_size, intermediate_size)

    # 2 tokens × 8 experts = 16 intermediate rows
    x_inter = mx.random.normal((2 * top_k, intermediate_size)).astype(mx.bfloat16)
    expert_ids = mx.random.randint(0, n_experts, shape=(2 * top_k,)).astype(mx.int32)
    scores = mx.random.uniform(shape=(2, top_k)).astype(mx.bfloat16)
    scores = scores / scores.sum(axis=-1, keepdims=True)
    mx.eval(x_inter, expert_ids, scores)

    # Reference
    ref = reference_down_reduce(x_inter, qw, scales, biases, expert_ids, scores, top_k)
    mx.eval(ref)

    # Fused kernel
    scores_flat = scores.reshape(-1)
    fused = gather_qmm_down_reduce(
        x_inter, qw, scales, biases, expert_ids, scores_flat,
        top_k=top_k, group_size=64, bits=4
    )
    mx.eval(fused)

    ref_np = np.array(ref.astype(mx.float32), dtype=np.float32)
    fused_np = np.array(fused.astype(mx.float32), dtype=np.float32)

    # Check each token separately
    for t in range(2):
        t_ref = ref_np[t]
        t_fused = fused_np[t]
        t_err = np.max(np.abs(t_ref - t_fused)) / (np.max(np.abs(t_ref)) + 1e-8)
        print(f"  token {t}: rel error = {t_err:.6f}")
        assert t_err < 0.02, f"Token {t} error too high: {t_err}"
    print("  PASSED")


def test_small_dimensions():
    """Test with smaller dimensions to verify edge cases."""
    print("Test: small dimensions (4 experts, top_k=2, 64→128)...")
    n_experts = 4
    top_k = 2
    intermediate_size = 64
    hidden_size = 128

    qw, scales, biases = make_quantized_switch_weights(n_experts, hidden_size, intermediate_size)

    x_inter = mx.random.normal((top_k, intermediate_size)).astype(mx.bfloat16)
    expert_ids = mx.array([1, 3], dtype=mx.int32)
    scores = mx.array([[0.6, 0.4]], dtype=mx.bfloat16)
    mx.eval(x_inter, expert_ids, scores)

    ref = reference_down_reduce(x_inter, qw, scales, biases, expert_ids, scores, top_k)
    mx.eval(ref)

    fused = gather_qmm_down_reduce(
        x_inter, qw, scales, biases, expert_ids, scores.reshape(-1),
        top_k=top_k, group_size=64, bits=4
    )
    mx.eval(fused)

    ref_np = np.array(ref.astype(mx.float32), dtype=np.float32)
    fused_np = np.array(fused.astype(mx.float32), dtype=np.float32)

    max_err = np.max(np.abs(ref_np - fused_np))
    rel_err = max_err / (np.max(np.abs(ref_np)) + 1e-8)
    print(f"  max abs error: {max_err:.6f}, rel error: {rel_err:.6f}")
    assert rel_err < 0.02, f"Relative error too high: {rel_err}"
    print("  PASSED")


if __name__ == "__main__":
    mx.random.seed(42)
    test_single_token()
    test_multi_token()
    test_small_dimensions()
    print("\nAll gather_qmm_down_reduce tests passed!")
