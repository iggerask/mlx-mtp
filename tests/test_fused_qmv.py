"""
Correctness test for fused_qmv kernel.

Tests that fused_qmv produces the same output as MLX's nn.QuantizedLinear
for the GatedDeltaNet projection fusion use case.
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_fused_moe._ext import fused_qmv


def make_quantized_linear(in_dim, out_dim, group_size=64, bits=4):
    """Create a QuantizedLinear layer with random weights."""
    layer = nn.Linear(in_dim, out_dim, bias=False)
    layer.set_dtype(mx.bfloat16)
    mx.eval(layer.parameters())
    ql = nn.QuantizedLinear.from_linear(layer, group_size=group_size, bits=bits)
    mx.eval(ql.parameters())
    return ql


def reference_qmv(x, ql):
    """Reference: use MLX's QuantizedLinear."""
    return ql(x)


def test_single_projection():
    """Test fused_qmv matches a single QuantizedLinear."""
    print("Test: single projection (2048 -> 4096)...")
    in_dim, out_dim = 2048, 4096
    ql = make_quantized_linear(in_dim, out_dim)

    x = mx.random.normal((1, 1, in_dim)).astype(mx.bfloat16)
    mx.eval(x)

    ref = reference_qmv(x, ql)
    mx.eval(ref)

    # Fused kernel
    x_flat = x.reshape(-1)
    biases = ql.biases if ql.biases is not None else mx.zeros_like(ql.scales)
    fused_out = fused_qmv(
        x_flat, ql.weight, ql.scales, biases,
        n_tokens=1, group_size=ql.group_size, bits=ql.bits
    )
    mx.eval(fused_out)

    ref_np = np.array(ref.reshape(1, -1).astype(mx.float32), dtype=np.float32)
    fused_np = np.array(fused_out.astype(mx.float32), dtype=np.float32)

    max_err = np.max(np.abs(ref_np - fused_np))
    rel_err = max_err / (np.max(np.abs(ref_np)) + 1e-8)
    print(f"  max abs error: {max_err:.6f}, rel error: {rel_err:.6f}")
    assert rel_err < 0.01, f"Relative error too high: {rel_err}"
    print("  PASSED")


def test_concatenated_projections():
    """Test fused_qmv with concatenated weights (the GDN use case)."""
    print("Test: concatenated projections (qkv=8192, z=4096, b=32, a=32)...")
    in_dim = 2048
    out_dims = [8192, 4096, 32, 32]  # qkv, z, b, a

    # Create 4 separate QuantizedLinear layers
    qls = [make_quantized_linear(in_dim, od) for od in out_dims]

    x = mx.random.normal((1, 1, in_dim)).astype(mx.bfloat16)
    mx.eval(x)

    # Reference: run each projection separately
    refs = [ql(x) for ql in qls]
    mx.eval(*refs)
    ref_cat = mx.concatenate([r.reshape(1, -1) for r in refs], axis=-1)

    # Fused: concatenate weights and run once
    weights = [ql.weight for ql in qls]
    scales_list = [ql.scales for ql in qls]
    biases_list = [
        ql.biases if ql.biases is not None else mx.zeros_like(ql.scales)
        for ql in qls
    ]

    cat_weight = mx.concatenate(weights, axis=0)
    cat_scales = mx.concatenate(scales_list, axis=0)
    cat_biases = mx.concatenate(biases_list, axis=0)
    mx.eval(cat_weight, cat_scales, cat_biases)

    x_flat = x.reshape(-1)
    fused_out = fused_qmv(
        x_flat, cat_weight, cat_scales, cat_biases,
        n_tokens=1, group_size=qls[0].group_size, bits=qls[0].bits
    )
    mx.eval(fused_out)

    ref_np = np.array(ref_cat.astype(mx.float32), dtype=np.float32)
    fused_np = np.array(fused_out.astype(mx.float32), dtype=np.float32)

    max_err = np.max(np.abs(ref_np - fused_np))
    rel_err = max_err / (np.max(np.abs(ref_np)) + 1e-8)
    print(f"  max abs error: {max_err:.6f}, rel error: {rel_err:.6f}")
    assert rel_err < 0.01, f"Relative error too high: {rel_err}"

    # Also verify split produces correct individual outputs
    split_indices = []
    cumsum = 0
    for od in out_dims[:-1]:
        cumsum += od
        split_indices.append(cumsum)

    parts = mx.split(fused_out, split_indices, axis=-1)
    for i, (part, ref) in enumerate(zip(parts, refs)):
        part_np = np.array(part.astype(mx.float32), dtype=np.float32)
        ref_np_i = np.array(ref.reshape(1, -1).astype(mx.float32), dtype=np.float32)
        part_err = np.max(np.abs(part_np - ref_np_i)) / (np.max(np.abs(ref_np_i)) + 1e-8)
        print(f"  proj {i} (out_dim={out_dims[i]}): rel error = {part_err:.6f}")
        assert part_err < 0.01, f"Split part {i} error too high: {part_err}"

    print("  PASSED")


def test_multi_token():
    """Test fused_qmv with 2 tokens (MTP batch verify case)."""
    print("Test: multi-token (2 tokens, 2048 -> 4096)...")
    in_dim, out_dim = 2048, 4096
    ql = make_quantized_linear(in_dim, out_dim)

    x = mx.random.normal((1, 2, in_dim)).astype(mx.bfloat16)
    mx.eval(x)

    ref = reference_qmv(x, ql)  # (1, 2, 4096)
    mx.eval(ref)

    x_flat = x.reshape(-1)
    biases = ql.biases if ql.biases is not None else mx.zeros_like(ql.scales)
    fused_out = fused_qmv(
        x_flat, ql.weight, ql.scales, biases,
        n_tokens=2, group_size=ql.group_size, bits=ql.bits
    )
    mx.eval(fused_out)

    ref_np = np.array(ref.reshape(2, -1).astype(mx.float32), dtype=np.float32)
    fused_np = np.array(fused_out.astype(mx.float32), dtype=np.float32)

    max_err = np.max(np.abs(ref_np - fused_np))
    rel_err = max_err / (np.max(np.abs(ref_np)) + 1e-8)
    print(f"  max abs error: {max_err:.6f}, rel error: {rel_err:.6f}")
    assert rel_err < 0.01, f"Relative error too high: {rel_err}"
    print("  PASSED")


if __name__ == "__main__":
    mx.random.seed(42)
    test_single_projection()
    test_concatenated_projections()
    test_multi_token()
    print("\nAll fused_qmv tests passed!")
