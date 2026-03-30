"""
Python implementation of fused gather_qmm_swiglu using mx.fast.metal_kernel.

NOTE: This is a standalone kernel — it runs as its own Metal command buffer
and does NOT participate in MLX's full-graph command buffer batching.
For optimal performance, a C++ primitive implementation is needed.

This module serves as:
1. A correctness reference implementation
2. A demonstration of the approach
3. A fallback for systems without the compiled C++ extension
"""

import mlx.core as mx
import mlx.nn as nn


# Build kernel once at module level (JIT-compiled on first call)
_fused_kernel = None


def _get_kernel():
    global _fused_kernel
    if _fused_kernel is not None:
        return _fused_kernel

    _fused_kernel = mx.fast.metal_kernel(
        name="fused_gather_qmm_swiglu",
        input_names=[
            "x", "gate_weight", "gate_scales", "gate_biases",
            "up_weight", "up_scales", "up_biases", "expert_inds",
        ],
        output_names=["output"],
        source=r"""
            uint out_d = thread_position_in_grid.x;
            uint expert_slot = thread_position_in_grid.y;

            int expert_id = expert_inds[expert_slot];

            uint n_groups_per_row = gate_scales_shape[2];
            uint intermediate_size = gate_scales_shape[1];
            uint packed_input_dim = gate_weight_shape[2];
            uint pack_factor = 8;
            uint input_dim = packed_input_dim * pack_factor;
            uint group_size = input_dim / n_groups_per_row;

            if (out_d >= intermediate_size) return;

            long gw_offset = (long)expert_id * intermediate_size * packed_input_dim
                           + (long)out_d * packed_input_dim;
            long uw_offset = gw_offset;

            long sc_offset = (long)expert_id * intermediate_size * n_groups_per_row
                           + (long)out_d * n_groups_per_row;

            float gate_acc = 0.0f;
            float up_acc = 0.0f;

            for (uint g = 0; g < n_groups_per_row; g++) {
                float g_scale = (float)gate_scales[sc_offset + g];
                float g_bias  = (float)gate_biases[sc_offset + g];
                float u_scale = (float)up_scales[sc_offset + g];
                float u_bias  = (float)up_biases[sc_offset + g];

                uint packed_start = g * (group_size / pack_factor);
                uint n_packed = group_size / pack_factor;

                for (uint p = 0; p < n_packed; p++) {
                    uint gate_packed = gate_weight[gw_offset + packed_start + p];
                    uint up_packed   = up_weight[uw_offset + packed_start + p];
                    uint base_idx = g * group_size + p * pack_factor;

                    for (uint b = 0; b < pack_factor; b++) {
                        float gw_val = float((gate_packed >> (b * 4)) & 0xF) * g_scale + g_bias;
                        float uw_val = float((up_packed >> (b * 4)) & 0xF) * u_scale + u_bias;
                        float xv = (float)x[base_idx + b];
                        gate_acc += xv * gw_val;
                        up_acc   += xv * uw_val;
                    }
                }
            }

            float result = (gate_acc / (1.0f + exp(-gate_acc))) * up_acc;
            output[expert_slot * intermediate_size + out_d] = (T)result;
        """,
        ensure_row_contiguous=True,
    )
    return _fused_kernel


def gather_qmm_swiglu_standalone(
    x: mx.array,
    gate_weight: mx.array,
    gate_scales: mx.array,
    gate_biases: mx.array,
    up_weight: mx.array,
    up_scales: mx.array,
    up_biases: mx.array,
    expert_indices: mx.array,
) -> mx.array:
    """
    Fused gather_qmm + SwiGLU for MoE decode.

    Computes: silu(x @ gate_weight[experts]) * (x @ up_weight[experts])
    in a single Metal dispatch.

    Args:
        x: Input activations [hidden_size] or [1, hidden_size]
        gate_weight: Packed 4-bit weights [n_experts, intermediate, packed_dim]
        gate_scales: Per-group scales [n_experts, intermediate, n_groups]
        gate_biases: Per-group biases [n_experts, intermediate, n_groups]
        up_weight: Same layout as gate_weight
        up_scales: Same layout as gate_scales
        up_biases: Same layout as gate_biases
        expert_indices: Selected expert IDs [top_k]

    Returns:
        Output [top_k, intermediate_size] in float16/bfloat16
    """
    kernel = _get_kernel()

    x_flat = x.reshape(-1)
    inds_flat = expert_indices.reshape(-1)
    top_k = inds_flat.shape[0]
    intermediate_size = gate_scales.shape[1]

    result = kernel(
        inputs=[x_flat, gate_weight, gate_scales, gate_biases,
                up_weight, up_scales, up_biases, inds_flat],
        template=[("T", mx.float16)],
        grid=(intermediate_size, top_k, 1),
        threadgroup=(min(intermediate_size, 256), 1, 1),
        output_shapes=[(top_k, intermediate_size)],
        output_dtypes=[mx.float16],
    )
    return result[0]


def patch_switchglu(model):
    """
    Patch a model's SwitchGLU layers to use the fused kernel for decode.

    NOTE: This uses the standalone kernel (not graph-native), so it won't
    benefit from command buffer batching. For full performance, use the
    C++ extension.

    Args:
        model: An MLX model with SwitchGLU layers

    Returns:
        Number of patched layers
    """
    tm = model.language_model if hasattr(model, "language_model") else model
    patched = 0

    for layer in tm.model.layers:
        moe = layer.get("mlp", None) if hasattr(layer, "get") else getattr(layer, "mlp", None)
        if moe is None:
            continue
        switch = getattr(moe, "switch_mlp", None)
        if switch is None:
            continue

        # Store reference to original forward for fallback
        original_call = type(switch).__call__

        gate_proj = switch.gate_proj
        up_proj = switch.up_proj
        down_proj = switch.down_proj

        def make_fused_call(gp, up, dp, orig_call):
            def fused_call(self, x, indices):
                # Only use fused path for single-token decode (seq_len=1)
                if x.shape[-2] == 1:
                    x_flat = x.reshape(-1)
                    inds_flat = indices.reshape(-1)

                    # Fused gate+up+SwiGLU
                    swiglu_out = gather_qmm_swiglu_standalone(
                        x_flat,
                        gp["weight"], gp["scales"], gp["biases"],
                        up["weight"], up["scales"], up["biases"],
                        inds_flat,
                    )

                    # Apply down_proj (still separate)
                    top_k = inds_flat.shape[0]
                    x_exp = mx.expand_dims(swiglu_out, (0, 1))  # [1, 1, top_k, intermediate]
                    result = dp(x_exp, indices.reshape(1, 1, -1), sorted_indices=False)
                    return result.squeeze(-2)
                else:
                    return orig_call(self, x, indices)
            return fused_call

        # Apply patch
        type(switch).__call__ = make_fused_call(gate_proj, up_proj, down_proj, original_call)
        patched += 1

    return patched
