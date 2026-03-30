"""
MLX Fused MoE Extension.

Provides a fused gather_qmm_swiglu primitive that replaces three separate
MLX operations (gate_proj gather_qmm + up_proj gather_qmm + SwiGLU activation)
with a single Metal kernel dispatch.

This extension integrates natively into MLX's computation graph, allowing
it to participate in command buffer batching alongside other model operations.

Usage:
    from mlx_fused_moe import gather_qmm_swiglu, patch_model

    # Direct API
    result = gather_qmm_swiglu(x, gate_weight, gate_scales, gate_biases,
                                up_weight, up_scales, up_biases, expert_indices)

    # Or patch an existing model
    patch_model(model)  # Replaces SwitchGLU forward with fused version
"""

__version__ = "0.1.0"

# These will be available once the C++ extension is built
# from ._ext import gather_qmm_swiglu

# Python-only fallback using mx.fast.metal_kernel (standalone, not graph-native)
from .python_impl import gather_qmm_swiglu_standalone, patch_switchglu
