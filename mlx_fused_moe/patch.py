"""
Monkey-patch SwitchGLU to use the fused gather_qmm_swiglu C++ primitive.

For decode (1-2 tokens), the fused kernel replaces:
  - gate_proj gather_qmm
  - up_proj gather_qmm
  - SwiGLU activation
with a single Metal dispatch, saving 2 DRAM round-trips per MoE layer.

The down_proj gather_qmm still runs separately (different dimensions).
For prefill (many tokens), falls back to the original implementation.

Supports multi-token inputs (e.g., MTP batch verify with 2 tokens).
Each token's expert set is processed independently in the same dispatch.

Shape trace for decode (batch=1, seq=1, hidden=2048, top_k=8, intermediate=512):
  Original:
    x: (1,1,2048) -> expand -> (1,1,1,1,2048)
    gate_proj/up_proj output: (1,1,8,1,512)
    activation output: (1,1,8,1,512)
    down_proj output: (1,1,8,1,2048)
    squeeze(-2): (1,1,8,2048)

  Fused (1 token):
    x_flat: (2048,)
    idx_flat: (8,)
    fused kernel output: (8, 512)
    reshape to: (1,1,8,1,512)
    down_proj: (1,1,8,1,2048)
    squeeze(-2): (1,1,8,2048)

  Fused (2 tokens, MTP batch verify):
    x_flat: (4096,)    = 2 * hidden_size
    idx_flat: (16,)     = 2 * top_k
    fused kernel output: (16, 512)
    reshape to: (1,2,8,1,512)
    down_proj: (1,2,8,1,2048)
    squeeze(-2): (1,2,8,2048)
"""

import mlx.core as mx
from mlx_fused_moe._ext import gather_qmm_swiglu


def _make_fused_call(original_call):
    """Create a patched __call__ that uses fused kernel for decode."""

    def fused_call(self, x, indices):
        # indices shape: (batch, seq, top_k)
        # Only fuse for small token counts (decode / MTP batch verify).
        # The threshold matches SwitchGLU's own sort threshold.
        if indices.size >= 64:
            return original_call(self, x, indices)

        gate_proj = self.gate_proj
        up_proj = self.up_proj
        down_proj = self.down_proj

        # Check if quantized (only support quantized for now)
        if not hasattr(gate_proj, 'group_size'):
            return original_call(self, x, indices)

        top_k = indices.shape[-1]

        # Flatten x to (n_tokens * hidden_size,) and indices to (n_tokens * top_k,)
        x_flat = x.reshape(-1)
        idx_flat = indices.reshape(-1).astype(mx.int32)

        gate_weight = gate_proj["weight"]
        gate_scales = gate_proj["scales"]
        gate_biases = gate_proj.get("biases")
        up_weight = up_proj["weight"]
        up_scales = up_proj["scales"]
        up_biases = up_proj.get("biases")

        # Handle None biases
        if gate_biases is None:
            gate_biases = mx.zeros_like(gate_scales)
        if up_biases is None:
            up_biases = mx.zeros_like(up_scales)

        # Fused kernel: returns (n_tokens * top_k, intermediate_size)
        fused_out = gather_qmm_swiglu(
            x_flat,
            gate_weight,
            gate_scales,
            gate_biases,
            up_weight,
            up_scales,
            up_biases,
            idx_flat,
            top_k=top_k,
            group_size=gate_proj.group_size,
            bits=gate_proj.bits,
        )

        # Reshape to match gather_qmm output: (*indices.shape, 1, intermediate)
        intermediate_size = fused_out.shape[-1]
        x_fused = fused_out.reshape(
            *indices.shape, 1, intermediate_size
        )

        # down_proj (not fused — different dimensions)
        x_out = down_proj(x_fused, indices, sorted_indices=False)

        # squeeze(-2) removes the size-1 "sequence within expert" dim
        return x_out.squeeze(-2)

    return fused_call


def patch_model(model, verbose=True):
    """
    Patch all SwitchGLU instances in the model to use the fused kernel.

    Returns the number of layers patched.
    """
    from mlx_lm.models.switch_layers import SwitchGLU

    lm = model.language_model if hasattr(model, "language_model") else model
    layers = lm.model.layers if hasattr(lm, "model") else lm.layers

    # Store original method
    original_call = SwitchGLU.__call__

    # Create the fused replacement
    fused_call = _make_fused_call(original_call)

    # Store and monkey-patch
    SwitchGLU._original_call = original_call
    SwitchGLU.__call__ = fused_call

    # Count patched layers
    patched = 0
    for layer in layers:
        moe = getattr(layer, "mlp", None)
        if moe is None:
            continue
        switch = getattr(moe, "switch_mlp", None)
        if switch is not None and isinstance(switch, SwitchGLU):
            patched += 1

    if verbose:
        print(f"[mlx_fused_moe] Patched {patched} SwitchGLU layers with fused kernel")

    return patched


def unpatch_model(model=None):
    """Restore original SwitchGLU forward."""
    from mlx_lm.models.switch_layers import SwitchGLU

    if hasattr(SwitchGLU, '_original_call'):
        SwitchGLU.__call__ = SwitchGLU._original_call
        del SwitchGLU._original_call
