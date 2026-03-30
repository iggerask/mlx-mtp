"""
Monkey-patch SparseMoeBlock to use both fused kernels for the full MoE path.

Replaces the entire SwitchGLU + score weighting + expert sum with:
  1. Fused gate_proj + up_proj + SwiGLU (existing kernel)
  2. Fused down_proj + score-weighted reduce (new kernel)

This patches at the SparseMoeBlock level (not SwitchGLU), giving us control
over the score weighting and expert reduction too.

For decode (1-2 tokens), the full fusion eliminates:
  - 2 DRAM round-trips from gate+up fusion
  - 1 additional DRAM round-trip from down_proj + reduce fusion
  - The reshape/squeeze operations between SwitchGLU and SparseMoeBlock
  - The element-wise score multiply and sum reduction

Falls back to original for large token counts (prefill).
"""

import mlx.core as mx
from mlx_fused_moe._ext import gather_qmm_swiglu, gather_qmm_down_reduce


def _make_fused_moe_call(original_call):
    """Create a patched __call__ for SparseMoeBlock."""

    def fused_moe_call(self, x):
        # Flatten batch dims: x is (batch, seq, hidden)
        batch_shape = x.shape[:-1]
        hidden_size = x.shape[-1]
        n_tokens = 1
        for d in batch_shape:
            n_tokens *= d

        # Only fuse for small token counts (decode / MTP batch verify)
        if n_tokens > 4:
            return original_call(self, x)

        switch_mlp = self.switch_mlp
        gate_proj = switch_mlp.gate_proj
        up_proj = switch_mlp.up_proj
        down_proj = switch_mlp.down_proj

        # Must be quantized
        if not hasattr(gate_proj, 'group_size'):
            return original_call(self, x)

        # --- Router (unchanged) ---
        gates = self.gate(x)
        gates = mx.softmax(gates, axis=-1, precise=True)
        k = self.top_k
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if self.norm_topk_prob:
            scores = scores / scores.sum(axis=-1, keepdims=True)

        # --- Fused gate+up+SwiGLU (existing kernel) ---
        x_flat = x.reshape(-1)
        idx_flat = inds.reshape(-1).astype(mx.int32)

        gate_biases = gate_proj.get("biases")
        up_biases = up_proj.get("biases")
        if gate_biases is None:
            gate_biases = mx.zeros_like(gate_proj["scales"])
        if up_biases is None:
            up_biases = mx.zeros_like(up_proj["scales"])

        intermediate = gather_qmm_swiglu(
            x_flat,
            gate_proj["weight"], gate_proj["scales"], gate_biases,
            up_proj["weight"], up_proj["scales"], up_biases,
            idx_flat,
            top_k=k,
            group_size=gate_proj.group_size,
            bits=gate_proj.bits,
        )
        # intermediate: (n_tokens * top_k, intermediate_size)

        # --- Fused down_proj + score-weighted reduce (NEW kernel) ---
        scores_flat = scores.reshape(n_tokens * k)
        down_biases = down_proj.get("biases")
        if down_biases is None:
            down_biases = mx.zeros_like(down_proj["scales"])

        y = gather_qmm_down_reduce(
            intermediate,
            down_proj["weight"], down_proj["scales"], down_biases,
            idx_flat, scores_flat,
            top_k=k,
            group_size=down_proj.group_size,
            bits=down_proj.bits,
        )
        # y: (n_tokens, hidden_size) — already reduced across experts!
        y = y.reshape(*batch_shape, hidden_size)

        # --- Shared expert (fused: gate+up+SwiGLU → down+reduce) ---
        se = self.shared_expert
        se_gate_proj = se.gate_proj
        se_up_proj = se.up_proj
        se_down_proj = se.down_proj

        # Reshape 2D weights to 3D (1 expert) for gather kernels
        se_gw = se_gate_proj["weight"][None]
        se_gs = se_gate_proj["scales"][None]
        se_gb = se_gate_proj.get("biases")
        se_gb = mx.zeros_like(se_gate_proj["scales"])[None] if se_gb is None else se_gb[None]
        se_uw = se_up_proj["weight"][None]
        se_us = se_up_proj["scales"][None]
        se_ub = se_up_proj.get("biases")
        se_ub = mx.zeros_like(se_up_proj["scales"])[None] if se_ub is None else se_ub[None]
        se_dw = se_down_proj["weight"][None]
        se_ds = se_down_proj["scales"][None]
        se_db = se_down_proj.get("biases")
        se_db = mx.zeros_like(se_down_proj["scales"])[None] if se_db is None else se_db[None]

        se_idx = mx.zeros(n_tokens, dtype=mx.int32)
        se_scores = mx.ones((n_tokens, 1), dtype=se_gs.dtype)

        se_intermediate = gather_qmm_swiglu(
            x_flat, se_gw, se_gs, se_gb, se_uw, se_us, se_ub, se_idx,
            top_k=1, group_size=se_gate_proj.group_size, bits=se_gate_proj.bits,
        )
        shared_y = gather_qmm_down_reduce(
            se_intermediate, se_dw, se_ds, se_db, se_idx, se_scores,
            top_k=1, group_size=se_down_proj.group_size, bits=se_down_proj.bits,
        )
        shared_y = shared_y.reshape(*batch_shape, hidden_size)
        shared_y = mx.sigmoid(self.shared_expert_gate(x)) * shared_y

        y = y + shared_y

        if self.sharding_group is not None:
            y = mx.distributed.all_sum(y, group=self.sharding_group)

        return y

    return fused_moe_call


_original_moe_call = None


def patch_moe_full(model, verbose=True):
    """Patch all SparseMoeBlock instances to use both fused kernels."""
    global _original_moe_call

    from mlx_lm.models.qwen3_next import Qwen3NextSparseMoeBlock

    lm = model.language_model if hasattr(model, "language_model") else model
    layers = lm.model.layers if hasattr(lm, "model") else lm.layers

    _original_moe_call = Qwen3NextSparseMoeBlock.__call__
    fused_call = _make_fused_moe_call(_original_moe_call)
    Qwen3NextSparseMoeBlock.__call__ = fused_call

    patched = sum(
        1 for layer in layers
        if hasattr(layer, "mlp") and isinstance(layer.mlp, Qwen3NextSparseMoeBlock)
    )

    if verbose:
        print(f"[mlx_fused_moe] Patched {patched} SparseMoeBlock layers with full fused MoE")

    return patched


def unpatch_moe_full(model=None):
    """Restore original SparseMoeBlock forward."""
    global _original_moe_call

    from mlx_lm.models.qwen3_next import Qwen3NextSparseMoeBlock

    if _original_moe_call is not None:
        Qwen3NextSparseMoeBlock.__call__ = _original_moe_call
        _original_moe_call = None
