"""
Monkey-patch GatedDeltaNet to fuse the 4 input projections into one dispatch.

Each GatedDeltaNet layer computes 4 quantized linear projections on the same x:
  - in_proj_qkv: 2048 → 8192 (Q, K, V for the recurrence)
  - in_proj_z:   2048 → 4096 (gating)
  - in_proj_a:   2048 → 32   (decay parameter)
  - in_proj_b:   2048 → 32   (beta parameter)

All 4 read x from DRAM separately. The fused kernel pre-concatenates the weights
and computes all outputs in one dispatch, reading x once.

For decode (S=1), this eliminates 3 redundant x reads and 3 dispatch round-trips
across 30 GatedDeltaNet layers.

For prefill (S>1), falls back to the original implementation.
"""

import mlx.core as mx
import mlx.nn as nn
from mlx_fused_moe._ext import fused_qmv


def _concat_quantized_weights(layers):
    """Pre-concatenate quantized weights for the 4 input projections.

    Returns list of (cat_weight, cat_scales, cat_biases, split_indices, group_size, bits)
    per layer.
    """
    results = []
    for proj_list in layers:
        weights = []
        scales_list = []
        biases_list = []
        group_size = None
        bits = None

        for proj in proj_list:
            weights.append(proj["weight"])
            scales_list.append(proj["scales"])
            biases_list.append(
                proj.get("biases") if proj.get("biases") is not None
                else mx.zeros_like(proj["scales"])
            )
            if group_size is None:
                group_size = proj.group_size
                bits = proj.bits

        cat_weight = mx.concatenate(weights, axis=0)
        cat_scales = mx.concatenate(scales_list, axis=0)
        cat_biases = mx.concatenate(biases_list, axis=0)

        # Split indices: cumulative output dims (for mx.split)
        dims = [w.shape[0] for w in weights]
        split_indices = []
        cumsum = 0
        for d in dims[:-1]:
            cumsum += d
            split_indices.append(cumsum)

        results.append((cat_weight, cat_scales, cat_biases, split_indices, group_size, bits))

    mx.eval(*[r[0] for r in results], *[r[1] for r in results], *[r[2] for r in results])
    return results


def _make_fused_gdn_call(original_call):
    """Create a patched __call__ that uses fused projection for decode."""

    def fused_call(self, inputs, mask=None, cache=None):
        B, S, _ = inputs.shape

        # Only fuse for decode (S=1 or S=2 for MTP batch verify)
        if S > 4 or not hasattr(self, '_fused_proj_weight'):
            return original_call(self, inputs, mask, cache)

        # Fused projection: one dispatch instead of 4
        x_flat = inputs.reshape(-1)  # (n_tokens * hidden_size,)
        n_tokens = B * S

        combined = fused_qmv(
            x_flat,
            self._fused_proj_weight,
            self._fused_proj_scales,
            self._fused_proj_biases,
            n_tokens=n_tokens,
            group_size=self._fused_group_size,
            bits=self._fused_bits,
        )
        # combined: (n_tokens, total_out_dim)
        combined = combined.reshape(B, S, -1)

        # Split back into qkv, z, b, a
        qkv, z_flat, b_flat, a_flat = mx.split(
            combined, self._fused_split_indices, axis=-1
        )

        # Now continue with the rest of the original GatedDeltaNet forward
        # Reshape z, b, a to expected shapes
        z = z_flat.reshape(B, S, self.num_v_heads, self.head_v_dim)
        b = b_flat  # (B, S, num_v_heads)
        a = a_flat  # (B, S, num_v_heads)

        # Conv1d path (same as original)
        if cache is not None and cache[0] is not None:
            conv_state = cache[0]
        else:
            conv_state = mx.zeros(
                (B, self.conv_kernel_size - 1, self.conv_dim),
                dtype=inputs.dtype,
            )

        if mask is not None:
            qkv = mx.where(mask[..., None], qkv, 0)
        conv_input = mx.concatenate([conv_state, qkv], axis=1)
        if cache is not None:
            cache[0] = conv_input[:, -(self.conv_kernel_size - 1):]
        conv_out = nn.silu(self.conv1d(conv_input))

        q, k, v = [
            t.reshape(B, S, h, d)
            for t, h, d in zip(
                mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                [self.head_k_dim, self.head_k_dim, self.head_v_dim],
            )
        ]

        from mlx_lm.models.gated_delta import gated_delta_update

        state = cache[1] if cache else None
        inv_scale = k.shape[-1] ** -0.5
        q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)

        out, state = gated_delta_update(
            q, k, v, a, b,
            self.A_log, self.dt_bias,
            state, mask,
            use_kernel=not self.training,
        )

        if cache is not None:
            cache[1] = state

        out = self.norm(out, z)
        out = self.out_proj(out.reshape(B, S, -1))

        if self.sharding_group is not None:
            out = mx.distributed.all_sum(out, group=self.sharding_group)

        return out

    return fused_call


def patch_deltanet(model, verbose=True):
    """
    Patch all GatedDeltaNet layers to use fused input projections.

    Pre-concatenates weights at patch time, then swaps __call__ for decode.
    Returns the number of layers patched.
    """
    from mlx_lm.models.qwen3_5 import GatedDeltaNet

    lm = model.language_model if hasattr(model, "language_model") else model
    layers = lm.model.layers if hasattr(lm, "model") else lm.layers

    # Collect projection lists for all GDN layers
    gdn_layers = []
    proj_lists = []
    for layer in layers:
        gdn = getattr(layer, "linear_attn", None)
        if gdn is None or not isinstance(gdn, GatedDeltaNet):
            continue
        # Check if quantized
        if not hasattr(gdn.in_proj_qkv, "group_size"):
            continue
        gdn_layers.append(gdn)
        proj_lists.append([
            gdn.in_proj_qkv,
            gdn.in_proj_z,
            gdn.in_proj_b,
            gdn.in_proj_a,
        ])

    if not gdn_layers:
        if verbose:
            print("[mlx_fused_moe] No quantized GatedDeltaNet layers found")
        return 0

    # Pre-concatenate weights
    if verbose:
        print(f"[mlx_fused_moe] Concatenating weights for {len(gdn_layers)} GatedDeltaNet layers...")
    cat_data = _concat_quantized_weights(proj_lists)

    # Attach concatenated weights to each layer
    for gdn, (cat_w, cat_s, cat_b, split_idx, gs, bits) in zip(gdn_layers, cat_data):
        gdn._fused_proj_weight = cat_w
        gdn._fused_proj_scales = cat_s
        gdn._fused_proj_biases = cat_b
        gdn._fused_split_indices = split_idx
        gdn._fused_group_size = gs
        gdn._fused_bits = bits

    # Monkey-patch __call__
    original_call = GatedDeltaNet.__call__
    GatedDeltaNet._original_call = original_call
    GatedDeltaNet.__call__ = _make_fused_gdn_call(original_call)

    if verbose:
        print(f"[mlx_fused_moe] Patched {len(gdn_layers)} GatedDeltaNet layers with fused projections")

    return len(gdn_layers)


def unpatch_deltanet(model=None):
    """Restore original GatedDeltaNet forward."""
    from mlx_lm.models.qwen3_5 import GatedDeltaNet

    if hasattr(GatedDeltaNet, '_original_call'):
        GatedDeltaNet.__call__ = GatedDeltaNet._original_call
        del GatedDeltaNet._original_call
