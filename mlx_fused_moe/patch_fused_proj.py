"""
Fuse GDN input projections to reduce dispatch count.

The GatedDeltaNet layer has 4 input projections from the same input:
  in_proj_qkv (2048→8192)  — feeds conv1d, then becomes q, k, v
  in_proj_z   (2048→4096)  — gating for output normalization
  in_proj_b   (2048→32)    — sigmoid decay weights (beta)
  in_proj_a   (2048→32)    — decay rate parameters (alpha)

All 4 are Q4 quantized matmuls with the same group_size and bits.
By concatenating the weights, we replace 4 dispatches with 1 + split.

For 1-token decode, this saves ~0.04ms per layer × 30 layers = ~1.1ms total.
"""

import mlx.core as mx
import mlx.nn as nn

_patched_layers = []


def _make_fused_gdn_call(original_call, fused_W, fused_S, fused_B, split_indices, gs, bits):
    """Create a patched __call__ that uses fused input projection."""

    def patched_call(self, inputs, mask=None, cache=None):
        B, S, _ = inputs.shape

        if self.sharding_group is not None:
            from mlx_lm.models.qwen3_5 import sum_gradients
            inputs = sum_gradients(self.sharding_group)(inputs)

        # Fused projection: one matmul instead of 4
        fused_out = mx.quantized_matmul(
            inputs.reshape(-1, inputs.shape[-1]),
            fused_W, fused_S, fused_B,
            transpose=True, group_size=gs, bits=bits,
        ).reshape(B, S, -1)

        # Split back into individual projections
        qkv, z_flat, b, a = mx.split(fused_out, split_indices, axis=-1)

        z = z_flat.reshape(B, S, self.num_v_heads, self.head_v_dim)

        # Rest of the forward pass is identical
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

        state = cache[1] if cache else None
        inv_scale = k.shape[-1] ** -0.5
        q = (inv_scale ** 2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)

        from mlx_lm.models.gated_delta import gated_delta_update
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

    return patched_call


def patch_fused_proj(model, verbose=True):
    """Patch all GatedDeltaNet layers to use fused input projections."""
    global _patched_layers

    lm = model.language_model if hasattr(model, "language_model") else model
    layers = lm.model.layers if hasattr(lm, "model") else lm.layers

    count = 0
    for layer in layers:
        gdn = getattr(layer, "linear_attn", None)
        if gdn is None:
            continue

        # Check it has the 4 separate projections
        qkv_proj = getattr(gdn, "in_proj_qkv", None)
        z_proj = getattr(gdn, "in_proj_z", None)
        b_proj = getattr(gdn, "in_proj_b", None)
        a_proj = getattr(gdn, "in_proj_a", None)
        if any(p is None for p in [qkv_proj, z_proj, b_proj, a_proj]):
            continue

        gs = qkv_proj.group_size
        bits = qkv_proj.bits

        # Concatenate Q4 weights along output dimension
        projs = [qkv_proj, z_proj, b_proj, a_proj]
        W_fused = mx.concatenate([p["weight"] for p in projs], axis=0)
        S_fused = mx.concatenate([p["scales"] for p in projs], axis=0)

        B_parts = []
        for p in projs:
            b = p.get("biases")
            if b is not None:
                B_parts.append(b)
            else:
                B_parts.append(mx.zeros_like(p["scales"]))
        B_fused = mx.concatenate(B_parts, axis=0)

        # Split indices: cumulative output dims (excluding last)
        out_dims = [p["scales"].shape[0] for p in projs]
        split_indices = []
        cum = 0
        for d in out_dims[:-1]:
            cum += d
            split_indices.append(cum)

        # Save original and patch
        original_call = type(gdn).__call__
        _patched_layers.append((gdn, original_call))

        # Bind the fused call to this instance
        import types
        gdn.__call__ = types.MethodType(
            lambda self, inputs, mask=None, cache=None,
                   _fn=_make_fused_gdn_call(original_call, W_fused, S_fused, B_fused,
                                             split_indices, gs, bits):
                _fn(self, inputs, mask, cache),
            gdn,
        )

        count += 1

    if verbose:
        print(f"[patch_fused_proj] Patched {count} GDN layers with fused input projections")


def unpatch_fused_proj():
    """Restore original GDN __call__ methods."""
    global _patched_layers
    for gdn, original_call in _patched_layers:
        if hasattr(gdn, "__call__"):
            delattr(gdn, "__call__")
    _patched_layers = []
