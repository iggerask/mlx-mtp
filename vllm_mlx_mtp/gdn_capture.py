"""
GDN intermediate state capture for zero-replay rejection.

During a multi-token batch verify (e.g., [token_0, draft1]), the GDN recurrence
processes token_0 then draft1 sequentially. Normally only the final state is saved.

This module uses a custom Metal kernel that outputs intermediate recurrent states
alongside the normal output, eliminating the need to split multi-token batches
into individual kernel calls. This saves ~30 extra kernel dispatches per step
(~0.9ms on M4 Pro) compared to the split approach.

Also saves/restores conv sliding window state for exact cache restoration.

Usage:
    capture = GDNStateCapture(model)
    capture.patch()
    ...
    capture.prepare(cache)            # saves pre-verify conv states, enables capture
    logits = model(verify_input, cache=cache)
    intermediates = capture.get_intermediates()
    mx.eval(logits, *intermediates)
    capture.disable()

    # On rejection:
    capture.restore(cache, position=0, n_kv_trim=1)
"""

from typing import Any, List, Optional, Tuple

import mlx.core as mx
from mlx_lm.models.cache import KVCache, ArraysCache

# Global state for capture (set during verify forward)
_capture_enabled = False
# List[mx.array] — _intermediates[layer_idx] = (B, T-1, Hv, Dv, Dk) tensor
_intermediates: List[mx.array] = []

# Store original function for unpatching
_original_gated_delta_update = None


def _capture_gated_delta_update(q, k, v, a, b, A_log, dt_bias,
                                state=None, mask=None, use_kernel=True):
    """Gated delta update that captures intermediate states via custom kernel.

    For T>1 with capture enabled: uses the capture kernel variant that outputs
    both the normal result and intermediate recurrent states in a single dispatch.

    For T=1 or capture disabled: passes through to the original function unchanged.
    """
    B, T, *_ = q.shape

    if _capture_enabled and T > 1:
        from .gdn_kernel import gated_delta_update_with_capture
        y, final_state, intermediates = gated_delta_update_with_capture(
            q, k, v, a, b, A_log, dt_bias, state, mask
        )
        _intermediates.append(intermediates)
        return y, final_state
    else:
        return _original_gated_delta_update(
            q, k, v, a, b, A_log, dt_bias, state, mask, use_kernel,
        )


class GDNStateCapture:
    """Manages GDN intermediate state capture for zero-replay rejection.

    Uses a custom Metal kernel that captures intermediate recurrent states
    during multi-token processing. This avoids splitting the batch into
    individual kernel calls, saving ~30 dispatches per decode step.
    """

    def __init__(self, model):
        self.model = model
        self._patched = False
        self._saved_conv: List[Optional[mx.array]] = []
        self._conv_kernel_size = 4  # Qwen3.5 default

    def patch(self):
        """Install monkey patches on gated_delta_update."""
        if self._patched:
            return
        global _original_gated_delta_update
        import mlx_lm.models.gated_delta as gd_module
        import mlx_lm.models.qwen3_5 as q35_module
        _original_gated_delta_update = gd_module.gated_delta_update
        gd_module.gated_delta_update = _capture_gated_delta_update
        q35_module.gated_delta_update = _capture_gated_delta_update
        self._patched = True

    def unpatch(self):
        """Remove monkey patches."""
        if not self._patched:
            return
        global _original_gated_delta_update
        import mlx_lm.models.gated_delta as gd_module
        import mlx_lm.models.qwen3_5 as q35_module
        gd_module.gated_delta_update = _original_gated_delta_update
        q35_module.gated_delta_update = _original_gated_delta_update
        _original_gated_delta_update = None
        self._patched = False

    def prepare(self, cache):
        """Save pre-verify conv states and enable capture.

        Must be called BEFORE the verify forward pass.
        """
        global _capture_enabled, _intermediates
        _capture_enabled = True
        _intermediates = []

        # Save pre-verify conv states (cache[0]) for each GDN layer.
        # Uses mx.array() to snapshot current graph node — lazy eval preserves
        # the correct pre-forward value even though the forward will overwrite cache.
        self._saved_conv = []
        self._conv_copies = []  # kept alive for eval in main graph
        for c in cache:
            if isinstance(c, ArraysCache):
                if c.cache[0] is not None:
                    copy = mx.array(c.cache[0])
                    self._saved_conv.append(copy)
                    self._conv_copies.append(copy)
                else:
                    self._saved_conv.append(None)

    def disable(self):
        """Disable intermediate state capture."""
        global _capture_enabled
        _capture_enabled = False

    def get_intermediates(self) -> List[mx.array]:
        """Get all captured intermediate arrays for mx.eval().

        Includes conv state copies that were snapshotted before verify forward.
        """
        flat = list(self._conv_copies)
        flat.extend(_intermediates)
        return flat

    def has_intermediates(self) -> bool:
        """Check if intermediate states were captured."""
        return len(_intermediates) > 0

    def _compute_intermediate_conv(self, saved_conv, current_conv, position, n_verify_tokens):
        """Compute intermediate conv state from saved and current.

        For kernel_size=4 (K-1=3 conv state rows) and N verify tokens:
        - saved_conv = last 3 rows BEFORE verify
        - current_conv = last 3 rows AFTER verify (includes all N tokens)
        - intermediate at position p = last 3 rows after processing p+1 tokens

        Formula: concat(saved_conv[p+1:], current_conv[token_start:token_start+p+1])
        where token_start = max(0, 3 - N) is where tokens begin in current_conv.
        """
        K1 = self._conv_kernel_size - 1  # 3
        N = n_verify_tokens
        p = position
        token_start = max(0, K1 - N)

        n_from_saved = max(0, K1 - (p + 1))
        if n_from_saved > 0:
            saved_part = saved_conv[:, (p + 1):, :]
            current_part = current_conv[:, token_start: token_start + (p + 1), :]
            return mx.concatenate([saved_part, current_part], axis=1)
        else:
            return current_conv[:, token_start: token_start + K1, :]

    def restore(self, cache, position: int = 0, n_kv_trim: int = 1):
        """Restore cache to an intermediate state (recurrent + conv + KV trim)."""
        n_verify_tokens = n_kv_trim + position + 1

        gdn_idx = 0
        for c in cache:
            if isinstance(c, KVCache):
                c.offset = max(0, c.offset - n_kv_trim)
            elif isinstance(c, ArraysCache):
                # Restore recurrent state from captured intermediates
                if gdn_idx < len(_intermediates):
                    im = _intermediates[gdn_idx]  # (B, T-1, Hv, Dv, Dk)
                    if position < im.shape[1]:
                        c.cache[1] = im[:, position]

                # Restore conv state
                if gdn_idx < len(self._saved_conv) and self._saved_conv[gdn_idx] is not None:
                    c.cache[0] = self._compute_intermediate_conv(
                        self._saved_conv[gdn_idx], c.cache[0],
                        position, n_verify_tokens,
                    )

                gdn_idx += 1
