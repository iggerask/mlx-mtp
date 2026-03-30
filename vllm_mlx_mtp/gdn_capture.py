"""
GDN intermediate state capture for zero-replay rejection.

During a multi-token batch verify (e.g., [token_0, draft1]), the GDN recurrence
processes token_0 then draft1 sequentially. Normally only the final state is saved.

This module patches gated_delta_update to SPLIT multi-token batches into individual
single-token Metal kernel calls, saving intermediate recurrent states between them.
This produces bit-identical results to the batch kernel (same operations, same order)
while capturing the intermediate states needed for zero-cost rejection.

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
# List[List[mx.array]] — _intermediates[layer_idx][position] = recurrent state
_intermediates: List[List[mx.array]] = []

# Store original function for unpatching
_original_gated_delta_update = None


def _split_gated_delta_update(q, k, v, a, b, A_log, dt_bias,
                               state=None, mask=None, use_kernel=True):
    """Split multi-token batch into individual Metal kernel calls.

    For T>1 with capture enabled: processes each token separately through the
    ORIGINAL gated_delta_update (which uses the Metal kernel), saving the
    recurrent state after each token except the last.

    For T=1 or capture disabled: passes through to the original function unchanged.
    """
    B, T, *_ = q.shape

    if _capture_enabled and T > 1:
        layer_states = []
        ys = []

        for t in range(T):
            y_t, state = _original_gated_delta_update(
                q[:, t:t+1], k[:, t:t+1], v[:, t:t+1],
                a[:, t:t+1], b[:, t:t+1],
                A_log, dt_bias, state,
                mask[:, t:t+1] if mask is not None else None,
                use_kernel=use_kernel,
            )
            ys.append(y_t)
            if t < T - 1:
                # Save state after this token (intermediate for rejection)
                layer_states.append(mx.array(state))

        _intermediates.append(layer_states)
        y = mx.concatenate(ys, axis=1)
        return y, state
    else:
        return _original_gated_delta_update(
            q, k, v, a, b, A_log, dt_bias, state, mask, use_kernel,
        )


class GDNStateCapture:
    """Manages GDN intermediate state capture for zero-replay rejection.

    Splits multi-token GDN batch processing into individual Metal kernel calls,
    capturing intermediate recurrent states. Also saves/restores conv state.
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
        gd_module.gated_delta_update = _split_gated_delta_update
        q35_module.gated_delta_update = _split_gated_delta_update
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

        # Save pre-verify conv states (cache[0]) for each GDN layer
        self._saved_conv = []
        copies = []
        for c in cache:
            if isinstance(c, ArraysCache):
                if c.cache[0] is not None:
                    copy = mx.array(c.cache[0])
                    self._saved_conv.append(copy)
                    copies.append(copy)
                else:
                    self._saved_conv.append(None)
        if copies:
            mx.eval(*copies)

    def disable(self):
        """Disable intermediate state capture."""
        global _capture_enabled
        _capture_enabled = False

    def get_intermediates(self) -> List[mx.array]:
        """Get all captured intermediate arrays for mx.eval()."""
        flat = []
        for layer_states in _intermediates:
            flat.extend(layer_states)
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
                # Restore recurrent state
                if gdn_idx < len(_intermediates):
                    layer_states = _intermediates[gdn_idx]
                    if position < len(layer_states):
                        c.cache[1] = layer_states[position]

                # Restore conv state
                if gdn_idx < len(self._saved_conv) and self._saved_conv[gdn_idx] is not None:
                    c.cache[0] = self._compute_intermediate_conv(
                        self._saved_conv[gdn_idx], c.cache[0],
                        position, n_verify_tokens,
                    )

                gdn_idx += 1
