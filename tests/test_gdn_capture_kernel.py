"""Tests for the GDN capture kernel — both unit and integration."""

import pytest
import mlx.core as mx
from mlx_lm.models.gated_delta import gated_delta_update, gated_delta_kernel, compute_g
from vllm_mlx_mtp.gdn_kernel import (
    gated_delta_update_with_capture,
    gated_delta_kernel_with_capture,
)


# Qwen3.5 GDN dimensions
B, Hk, Dk = 1, 4, 32
Hv, Dv = 8, 128


@pytest.fixture
def gdn_inputs():
    """Create random GDN inputs matching Qwen3.5 dimensions."""
    mx.random.seed(42)
    state = mx.random.normal((B, Hv, Dv, Dk)).astype(mx.bfloat16)
    A_log = mx.random.normal((Hv,)).astype(mx.bfloat16)
    dt_bias = mx.random.normal((Hv,)).astype(mx.bfloat16)
    mx.eval(state, A_log, dt_bias)
    return state, A_log, dt_bias


def _make_token_inputs(T):
    q = mx.random.normal((B, T, Hk, Dk)).astype(mx.bfloat16)
    k = mx.random.normal((B, T, Hk, Dk)).astype(mx.bfloat16)
    v = mx.random.normal((B, T, Hv, Dv)).astype(mx.bfloat16)
    a = mx.random.normal((B, T, Hv)).astype(mx.bfloat16)
    b = mx.random.normal((B, T, Hv)).astype(mx.bfloat16)
    mask = mx.ones((B, T), dtype=mx.bool_)
    mx.eval(q, k, v, a, b, mask)
    return q, k, v, a, b, mask


class TestCaptureKernelCorrectness:
    """Verify capture kernel produces bit-identical output and correct intermediates."""

    def test_t2_output_matches(self, gdn_inputs):
        state, A_log, dt_bias = gdn_inputs
        q, k, v, a, b, mask = _make_token_inputs(2)

        y_ref, s_ref = gated_delta_update(q, k, v, a, b, A_log, dt_bias, state, mask)
        y_cap, s_cap, im = gated_delta_update_with_capture(
            q, k, v, a, b, A_log, dt_bias, state, mask
        )
        mx.eval(y_ref, s_ref, y_cap, s_cap, im)

        assert mx.allclose(y_ref, y_cap, atol=1e-4).item()
        assert mx.allclose(s_ref, s_cap, atol=1e-4).item()

    def test_t2_intermediate_matches_split(self, gdn_inputs):
        state, A_log, dt_bias = gdn_inputs
        q, k, v, a, b, mask = _make_token_inputs(2)

        # Get intermediate via capture kernel
        _, _, im = gated_delta_update_with_capture(
            q, k, v, a, b, A_log, dt_bias, state, mask
        )
        mx.eval(im)

        # Get intermediate via split (process only token 0)
        _, state_after_t0 = gated_delta_update(
            q[:, 0:1], k[:, 0:1], v[:, 0:1], a[:, 0:1], b[:, 0:1],
            A_log, dt_bias, state, mask[:, 0:1],
        )
        mx.eval(state_after_t0)

        assert im.shape == (B, 1, Hv, Dv, Dk)
        assert mx.allclose(im[:, 0], state_after_t0, atol=1e-4).item()

    def test_t3_intermediates(self, gdn_inputs):
        state, A_log, dt_bias = gdn_inputs
        q, k, v, a, b, mask = _make_token_inputs(3)

        y_ref, s_ref = gated_delta_update(q, k, v, a, b, A_log, dt_bias, state, mask)
        y_cap, s_cap, im = gated_delta_update_with_capture(
            q, k, v, a, b, A_log, dt_bias, state, mask
        )
        mx.eval(y_ref, s_ref, y_cap, s_cap, im)

        assert mx.allclose(y_ref, y_cap, atol=1e-4).item()
        assert im.shape == (B, 2, Hv, Dv, Dk)

        # Verify each intermediate via split
        for t in range(2):
            _, state_at_t = gated_delta_update(
                q[:, :t + 1], k[:, :t + 1], v[:, :t + 1],
                a[:, :t + 1], b[:, :t + 1],
                A_log, dt_bias, state, mask[:, :t + 1],
            )
            mx.eval(state_at_t)
            assert mx.allclose(im[:, t], state_at_t, atol=1e-4).item(), (
                f"intermediate[{t}] mismatch"
            )

    def test_t1_no_intermediates(self, gdn_inputs):
        state, A_log, dt_bias = gdn_inputs
        q, k, v, a, b, mask = _make_token_inputs(1)

        y_ref, s_ref = gated_delta_update(q, k, v, a, b, A_log, dt_bias, state, mask)
        y_cap, s_cap, im = gated_delta_update_with_capture(
            q, k, v, a, b, A_log, dt_bias, state, mask
        )
        mx.eval(y_ref, s_ref, y_cap, s_cap, im)

        assert mx.allclose(y_ref, y_cap, atol=1e-4).item()
        assert mx.allclose(s_ref, s_cap, atol=1e-4).item()

    def test_no_mask(self, gdn_inputs):
        state, A_log, dt_bias = gdn_inputs
        q, k, v, a, b, _ = _make_token_inputs(2)

        y_ref, s_ref = gated_delta_update(q, k, v, a, b, A_log, dt_bias, state, None)
        y_cap, s_cap, im = gated_delta_update_with_capture(
            q, k, v, a, b, A_log, dt_bias, state, None
        )
        mx.eval(y_ref, s_ref, y_cap, s_cap, im)

        assert mx.allclose(y_ref, y_cap, atol=1e-4).item()
        assert mx.allclose(s_ref, s_cap, atol=1e-4).item()
