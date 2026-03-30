"""
Custom gated_delta_update kernel that captures intermediate recurrent states.

The standard MLX kernel processes T timesteps and only outputs the final state.
This variant writes the state after each timestep to an intermediate buffer,
enabling zero-replay rejection in MTP speculative decoding without splitting
the T-token batch into T separate kernel calls.

For T=2 (the common MTP K=1 case), this saves 30 extra kernel dispatches
(one per GDN layer) = ~0.9ms per decode step.
"""

from typing import Optional, Tuple

import mlx.core as mx


def _make_capture_kernel(has_mask=False, vectorized=False):
    """Create a gated_delta kernel that writes intermediate states."""
    if not mx.metal.is_available():
        return None

    mask_source = "mask[b_idx * T + t]" if has_mask else "true"

    if vectorized:
        g_comment = "// g: [B, T, Hv, Dk]"
        g_setup = "auto g_ = g + (b_idx * T * Hv + hv_idx) * Dk;"
        g_access = "g_[s_idx]"
        g_advance = "g_ += Hv * Dk;"
    else:
        g_comment = "// g: [B, T, Hv]"
        g_setup = "auto g_ = g + b_idx * T * Hv;"
        g_access = "g_[hv_idx]"
        g_advance = "g_ += Hv;"

    # The intermediate_states buffer has shape (B, T-1, Hv, Dv, Dk)
    # For T=1 this buffer is empty (size 0), kernel skips writes.
    # Layout: [b_idx, t, hv_idx, dv_idx, dk_idx] contiguous in dk_idx.
    source = f"""
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        // q, k: [B, T, Hk, Dk]
        auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

        // v, y: [B, T, Hv, Dv]
        auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
        y += b_idx * T * Hv * Dv + hv_idx * Dv;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        // state_in, state_out: [B, Hv, Dv, Dk]
        auto i_state = state_in + (n * Dv + dv_idx) * Dk;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk;

        // intermediate_states: [B, T-1, Hv, Dv, Dk]
        // Stride: b -> (T-1)*Hv*Dv*Dk, t -> Hv*Dv*Dk, hv -> Dv*Dk, dv -> Dk
        auto im_stride_b = (T - 1) * Hv * Dv * Dk;
        auto im_base = intermediate_states + b_idx * im_stride_b
                        + hv_idx * Dv * Dk + dv_idx * Dk;

        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {{
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = static_cast<float>(i_state[s_idx]);
        }}

        {g_comment}
        {g_setup}
        auto beta_ = beta + b_idx * T * Hv;

        for (int t = 0; t < T; ++t) {{
          if ({mask_source}) {{
            float kv_mem = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {{
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = state[i] * {g_access};
              kv_mem += state[i] * k_[s_idx];
            }}
            kv_mem = simd_sum(kv_mem);

            auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

            float out = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {{
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = state[i] + k_[s_idx] * delta;
              out += state[i] * q_[s_idx];
            }}
            out = simd_sum(out);
            if (thread_index_in_simdgroup == 0) {{
              y[dv_idx] = static_cast<InT>(out);
            }}
          }}

          // Write intermediate state (all except last timestep)
          if (t < T - 1) {{
            auto im_ptr = im_base + t * Hv * Dv * Dk;
            for (int i = 0; i < n_per_t; ++i) {{
              auto s_idx = n_per_t * dk_idx + i;
              im_ptr[s_idx] = static_cast<InT>(state[i]);
            }}
          }}

          // Advance to next timestep
          q_ += Hk * Dk;
          k_ += Hk * Dk;
          v_ += Hv * Dv;
          y += Hv * Dv;
          {g_advance}
          beta_ += Hv;
        }}
        for (int i = 0; i < n_per_t; ++i) {{
          auto s_idx = n_per_t * dk_idx + i;
          o_state[s_idx] = static_cast<InT>(state[i]);
        }}
    """

    inputs = ["q", "k", "v", "g", "beta", "state_in", "T"]
    if has_mask:
        inputs.append("mask")

    suffix = "_capture"
    if vectorized:
        suffix += "_vec"
    if has_mask:
        suffix += "_mask"

    return mx.fast.metal_kernel(
        name=f"gated_delta_step{suffix}",
        input_names=inputs,
        output_names=["y", "state_out", "intermediate_states"],
        source=source,
    )


# Create kernel variants at module load
_capture_kernel = _make_capture_kernel(has_mask=False, vectorized=False)
_capture_kernel_masked = _make_capture_kernel(has_mask=True, vectorized=False)
_capture_kernel_vec = _make_capture_kernel(has_mask=False, vectorized=True)
_capture_kernel_vec_masked = _make_capture_kernel(has_mask=True, vectorized=True)


def gated_delta_kernel_with_capture(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
    mask: Optional[mx.array] = None,
) -> Tuple[mx.array, mx.array, mx.array]:
    """Run gated delta update and capture intermediate states.

    Returns:
        y: (B, T, Hv, Dv) — output
        state_out: (B, Hv, Dv, Dk) — final state
        intermediate_states: (B, T-1, Hv, Dv, Dk) — state after each token except last
    """
    B, T, Hk, Dk = k.shape
    Hv, Dv = v.shape[2:]
    input_type = q.dtype

    if g.ndim == 4:
        kernel = _capture_kernel_vec
        inputs = [q, k, v, g, beta, state, T]
        if mask is not None:
            kernel = _capture_kernel_vec_masked
            inputs.append(mask)
    else:
        kernel = _capture_kernel
        inputs = [q, k, v, g, beta, state, T]
        if mask is not None:
            kernel = _capture_kernel_masked
            inputs.append(mask)

    return kernel(
        inputs=inputs,
        template=[
            ("InT", input_type),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
        ],
        grid=(32, Dv, B * Hv),
        threadgroup=(32, 4, 1),
        output_shapes=[
            (B, T, Hv, Dv),           # y
            state.shape,               # state_out (B, Hv, Dv, Dk)
            (B, max(T - 1, 1), Hv, Dv, Dk),  # intermediate_states
        ],
        output_dtypes=[input_type, input_type, input_type],
    )


def gated_delta_update_with_capture(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    a: mx.array,
    b: mx.array,
    A_log: mx.array,
    dt_bias: mx.array,
    state: Optional[mx.array] = None,
    mask: Optional[mx.array] = None,
) -> Tuple[mx.array, mx.array, mx.array]:
    """Like gated_delta_update but also returns intermediate states.

    Returns:
        y: (B, T, Hv, Dv)
        state: (B, Hv, Dv, Dk) — final state
        intermediate_states: (B, T-1, Hv, Dv, Dk) — state after tokens 0..T-2
    """
    from mlx_lm.models.gated_delta import compute_g

    beta = mx.sigmoid(b)
    g = compute_g(A_log, a, dt_bias)
    if state is None:
        B, _, Hk, Dk = q.shape
        Hv, Dv = v.shape[-2:]
        state = mx.zeros((B, Hv, Dv, Dk), dtype=q.dtype)

    return gated_delta_kernel_with_capture(q, k, v, g, beta, state, mask)
