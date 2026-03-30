#!/usr/bin/env python3
"""Test that the capture kernel produces identical results to the original."""

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.gated_delta import gated_delta_update, gated_delta_kernel, compute_g
from vllm_mlx_mtp.gdn_kernel import gated_delta_update_with_capture, gated_delta_kernel_with_capture

mx.random.seed(42)

# Qwen3.5 GDN dimensions
B, T, Hk, Dk = 1, 2, 4, 32
Hv, Dv = 8, 128  # Hv = 2 * Hk for this model

# Create random inputs
q = mx.random.normal((B, T, Hk, Dk)).astype(mx.bfloat16)
k = mx.random.normal((B, T, Hk, Dk)).astype(mx.bfloat16)
v = mx.random.normal((B, T, Hv, Dv)).astype(mx.bfloat16)
a = mx.random.normal((B, T, Hv))
b = mx.random.normal((B, T, Hv))
A_log = mx.random.normal((Hv,))
dt_bias = mx.random.normal((Hv,))
state = mx.random.normal((B, Hv, Dv, Dk)).astype(mx.bfloat16)
mask = mx.ones((B, T), dtype=mx.bool_)

# Cast to match
a = a.astype(mx.bfloat16)
b = b.astype(mx.bfloat16)
A_log = A_log.astype(mx.bfloat16)
dt_bias = dt_bias.astype(mx.bfloat16)

mx.eval(q, k, v, a, b, A_log, dt_bias, state, mask)

print("=== Test 1: T=2 output and final state match ===")
y_ref, state_ref = gated_delta_update(q, k, v, a, b, A_log, dt_bias, state, mask)
y_cap, state_cap, intermediates = gated_delta_update_with_capture(q, k, v, a, b, A_log, dt_bias, state, mask)
mx.eval(y_ref, state_ref, y_cap, state_cap, intermediates)

y_err = mx.abs(y_ref.astype(mx.float32) - y_cap.astype(mx.float32)).max().item()
s_err = mx.abs(state_ref.astype(mx.float32) - state_cap.astype(mx.float32)).max().item()
print(f"  y max error: {y_err:.6f}")
print(f"  state max error: {s_err:.6f}")
print(f"  intermediates shape: {intermediates.shape}")
assert y_err < 0.01, f"y error too large: {y_err}"
assert s_err < 0.01, f"state error too large: {s_err}"

print("\n=== Test 2: Intermediate state matches split approach ===")
# Run T=1 for first token to get intermediate state (reference)
y1_ref, state_after_t0 = gated_delta_update(
    q[:, 0:1], k[:, 0:1], v[:, 0:1], a[:, 0:1], b[:, 0:1],
    A_log, dt_bias, state, mask[:, 0:1]
)
mx.eval(y1_ref, state_after_t0)

# Compare with captured intermediate
intermediate_t0 = intermediates[:, 0]  # (B, Hv, Dv, Dk)
mx.eval(intermediate_t0)

im_err = mx.abs(state_after_t0.astype(mx.float32) - intermediate_t0.astype(mx.float32)).max().item()
print(f"  intermediate[0] vs split state: max error = {im_err:.6f}")
assert im_err < 0.01, f"intermediate error too large: {im_err}"

print("\n=== Test 3: T=3 with 2 intermediates ===")
T3 = 3
q3 = mx.random.normal((B, T3, Hk, Dk)).astype(mx.bfloat16)
k3 = mx.random.normal((B, T3, Hk, Dk)).astype(mx.bfloat16)
v3 = mx.random.normal((B, T3, Hv, Dv)).astype(mx.bfloat16)
a3 = mx.random.normal((B, T3, Hv)).astype(mx.bfloat16)
b3 = mx.random.normal((B, T3, Hv)).astype(mx.bfloat16)
mask3 = mx.ones((B, T3), dtype=mx.bool_)
mx.eval(q3, k3, v3, a3, b3, mask3)

y3_ref, s3_ref = gated_delta_update(q3, k3, v3, a3, b3, A_log, dt_bias, state, mask3)
y3_cap, s3_cap, im3 = gated_delta_update_with_capture(q3, k3, v3, a3, b3, A_log, dt_bias, state, mask3)
mx.eval(y3_ref, s3_ref, y3_cap, s3_cap, im3)

y3_err = mx.abs(y3_ref.astype(mx.float32) - y3_cap.astype(mx.float32)).max().item()
s3_err = mx.abs(s3_ref.astype(mx.float32) - s3_cap.astype(mx.float32)).max().item()
print(f"  y max error: {y3_err:.6f}")
print(f"  state max error: {s3_err:.6f}")
print(f"  intermediates shape: {im3.shape}")

# Check both intermediates via split
for t in range(T3 - 1):
    _, state_at_t = gated_delta_update(
        q3[:, :t+1], k3[:, :t+1], v3[:, :t+1], a3[:, :t+1], b3[:, :t+1],
        A_log, dt_bias, state, mask3[:, :t+1]
    )
    mx.eval(state_at_t)
    err = mx.abs(state_at_t.astype(mx.float32) - im3[:, t].astype(mx.float32)).max().item()
    print(f"  intermediate[{t}] vs split: max error = {err:.6f}")
    assert err < 0.01, f"intermediate[{t}] error too large: {err}"

print("\n=== Test 4: T=1 (no intermediates) ===")
y1, s1, im1 = gated_delta_update_with_capture(
    q[:, 0:1], k[:, 0:1], v[:, 0:1], a[:, 0:1], b[:, 0:1],
    A_log, dt_bias, state, mask[:, 0:1]
)
mx.eval(y1, s1, im1)
print(f"  intermediates shape: {im1.shape}")
y1_ref2, s1_ref2 = gated_delta_update(
    q[:, 0:1], k[:, 0:1], v[:, 0:1], a[:, 0:1], b[:, 0:1],
    A_log, dt_bias, state, mask[:, 0:1]
)
mx.eval(y1_ref2, s1_ref2)
y1_err = mx.abs(y1.astype(mx.float32) - y1_ref2.astype(mx.float32)).max().item()
s1_err = mx.abs(s1.astype(mx.float32) - s1_ref2.astype(mx.float32)).max().item()
print(f"  y max error: {y1_err:.6f}")
print(f"  state max error: {s1_err:.6f}")

print("\n=== Test 5: Without mask ===")
y_nm, s_nm = gated_delta_update(q, k, v, a, b, A_log, dt_bias, state, None)
y_nm_c, s_nm_c, im_nm = gated_delta_update_with_capture(q, k, v, a, b, A_log, dt_bias, state, None)
mx.eval(y_nm, s_nm, y_nm_c, s_nm_c, im_nm)
y_nm_err = mx.abs(y_nm.astype(mx.float32) - y_nm_c.astype(mx.float32)).max().item()
s_nm_err = mx.abs(s_nm.astype(mx.float32) - s_nm_c.astype(mx.float32)).max().item()
print(f"  y max error: {y_nm_err:.6f}")
print(f"  state max error: {s_nm_err:.6f}")

# Performance test
import time

print("\n=== Performance: T=2 ===")

beta = mx.sigmoid(b)
g = compute_g(A_log, a, dt_bias)
mx.eval(beta, g)

def time_it(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    mx.synchronize()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        mx.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return min(times)

# Original kernel
ms_orig = time_it(lambda: mx.eval(*gated_delta_kernel(q, k, v, g, beta, state, mask)))
print(f"  Original kernel (T=2): {ms_orig:.3f}ms")

# Capture kernel
ms_cap = time_it(lambda: mx.eval(*gated_delta_kernel_with_capture(q, k, v, g, beta, state, mask)))
print(f"  Capture kernel (T=2):  {ms_cap:.3f}ms")
print(f"  Overhead: {ms_cap - ms_orig:.3f}ms")

# Split approach (current)
from mlx_lm.models.gated_delta import gated_delta_update as gdu_ref
def run_split():
    y1, s1 = gdu_ref(q[:, 0:1], k[:, 0:1], v[:, 0:1], a[:, 0:1], b[:, 0:1], A_log, dt_bias, state, mask[:, 0:1])
    im = mx.array(s1)
    y2, s2 = gdu_ref(q[:, 1:2], k[:, 1:2], v[:, 1:2], a[:, 1:2], b[:, 1:2], A_log, dt_bias, s1, mask[:, 1:2])
    mx.eval(y1, y2, s2, im)

ms_split = time_it(run_split)
print(f"  Split approach (T=2):  {ms_split:.3f}ms")
print(f"  Savings vs split: {ms_split - ms_cap:.3f}ms")

print("\nAll tests passed!")
