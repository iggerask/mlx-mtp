#!/usr/bin/env python3
"""
Benchmark fused gather_qmm_swiglu vs separate ops for MoE decode.

Tests a custom Metal kernel that fuses:
  gate_out = gather_qmm(x, gate_weight)
  up_out   = gather_qmm(x, up_weight)
  result   = silu(gate_out) * up_out

into a single Metal dispatch, eliminating intermediate DRAM writes.

We test this in isolation first to measure the per-MoE-block savings,
then estimate the full model impact.
"""

import time
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

MODEL_NAME = "mlx-community/Qwen3.5-35B-A3B-4bit"


def get_moe_params(model):
    """Extract MoE parameters from first layer."""
    tm = model.language_model if hasattr(model, "language_model") else model
    layer0 = tm.model.layers[0]
    moe = layer0["mlp"]

    switch = moe.switch_mlp
    gate_proj = switch.gate_proj
    up_proj = switch.up_proj
    down_proj = switch.down_proj

    # Get weight shapes
    # QuantizedSwitchLinear stores: weight (packed), scales, biases
    print(f"gate_proj weight keys: {list(gate_proj.keys())}")
    gw = gate_proj["weight"]
    gs = gate_proj["scales"]
    gb = gate_proj["biases"]
    print(f"  gate weight shape: {gw.shape}, dtype: {gw.dtype}")
    print(f"  gate scales shape: {gs.shape}, dtype: {gs.dtype}")
    print(f"  gate biases shape: {gb.shape}, dtype: {gb.dtype}")

    uw = up_proj["weight"]
    us = up_proj["scales"]
    ub = up_proj["biases"]
    print(f"  up weight shape:   {uw.shape}")

    dw = down_proj["weight"]
    ds = down_proj["scales"]
    db = down_proj["biases"]
    print(f"  down weight shape: {dw.shape}")

    hidden_size = tm.args.hidden_size
    print(f"  hidden_size: {hidden_size}")
    print(f"  top_k: {moe.top_k}")
    print(f"  num_experts: {moe.num_experts}")

    return moe, switch, gate_proj, up_proj, down_proj


def benchmark_separate_ops(moe, switch, x, expert_inds, N=50):
    """Benchmark the current separate gather_qmm approach."""
    mx.eval(x, expert_inds)

    # Warmup
    for _ in range(10):
        out = switch(x, expert_inds)
        mx.eval(out)

    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        out = switch(x, expert_inds)
        mx.eval(out)
    mx.synchronize()
    t1 = time.perf_counter()

    return (t1 - t0) / N * 1000


def benchmark_full_moe(moe, x, N=50):
    """Benchmark the full MoE block (router + switch + shared expert)."""
    mx.eval(x)

    for _ in range(10):
        out = moe(x)
        mx.eval(out)

    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        out = moe(x)
        mx.eval(out)
    mx.synchronize()
    t1 = time.perf_counter()

    return (t1 - t0) / N * 1000


def build_fused_kernel():
    """Build fused gather_qmm_swiglu kernel using mx.fast.metal_kernel."""
    # This kernel computes gate+up+SwiGLU for one output element per thread.
    # For decode (seq_len=1): input is [1, 1, hidden_size],
    # output is [top_k, intermediate_size].
    kernel = mx.fast.metal_kernel(
        name="fused_gather_qmm_swiglu",
        input_names=[
            "x",           # [hidden_size] float16
            "gate_weight", # [n_experts, intermediate_size, hidden_size/8] uint32 packed
            "gate_scales", # [n_experts, intermediate_size, n_groups] float16
            "gate_biases", # [n_experts, intermediate_size, n_groups] float16
            "up_weight",   # same layout as gate
            "up_scales",
            "up_biases",
            "expert_inds", # [top_k] int32
        ],
        output_names=["output"],  # [top_k, intermediate_size] float16
        source=r"""
            // Thread computes one output element for one expert
            uint out_d = thread_position_in_grid.x;       // output dimension
            uint expert_slot = thread_position_in_grid.y;  // which active expert

            // Read expert index
            int expert_id = expert_inds[expert_slot];

            // Constants derived from input shapes
            // gate_scales shape: [n_experts, intermediate_size, n_groups]
            uint n_groups_per_row = gate_scales_shape[2];
            uint intermediate_size = gate_scales_shape[1];
            // gate_weight shape: [n_experts, intermediate_size, packed_input_dim]
            uint packed_input_dim = gate_weight_shape[2];
            uint pack_factor = 8;  // 4-bit quantization: 8 values per uint32
            uint input_dim = packed_input_dim * pack_factor;
            uint group_size = input_dim / n_groups_per_row;

            if (out_d >= intermediate_size) return;

            // Weight row: [expert_id, out_d, :packed_input_dim]
            long gw_offset = (long)expert_id * intermediate_size * packed_input_dim
                           + (long)out_d * packed_input_dim;
            long uw_offset = gw_offset;  // same layout

            // Scale row: [expert_id, out_d, :n_groups]
            long sc_offset = (long)expert_id * intermediate_size * n_groups_per_row
                           + (long)out_d * n_groups_per_row;

            float gate_acc = 0.0f;
            float up_acc = 0.0f;

            for (uint g = 0; g < n_groups_per_row; g++) {
                float g_scale = (float)gate_scales[sc_offset + g];
                float g_bias  = (float)gate_biases[sc_offset + g];
                float u_scale = (float)up_scales[sc_offset + g];
                float u_bias  = (float)up_biases[sc_offset + g];

                uint packed_start = g * (group_size / pack_factor);
                uint n_packed = group_size / pack_factor;

                for (uint p = 0; p < n_packed; p++) {
                    uint gate_packed = gate_weight[gw_offset + packed_start + p];
                    uint up_packed   = up_weight[uw_offset + packed_start + p];

                    uint base_idx = g * group_size + p * pack_factor;

                    for (uint b = 0; b < pack_factor; b++) {
                        float gw_val = float((gate_packed >> (b * 4)) & 0xF) * g_scale + g_bias;
                        float uw_val = float((up_packed >> (b * 4)) & 0xF) * u_scale + u_bias;
                        float xv = (float)x[base_idx + b];
                        gate_acc += xv * gw_val;
                        up_acc   += xv * uw_val;
                    }
                }
            }

            // SwiGLU: silu(gate) * up
            float result = (gate_acc / (1.0f + exp(-gate_acc))) * up_acc;

            output[expert_slot * intermediate_size + out_d] = (T)result;
        """,
        ensure_row_contiguous=True,
    )
    return kernel


def benchmark_fused_kernel(kernel, gate_proj, up_proj, x_flat, expert_inds, N=50):
    """Benchmark the fused kernel."""
    gw = gate_proj["weight"]
    gs = gate_proj["scales"]
    gb = gate_proj["biases"]
    uw = up_proj["weight"]
    us = up_proj["scales"]
    ub = up_proj["biases"]

    # Output shape: [top_k, intermediate_size]
    top_k = expert_inds.shape[-1]
    intermediate_size = gs.shape[1]  # scales shape: [n_experts, intermediate_size, n_groups]

    inds_flat = expert_inds.reshape(-1)

    # Warmup
    for _ in range(10):
        result = kernel(
            inputs=[x_flat, gw, gs, gb, uw, us, ub, inds_flat],
            template=[("T", mx.float16)],
            grid=(intermediate_size, top_k, 1),
            threadgroup=(min(intermediate_size, 256), 1, 1),
            output_shapes=[(top_k, intermediate_size)],
            output_dtypes=[mx.float16],
        )
        mx.eval(result[0])

    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        result = kernel(
            inputs=[x_flat, gw, gs, gb, uw, us, ub, inds_flat],
            template=[("T", mx.float16)],
            grid=(intermediate_size, top_k, 1),
            threadgroup=(min(intermediate_size, 256), 1, 1),
            output_shapes=[(top_k, intermediate_size)],
            output_dtypes=[mx.float16],
        )
        mx.eval(result[0])
    mx.synchronize()
    t1 = time.perf_counter()

    return (t1 - t0) / N * 1000, result[0]


def verify_correctness(kernel, moe, switch, gate_proj, up_proj, x, expert_inds):
    """Check that fused kernel matches separate ops."""
    # Reference: SwitchGLU(x, expert_inds)
    ref = switch(x, expert_inds)
    mx.eval(ref)

    # Fused kernel
    x_flat = x.reshape(-1)
    inds_flat = expert_inds.reshape(-1)
    top_k = expert_inds.shape[-1]
    intermediate_size = gate_proj["scales"].shape[1]

    result = kernel(
        inputs=[x_flat, gate_proj["weight"], gate_proj["scales"], gate_proj["biases"],
                up_proj["weight"], up_proj["scales"], up_proj["biases"], inds_flat],
        template=[("T", mx.float16)],
        grid=(intermediate_size, top_k, 1),
        threadgroup=(min(intermediate_size, 256), 1, 1),
        output_shapes=[(top_k, intermediate_size)],
        output_dtypes=[mx.float16],
    )
    mx.eval(result[0])

    # Note: the fused kernel only computes gate+up+SwiGLU (not down_proj)
    # The reference SwitchGLU includes down_proj too.
    # So we can only compare intermediate results.

    # Instead, compute reference gate+up+SwiGLU manually
    x_exp = mx.expand_dims(x, (-2, -3))  # [1, 1, 1, 1, hidden_size]
    x_up = up_proj(x_exp, expert_inds, sorted_indices=False)
    x_gate = gate_proj(x_exp, expert_inds, sorted_indices=False)
    # SwiGLU activation
    ref_swiglu = nn.silu(x_gate) * x_up
    mx.eval(ref_swiglu)

    # Reshape for comparison
    ref_vals = ref_swiglu.reshape(top_k, -1)
    fused_vals = result[0]

    # Compare
    diff = mx.abs(ref_vals.astype(mx.float32) - fused_vals.astype(mx.float32))
    mx.eval(diff)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"  Max abs diff:  {max_diff:.6f}")
    print(f"  Mean abs diff: {mean_diff:.6f}")
    print(f"  Ref range:     [{ref_vals.min().item():.4f}, {ref_vals.max().item():.4f}]")
    print(f"  Fused range:   [{fused_vals.min().item():.4f}, {fused_vals.max().item():.4f}]")

    ok = max_diff < 0.1  # Allow some tolerance for quantization
    print(f"  Correctness:   {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    print("Loading model...")
    model, tok = load(MODEL_NAME)

    moe, switch, gate_proj, up_proj, down_proj = get_moe_params(model)
    tm = model.language_model if hasattr(model, "language_model") else model
    hidden_size = tm.args.hidden_size

    # Create test inputs
    x = mx.random.normal((1, 1, hidden_size)).astype(mx.float16)
    gates = moe.gate(x)
    gates = mx.softmax(gates, axis=-1, precise=True)
    k = moe.top_k
    expert_inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
    mx.eval(x, expert_inds)

    print(f"\nExpert indices shape: {expert_inds.shape}")
    print(f"Selected experts: {expert_inds.reshape(-1).tolist()}")

    # Build fused kernel
    print("\nBuilding fused kernel...")
    kernel = build_fused_kernel()

    # Correctness check
    print("\nCorrectness verification:")
    ok = verify_correctness(kernel, moe, switch, gate_proj, up_proj, x, expert_inds)
    if not ok:
        print("CORRECTNESS FAILED — aborting benchmark")
        return

    # Benchmark
    print(f"\n{'='*60}")
    print("Performance Comparison (isolated, per-eval)")
    print(f"{'='*60}")

    # Separate ops (SwitchGLU = gate+up+SwiGLU+down)
    switch_ms = benchmark_separate_ops(moe, switch, x, expert_inds)
    print(f"  SwitchGLU (gate+up+SwiGLU+down): {switch_ms:.3f}ms")

    # Full MoE block
    moe_ms = benchmark_full_moe(moe, x)
    print(f"  Full MoE block:                  {moe_ms:.3f}ms")

    # Fused kernel (gate+up+SwiGLU only, no down_proj)
    x_flat = x.reshape(-1)
    fused_ms, _ = benchmark_fused_kernel(kernel, gate_proj, up_proj, x_flat, expert_inds)
    print(f"  Fused gate+up+SwiGLU:            {fused_ms:.3f}ms")

    # Estimate: what fraction of SwitchGLU is gate+up+SwiGLU vs down_proj?
    # SwitchGLU does 3 gather_qmm (gate, up, down) + activation
    # gate+up+SwiGLU = 2/3 of the matmul work + activation
    estimated_gate_up_ms = switch_ms * 2 / 3
    savings_per_block = estimated_gate_up_ms - fused_ms
    print(f"\n  Estimated gate+up+SwiGLU (2/3 of switch): {estimated_gate_up_ms:.3f}ms")
    print(f"  Fused kernel savings per block: {savings_per_block:.3f}ms")
    print(f"  40 layers × savings: {savings_per_block * 40:.2f}ms")

    # But remember: isolated eval has ~100μs overhead per call
    # In the full model graph (one eval), this overhead is amortized
    print(f"\n  NOTE: These isolated timings include ~100μs per-eval overhead.")
    print(f"  In the full model (one eval for all 2612 ops), per-op overhead is ~15μs.")
    print(f"  Realistic in-graph savings would be much smaller.")

    # Full model decode step for reference
    cache = make_prompt_cache(model)
    prompt = mx.array(tok.encode("Hello"))
    logits = model(prompt[None], cache=cache)
    mx.eval(logits, *[c.state for c in cache if hasattr(c, "state")])

    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        logits = model(mx.array([[1]]), cache=cache)
        mx.eval(logits)
    mx.synchronize()
    t1 = time.perf_counter()
    full_step_ms = (t1 - t0) / 20 * 1000
    print(f"\n  Full model decode step: {full_step_ms:.1f}ms")
    print(f"  Theoretical: {full_step_ms}ms - {savings_per_block * 40:.1f}ms = {full_step_ms - savings_per_block * 40:.1f}ms")
    print(f"  BUT this requires graph-native integration (C++ primitive), not standalone kernel")


if __name__ == "__main__":
    main()
