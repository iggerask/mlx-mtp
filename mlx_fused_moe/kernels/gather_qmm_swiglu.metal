/**
 * Fused Gather-QMV-SwiGLU Metal Kernel (SIMD-optimized, multi-token)
 *
 * Fuses three operations into a single dispatch for MoE decode:
 *   1. gather_qmv(x, gate_weight) -> gate_out
 *   2. gather_qmv(x, up_weight)   -> up_out
 *   3. silu(gate_out) * up_out     -> final_out
 *
 * Uses the same SIMD group tiling strategy as MLX's qmv_fast:
 *   - 2 simdgroups per threadgroup, each computing 4 output rows
 *   - Cooperative x loading across 32 threads (512 elements per iteration)
 *   - Pre-division trick for fast 4-bit dequantization
 *   - simd_sum for cross-thread reduction
 *
 * Supports 1-N tokens (single decode or MTP batch verify).
 * Grid: (n_tokens, ceil(output_dim/8), top_k)
 *
 * Weight format: 4-bit quantized (MLX affine format)
 *   - weight: packed uint32, each holding 8 x 4-bit values
 *   - scales/biases: float16 or bfloat16, per group
 *   - group_size: typically 64
 */

#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

static constant constexpr const int SIMD_SIZE = 32;

inline float silu(float x) {
    return x / (1.0f + exp(-x));
}

template<typename T>
void gather_qmv_swiglu_impl(
    const device T* x,
    const device uint32_t* gate_weight,
    const device T* gate_scales,
    const device T* gate_biases,
    const device uint32_t* up_weight,
    const device T* up_scales,
    const device T* up_biases,
    const device int* expert_inds,
    device T* output,
    constant int& input_dim,
    constant int& output_dim,
    constant int& group_size,
    constant int& n_experts,
    constant int& top_k,
    constant int& n_tokens,
    uint3 tid,
    uint simd_gid,
    uint simd_lid)
{
    // Constants for 4-bit quantization (matching MLX qmv_fast)
    constexpr int packs_per_thread = 2;
    constexpr int pack_factor = 8;       // 32 / 4 bits
    constexpr int bytes_per_pack = 4;    // 32 / 8
    constexpr int values_per_thread = pack_factor * packs_per_thread; // 16
    constexpr int block_size = values_per_thread * SIMD_SIZE;         // 512
    constexpr int num_simdgroups = 2;
    constexpr int results_per_simdgroup = 4;

    int token_idx = tid.x;       // which token (0..n_tokens-1)
    int expert_slot = tid.z;     // which expert for this token (0..top_k-1)

    int expert_id = expert_inds[token_idx * top_k + expert_slot];
    if (expert_id < 0 || expert_id >= n_experts) return;

    int out_row = tid.y * (num_simdgroups * results_per_simdgroup)
                + simd_gid * results_per_simdgroup;
    if (out_row >= output_dim) return;

    // Byte-stride calculations
    int in_vec_size_w = input_dim * bytes_per_pack / pack_factor; // bytes per weight row
    int in_vec_size_g = input_dim / group_size;                   // groups per row
    int scale_step = group_size / values_per_thread;              // threads per group

    // Expert base offsets (shared across tokens — weights are per-expert, not per-token)
    long expert_w_byte = (long)expert_id * output_dim * in_vec_size_w;
    long expert_s_off  = (long)expert_id * output_dim * in_vec_size_g;

    // Weight pointers (byte-addressed via uint8_t*)
    const device uint8_t* g_ws = (const device uint8_t*)gate_weight
        + expert_w_byte + out_row * in_vec_size_w
        + simd_lid * packs_per_thread * bytes_per_pack;
    const device uint8_t* u_ws = (const device uint8_t*)up_weight
        + expert_w_byte + out_row * in_vec_size_w
        + simd_lid * packs_per_thread * bytes_per_pack;

    // Scale/bias pointers
    const device T* g_sl = gate_scales + expert_s_off
        + out_row * in_vec_size_g + simd_lid / scale_step;
    const device T* g_bl = gate_biases + expert_s_off
        + out_row * in_vec_size_g + simd_lid / scale_step;
    const device T* u_sl = up_scales + expert_s_off
        + out_row * in_vec_size_g + simd_lid / scale_step;
    const device T* u_bl = up_biases + expert_s_off
        + out_row * in_vec_size_g + simd_lid / scale_step;

    // Input pointer — offset by token index
    const device T* xp = x + token_idx * input_dim + simd_lid * values_per_thread;

    float x_thread[values_per_thread];
    float gate_acc[results_per_simdgroup] = {0};
    float up_acc[results_per_simdgroup] = {0};

    // Main loop: cooperative x loading + dual dot products
    for (int k = 0; k < input_dim; k += block_size) {
        // Load x with pre-division trick for 4-bit
        float x_sum = 0;
        for (int i = 0; i < values_per_thread; i += 4) {
            float v0 = (float)xp[i];
            float v1 = (float)xp[i + 1];
            float v2 = (float)xp[i + 2];
            float v3 = (float)xp[i + 3];
            x_sum += v0 + v1 + v2 + v3;
            x_thread[i]     = v0;
            x_thread[i + 1] = v1 / 16.0f;
            x_thread[i + 2] = v2 / 256.0f;
            x_thread[i + 3] = v3 / 4096.0f;
        }

        // Dot product for each of 4 output rows (both gate and up)
        for (int row = 0; row < results_per_simdgroup; row++) {
            const device uint16_t* gw = (const device uint16_t*)(g_ws + row * in_vec_size_w);
            const device uint16_t* uw = (const device uint16_t*)(u_ws + row * in_vec_size_w);
            float gs = (float)g_sl[row * in_vec_size_g];
            float gb = (float)g_bl[row * in_vec_size_g];
            float us = (float)u_sl[row * in_vec_size_g];
            float ub = (float)u_bl[row * in_vec_size_g];

            float g_dot = 0, u_dot = 0;
            for (int i = 0; i < values_per_thread / 4; i++) {
                g_dot += x_thread[4 * i]     * (float)(gw[i] & 0x000f)
                       + x_thread[4 * i + 1] * (float)(gw[i] & 0x00f0)
                       + x_thread[4 * i + 2] * (float)(gw[i] & 0x0f00)
                       + x_thread[4 * i + 3] * (float)(gw[i] & 0xf000);
                u_dot += x_thread[4 * i]     * (float)(uw[i] & 0x000f)
                       + x_thread[4 * i + 1] * (float)(uw[i] & 0x00f0)
                       + x_thread[4 * i + 2] * (float)(uw[i] & 0x0f00)
                       + x_thread[4 * i + 3] * (float)(uw[i] & 0xf000);
            }

            gate_acc[row] += gs * g_dot + x_sum * gb;
            up_acc[row]   += us * u_dot + x_sum * ub;
        }

        // Advance all pointers by one block
        g_ws += block_size * bytes_per_pack / pack_factor;
        u_ws += block_size * bytes_per_pack / pack_factor;
        g_sl += block_size / group_size;
        g_bl += block_size / group_size;
        u_sl += block_size / group_size;
        u_bl += block_size / group_size;
        xp   += block_size;
    }

    // SIMD reduction + fused SiLU activation + write
    // Output layout: (n_tokens * top_k, output_dim) — flat
    int out_base = (token_idx * top_k + expert_slot) * output_dim;
    for (int row = 0; row < results_per_simdgroup; row++) {
        float g = simd_sum(gate_acc[row]);
        float u = simd_sum(up_acc[row]);
        if (simd_lid == 0 && out_row + row < output_dim) {
            output[out_base + out_row + row] = (T)(silu(g) * u);
        }
    }
}

// float16 entry point
kernel void gather_qmm_swiglu_f16(
    device const half* x                [[buffer(0)]],
    device const uint* gate_weight      [[buffer(1)]],
    device const half* gate_scales      [[buffer(2)]],
    device const half* gate_biases      [[buffer(3)]],
    device const uint* up_weight        [[buffer(4)]],
    device const half* up_scales        [[buffer(5)]],
    device const half* up_biases        [[buffer(6)]],
    device const int* expert_inds       [[buffer(7)]],
    device half* output                 [[buffer(8)]],
    constant int& input_dim             [[buffer(9)]],
    constant int& output_dim            [[buffer(10)]],
    constant int& group_size            [[buffer(11)]],
    constant int& n_experts             [[buffer(12)]],
    constant int& top_k                 [[buffer(13)]],
    constant int& n_tokens              [[buffer(14)]],
    uint3 tid                           [[threadgroup_position_in_grid]],
    uint simd_gid                       [[simdgroup_index_in_threadgroup]],
    uint simd_lid                       [[thread_index_in_simdgroup]])
{
    gather_qmv_swiglu_impl(
        x, gate_weight, gate_scales, gate_biases,
        up_weight, up_scales, up_biases,
        expert_inds, output,
        input_dim, output_dim, group_size, n_experts,
        top_k, n_tokens,
        tid, simd_gid, simd_lid);
}

// bfloat16 entry point
kernel void gather_qmm_swiglu_bf16(
    device const bfloat* x              [[buffer(0)]],
    device const uint* gate_weight      [[buffer(1)]],
    device const bfloat* gate_scales    [[buffer(2)]],
    device const bfloat* gate_biases    [[buffer(3)]],
    device const uint* up_weight        [[buffer(4)]],
    device const bfloat* up_scales      [[buffer(5)]],
    device const bfloat* up_biases      [[buffer(6)]],
    device const int* expert_inds       [[buffer(7)]],
    device bfloat* output               [[buffer(8)]],
    constant int& input_dim             [[buffer(9)]],
    constant int& output_dim            [[buffer(10)]],
    constant int& group_size            [[buffer(11)]],
    constant int& n_experts             [[buffer(12)]],
    constant int& top_k                 [[buffer(13)]],
    constant int& n_tokens              [[buffer(14)]],
    uint3 tid                           [[threadgroup_position_in_grid]],
    uint simd_gid                       [[simdgroup_index_in_threadgroup]],
    uint simd_lid                       [[thread_index_in_simdgroup]])
{
    gather_qmv_swiglu_impl(
        x, gate_weight, gate_scales, gate_biases,
        up_weight, up_scales, up_biases,
        expert_inds, output,
        input_dim, output_dim, group_size, n_experts,
        top_k, n_tokens,
        tid, simd_gid, simd_lid);
}
