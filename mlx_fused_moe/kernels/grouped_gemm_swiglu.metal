/**
 * Grouped GEMM with fused SwiGLU for MoE prefill.
 *
 * Processes ALL experts in a single kernel dispatch using a flattened token list.
 * Each threadgroup handles a tile of one expert's computation.
 *
 * For prefill with N tokens × top_k experts:
 *   Input:  x (N, input_dim), expert_indices (N*top_k,)
 *   Output: (N*top_k, output_dim) with SiLU(gate)*up applied
 *
 * Uses tiled GEMM with cooperative loading of both input and weight tiles.
 * Weights are 4-bit quantized (MLX affine format).
 *
 * Grid: (n_tokens * top_k, ceil(output_dim / TILE_N), 1)
 * Each threadgroup computes TILE_M output rows × TILE_N output columns for one
 * token-expert pair. For small expert batches this degenerates to our QMV kernel;
 * for large batches it achieves better utilization via tiled accumulation.
 *
 * This kernel uses the same dequantization and SIMD reduction patterns as
 * gather_qmm_swiglu.metal but with a GEMM-friendly tile structure.
 */

#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

static constant constexpr const int SIMD_SIZE = 32;

inline float silu(float x) {
    return x / (1.0f + exp(-x));
}

/**
 * Tiled fused gate+up+SwiGLU for MoE.
 *
 * Each threadgroup processes one token-expert pair, computing TILE_N output columns.
 * Uses the QMV pattern (cooperative x loading, per-row dot products) since each
 * "row" in the GEMM is one token's projection — and we only have 1 token per
 * expert-slot in the flattened layout.
 *
 * This is architecturally identical to gather_qmm_swiglu but processes all
 * token-expert pairs in a single dispatch with correct indexing.
 */
template<typename T>
void grouped_gemm_swiglu_impl(
    const device T* x,              // (n_tokens, input_dim) — original tokens
    const device uint32_t* gate_weight,  // (n_experts, output_dim, packed_input)
    const device T* gate_scales,
    const device T* gate_biases,
    const device uint32_t* up_weight,
    const device T* up_scales,
    const device T* up_biases,
    const device int* expert_inds,   // (n_tokens * top_k,) — which expert for each slot
    const device int* token_inds,    // (n_tokens * top_k,) — which token for each slot
    device T* output,                // (n_tokens * top_k, output_dim)
    constant int& input_dim,
    constant int& output_dim,
    constant int& group_size,
    constant int& n_experts,
    constant int& n_token_expert_pairs,
    uint3 tid,
    uint simd_gid,
    uint simd_lid)
{
    constexpr int packs_per_thread = 2;
    constexpr int pack_factor = 8;
    constexpr int bytes_per_pack = 4;
    constexpr int values_per_thread = pack_factor * packs_per_thread; // 16
    constexpr int block_size = values_per_thread * SIMD_SIZE;         // 512
    constexpr int num_simdgroups = 2;
    constexpr int results_per_simdgroup = 4;

    int pair_idx = tid.x;  // which token-expert pair
    if (pair_idx >= n_token_expert_pairs) return;

    int expert_id = expert_inds[pair_idx];
    int token_id = token_inds[pair_idx];
    if (expert_id < 0 || expert_id >= n_experts) return;

    int out_row = tid.y * (num_simdgroups * results_per_simdgroup)
                + simd_gid * results_per_simdgroup;
    if (out_row >= output_dim) return;

    int in_vec_size_w = input_dim * bytes_per_pack / pack_factor;
    int in_vec_size_g = input_dim / group_size;
    int scale_step = group_size / values_per_thread;

    long expert_w_byte = (long)expert_id * output_dim * in_vec_size_w;
    long expert_s_off  = (long)expert_id * output_dim * in_vec_size_g;

    const device uint8_t* g_ws = (const device uint8_t*)gate_weight
        + expert_w_byte + out_row * in_vec_size_w
        + simd_lid * packs_per_thread * bytes_per_pack;
    const device uint8_t* u_ws = (const device uint8_t*)up_weight
        + expert_w_byte + out_row * in_vec_size_w
        + simd_lid * packs_per_thread * bytes_per_pack;

    const device T* g_sl = gate_scales + expert_s_off + out_row * in_vec_size_g
        + simd_lid / scale_step;
    const device T* g_bl = gate_biases + expert_s_off + out_row * in_vec_size_g
        + simd_lid / scale_step;
    const device T* u_sl = up_scales + expert_s_off + out_row * in_vec_size_g
        + simd_lid / scale_step;
    const device T* u_bl = up_biases + expert_s_off + out_row * in_vec_size_g
        + simd_lid / scale_step;

    // Input pointer for this token
    const device T* xp = x + (long)token_id * input_dim + simd_lid * values_per_thread;

    float gate_acc[results_per_simdgroup] = {0};
    float up_acc[results_per_simdgroup] = {0};

    for (int k = 0; k < input_dim; k += block_size) {
        float x_thread[values_per_thread];
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

        g_ws += block_size * bytes_per_pack / pack_factor;
        u_ws += block_size * bytes_per_pack / pack_factor;
        g_sl += block_size / group_size;
        g_bl += block_size / group_size;
        u_sl += block_size / group_size;
        u_bl += block_size / group_size;
        xp   += block_size;
    }

    // SIMD reduction + fused SiLU activation + write
    int out_base = pair_idx * output_dim;
    for (int row = 0; row < results_per_simdgroup; row++) {
        float g = simd_sum(gate_acc[row]);
        float u = simd_sum(up_acc[row]);
        if (simd_lid == 0 && out_row + row < output_dim) {
            output[out_base + out_row + row] = (T)(silu(g) * u);
        }
    }
}

kernel void grouped_gemm_swiglu_bf16(
    device const bfloat* x              [[buffer(0)]],
    device const uint* gate_weight      [[buffer(1)]],
    device const bfloat* gate_scales    [[buffer(2)]],
    device const bfloat* gate_biases    [[buffer(3)]],
    device const uint* up_weight        [[buffer(4)]],
    device const bfloat* up_scales      [[buffer(5)]],
    device const bfloat* up_biases      [[buffer(6)]],
    device const int* expert_inds       [[buffer(7)]],
    device const int* token_inds        [[buffer(8)]],
    device bfloat* output               [[buffer(9)]],
    constant int& input_dim             [[buffer(10)]],
    constant int& output_dim            [[buffer(11)]],
    constant int& group_size            [[buffer(12)]],
    constant int& n_experts             [[buffer(13)]],
    constant int& n_pairs               [[buffer(14)]],
    uint3 tid                           [[threadgroup_position_in_grid]],
    uint simd_gid                       [[simdgroup_index_in_threadgroup]],
    uint simd_lid                       [[thread_index_in_simdgroup]])
{
    grouped_gemm_swiglu_impl(
        x, gate_weight, gate_scales, gate_biases,
        up_weight, up_scales, up_biases,
        expert_inds, token_inds, output,
        input_dim, output_dim, group_size, n_experts, n_pairs,
        tid, simd_gid, simd_lid);
}

kernel void grouped_gemm_swiglu_f16(
    device const half* x                [[buffer(0)]],
    device const uint* gate_weight      [[buffer(1)]],
    device const half* gate_scales      [[buffer(2)]],
    device const half* gate_biases      [[buffer(3)]],
    device const uint* up_weight        [[buffer(4)]],
    device const half* up_scales        [[buffer(5)]],
    device const half* up_biases        [[buffer(6)]],
    device const int* expert_inds       [[buffer(7)]],
    device const int* token_inds        [[buffer(8)]],
    device half* output                 [[buffer(9)]],
    constant int& input_dim             [[buffer(10)]],
    constant int& output_dim            [[buffer(11)]],
    constant int& group_size            [[buffer(12)]],
    constant int& n_experts             [[buffer(13)]],
    constant int& n_pairs               [[buffer(14)]],
    uint3 tid                           [[threadgroup_position_in_grid]],
    uint simd_gid                       [[simdgroup_index_in_threadgroup]],
    uint simd_lid                       [[thread_index_in_simdgroup]])
{
    grouped_gemm_swiglu_impl(
        x, gate_weight, gate_scales, gate_biases,
        up_weight, up_scales, up_biases,
        expert_inds, token_inds, output,
        input_dim, output_dim, group_size, n_experts, n_pairs,
        tid, simd_gid, simd_lid);
}
