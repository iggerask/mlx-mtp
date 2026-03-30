/**
 * Fused Gather-QMM Down-Proj + Score-Weighted Reduce (SIMD-optimized)
 *
 * Fuses the second half of MoE into a single dispatch:
 *   1. For each expert: gather_qmv(x_intermediate, down_weight[expert])
 *   2. Multiply each expert's output by its routing score
 *   3. Sum across all top_k experts
 *
 * Same SIMD group tiling as gather_qmm_swiglu:
 *   - 2 simdgroups per threadgroup, each computing 4 output rows
 *   - Cooperative x loading across 32 threads
 *   - Pre-division trick for fast 4-bit dequantization
 *   - simd_sum for cross-thread reduction
 *
 * Grid: (n_tokens, ceil(hidden_size/8), 1)
 * Threadgroup: (64, 1, 1)
 *
 * Key difference from gather_qmm_swiglu: experts are looped INSIDE the
 * kernel (not dispatched as separate z-slices). Each expert reads its own
 * x_intermediate segment and accumulates into a shared output with scores.
 */

#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

static constant constexpr const int SIMD_SIZE = 32;

template<typename T>
void gather_qmm_down_reduce_impl(
    const device T* x_intermediate,
    const device uint32_t* down_weight,
    const device T* down_scales,
    const device T* down_biases,
    const device int* expert_inds,
    const device T* scores,
    device T* output,
    constant int& intermediate_size,
    constant int& hidden_size,
    constant int& group_size,
    constant int& n_experts,
    constant int& top_k,
    constant int& n_tokens,
    uint3 tid,
    uint simd_gid,
    uint simd_lid)
{
    // Constants for 4-bit quantization
    constexpr int packs_per_thread = 2;
    constexpr int pack_factor = 8;       // 32 / 4 bits
    constexpr int bytes_per_pack = 4;    // 32 / 8
    constexpr int values_per_thread = pack_factor * packs_per_thread; // 16
    constexpr int block_size = values_per_thread * SIMD_SIZE;         // 512
    constexpr int num_simdgroups = 2;
    constexpr int results_per_simdgroup = 4;

    int token_idx = tid.x;

    int out_row = tid.y * (num_simdgroups * results_per_simdgroup)
                + simd_gid * results_per_simdgroup;
    if (out_row >= hidden_size) return;

    // Byte-stride calculations for down_proj weights
    // down_weight shape: (n_experts, hidden_size, intermediate_size/pack_factor)
    int in_vec_size_w = intermediate_size * bytes_per_pack / pack_factor; // bytes per weight row
    int in_vec_size_g = intermediate_size / group_size;                   // groups per row
    int scale_step = group_size / values_per_thread;                     // threads per group

    // Accumulated output across all experts (score-weighted)
    float acc[results_per_simdgroup] = {0};

    // Loop over experts for this token
    for (int e = 0; e < top_k; e++) {
        int expert_id = expert_inds[token_idx * top_k + e];
        if (expert_id < 0 || expert_id >= n_experts) continue;

        float score = (float)scores[token_idx * top_k + e];

        // Expert weight base offsets
        long expert_w_byte = (long)expert_id * hidden_size * in_vec_size_w;
        long expert_s_off  = (long)expert_id * hidden_size * in_vec_size_g;

        // Weight pointers for this expert's down_proj rows
        const device uint8_t* ws = (const device uint8_t*)down_weight
            + expert_w_byte + out_row * in_vec_size_w
            + simd_lid * packs_per_thread * bytes_per_pack;

        // Scale/bias pointers
        const device T* sl = down_scales + expert_s_off
            + out_row * in_vec_size_g + simd_lid / scale_step;
        const device T* bl = down_biases + expert_s_off
            + out_row * in_vec_size_g + simd_lid / scale_step;

        // Input pointer: x_intermediate for this expert slot
        // x_intermediate layout: (n_tokens * top_k, intermediate_size)
        const device T* xp = x_intermediate
            + (token_idx * top_k + e) * intermediate_size
            + simd_lid * values_per_thread;

        float x_thread[values_per_thread];
        float expert_result[results_per_simdgroup] = {0};

        // Inner dot product loop over intermediate_size
        for (int k = 0; k < intermediate_size; k += block_size) {
            // Load x with pre-division trick for 4-bit
            // Guard against reading past intermediate_size when block_size > remaining
            int remaining = intermediate_size - k;
            int my_start = simd_lid * values_per_thread;
            float x_sum = 0;
            for (int i = 0; i < values_per_thread; i += 4) {
                bool valid = (my_start + i + 3) < remaining;
                float v0 = valid ? (float)xp[i]     : 0.0f;
                float v1 = valid ? (float)xp[i + 1] : 0.0f;
                float v2 = valid ? (float)xp[i + 2] : 0.0f;
                float v3 = valid ? (float)xp[i + 3] : 0.0f;
                x_sum += v0 + v1 + v2 + v3;
                x_thread[i]     = v0;
                x_thread[i + 1] = v1 / 16.0f;
                x_thread[i + 2] = v2 / 256.0f;
                x_thread[i + 3] = v3 / 4096.0f;
            }

            // Dot product for each of 4 output rows
            // Only threads with valid data contribute
            if (my_start < remaining) {
                for (int row = 0; row < results_per_simdgroup; row++) {
                    const device uint16_t* ww = (const device uint16_t*)(ws + row * in_vec_size_w);
                    float s = (float)sl[row * in_vec_size_g];
                    float b = (float)bl[row * in_vec_size_g];

                    float dot = 0;
                    for (int i = 0; i < values_per_thread / 4; i++) {
                        dot += x_thread[4 * i]     * (float)(ww[i] & 0x000f)
                             + x_thread[4 * i + 1] * (float)(ww[i] & 0x00f0)
                             + x_thread[4 * i + 2] * (float)(ww[i] & 0x0f00)
                             + x_thread[4 * i + 3] * (float)(ww[i] & 0xf000);
                    }

                    expert_result[row] += s * dot + x_sum * b;
                }
            }

            // Advance pointers
            ws += block_size * bytes_per_pack / pack_factor;
            sl += block_size / group_size;
            bl += block_size / group_size;
            xp += block_size;
        }

        // SIMD reduce this expert's contribution and accumulate with score
        for (int row = 0; row < results_per_simdgroup; row++) {
            float r = simd_sum(expert_result[row]);
            acc[row] += score * r;
        }
    }

    // Write final accumulated output
    // Output layout: (n_tokens, hidden_size)
    int out_base = token_idx * hidden_size;
    for (int row = 0; row < results_per_simdgroup; row++) {
        if (simd_lid == 0 && out_row + row < hidden_size) {
            output[out_base + out_row + row] = (T)acc[row];
        }
    }
}

// float16 entry point
kernel void gather_qmm_down_reduce_f16(
    device const half* x_intermediate   [[buffer(0)]],
    device const uint* down_weight      [[buffer(1)]],
    device const half* down_scales      [[buffer(2)]],
    device const half* down_biases      [[buffer(3)]],
    device const int* expert_inds       [[buffer(4)]],
    device const half* scores           [[buffer(5)]],
    device half* output                 [[buffer(6)]],
    constant int& intermediate_size     [[buffer(7)]],
    constant int& hidden_size           [[buffer(8)]],
    constant int& group_size            [[buffer(9)]],
    constant int& n_experts             [[buffer(10)]],
    constant int& top_k                 [[buffer(11)]],
    constant int& n_tokens              [[buffer(12)]],
    uint3 tid                           [[threadgroup_position_in_grid]],
    uint simd_gid                       [[simdgroup_index_in_threadgroup]],
    uint simd_lid                       [[thread_index_in_simdgroup]])
{
    gather_qmm_down_reduce_impl(
        x_intermediate, down_weight, down_scales, down_biases,
        expert_inds, scores, output,
        intermediate_size, hidden_size, group_size, n_experts,
        top_k, n_tokens,
        tid, simd_gid, simd_lid);
}

// bfloat16 entry point
kernel void gather_qmm_down_reduce_bf16(
    device const bfloat* x_intermediate [[buffer(0)]],
    device const uint* down_weight      [[buffer(1)]],
    device const bfloat* down_scales    [[buffer(2)]],
    device const bfloat* down_biases    [[buffer(3)]],
    device const int* expert_inds       [[buffer(4)]],
    device const bfloat* scores         [[buffer(5)]],
    device bfloat* output               [[buffer(6)]],
    constant int& intermediate_size     [[buffer(7)]],
    constant int& hidden_size           [[buffer(8)]],
    constant int& group_size            [[buffer(9)]],
    constant int& n_experts             [[buffer(10)]],
    constant int& top_k                 [[buffer(11)]],
    constant int& n_tokens              [[buffer(12)]],
    uint3 tid                           [[threadgroup_position_in_grid]],
    uint simd_gid                       [[simdgroup_index_in_threadgroup]],
    uint simd_lid                       [[thread_index_in_simdgroup]])
{
    gather_qmm_down_reduce_impl(
        x_intermediate, down_weight, down_scales, down_biases,
        expert_inds, scores, output,
        intermediate_size, hidden_size, group_size, n_experts,
        top_k, n_tokens,
        tid, simd_gid, simd_lid);
}
