/**
 * Fused Quantized Matrix-Vector Multiply (SIMD-optimized, multi-token)
 *
 * A straight quantized GEMV matching MLX's qmv_fast pattern.
 * Used for fusing multiple projections that share the same input x
 * (e.g., GatedDeltaNet's 4 input projections concatenated into one).
 *
 * Thread layout:
 *   - 2 simdgroups per threadgroup, each computing 4 output rows
 *   - Cooperative x loading across 32 threads (512 elements per iteration)
 *   - Pre-division trick for fast 4-bit dequantization
 *   - simd_sum for cross-thread reduction
 *
 * Grid: (n_tokens, ceil(output_dim/8), 1)
 *
 * Weight format: 4-bit quantized (MLX affine format)
 */

#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

static constant constexpr const int SIMD_SIZE = 32;

template<typename T>
void fused_qmv_impl(
    const device T* x,
    const device uint32_t* weight,
    const device T* scales,
    const device T* biases,
    device T* output,
    constant int& input_dim,
    constant int& output_dim,
    constant int& group_size,
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

    int token_idx = tid.x;

    int out_row = tid.y * (num_simdgroups * results_per_simdgroup)
                + simd_gid * results_per_simdgroup;
    if (out_row >= output_dim) return;

    // Byte-stride calculations
    int in_vec_size_w = input_dim * bytes_per_pack / pack_factor; // bytes per weight row
    int in_vec_size_g = input_dim / group_size;                   // groups per row
    int scale_step = group_size / values_per_thread;              // threads per group

    // Weight pointers (byte-addressed via uint8_t*)
    const device uint8_t* ws = (const device uint8_t*)weight
        + out_row * in_vec_size_w
        + simd_lid * packs_per_thread * bytes_per_pack;

    // Scale/bias pointers
    const device T* sl = scales + out_row * in_vec_size_g + simd_lid / scale_step;
    const device T* bl = biases + out_row * in_vec_size_g + simd_lid / scale_step;

    // Input pointer — offset by token index
    const device T* xp = x + token_idx * input_dim + simd_lid * values_per_thread;

    float x_thread[values_per_thread];
    float result[results_per_simdgroup] = {0};

    // Main loop: cooperative x loading + dot products
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

        // Dot product for each of 4 output rows
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

            result[row] += s * dot + x_sum * b;
        }

        // Advance pointers
        ws += block_size * bytes_per_pack / pack_factor;
        sl += block_size / group_size;
        bl += block_size / group_size;
        xp += block_size;
    }

    // SIMD reduction + write
    int out_base = token_idx * output_dim;
    for (int row = 0; row < results_per_simdgroup; row++) {
        float r = simd_sum(result[row]);
        if (simd_lid == 0 && out_row + row < output_dim) {
            output[out_base + out_row + row] = (T)r;
        }
    }
}

// float16 entry point
kernel void fused_qmv_f16(
    device const half* x                [[buffer(0)]],
    device const uint* weight           [[buffer(1)]],
    device const half* scales           [[buffer(2)]],
    device const half* biases           [[buffer(3)]],
    device half* output                 [[buffer(4)]],
    constant int& input_dim             [[buffer(5)]],
    constant int& output_dim            [[buffer(6)]],
    constant int& group_size            [[buffer(7)]],
    constant int& n_tokens              [[buffer(8)]],
    uint3 tid                           [[threadgroup_position_in_grid]],
    uint simd_gid                       [[simdgroup_index_in_threadgroup]],
    uint simd_lid                       [[thread_index_in_simdgroup]])
{
    fused_qmv_impl(x, weight, scales, biases, output,
                    input_dim, output_dim, group_size, n_tokens,
                    tid, simd_gid, simd_lid);
}

// bfloat16 entry point
kernel void fused_qmv_bf16(
    device const bfloat* x              [[buffer(0)]],
    device const uint* weight           [[buffer(1)]],
    device const bfloat* scales         [[buffer(2)]],
    device const bfloat* biases         [[buffer(3)]],
    device bfloat* output               [[buffer(4)]],
    constant int& input_dim             [[buffer(5)]],
    constant int& output_dim            [[buffer(6)]],
    constant int& group_size            [[buffer(7)]],
    constant int& n_tokens              [[buffer(8)]],
    uint3 tid                           [[threadgroup_position_in_grid]],
    uint simd_gid                       [[simdgroup_index_in_threadgroup]],
    uint simd_lid                       [[thread_index_in_simdgroup]])
{
    fused_qmv_impl(x, weight, scales, biases, output,
                    input_dim, output_dim, group_size, n_tokens,
                    tid, simd_gid, simd_lid);
}
