#pragma once

#include "mlx/mlx.h"
#include "mlx/primitives.h"

namespace mlx_fused_moe {

/**
 * Fused gather_qmm + SwiGLU for MoE decode.
 *
 * Computes in one Metal dispatch:
 *   gate_out = dequant_matmul(x, gate_weight[expert_indices])
 *   up_out   = dequant_matmul(x, up_weight[expert_indices])
 *   result   = silu(gate_out) * up_out
 */
/// Set the directory where mlx_fused_moe.metallib is located.
void set_metallib_path(const std::string& path);

/// Get the current metallib path.
const std::string& get_metallib_path();

mlx::core::array gather_qmm_swiglu(
    const mlx::core::array& x,
    const mlx::core::array& gate_weight,
    const mlx::core::array& gate_scales,
    const mlx::core::array& gate_biases,
    const mlx::core::array& up_weight,
    const mlx::core::array& up_scales,
    const mlx::core::array& up_biases,
    const mlx::core::array& expert_indices,
    int top_k,
    int group_size = 64,
    int bits = 4,
    mlx::core::StreamOrDevice s = {});

class GatherQMMSwiGLU : public mlx::core::UnaryPrimitive {
 public:
  explicit GatherQMMSwiGLU(
      mlx::core::Stream stream, int top_k, int group_size, int bits,
      std::string metallib_path = "")
      : UnaryPrimitive(stream),
        top_k_(top_k),
        group_size_(group_size),
        bits_(bits),
        metallib_path_(std::move(metallib_path)) {}

  void eval_cpu(
      const std::vector<mlx::core::array>& inputs,
      mlx::core::array& out) override;
  void eval_gpu(
      const std::vector<mlx::core::array>& inputs,
      mlx::core::array& out) override;

  DEFINE_NAME(GatherQMMSwiGLU)
  DEFINE_DEFAULT_IS_EQUIVALENT()

 private:
  int top_k_;
  int group_size_;
  int bits_;
  std::string metallib_path_;
};

/**
 * Fused quantized matrix-vector multiply.
 *
 * Single dispatch quantized GEMV for pre-concatenated weight matrices.
 * Used to fuse multiple projections sharing the same input x.
 */
mlx::core::array fused_qmv(
    const mlx::core::array& x,
    const mlx::core::array& weight,
    const mlx::core::array& scales,
    const mlx::core::array& biases,
    int n_tokens,
    int group_size = 64,
    int bits = 4,
    mlx::core::StreamOrDevice s = {});

class FusedQMV : public mlx::core::UnaryPrimitive {
 public:
  explicit FusedQMV(
      mlx::core::Stream stream, int group_size, int bits,
      std::string metallib_path = "")
      : UnaryPrimitive(stream),
        group_size_(group_size),
        bits_(bits),
        metallib_path_(std::move(metallib_path)) {}

  void eval_cpu(
      const std::vector<mlx::core::array>& inputs,
      mlx::core::array& out) override;
  void eval_gpu(
      const std::vector<mlx::core::array>& inputs,
      mlx::core::array& out) override;

  DEFINE_NAME(FusedQMV)
  DEFINE_DEFAULT_IS_EQUIVALENT()

 private:
  int group_size_;
  int bits_;
  std::string metallib_path_;
};

/**
 * Fused gather_qmm down_proj + score-weighted reduce.
 *
 * Computes in one Metal dispatch for each token:
 *   for each expert e in top_k:
 *     out += score[e] * dequant_matmul(x_intermediate[e], down_weight[expert_id[e]])
 *
 * Output is already reduced across experts.
 */
mlx::core::array gather_qmm_down_reduce(
    const mlx::core::array& x_intermediate,
    const mlx::core::array& down_weight,
    const mlx::core::array& down_scales,
    const mlx::core::array& down_biases,
    const mlx::core::array& expert_indices,
    const mlx::core::array& scores,
    int top_k,
    int group_size = 64,
    int bits = 4,
    mlx::core::StreamOrDevice s = {});

class GatherQMMDownReduce : public mlx::core::UnaryPrimitive {
 public:
  explicit GatherQMMDownReduce(
      mlx::core::Stream stream, int top_k, int group_size, int bits,
      std::string metallib_path = "")
      : UnaryPrimitive(stream),
        top_k_(top_k),
        group_size_(group_size),
        bits_(bits),
        metallib_path_(std::move(metallib_path)) {}

  void eval_cpu(
      const std::vector<mlx::core::array>& inputs,
      mlx::core::array& out) override;
  void eval_gpu(
      const std::vector<mlx::core::array>& inputs,
      mlx::core::array& out) override;

  DEFINE_NAME(GatherQMMDownReduce)
  DEFINE_DEFAULT_IS_EQUIVALENT()

 private:
  int top_k_;
  int group_size_;
  int bits_;
  std::string metallib_path_;
};

}  // namespace mlx_fused_moe
