#include "ops.h"

#include "mlx/backend/metal/device.h"

namespace mlx_fused_moe {

using namespace mlx::core;

static std::string g_metallib_path;

void set_metallib_path(const std::string& path) {
  g_metallib_path = path;
}

const std::string& get_metallib_path() {
  return g_metallib_path;
}

array gather_qmm_swiglu(
    const array& x,
    const array& gate_weight,
    const array& gate_scales,
    const array& gate_biases,
    const array& up_weight,
    const array& up_scales,
    const array& up_biases,
    const array& expert_indices,
    int top_k,
    int group_size,
    int bits,
    StreamOrDevice s) {

  // gate_scales: [n_experts, intermediate_size, n_groups]
  int intermediate_size = gate_scales.shape(1);

  // Output shape: (n_tokens * top_k, intermediate_size)
  int n_total_experts = expert_indices.size();
  Shape out_shape = {n_total_experts, intermediate_size};

  return array(
      std::move(out_shape),
      gate_scales.dtype(),
      std::make_shared<GatherQMMSwiGLU>(to_stream(s), top_k, group_size, bits, g_metallib_path),
      {x, gate_weight, gate_scales, gate_biases,
       up_weight, up_scales, up_biases, expert_indices});
}

void GatherQMMSwiGLU::eval_cpu(
    const std::vector<array>& inputs,
    array& out) {
  throw std::runtime_error(
      "[GatherQMMSwiGLU] CPU not implemented — Metal GPU required.");
}

void GatherQMMSwiGLU::eval_gpu(
    const std::vector<array>& inputs,
    array& out) {
  auto& s = stream();
  auto& d = metal::device(s.device);

  // Allocate output
  out.set_data(allocator::malloc(out.nbytes()));

  auto& x = inputs[0];
  auto& gate_weight = inputs[1];
  auto& gate_scales = inputs[2];
  auto& gate_biases = inputs[3];
  auto& up_weight = inputs[4];
  auto& up_scales = inputs[5];
  auto& up_biases = inputs[6];
  auto& expert_indices = inputs[7];

  int intermediate_size = gate_scales.shape(1);
  int n_experts = gate_weight.shape(0);
  int packed_input_dim = gate_weight.shape(2);
  int input_dim = packed_input_dim * (32 / bits_);
  int n_tokens = expert_indices.size() / top_k_;

  // Load our custom metallib (with explicit path if set)
  auto lib = d.get_library("mlx_fused_moe", metallib_path_);

  // Select kernel based on dtype (float16 vs bfloat16)
  std::string kernel_name;
  if (gate_scales.dtype() == bfloat16) {
    kernel_name = "gather_qmm_swiglu_bf16";
  } else {
    kernel_name = "gather_qmm_swiglu_f16";
  }
  auto kernel = d.get_kernel(kernel_name, lib);

  auto& enc = d.get_command_encoder(s.index);
  enc.set_compute_pipeline_state(kernel);

  enc.set_input_array(x, 0);
  enc.set_input_array(gate_weight, 1);
  enc.set_input_array(gate_scales, 2);
  enc.set_input_array(gate_biases, 3);
  enc.set_input_array(up_weight, 4);
  enc.set_input_array(up_scales, 5);
  enc.set_input_array(up_biases, 6);
  enc.set_input_array(expert_indices, 7);
  enc.set_output_array(out, 8);

  enc.set_bytes(input_dim, 9);
  enc.set_bytes(intermediate_size, 10);
  enc.set_bytes(group_size_, 11);
  enc.set_bytes(n_experts, 12);
  enc.set_bytes(top_k_, 13);
  enc.set_bytes(n_tokens, 14);

  // SIMD-tiled dispatch: 2 simdgroups × 32 threads = 64 threads per threadgroup
  // Each threadgroup computes 8 output rows (4 per simdgroup)
  // Grid: (n_tokens, ceil(output_dim/8), top_k)
  int rows_per_tg = 8;
  int n_tg_y = (intermediate_size + rows_per_tg - 1) / rows_per_tg;
  auto grid = MTL::Size(n_tokens, n_tg_y, top_k_);
  auto group = MTL::Size(64, 1, 1);
  enc.dispatch_threadgroups(grid, group);
}

array fused_qmv(
    const array& x,
    const array& weight,
    const array& scales,
    const array& biases,
    int n_tokens,
    int group_size,
    int bits,
    StreamOrDevice s) {

  int output_dim = weight.shape(0);
  Shape out_shape = {n_tokens, output_dim};

  return array(
      std::move(out_shape),
      scales.dtype(),
      std::make_shared<FusedQMV>(to_stream(s), group_size, bits, g_metallib_path),
      {x, weight, scales, biases});
}

void FusedQMV::eval_cpu(
    const std::vector<array>& inputs,
    array& out) {
  throw std::runtime_error(
      "[FusedQMV] CPU not implemented — Metal GPU required.");
}

void FusedQMV::eval_gpu(
    const std::vector<array>& inputs,
    array& out) {
  auto& s = stream();
  auto& d = metal::device(s.device);

  out.set_data(allocator::malloc(out.nbytes()));

  auto& x = inputs[0];
  auto& weight = inputs[1];
  auto& scales = inputs[2];
  auto& biases = inputs[3];

  int output_dim = weight.shape(0);
  int packed_input_dim = weight.shape(1);
  int input_dim = packed_input_dim * (32 / bits_);
  int n_tokens = out.shape(0);

  auto lib = d.get_library("mlx_fused_moe", metallib_path_);

  std::string kernel_name;
  if (scales.dtype() == bfloat16) {
    kernel_name = "fused_qmv_bf16";
  } else {
    kernel_name = "fused_qmv_f16";
  }
  auto kernel = d.get_kernel(kernel_name, lib);

  auto& enc = d.get_command_encoder(s.index);
  enc.set_compute_pipeline_state(kernel);

  enc.set_input_array(x, 0);
  enc.set_input_array(weight, 1);
  enc.set_input_array(scales, 2);
  enc.set_input_array(biases, 3);
  enc.set_output_array(out, 4);

  enc.set_bytes(input_dim, 5);
  enc.set_bytes(output_dim, 6);
  enc.set_bytes(group_size_, 7);
  enc.set_bytes(n_tokens, 8);

  int rows_per_tg = 8;
  int n_tg_y = (output_dim + rows_per_tg - 1) / rows_per_tg;
  auto grid = MTL::Size(n_tokens, n_tg_y, 1);
  auto group = MTL::Size(64, 1, 1);
  enc.dispatch_threadgroups(grid, group);
}

array gather_qmm_down_reduce(
    const array& x_intermediate,
    const array& down_weight,
    const array& down_scales,
    const array& down_biases,
    const array& expert_indices,
    const array& scores,
    int top_k,
    int group_size,
    int bits,
    StreamOrDevice s) {

  // down_scales: [n_experts, hidden_size, n_groups]
  int hidden_size = down_scales.shape(1);
  int n_tokens = expert_indices.size() / top_k;

  // Output shape: (n_tokens, hidden_size) — already reduced across experts
  Shape out_shape = {n_tokens, hidden_size};

  return array(
      std::move(out_shape),
      down_scales.dtype(),
      std::make_shared<GatherQMMDownReduce>(to_stream(s), top_k, group_size, bits, g_metallib_path),
      {x_intermediate, down_weight, down_scales, down_biases, expert_indices, scores});
}

void GatherQMMDownReduce::eval_cpu(
    const std::vector<array>& inputs,
    array& out) {
  throw std::runtime_error(
      "[GatherQMMDownReduce] CPU not implemented — Metal GPU required.");
}

void GatherQMMDownReduce::eval_gpu(
    const std::vector<array>& inputs,
    array& out) {
  auto& s = stream();
  auto& d = metal::device(s.device);

  out.set_data(allocator::malloc(out.nbytes()));

  auto& x_intermediate = inputs[0];
  auto& down_weight = inputs[1];
  auto& down_scales = inputs[2];
  auto& down_biases = inputs[3];
  auto& expert_indices = inputs[4];
  auto& scores = inputs[5];

  int hidden_size = down_scales.shape(1);
  int n_experts = down_weight.shape(0);
  int packed_input_dim = down_weight.shape(2);
  int intermediate_size = packed_input_dim * (32 / bits_);
  int n_tokens = expert_indices.size() / top_k_;

  auto lib = d.get_library("mlx_fused_moe", metallib_path_);

  std::string kernel_name;
  if (down_scales.dtype() == bfloat16) {
    kernel_name = "gather_qmm_down_reduce_bf16";
  } else {
    kernel_name = "gather_qmm_down_reduce_f16";
  }
  auto kernel = d.get_kernel(kernel_name, lib);

  auto& enc = d.get_command_encoder(s.index);
  enc.set_compute_pipeline_state(kernel);

  enc.set_input_array(x_intermediate, 0);
  enc.set_input_array(down_weight, 1);
  enc.set_input_array(down_scales, 2);
  enc.set_input_array(down_biases, 3);
  enc.set_input_array(expert_indices, 4);
  enc.set_input_array(scores, 5);
  enc.set_output_array(out, 6);

  enc.set_bytes(intermediate_size, 7);
  enc.set_bytes(hidden_size, 8);
  enc.set_bytes(group_size_, 9);
  enc.set_bytes(n_experts, 10);
  enc.set_bytes(top_k_, 11);
  enc.set_bytes(n_tokens, 12);

  // Grid: (n_tokens, ceil(hidden_size/8), 1)
  int rows_per_tg = 8;
  int n_tg_y = (hidden_size + rows_per_tg - 1) / rows_per_tg;
  auto grid = MTL::Size(n_tokens, n_tg_y, 1);
  auto group = MTL::Size(64, 1, 1);
  enc.dispatch_threadgroups(grid, group);
}

// ---- GroupedGEMMSwiGLU ----

array grouped_gemm_swiglu(
    const array& x,
    const array& gate_weight,
    const array& gate_scales,
    const array& gate_biases,
    const array& up_weight,
    const array& up_scales,
    const array& up_biases,
    const array& expert_indices,
    const array& token_indices,
    int group_size,
    int bits,
    StreamOrDevice s) {
  auto stream = to_stream(s);
  int n_pairs = expert_indices.size();
  int output_dim = gate_scales.shape(1);

  auto out = array(
      {n_pairs, output_dim},
      gate_scales.dtype(),
      std::make_shared<GroupedGEMMSwiGLU>(stream, group_size, bits, get_metallib_path()),
      {x, gate_weight, gate_scales, gate_biases,
       up_weight, up_scales, up_biases,
       expert_indices, token_indices});
  return out;
}

void GroupedGEMMSwiGLU::eval_cpu(
    const std::vector<array>& inputs,
    array& out) {
  throw std::runtime_error("GroupedGEMMSwiGLU only supports GPU");
}

void GroupedGEMMSwiGLU::eval_gpu(
    const std::vector<array>& inputs,
    array& out) {
  auto& s = stream();
  auto& d = metal::device(s.device);

  out.set_data(allocator::malloc(out.nbytes()));

  auto& x = inputs[0];
  auto& gate_weight = inputs[1];
  auto& gate_scales = inputs[2];
  auto& gate_biases = inputs[3];
  auto& up_weight = inputs[4];
  auto& up_scales = inputs[5];
  auto& up_biases = inputs[6];
  auto& expert_indices = inputs[7];
  auto& token_indices = inputs[8];

  int output_dim = gate_scales.shape(1);
  int n_experts = gate_weight.shape(0);
  int packed_input_dim = gate_weight.shape(2);
  int input_dim = packed_input_dim * (32 / bits_);
  int n_pairs = expert_indices.size();

  auto lib = d.get_library("mlx_fused_moe", metallib_path_);

  std::string kernel_name;
  if (gate_scales.dtype() == bfloat16) {
    kernel_name = "grouped_gemm_swiglu_bf16";
  } else {
    kernel_name = "grouped_gemm_swiglu_f16";
  }
  auto kernel = d.get_kernel(kernel_name, lib);

  auto& enc = d.get_command_encoder(s.index);
  enc.set_compute_pipeline_state(kernel);

  enc.set_input_array(x, 0);
  enc.set_input_array(gate_weight, 1);
  enc.set_input_array(gate_scales, 2);
  enc.set_input_array(gate_biases, 3);
  enc.set_input_array(up_weight, 4);
  enc.set_input_array(up_scales, 5);
  enc.set_input_array(up_biases, 6);
  enc.set_input_array(expert_indices, 7);
  enc.set_input_array(token_indices, 8);
  enc.set_output_array(out, 9);

  enc.set_bytes(input_dim, 10);
  enc.set_bytes(output_dim, 11);
  enc.set_bytes(group_size_, 12);
  enc.set_bytes(n_experts, 13);
  enc.set_bytes(n_pairs, 14);

  // Grid: (n_pairs, ceil(output_dim/8), 1)
  int rows_per_tg = 8;
  int n_tg_y = (output_dim + rows_per_tg - 1) / rows_per_tg;
  auto grid = MTL::Size(n_pairs, n_tg_y, 1);
  auto group = MTL::Size(64, 1, 1);
  enc.dispatch_threadgroups(grid, group);
}

}  // namespace mlx_fused_moe
