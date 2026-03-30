#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>

#include "mlx/mlx.h"
#include "ops.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace mlx::core;

NB_MODULE(_ext, m) {
  m.doc() = "MLX Fused MoE Extension";

  // Import mlx.core so nanobind can find mlx::core::array type
  nb::module_::import_("mlx.core");

  // Auto-detect metallib path from this module's location
  {
    auto path = nb::module_::import_("pathlib").attr("Path")(m.attr("__file__"));
    auto metallib = path.attr("parent") / nb::str("mlx_fused_moe.metallib");
    auto metallib_str = nb::str(metallib.attr("resolve")());
    mlx_fused_moe::set_metallib_path(nb::cast<std::string>(metallib_str));
  }

  m.def(
      "gather_qmm_swiglu",
      &mlx_fused_moe::gather_qmm_swiglu,
      "x"_a,
      "gate_weight"_a,
      "gate_scales"_a,
      "gate_biases"_a,
      "up_weight"_a,
      "up_scales"_a,
      "up_biases"_a,
      "expert_indices"_a,
      "top_k"_a,
      "group_size"_a = 64,
      "bits"_a = 4,
      "stream"_a = nb::none(),
      "Fused gather_qmm + SwiGLU for MoE decode.");

  m.def(
      "fused_qmv",
      &mlx_fused_moe::fused_qmv,
      "x"_a,
      "weight"_a,
      "scales"_a,
      "biases"_a,
      "n_tokens"_a,
      "group_size"_a = 64,
      "bits"_a = 4,
      "stream"_a = nb::none(),
      "Fused quantized matrix-vector multiply for concatenated projections.");

  m.def(
      "gather_qmm_down_reduce",
      &mlx_fused_moe::gather_qmm_down_reduce,
      "x_intermediate"_a,
      "down_weight"_a,
      "down_scales"_a,
      "down_biases"_a,
      "expert_indices"_a,
      "scores"_a,
      "top_k"_a,
      "group_size"_a = 64,
      "bits"_a = 4,
      "stream"_a = nb::none(),
      "Fused gather_qmm down_proj + score-weighted reduce for MoE decode.");

  m.def(
      "grouped_gemm_swiglu",
      &mlx_fused_moe::grouped_gemm_swiglu,
      "x"_a,
      "gate_weight"_a,
      "gate_scales"_a,
      "gate_biases"_a,
      "up_weight"_a,
      "up_scales"_a,
      "up_biases"_a,
      "expert_indices"_a,
      "token_indices"_a,
      "group_size"_a = 64,
      "bits"_a = 4,
      "stream"_a = nb::none(),
      "Grouped GEMM with fused SwiGLU for MoE prefill.");
}
