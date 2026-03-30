cmake_minimum_required(VERSION 3.25)
project(mlx_fused_moe LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find MLX (MLX_DIR is set by mlx.extension)
find_package(MLX CONFIG REQUIRED)

# Find Python and nanobind
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

# Build Metal shader library
mlx_build_metallib(
    TARGET mlx_fused_moe_metallib
    TITLE mlx_fused_moe
    OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
    SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/kernels/gather_qmm_swiglu.metal
        ${CMAKE_CURRENT_SOURCE_DIR}/kernels/fused_qmv.metal
        ${CMAKE_CURRENT_SOURCE_DIR}/kernels/gather_qmm_down_reduce.metal
        ${CMAKE_CURRENT_SOURCE_DIR}/kernels/grouped_gemm_swiglu.metal
)

# Build Python extension module
nanobind_add_module(
    _ext
    NB_STATIC
    STABLE_ABI
    LTO
    NOMINSIZE
    NB_DOMAIN mlx
    cpp/ops.cpp
    cpp/bindings.cpp
)

target_include_directories(_ext PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(_ext PRIVATE mlx)
add_dependencies(_ext mlx_fused_moe_metallib)
