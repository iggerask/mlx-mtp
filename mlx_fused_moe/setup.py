"""Build script for the mlx_fused_moe C++ extension."""

from mlx.extension import CMakeExtension, CMakeBuild
from setuptools import setup

setup(
    name="mlx_fused_moe",
    version="0.1.0",
    description="Fused gather_qmm_swiglu Metal kernel for MLX MoE models",
    author="mlx-mtp",
    ext_modules=[CMakeExtension("mlx_fused_moe._ext")],
    cmdclass={"build_ext": CMakeBuild},
    packages=["mlx_fused_moe"],
    package_data={"mlx_fused_moe": ["*.metallib", "*.dylib"]},
    install_requires=["mlx>=0.30.0"],
    python_requires=">=3.10",
)
