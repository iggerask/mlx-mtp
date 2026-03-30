"""Build Cython extensions for MTP fast decode loop."""

from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension(
        "vllm_mlx_mtp._fast_loop",
        ["vllm_mlx_mtp/_fast_loop.pyx"],
    ),
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": 3,
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
        },
    ),
)
