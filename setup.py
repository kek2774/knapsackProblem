"""
Сборка C++ модуля knapsack_cpp.

Использование:
    pip install pybind11   (если ещё не стоит)
    python setup.py build_ext --inplace

После этого в текущей папке появится knapsack_cpp*.so (Linux/macOS)
или knapsack_cpp*.pyd (Windows), и его можно импортировать:
    import knapsack_cpp
"""

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext_modules = [
    Pybind11Extension(
        "knapsack_cpp",
        ["knapsack_cpp.cpp"],
        extra_compile_args=["-O2", "-std=c++17"],
    ),
]

setup(
    name="knapsack_cpp",
    version="1.0.0",
    author="",
    description="C++ backend for the knapsack benchmark",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.8",
)
