from Cython.Build import cythonize
from setuptools import Extension, setup

ext_modules = [
    Extension(
        "cython_worker",
        ["src/workers/cython_python/cython_worker.pyx"],
        extra_compile_args=["-fopenmp"],
        extra_link_args=["-fopenmp"],
    )
]

setup(ext_modules=cythonize(ext_modules))
