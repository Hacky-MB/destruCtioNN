from distutils.core import setup
from Cython.Build import cythonize

# launch as python setup.py build_ext --inplace

setup(
    ext_modules=cythonize("optimalized.pyx")
)
