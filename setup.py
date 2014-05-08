"""
Setup for compiling cache

Jonas Toft Arnfred, 2013-04-22
"""
from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("cache.pyx")
)
