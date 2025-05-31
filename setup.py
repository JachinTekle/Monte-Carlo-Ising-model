from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "my_module",                 # Name des zu erzeugenden Moduls
        ["my_module.pyx"],           # Quellcode-Datei
        include_dirs=[numpy.get_include()]  # Pfad zu den NumPy-Headern
    )
]

setup(
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"}),
)