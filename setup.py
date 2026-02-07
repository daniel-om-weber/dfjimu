import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class OptionalBuildExt(build_ext):
    """build_ext that warns instead of crashing when compilation fails."""

    def build_extension(self, ext):
        try:
            super().build_extension(ext)
        except Exception as e:
            print(f"WARNING: Failed to build extension {ext.name}: {e}")
            print("The package will use pure Python fallbacks (slower).")


def get_extensions():
    try:
        import numpy as np
    except ImportError:
        print("WARNING: NumPy not available. "
              "Skipping C extension compilation.")
        return []

    try:
        from Cython.Build import cythonize
        USE_CYTHON = True
    except ImportError:
        USE_CYTHON = False

    ext = '.pyx' if USE_CYTHON else '.c'
    extensions = [
        Extension(
            "dfjimu._cython.core",
            [f"src/dfjimu/_cython/core{ext}"],
            include_dirs=[np.get_include()],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        ),
        Extension(
            "dfjimu._cython.optimizer",
            [f"src/dfjimu/_cython/optimizer{ext}"],
            include_dirs=[np.get_include()],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        ),
        Extension(
            "dfjimu._cython.lever_arms",
            [f"src/dfjimu/_cython/lever_arms{ext}"],
            include_dirs=[np.get_include()],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        ),
    ]

    if USE_CYTHON:
        extensions = cythonize(extensions, compiler_directives={'language_level': "3"})

    return extensions


setup(
    ext_modules=get_extensions(),
    cmdclass={'build_ext': OptionalBuildExt},
)
