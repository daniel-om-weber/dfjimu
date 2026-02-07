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
        from Cython.Build import cythonize
    except ImportError:
        print("WARNING: Cython or NumPy not available. "
              "Skipping C extension compilation.")
        return []

    extensions = [
        Extension(
            "dfjimu._cython.core",
            ["src/dfjimu/_cython/core.pyx"],
            include_dirs=[np.get_include()],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        ),
        Extension(
            "dfjimu._cython.optimizer",
            ["src/dfjimu/_cython/optimizer.pyx"],
            include_dirs=[np.get_include()],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        ),
    ]
    return cythonize(extensions, compiler_directives={'language_level': "3"})


setup(
    ext_modules=get_extensions(),
    cmdclass={'build_ext': OptionalBuildExt},
)
