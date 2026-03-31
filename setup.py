"""CMake-based build for pybind11 extension module."""

import subprocess
from pathlib import Path

import pybind11
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


PROJECT_ROOT = Path(__file__).resolve().parent


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        super().__init__(name, sources=[])
        self.sourcedir = str((PROJECT_ROOT / Path(sourcedir)).resolve())


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        extdir = Path(self.get_ext_fullpath(ext.name)).resolve().parent
        cfg = "Release"

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
            f"-Dpybind11_DIR={pybind11.get_cmake_dir()}",
        ]

        build_temp = Path(self.build_temp) / ext.name
        build_temp.mkdir(parents=True, exist_ok=True)

        subprocess.run(
            ["cmake", ext.sourcedir, *cmake_args],
            cwd=build_temp, check=True,
        )
        subprocess.run(
            ["cmake", "--build", ".", "--config", cfg, "--target", "_core"],
            cwd=build_temp, check=True,
        )


setup(
    ext_modules=[CMakeExtension("evtq._core")],
    cmdclass={"build_ext": CMakeBuild},
)
