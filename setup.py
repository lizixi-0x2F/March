"""
setup.py — build hook that compiles libmarch.so via make before install.

    uv pip install git+https://github.com/lizixi-0x2F/March
    pip install git+https://github.com/lizixi-0x2F/March
"""
import os
import subprocess
from setuptools import setup
from setuptools.command.build_ext import build_ext


class MakeBuild(build_ext):
    def run(self):
        march_dir = os.path.join(os.path.dirname(__file__), "march")
        subprocess.run(["make", "all"], cwd=march_dir, check=True)
        # Copy libmarch.so into the build lib dir so it gets packaged
        import shutil
        src = os.path.join(march_dir, "libmarch.so")
        dst_dir = os.path.join(self.build_lib, "march")
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy2(src, dst_dir)


setup(cmdclass={"build_ext": MakeBuild})
