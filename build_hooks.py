"""
build_hooks.py — setuptools build extension hook.
Runs `make` inside the march/ directory to produce libmarch.so
before the Python package is installed.
"""
import subprocess
import os
from setuptools.command.build_ext import build_ext


class MakeBuild(build_ext):
    def run(self):
        march_dir = os.path.join(os.path.dirname(__file__), "march")
        subprocess.run(["make", "all"], cwd=march_dir, check=True)
