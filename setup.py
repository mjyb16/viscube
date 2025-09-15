#!/usr/bin/env python

import glob
import os
import sys
from shutil import rmtree

from setuptools import Command, find_packages, setup
from setuptools.command.test import test as TestCommand


def read(fname):
    with open(os.path.join(os.path.dirname(__file__), fname)) as f:
        return f.read()


install_requires = read("requirements.txt").splitlines()

# Dynamically determine extra dependencies
extras_require = {}
extra_req_files = glob.glob("requirements-*.txt")
for extra_req_file in extra_req_files:
    name = os.path.splitext(extra_req_file)[0].replace("requirements-", "", 1)
    extras_require[name] = read(extra_req_file).splitlines()

if extras_require:
    extras_require["all"] = sorted({x for v in extras_require.values() for x in v})

# Import meta data from __meta__.py (no version here)
meta = {}
exec(read("supermage/__meta__.py"), meta)

# README handling
possible_readme_names = ["README.rst", "README.md", "README.txt", "README"]
long_description = meta["description"]
readme_fname = ""
for fname in possible_readme_names:
    try:
        long_description = read(fname)
    except IOError:
        continue
    else:
        readme_fname = fname
        break

readme_ext = os.path.splitext(readme_fname)[1]
if readme_ext.lower() == ".rst":
    long_description_content_type = "text/x-rst"
elif readme_ext.lower() == ".md":
    long_description_content_type = "text/markdown"
else:
    long_description_content_type = "text/plain"


class PyTest(TestCommand):
    """Support setup.py test."""

    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest
        pytest.main(self.test_args)


class UploadCommand(Command):
    """Support setup.py upload (local convenience)."""

    description = "Build and publish the package."
    user_options = []

    @staticmethod
    def status(s):
        print("\033[1m{0}\033[0m".format(s))

    def initialize_options(self): pass
    def finalize_options(self): pass

    def run(self):
        try:
            self.status("Removing previous builds...")
            here = os.path.abspath(os.path.dirname(__file__))
            rmtree(os.path.join(here, "dist"))
        except OSError:
            pass

        self.status("Building Source and Wheel (universal) distribution...")
        os.system(f"{sys.executable} -m build")

        self.status("Uploading the package to PyPI via Twine...")
        os.system("twine upload dist/*")

        # NOTE: We do NOT create/push tags here anymore.
        # Version is derived from existing git tags via setuptools_scm.
        sys.exit()


setup(
    name=meta["name"],
    # Version comes from git tags via setuptools_scm:
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    package_dir={meta["name"]: os.path.join(".", meta["path"])},
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require=extras_require,
    # Metadata to display on PyPI
    author=meta["author"],
    author_email=meta["author_email"],
    description=meta["description"],
    long_description=long_description,
    long_description_content_type=long_description_content_type,
    license=meta["license"],
    url=meta["url"],
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    cmdclass={
        "test": PyTest,
        "upload": UploadCommand,
    },
)
