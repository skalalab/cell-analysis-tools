#!/usr/bin/env python

from pathlib import Path

from setuptools import setup, find_packages

THIS_DIR = Path(__file__).parent

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="flim-tools-econtreras",
    version="0.0.1",
    author="Emmanuel Contreras",
    author_email="econtreras@wisc.edu",
    long_description=Path("README.md").read_text(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
    ],
    packages=find_packages(exclude=["tests", "dev"]),
    install_requires=Path("requirements.txt").read_text().splitlines(),
)
