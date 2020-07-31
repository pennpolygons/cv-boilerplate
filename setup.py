#!/usr/bin/env python

import io
import os

from setuptools import setup

# Package meta-data.
NAME = "Research"
DESCRIPTION = "My short description for my research project."
URL = "https://github.com/pennpolygons/cv-boilerplate"
EMAIL = "jane@institution.com"
AUTHOR = "Jane Doe"
REQUIRES_PYTHON = ">=3.5.0"
VERSION = "0.1.0"
PACKAGES = ["research"]

# What packages are required for this module to be executed?
REQUIRED = [
    "hydra-core>=1.0.0-rc1",
    "omegaconf>=2.0.1rc6",
    "opencv-python>=4.2.0",
    "matplotlib>=3.2.1",
    "visdom>=0.1.8.9",
    "torch>=1.5.0",
    "pytorch-ignite>=0.3.0",
    "torchvision>=0.6.0",
]

# What packages are optional?
EXTRAS = {
    # 'fancy feature': ['django'],
}

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    # CHANGEME:
    packages=PACKAGES,
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license="MIT",
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
)
