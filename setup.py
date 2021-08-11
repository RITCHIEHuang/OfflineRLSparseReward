#!/usr/bin/env python3
import os
from setuptools import setup

setup(
    name='mbsorl',
    version='0.0.1',
    python_requires=">=3.6",
    install_requires=[
        "loguru",
        "gym",
        "d3rlpy",
        "numpy",
    ],
    
)
