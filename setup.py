#!/usr/bin/env python3
import os
from setuptools import setup

setup(
    name="mbsorl",
    version="0.0.1",
    python_requires=">=3.6",
    install_requires=[
        "loguru",
        "gym",
        "seaborn",
        "PyYAML",
        "numpy",
        "scipy",
        "numpy",
        "matplotlib",
        "absl-py",
        "gin-config",
        "wandb",
        "transformers",
        "ml_collections"
        # "scipy",
        # "neorl @ git+https://agit.ai/Polixir/neorl.git",
        # "OfflineRL @ git+https://agit.ai/Polixir/OfflineRL.git@master#egg=OfflineRL",
    ],
)
