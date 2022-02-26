#!/usr/bin/env python3
import os
from setuptools import setup

setup(
    name="offlinerlsparse",
    version="0.0.1",
    python_requires=">=3.7",
    install_requires=[
        "absl-py",
        "loguru",
        "gym",
        "seaborn",
        "PyYAML",
        "numpy",
        "scipy",
        "numpy",
        "matplotlib",
        "wandb",
        # decision transformer
        "transformers",
        # rec_sim
        "gin-config",
        "dopamine-rl",
        # d4rl
        "h5py",
        "mujoco_py",
        "pybullet",
        "dm_control",
#        "mjrl",
        "stable-baselines3",
        "numba",
        "ray[tune]",
        # gdown
        "gdown",
    ],
)
