"""Editable installation metadata for the local AMP learning library."""

from setuptools import find_packages, setup


setup(
    name="rsl-rl-amp",
    version="0.1.0",
    description="TensorDict PPO and adversarial motion priors for Unitree RL Lab",
    packages=find_packages(include=["rsl_rl_amp", "rsl_rl_amp.*"]),
    python_requires=">=3.10",
    install_requires=["numpy>=1.24", "torch>=2.6", "tensordict>=0.7", "gymnasium", "GitPython"],
)
