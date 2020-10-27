#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="SGGM",
    version="0.0.0",
    description="Scalable Geometrical Generative Models",
    author="Pierre Segonne",
    author_email="pierre.segonne@protonmail.com",
    url="https://github.com/pierresegonne/SGGM",
    install_requires=["pytorch-lightning"],
    packages=find_packages(),
)
