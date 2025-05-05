import os
from setuptools import setup, find_packages

setup(
    name="neuro_fuzzy_multiagent",
    version="0.1.0",
    description="Enviroment Independent dynamic self Orgenizing Neuro Fuzzy Multi Agent System",
    author="Dumith Salinda",
    author_email="dumithrathnayaka@yahoo.com",
    packages=find_packages(include=["neuro_fuzzy_multiagent", "neuro_fuzzy_multiagent.*"]),
    install_requires=[
        line.strip() for line in open("requirements.txt")
        if line.strip() and not line.startswith("#")
    ],
    include_package_data=True,
    license="MIT",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
)
