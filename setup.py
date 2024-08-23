from setuptools import find_packages, setup

with open("README.md", "r") as fh:
  long_description = fh.read()

# Read in requirements
requirements = [
    requirement.strip() for requirement in open('requirements.txt').readlines()
]

setup(
    name="dft_exconditions",
    version="0.0.2",
    author="Ryan Pederson",
    author_email="pedersor@uci.edu",
    description="A library for checking exact conditions in DFT approximations.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pedersor/dft_exconditions",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
    ],
    python_requires=('>=3.8'),
    install_requires=requirements,
    packages=find_packages(),
)
