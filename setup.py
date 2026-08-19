from pathlib import Path
from setuptools import setup, find_packages

long_description = Path("README.md").read_text(encoding="utf-8")

setup(name='suppnet',
      version='0.0.1',
      description="SUPPNet: Neural network for stellar spectrum normalisation",
      author="Tomasz Różański",
      author_email="tomasz.rozanski@uwr.edu.pl",
      long_description=long_description,
      long_description_content_type="text/markdown",
      url="https://github.com/RozanskiT/suppnet.git",
      packages=find_packages(),
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        ],
      python_requires='>=3.12.3',
      install_requires=[
          'numpy==2.5.1',
          'scipy==1.18.0',
          'PySide6==6.11.1',
          'matplotlib==3.11.1',
          'pandas==3.0.5',
          'torch==2.13.0',
          'h5py==3.16.0',
      ],
)

# Install in editable/development mode with pip:
# pip install -e .

# Or for a regular install:
# pip install .