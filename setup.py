from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

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
          'numpy==2.4.3',
          'scipy==1.17.1',
          'PySide6==6.10.2',
          'matplotlib==3.10.8',
          'pandas==3.0.1',
          'tensorflow==2.21.0',
      ],
)

# Install in editable/development mode with pip:
# pip install -e .

# Or for a regular install:
# pip install .