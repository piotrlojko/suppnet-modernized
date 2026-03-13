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
      python_requires='>=3.12',
      install_requires=[
          'numpy==2.2.6',
          'scipy==1.15.2',
          'PySide6==6.8.3',
          'matplotlib==3.9.4',
          'pandas==2.2.3',
          'tensorflow==2.18.0',
      ],
)

# Way to install in develop mode
# python setup.py develop
# https://setuptools.readthedocs.io/en/latest/setuptools.html#development-mode

# or simply install
# python setup.py install