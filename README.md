# SUPPNet: Neural network for stellar spectrum normalisation

---

[__SUPPNet: Neural network for stellar spectrum normalisation__](https://rozanskit.com/suppnet/)\
[Różański Tomasz](https://rozanskit.com/)<sup>1</sup>, Niemczura Ewa<sup>1</sup>, Lemiesz Jakub<sup>2</sup>, Posiłek Natalia<sup>1</sup>, Różański Paweł<sup>3</sup>

![Here should be example_run.gif](gifs/example_run.gif) 

<sup><sub>1. Astronomical Institute, University of Wrocław, Kopernika 11, 51-622 Wrocław, Poland 2. Department  of  Computer  Science,  Faculty  of  Fundamental  Problems  of  Technology,  Wrocław  University  of  Science  and Technology, Wrocław, Poland 3. Faculty  of  Electronics,  Wrocław  University  of  Science  and Technology, Wrocław</sup></sub>

---

## Installing Guide
SUPPNet can be installed in several simple steps. If you want to test SUPPNet on-line version please check the [link](https://rozanskit.com/suppnet/) (recommended Chrome browser).

### 0. Prerequisites

Install [Python 3.12](https://www.python.org/downloads/) or later. No Anaconda required — all dependencies are available via `pip`.

### 1. Download repository

Download `suppnet` repository by:
```
git clone https://github.com/RozanskiT/suppnet.git
```
Now change the directory to `suppnet`:
```
cd suppnet
```

### 2. Create and activate a virtual environment

Create a virtual environment (recommended):
```
python -m venv .venv
```
Activate it:
- On Linux/macOS:
  ```
  source .venv/bin/activate
  ```
- On Windows:
  ```
  .venv\Scripts\activate
  ```

### 3. Install dependencies

Install all required packages with pip:
```
pip install -r requirements.txt
```

## Creating symbolic link in local '~/bin/' directory

The package contains the script SUPPNET.sh which enables user to use suppnet from any place in the system by simply calling `SUPPNET` command. To create symbolic link please make sure that you have local `~/bin` directory by running:
```
ls ~/bin
```
if you do not have `~/bin` directory you can create one by running: `mkdir ~/bin`. Then create link:
```
ln -s ${PWD}/SUPPNET.sh ~/bin/SUPPNET
```
then inspect the result by:
```
ls -l ~/bin/SUPPNET
```
you should see something like:
```
lrwxrwxrwx 1 tr tr 37 wrz 23 11:20 /home/tr/bin/SUPPNET -> /home/tr/repos/suppnet-dev/SUPPNET.sh
```
**Important:** Before running the `SUPPNET` command, activate your virtual environment:
```
source .venv/bin/activate
```
To test if everything runs correctly just run:
```
SUPPNET
```

## Python script usage
After successful environment setup and linking the script SUPPNET in your personal `bin` directory you should be able to use SUPPNet. Spectra that you are working with should't have header: the first column should contain wavelengths in angstroms (nanometers possible but then you need to change the sampling value from default 0.05 to 0.005), the second should contain flux. Start with:
```
SUPPNET
```
The program window should pop-up and from now you can normalise your spectra. Typical usage scenarios are:

1. Spectrum-by-spectrum normalisation using interactive app:
```
SUPPNET [--segmentation] [--sampling RESAMPLING_STEP=0.05] [--weights WHICH_WEIGHTS=active|synth|emission]
```
2. Normalisation of group of spectra without any supervision:
```
SUPPNET --quiet [--sampling RESAMPLING_STEP=0.05] [--weights WHICH_WEIGHTS=active|synth|emission] [--skip number_of_rows_to_skip=0] path_to_spec_1.txt [path_to_spec_2.txt ...]
```
3. Manual inspection and correction of previously normalised spectrum, SUPPNet will not be loaded (often used in pair with 2.):
```
SUPPNET [--segmentation] --path path_to_processing_results.all
```

You can always remind yourself the typical usage by writing:
```
SUPPNET --help
```

### --sampling and --weights options

- `--sampling`, default=0.05, sampling option enable user to adjust the resampling that the neural network is using for a pseudo-continuum prediction, (If working with wavelengths in nm should be changed to 0.005),
- `--weights`, default=active, set of weights that can be used, __active__ is a default one, __emission__ should be used for objects that show wide emission lines, __synth__ is a weights trained only on synthetic spectra and shouldn't be used.

## SUPPNet as python module

You can install and use `suppnet` as regular Python module. First activate your virtual environment, then call:
```
pip install -e .
```
For an example usage check the notebook in `notebooks` directory.