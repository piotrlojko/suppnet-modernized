> [!IMPORTANT]
> ## This repo contains a vibe-coded, modernized fork of the original `SUPPNet` software. The code itself is functional and tested.
> The main goal was to create a sleek and modern `SUPPNet` release, tailored for the latest versions of Python (v3.14) and
> the required packages (e.g., PyTorch, Pandas). Another incentive was to transition from the original `Anaconda`-rooted
> solution into a pure `venv`-based one, since `Anaconda` is known for having trouble with resolving its databases.
>
> **NOTE:** The neural network now runs on `PyTorch` instead of `TensorFlow`.
> 
> *All the credit belongs to **Tomasz Różański** and **Collaborators***


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

Install [Python 3.12.3](https://www.python.org/downloads/) or later — including Python 3.14. All dependencies are
available via `pip`.

### 1. Download repository

Download `suppnet` repository by:
```
git clone https://github.com/piotrlojko/suppnet-modernized.git
```
Now change the directory to `suppnet-modernized`:
```
cd suppnet-modernized
```

### 2. Create and activate a virtual environment

Create a dedicated virtual environment named `suppnet_env` in your home directory (recommended):
```
python -m venv ~/suppnet_env
```
Activate it:
- On Linux/macOS:
  ```
  source ~/suppnet_env/bin/activate
  ```
- On Windows:
  ```
  %USERPROFILE%\suppnet_env\Scripts\activate
  ```

### 3. Install dependencies 
#### 3.1 Nvidia GPU:

Install all required packages (including `PyTorch` for CUDA) with pip:
```
pip install -r requirements-nvidia.txt
```
#### 3.2 Intel/AMD GPUs or CPUs only:
```
pip install -r requirements-no-torch.txt
```
For intel GPUs 
```
pip install torch[==2.13.0] --index-url https://download.pytorch.org/whl/xpu 
```
For AMD with ROCm platform
```
pip install torch[==2.13.0] --index-url https://download.pytorch.org/whl/rocm7.2 
```

For CPU-based use only 
```
pip install torch[==2.13.0] --index-url https://download.pytorch.org/whl/cpu 
```

## Creating symbolic link in local '~/bin/' directory

The package contains the script SUPPNET.sh, which enables the user to use SUPPNet from any place in the system by simply calling `SUPPNET` command. To create such a symbolic link, please make sure that you have a local `~/bin` directory by running:
```
ls ~/bin
```
If you do not have `~/bin` directory, you can create one by running: `mkdir ~/bin`. **You must ensure that your local `~/bin` is on PATH.**
Then, **from within the suppnet directory**, create a symlink:
```
cd ~/suppnet-modernized
ln -s $(pwd)/SUPPNET.sh ~/bin/SUPPNET
```
Inspect the result by:
```
ls -l ~/bin/SUPPNET
```
You should see something like:
```
lrwxrwxrwx 1 tr tr 37 wrz 23 11:20 /home/piotr/bin/SUPPNET -> /home/piotr/suppnet-modernized/SUPPNET.sh
```
**Important:** Before running the `SUPPNET` command, activate your virtual environment:
```
source ~/suppnet_env/bin/activate
```
To test if everything runs correctly, just run:
```
SUPPNET
```

## Creating a system-wide symbolic link in the '/usr/bin/' directory

If you have superuser privileges, you can run:
```
cd ~/suppnet-modenized
sudo ln -s $(pwd)/SUPPNET.sh /usr/bin/SUPPNET
```
to enable SUPPNet for all users.

## Unlinking `SUPPNet`

If you want to get rid of this code once and for all, you should also consider removing any existing symlinks. Run:
```
type -a SUPPNET
```
and inspect the result to find a path to your `SUPPNet` symbolic link. Then simply remove it with:
```
sudo rm /path/to/the/symlink
```

## Python script usage
After successful environment setup and linking the script SUPPNET in your personal `bin` directory, you should be able to use SUPPNet. Spectra that you are working with shouldn't have any header: the first column is to contain wavelengths in angstroms (nanometers possible, but then you need to change the sampling value from default 0.05 to 0.005), the second should contain flux. Start with:
```
SUPPNET
```
The program GUI window should pop up, and from then on, you are good to go to normalise some spectra! Typical usage scenarios are:

1. Spectrum-by-spectrum normalisation using an interactive app:
```
SUPPNET [--segmentation] [--sampling RESAMPLING_STEP=0.05] [--weights WHICH_WEIGHTS=active|synth|emission] [--device DEVICE]
```
2. Normalisation of a group of spectra without any supervision:
```
SUPPNET --quiet [--sampling RESAMPLING_STEP=0.05] [--smoothing SMOOTHING_FACTOR=1.0] [--weights WHICH_WEIGHTS=active|synth|emission] [--device DEVICE] [--skip number_of_rows_to_skip=0] path_to_spec_1.txt [path_to_spec_2.txt ...]
```
3. Manual inspection and correction of previously normalised spectrum, SUPPNet will not be loaded (often used in pair with 2.):
```
SUPPNET [--segmentation] --path path_to_processing_results.all
```

You can always remind yourself of the usage by writing:
```
SUPPNET --help
```

### --sampling, --smoothing, --weights and --device options

- `--sampling`, default=0.05, sampling option enables the user to adjust the resampling that the neural network is using for a pseudo-continuum prediction, (If working with wavelengths in nm should be changed to 0.005),
- `--smoothing`, default=1.0, sets the pseudo-continuum smoothing factor used in quiet mode; values below 0.05 are clamped to 0.05 to keep spline fitting responsive and numerically stable,
- `--weights`, default=active, set of weights that can be used, __active__ is a default one, __emission__ should be used for objects that show wide emission lines, __synth__ is a set of weights trained only on synthetic spectra and shouldn't be used for doing science,
- `--device`, default=best available, the device the neural network runs on (`cpu`, `cuda`, `xpu`, `mps`); see [Neural network backend](#neural-network-backend).

## SUPPNet as a Python module

You can install and use `suppnet` as a regular Python module. First, activate your virtual environment, then call:
```
pip install -e .
```
For an example usage, check the notebook in `notebooks` directory.
