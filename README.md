# Accelerating Multicomponent Phase-Coexistence Calculations with Machine Learning

<br />
<img src="./website/overview.png" />
<br />

## Install Instructions

Please install using the `setup.py`.

```console
$ git clone https://github.com/webbtheosim/ml-ternary-phase.git
$ cd ml-ternary-phase
$ conda create --name phase python=3.8.16
$ conda activate phase
$ # or source activate phase
$ pip install -e .
```

## Download Data and Results
Select a disk location for data storage and update the directory paths before running the program. Download the required data from Zenodo [here](https://doi.org/10.5281/zenodo.13776946).
- **DATA_DIR**: Stores data pickle files (approx. 400 MB).

Additional training weights and results can be downloaded [here](https://drive.google.com/drive/folders/).
- **TRAIN_RESULT_DIR** or **HIST_DIR** (optional): Stores training results in pickle format (approx. 2.5 GB).
- **MODEL_DIR** (optional): Stores training weights in .h5 format (approx. 1.3 GB).

To train from scratch, only the **DATA_DIR** is required.

