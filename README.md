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

Additional training weights and results can be downloaded [here](https://drive.google.com/drive/folders/1BjnHbVxUHIj7Gj4wliY5N8mMjR9abFwi?usp=sharing).
- **TRAIN_RESULT_DIR**: Stores training results in pickle format (approx. 5 GB).
- **RESULT_DIR**: Stores results in pickle format for analysis and plotting (approx. 229 GB).
- **MODEL_DIR**: Stores training weights in .h5 format (approx. 14 MB).
- **OPT_RESULT_DIR**: Stores post-ML Newton-CG optimization results in pickle format (approx. 78 MB)
- **PICKLE_INNER_PATH**: Stores training results of hyperparameter tuning (approx. 16 MB)

To train from scratch, only the **DATA_DIR** is required.

```python
# LOAD SIMULATION DATA
DATA_DIR = "your/custom/dir/"

filename = os.path.join(DATA_DIR, f"data_clean.pickle")
with open(filename, "rb") as handle:
    (x, y_c, y_r, phase_idx, num_phase, max_phase) = pickle.load(handle)
```
- **x**: Input $\mathbf{x}=(\chi_\mathrm{AB}, \chi_\mathrm{BC}, \chi_\mathrm{AC}, v_\mathrm{A}, v_\mathrm{B}, v_\mathrm{C})\in \mathbb{R}^8$.
- **y_c**: Output one-hot encoded classification vector $\mathbf{y}_\mathrm{c}\in \mathbb{R}^3$.
- **y_r**: Output equilibrium composition and abundance vector y<sub>r</sub> = (&phi;<sub>A</sub><sup>&alpha;</sup>, &phi;<sub>B</sub><sup>&alpha;</sup>, &phi;<sub>A</sub><sup>&beta;</sup>, &phi;<sub>B</sub><sup>&beta;</sup>, &phi;<sub>A</sub><sup>&gamma;</sup>, &phi;<sub>B</sub><sup>&gamma;</sup>, w<sup>&alpha;</sup>, w<sup>&beta;</sup>, w<sup>&gamma;</sup>) &isin; ℝ<sup>9</sup>.
- **phase_idx**: A single integer indicating which unique phase system it belongs to.
- **num_phase**: A single integer indicates the number of equilibrium phases the input splits into.
- **max_phase**: A single integer indicates the maximum number of equilibrium phases the system splits into.

## File Structure

### Notebooks
The `notebook` folder contains Jupyter notebooks for reproducing figures and tables:
- `result.ipynb`: Visualizes all result figures from the paper.
- `optimization.ipynb`: Generates coexistence curves using ML predictions and post-ML Newton-CG optimization.

### Machine Learning Core
The `mlphase` folder contains the core ML code:
- `run_innercv.py`: Performs hyperparameter tuning using 10% of the training data (inner loop of nested five-fold cross-validation).
- `run_outercv.py`: Conducts production run training using the best hyperparameter combinations (outer loop of nested five-fold cross-validation).
- `run_opt.py`: Executes post-ML Newton-CG optimization for coexistence curve predictions.
- `analysis/`: Scripts for calculating and evaluating accuracy metrics of model predictions.
- `data/`: Scripts for data splitting and loading.
- `models/`: Defines architecture and implementation of various machine learning models.
- `plot/`: Scripts for creating visual representations of data, including figures for analysis and publication.

### Data Generation
The `hull_creation` folder contains code for convex hull data generation.

### Results
- `result_pickle/`: Contains temporary files used for figure preparation.
- `result_csv/`: Contains temporary files used for table creation.

Note: To generate your own results, set `reload=True` in the notebooks.

### Job Submission
The `submit` folder contains job submission scripts for High-Performance Computing environments:
- `innercv.submit`: Neural network hyperparameter tuning.
- `outercv.submit`: Neural network production run training.
- `post_opt.submit`: Post-ML optimization.