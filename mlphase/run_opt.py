import os
import pickle
import itertools
from timeit import default_timer as timer

import numpy as np
import proplot as pplt
from autograd import grad, hessian
from scipy.optimize import minimize
from tqdm import tqdm

from mlphase.plot import (
    gen_ternary_boundary_post,
    gen_ternary_boundary_ml,
    gen_ternary_scatter,
)

from mlphase.analysis.opt import min_2phase, min_3phase, stability_condition

# Initialize color cycles for plotting
COLORS = []
colors = pplt.Cycle("ggplot")
for color in colors:
    COLORS.append(color["color"])
colors = pplt.Cycle("default")
for color in colors:
    COLORS.append(color["color"])

# Directories for saving plots and optimization results
PLOT_DIR = "/scratch/gpfs/sj0161/mlphase/fig_ternary/"
OPT_RESULT_DIR = "/scratch/gpfs/sj0161/mlphase/opt_pickle/"
DATA_DIR = "/scratch/gpfs/sj0161/mlphase/data/"

# 0, 1, 2 are phi_a^alpha, phi_b^alpha, phi_c^alpha (c depends on a and b, not considered)
# 3, 4, 5 are phi_a^beta, phi_b^beta, phi_c^beta
# 6, 7, 8 are phi_a^gamma, phi_b^gamma, phi_c^gamma
# 9, 10, 11 are w^alpha, w^beta, w^gamma


def generate_plots(
    opt_file,
    yc_test,
    yr_test,
    phase_2_pred,
    phase_3_pred,
    phase_2_pred_ml,
    phase_3_pred_ml,
    x_test,
    yc_pred,
    valid_idx,
    valid_idx_3,
    PLOT_DIR,
    COLORS,
    three_phase,
):
    """
    Generates ternary boundary and scatter plots for phase predictions.
    """
    # Generate ternary boundary plot for post-ML optimization predictions
    gen_ternary_boundary_post(
        file=opt_file,
        yc_test=yc_test,
        yr_test=yr_test,
        phase_2_pred=phase_2_pred,
        phase_3_pred=phase_3_pred,
        PLOT_DIR=PLOT_DIR,
        COLORS=COLORS,
        three_phase=three_phase,
        format="png",
    )

    # Filter ML predictions based on valid indices
    phase_2_pred_ml_in = (
        phase_2_pred_ml if valid_idx is None else phase_2_pred_ml[valid_idx]
    )
    phase_3_pred_ml_in = (
        phase_3_pred_ml if valid_idx_3 is None else phase_3_pred_ml[valid_idx_3]
    )

    # Generate ternary boundary plot for ML predictions
    gen_ternary_boundary_ml(
        file=opt_file,
        yc_test=yc_test,
        yr_test=yr_test,
        phase_2_pred=phase_2_pred_ml_in,
        phase_3_pred=phase_3_pred_ml_in,
        PLOT_DIR=PLOT_DIR,
        COLORS=COLORS,
        three_phase=three_phase,
        format="png",
    )

    # Generate ternary scatter plot
    gen_ternary_scatter(
        file=opt_file,
        x_test=x_test,
        yc_test=yc_test,
        yc_pred=yc_pred,
        yr_test=yr_test,
        PLOT_DIR=PLOT_DIR,
        COLORS=COLORS,
        three_phase=three_phase,
        format="png",
    )


def main(
    pickle_file,
    opt_result_dir,
    index=207,
    plot_fig=True,
    EPS=1e-7,
    rerun=False,
):
    with open(pickle_file, "rb") as handle:
        yc_test_all = pickle.load(handle)
        yc_pred_all = pickle.load(handle)
        yr_test_all = pickle.load(handle)
        yr_pred_all = pickle.load(handle)
        x_test_all = pickle.load(handle)
        phase_idx_test = pickle.load(handle)

    # Get unique phase indices
    phase_idx_unique = np.unique(phase_idx_test).astype("int")
    idx_unique = phase_idx_unique[index]

    base_name = os.path.basename(pickle_file)
    base_name = base_name.replace("MLPHASEOUTER", "OPT").replace(
        ".pickle", f"-{int(idx_unique)}.pickle"
    )

    print(base_name)

    opt_file = os.path.join(opt_result_dir, base_name)

    row_idx = np.where(phase_idx_test == idx_unique)[0]

    # Extract test data for the current phase index
    x_test = x_test_all[row_idx]
    yc_test = yc_test_all[row_idx].argmax(axis=1)
    yc_pred = yc_pred_all[row_idx].argmax(axis=1)
    yr_pred = yr_pred_all[row_idx]
    yr_test = yr_test_all[row_idx]

    idx_ml_1 = np.where(yc_pred == 1)[0]
    idx_ml_2 = np.where(yc_pred == 2)[0]

    phase_2_pred_ml = yr_pred[idx_ml_1][:, [0, 1, 3, 4]]
    phase_3_pred_ml = yr_pred[idx_ml_2][:, [0, 1, 3, 4, 6, 7]]

    # Determine if three-phase system
    if yc_test.max() == 2:
        three_phase = True
    else:
        three_phase = False

    # Extract parameters from test data
    v_a, v_c, v_b, chi_ac, chi_ab, chi_bc, _, _ = x_test[0]

    if os.path.exists(opt_file) and not rerun:
        with open(opt_file, "rb") as handle:
            y_opt_2 = pickle.load(handle)
            v_fun_2 = pickle.load(handle)
            y_opt_3 = pickle.load(handle)
            v_fun_3 = pickle.load(handle)
            time = pickle.load(handle)
            n_run_2 = pickle.load(handle)
            n_run_3 = pickle.load(handle)

        phase_2_pred = y_opt_2
        phase_3_pred = y_opt_3

        valid_idx = None
        valid_idx_3 = None

        # Validate two-phase predictions
        if len(y_opt_2) > 1:
            D1 = stability_condition(
                y_opt_2[:, 0], y_opt_2[:, 1], v_a, v_c, v_b, chi_ab, chi_bc, chi_ac
            )
            D2 = stability_condition(
                y_opt_2[:, 2], y_opt_2[:, 3], v_a, v_c, v_b, chi_ab, chi_bc, chi_ac
            )
            flag_conv = v_fun_2 < -1 + EPS

            valid_idx = np.where((D1 >= 0) & (D2 >= 0) & flag_conv)[0]

            phase_2_pred = np.copy(phase_2_pred)[valid_idx]

        # Validate three-phase predictions
        if three_phase and len(y_opt_3) > 1:
            D1 = stability_condition(
                y_opt_3[:, 0], y_opt_3[:, 1], v_a, v_c, v_b, chi_ab, chi_bc, chi_ac
            )
            D2 = stability_condition(
                y_opt_3[:, 2], y_opt_3[:, 3], v_a, v_c, v_b, chi_ab, chi_bc, chi_ac
            )
            D3 = stability_condition(
                y_opt_3[:, 4], y_opt_3[:, 5], v_a, v_c, v_b, chi_ab, chi_bc, chi_ac
            )
            flag_conv_3 = v_fun_3 < EPS

            valid_idx_3 = np.where((D1 >= 0) & (D2 >= 0) & (D3 >= 0) & flag_conv_3)[0]

            phase_3_pred = np.copy(phase_3_pred)[valid_idx_3]

        # Calculate average time per run
        if len(y_opt_3) > 1 and len(y_opt_2) > 1:
            time = time / (len(y_opt_3) + len(y_opt_2))
        elif len(y_opt_3) <= 1 and len(y_opt_2) > 1:
            time = time / len(y_opt_2)
        else:
            time = np.nan

    else:
        t0 = timer()
        x_test = x_test_all[row_idx]
        yc_test = yc_test_all[row_idx].argmax(axis=1)
        yc_pred = yc_pred_all[row_idx].argmax(axis=1)
        yr_pred = yr_pred_all[row_idx]
        yr_test = yr_test_all[row_idx]

        idx = np.where(yc_pred == 1)[0]

        def wrapped_func(params):
            return min_2phase(params, X_fixed)

        jac = grad(wrapped_func)
        hess = hessian(wrapped_func)

        y_opt_2 = []
        v_fun_2 = []
        n_run_2 = []
        for i in tqdm(idx, total=len(idx)):
            X_fixed = x_test[i, :]
            y1 = yr_pred[i, 0:2]
            y2 = yr_pred[i, 3:5]

            initial_guess = np.concatenate([y1, y2])

            result = minimize(
                wrapped_func,
                initial_guess,
                method="Newton-CG",
                jac=jac,
                hess=hess,
                options={"maxiter": 10000},
            )
            y_opt_2.append(result.x)
            v_fun_2.append(result.fun)
            n_run_2.append(result.nit)

        y_opt_2 = np.array(y_opt_2)
        v_fun_2 = np.array(v_fun_2)
        n_run_2 = np.array(n_run_2)

        if yc_test.max() == 2 and len(np.where(yc_pred == 2)[0]) > 0:
            three_phase = True
            idx = np.where(yc_pred == 2)[0]

            def wrapped_func(params):
                return min_3phase(params, X_fixed)

            jac = grad(wrapped_func)
            hess = hessian(wrapped_func)

            y_opt_3 = []
            v_fun_3 = []
            n_run_3 = []
            for i in tqdm(idx, total=len(idx)):
                X_fixed = x_test[i, :]
                y1 = yr_pred[i, 0:2]
                y2 = yr_pred[i, 3:5]
                y3 = yr_pred[i, 6:8]

                initial_guess = np.concatenate([y1, y2, y3])

                result = minimize(
                    wrapped_func,
                    initial_guess,
                    method="Newton-CG",
                    jac=jac,
                    hess=hess,
                    options={"maxiter": 10000},
                )
                y_opt_3.append(result.x)
                v_fun_3.append(result.fun)
                n_run_3.append(result.nit)
        else:
            three_phase = False
            y_opt_3 = [-1]
            v_fun_3 = [-1]
            n_run_3 = [-1]

        y_opt_3 = np.array(y_opt_3)
        v_fun_3 = np.array(v_fun_3)
        n_run_3 = np.array(n_run_3)

        t1 = timer()

        with open(opt_file, "wb") as handle:
            pickle.dump(y_opt_2, handle)
            pickle.dump(v_fun_2, handle)
            pickle.dump(y_opt_3, handle)
            pickle.dump(v_fun_3, handle)
            pickle.dump(t1 - t0, handle)
            pickle.dump(n_run_2, handle)
            pickle.dump(n_run_3, handle)

        print(opt_file)

        valid_idx = None
        valid_idx_3 = None

        # Validate two-phase predictions
        if len(y_opt_2) > 1:
            D1 = stability_condition(
                y_opt_2[:, 0], y_opt_2[:, 1], v_a, v_c, v_b, chi_ab, chi_bc, chi_ac
            )
            D2 = stability_condition(
                y_opt_2[:, 2], y_opt_2[:, 3], v_a, v_c, v_b, chi_ab, chi_bc, chi_ac
            )
            flag_conv = v_fun_2 < -1 + EPS

            valid_idx = np.where((D1 >= 0) & (D2 >= 0) & flag_conv)[0]

        # Validate three-phase predictions
        if three_phase and len(y_opt_3) > 1:
            D1 = stability_condition(
                y_opt_3[:, 0], y_opt_3[:, 1], v_a, v_c, v_b, chi_ab, chi_bc, chi_ac
            )
            D2 = stability_condition(
                y_opt_3[:, 2], y_opt_3[:, 3], v_a, v_c, v_b, chi_ab, chi_bc, chi_ac
            )
            D3 = stability_condition(
                y_opt_3[:, 4], y_opt_3[:, 5], v_a, v_c, v_b, chi_ab, chi_bc, chi_ac
            )
            flag_conv_3 = v_fun_3 < EPS

            valid_idx_3 = np.where((D1 >= 0) & (D2 >= 0) & (D3 >= 0) & flag_conv_3)[0]

        # Filter valid predictions
        if valid_idx is not None:
            phase_2_pred = y_opt_2[valid_idx]
        else:
            phase_2_pred = y_opt_2
        if valid_idx_3 is not None:
            phase_3_pred = y_opt_3[valid_idx_3]
        else:
            phase_3_pred = y_opt_3

    if plot_fig:
        generate_plots(
            opt_file=opt_file,
            yc_test=yc_test,
            yr_test=yr_test,
            phase_2_pred=phase_2_pred,
            phase_3_pred=phase_3_pred,
            phase_2_pred_ml=phase_2_pred_ml,
            phase_3_pred_ml=phase_3_pred_ml,
            x_test=x_test,
            yc_pred=yc_pred,
            valid_idx=valid_idx,
            valid_idx_3=valid_idx_3,
            PLOT_DIR=PLOT_DIR,
            COLORS=COLORS,
            three_phase=three_phase,
        )
        print("figure done")


def job_array(idx, max_idx):
    # List of pickle files to process
    # Fold 1
    files = [
        "/scratch/gpfs/sj0161/mlphase/pickle/MLPHASEOUTER-1-1-5000-500-0.001-0-1.0-256-base.pickle",
        "/scratch/gpfs/sj0161/mlphase/pickle/MLPHASEOUTER-1-1-5000-500-0.001-0-0.1-256-base.pickle",
        "/scratch/gpfs/sj0161/mlphase/pickle/MLPHASEOUTER-1-1-5000-500-0.001-0-1.0-256-softbase.pickle",
        "/scratch/gpfs/sj0161/mlphase/pickle/MLPHASEOUTER-1-1-5000-500-0.001-0-0.1-256-softbase.pickle",
        "/scratch/gpfs/sj0161/mlphase/pickle/MLPHASEOUTER-1-1-5000-500-0.001-0-0.1-256-softpir.pickle",
        "/scratch/gpfs/sj0161/mlphase/pickle/MLPHASEOUTER-1-1-5000-500-0.001-0-1.0-256-softpir.pickle",
    ]

    files = sorted(files)

    indices = np.arange(208).astype("int")
    combs = itertools.product(indices, files)
    combs = list(combs)

    size = len(combs) // max_idx
    start = idx * size
    end = min((idx + 1) * size, len(combs))

    return combs[start:end]


if __name__ == "__main__":
    idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
    max_idx = int(os.environ["SLURM_ARRAY_TASK_MAX"]) + 1
    combs = job_array(idx, max_idx)

    for comb in combs:
        index, pickle_file = comb
        print(index, pickle_file)
        main(
            pickle_file=pickle_file,
            opt_result_dir=OPT_RESULT_DIR,
            index=index,
            plot_fig=False,
            rerun=True,
        )
