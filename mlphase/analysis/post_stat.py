import os
import pickle
import numpy as np
import itertools
from tqdm import tqdm
import pandas as pd

# Output vector indices
# 0, 1, 2 are phi_a^alpha, phi_b^alpha, phi_c^alpha (c depends on a and b, not considered)
# 3, 4, 5 are phi_a^beta, phi_b^beta, phi_c^beta
# 6, 7, 8 are phi_a^gamma, phi_b^gamma, phi_c^gamma
# 9, 10, 11 are w^alpha, w^beta, w^gamma

THRESHOLD_DIS = 1e-9

# List of file paths
FILES = [
    "/scratch/gpfs/sj0161/mlphase/pickle/MLPHASEOUTER-1-1-5000-500-0.001-0-1.0-256-base.pickle",
    "/scratch/gpfs/sj0161/mlphase/pickle/MLPHASEOUTER-1-1-5000-500-0.001-0-0.1-256-base.pickle",
    "/scratch/gpfs/sj0161/mlphase/pickle/MLPHASEOUTER-1-1-5000-500-0.001-0-1.0-256-softbase.pickle",
    "/scratch/gpfs/sj0161/mlphase/pickle/MLPHASEOUTER-1-1-5000-500-0.001-0-0.1-256-softbase.pickle",
    "/scratch/gpfs/sj0161/mlphase/pickle/MLPHASEOUTER-1-1-5000-500-0.001-0-0.1-256-softpir.pickle",
    "/scratch/gpfs/sj0161/mlphase/pickle/MLPHASEOUTER-1-1-5000-500-0.001-0-1.0-256-softpir.pickle",
]

# Load the phase index test data from the first file
with open(FILES[0], "rb") as handle:
    for _ in range(5):
        _ = pickle.load(handle)
    phase_idx_test = pickle.load(handle)

# Get unique phase indices
PHASE_IDX_UNIQUE = np.unique(phase_idx_test).astype(int)

# Directory for optimization results
OPT_RESULT_DIR = "/scratch/gpfs/sj0161/mlphase/opt_pickle/"


def check_file_exist(idx):
    """
    Check if optimization result files exist for a given phase index.
    """
    idx_unique = PHASE_IDX_UNIQUE[idx]

    for file in FILES:
        base_name = os.path.basename(file)
        opt_file_name = base_name.replace("MLPHASEOUTER", "OPT").replace(
            ".pickle", f"-{int(idx_unique)}.pickle"
        )
        opt_file = os.path.join(OPT_RESULT_DIR, opt_file_name)

        if not os.path.exists(opt_file):
            return False

    return True


def get_opt_file_names(idx):
    """
    Generate a list of optimization file paths for a given phase index.
    """
    idx_unique = PHASE_IDX_UNIQUE[idx]
    opt_files = []

    for file in FILES:
        base_name = os.path.basename(file)
        opt_file_name = base_name.replace("MLPHASEOUTER", "OPT").replace(
            ".pickle", f"-{int(idx_unique)}.pickle"
        )
        opt_file = os.path.join(OPT_RESULT_DIR, opt_file_name)
        opt_files.append(opt_file)

    return opt_files


def load_ml_pickle(ml_file, idx):
    """
    Load data from a pickle file and process it based on a given index.
    """
    with open(ml_file, "rb") as handle:
        yc_test_all = pickle.load(handle)
        yc_pred_all = pickle.load(handle)
        yr_test_all = pickle.load(handle)
        yr_pred_all = pickle.load(handle)
        x_test_all = pickle.load(handle)
        phase_idx_test = pickle.load(handle)

    # Get unique phase indices and the specific index for this run
    phase_idx_unique = np.unique(phase_idx_test).astype(int)
    idx_unique = phase_idx_unique[idx]

    # Get the row indices where the phase index matches the unique index
    row_idx = np.where(phase_idx_test == idx_unique)[0]
    x_test_out = x_test_all[row_idx[0]]

    # Get the predictions and test data for yc and yr
    yc_pred = yc_pred_all[row_idx].argmax(axis=1)
    yr_pred = yr_pred_all[row_idx]
    yr_test = yr_test_all[row_idx]
    yc_test = yc_test_all[row_idx].argmax(axis=1)

    # Filter predictions and test data based on yc_pred values

    # 0, 1, 2 are phi_a^alpha, phi_b^alpha, phi_c^alpha (c depends on a and b, not considered)
    # 3, 4, 5 are phi_a^beta, phi_b^beta, phi_c^beta
    # 6, 7, 8 are phi_a^gamma, phi_b^gamma, phi_c^gamma
    # 9, 10, 11 are w^alpha, w^beta, w^gamma
    
    idx_2 = np.where(yc_pred == 1)[0]
    yr_pred_2 = yr_pred[idx_2][:, [0, 1, 3, 4]]
    yr_test_2 = yr_test[idx_2][:, [0, 1, 3, 4]]

    idx_3 = np.where(yc_pred == 2)[0]
    yr_pred_3 = yr_pred[idx_3][:, [0, 1, 3, 4, 6, 7]]
    yr_test_3 = yr_test[idx_3][:, [0, 1, 3, 4, 6, 7]]

    return x_test_out, yr_pred_2, yr_test_2, yr_pred_3, yr_test_3, yc_test


def load_opt_pickle(opt_file, ml_file, idx, THRESHOLD=np.inf):
    """
    Load and process optimization and machine learning results from pickle files.
    """
    time_out = None
    file_2 = file_3 = None
    x_test_out_2 = yr_true_2_out = yr_pred_2_ml_out = y_opt_2_new = None
    x_test_out_3 = yr_true_3_out = yr_pred_3_ml_out = y_opt_3_new = None

    with open(opt_file, "rb") as handle:
        y_opt_2_raw, v_fun_2_raw, y_opt_3_raw, v_fun_3_raw, t_raw = [
            pickle.load(handle) for _ in range(5)
        ]

        if len(y_opt_2_raw) > 1:
            file_2 = opt_file
            good_2 = np.where(v_fun_2_raw < -1 + THRESHOLD)[0]
            y_opt_2_new = y_opt_2_raw[good_2]

            if len(y_opt_2_new) > 1:
                time_out = t_raw / (len(y_opt_2_raw) + len(y_opt_3_raw))
                x_test_out_2, yr_pred_2, yr_test_2, yr_pred_3, yr_test_3, yc_test = (
                    load_ml_pickle(ml_file, idx)
                )
                yr_pred_2_ml = yr_pred_2[good_2]
                yr_true_2_ml = yr_test_2[good_2]

                idx_dist_2 = filter_by_composition(y_opt_2_new)

                yr_true_2_out = yr_true_2_ml[idx_dist_2]
                yr_pred_2_ml_out = yr_pred_2_ml[idx_dist_2]
                y_opt_2_new = y_opt_2_new[idx_dist_2]

                good_3 = np.where(v_fun_3_raw < THRESHOLD)[0]
                y_opt_3_new = y_opt_3_raw[good_3]
                if yc_test.max() == 2 and len(y_opt_3_new) > 1:
                    file_3 = opt_file
                    yr_pred_3_ml = yr_pred_3[good_3]
                    yr_true_3_ml = yr_test_3[good_3]

                    idx_dist_3 = filter_by_composition(y_opt_3_new)

                    yr_true_3_out = yr_true_3_ml[idx_dist_3]
                    yr_pred_3_ml_out = yr_pred_3_ml[idx_dist_3]
                    y_opt_3_new = y_opt_3_new[idx_dist_3]
                    x_test_out_3 = x_test_out_2

    return (
        time_out,
        file_2,
        file_3,
        x_test_out_2,
        yr_true_2_out,
        yr_pred_2_ml_out,
        y_opt_2_new,
        x_test_out_3,
        yr_true_3_out,
        yr_pred_3_ml_out,
        y_opt_3_new,
    )


def filter_by_distance(y_opt, threshold):
    """
    Filter optimization results based on distance for two-phase case.
    """
    a = y_opt[:, [0, 1]]
    b = y_opt[:, [2, 3]]
    distances = np.linalg.norm(a - b, axis=1)
    return np.where(distances >= threshold)[0]


def filter_by_distance_3_phase(y_opt, threshold):
    """
    Filter optimization results based on distance for three-phase case.
    """
    a = y_opt[:, [0, 1]]
    b = y_opt[:, [2, 3]]
    c = y_opt[:, [4, 5]]
    dis_1 = np.linalg.norm(a - b, axis=1)
    dis_2 = np.linalg.norm(b - c, axis=1)
    dis_3 = np.linalg.norm(a - c, axis=1)
    return np.where((dis_1 >= threshold) & (dis_2 >= threshold) & (dis_3 >= threshold))[
        0
    ]


def filter_by_composition(y_opt):
    """
    Composition >= 0 and <= 1.
    """
    vmin = y_opt.min(axis=1)
    vmax = y_opt.min(axis=1)

    return np.where((vmin >= 0) & (vmax <= 1))[0]


def post_performance_raw(idx=207, THRESHOLD=np.inf):
    """
    Process and load optimization performance results for a given index.
    """
    if not check_file_exist(idx):
        print(f"# {idx} does not exist")
        return None
    else:
        opt_files = get_opt_file_names(idx)

        results = []
        for i in range(len(opt_files)):
            result = load_opt_pickle(
                opt_file=opt_files[i], ml_file=FILES[i], idx=idx, THRESHOLD=THRESHOLD
            )
            results.append(result)

    return results


def post_performance_load(post_result_pickle, rerun=False, THRESHOLD=np.inf):
    """
    Load or generate and save post-performance results.
    """
    if os.path.exists(post_result_pickle) and not rerun:
        with open(post_result_pickle, "rb") as handle:
            results = pickle.load(handle)
    else:
        results = []
        for idx in tqdm(range(208)):
            result = post_performance_raw(idx=idx, THRESHOLD=THRESHOLD)
            results.append(result)

        with open(post_result_pickle, "wb") as handle:
            pickle.dump(results, handle)

    print("# post-performance result loaded")
    return results


def stability_condition(phi_s, phi_p, vs, vc, vp, chi_ps, chi_pc, chi_sc):
    return (1 / (vp * phi_p) + 1 / (vc * (1 - phi_s - phi_p)) - 2 * chi_pc) * (
        1 / (vs * phi_s) + 1 / (vc * (1 - phi_s - phi_p)) - 2 * chi_sc
    ) - (1 / (vc * (1 - phi_s - phi_p)) + chi_ps - chi_pc - chi_sc) ** 2


def dis3(y_pred, x_test):
    """
    Evaluate the stability condition for three sets of volume fractions.
    """
    vs, vc, vp, chi_sc, chi_ps, chi_pc, _, _ = x_test

    out = 0

    if (
        stability_condition(y_pred[0], y_pred[1], vs, vc, vp, chi_ps, chi_pc, chi_sc)
        > 0
    ):
        out += 1

    if (
        stability_condition(y_pred[2], y_pred[3], vs, vc, vp, chi_ps, chi_pc, chi_sc)
        > 0
    ):
        out += 1

    if (
        stability_condition(y_pred[4], y_pred[5], vs, vc, vp, chi_ps, chi_pc, chi_sc)
        > 0
    ):
        out += 1

    return out


def dis2(y_pred, x_test):
    """
    Evaluate the stability condition for two sets of volume fractions.
    """
    vs, vc, vp, chi_sc, chi_ps, chi_pc, _, _ = x_test

    out = 0

    if (
        stability_condition(y_pred[0], y_pred[1], vs, vc, vp, chi_ps, chi_pc, chi_sc)
        > 0
    ):
        out += 1

    if (
        stability_condition(y_pred[2], y_pred[3], vs, vc, vp, chi_ps, chi_pc, chi_sc)
        > 0
    ):
        out += 1

    return out


def find_same_index(results):
    index_2 = []
    index_3 = []

    for i in [0, 2, 4, 1, 3, 5]:
        k_2 = []
        k_3 = []

        for k, result in enumerate(results):
            if result is not None:
                (
                    time_out,
                    file_2,
                    file_3,
                    x_test_out_2,
                    yr_true_2_out,
                    yr_pred_2_ml_out,
                    y_opt_2_new,
                    x_test_out_3,
                    yr_true_3_out,
                    yr_pred_3_ml_out,
                    y_opt_3_new,
                ) = result[i]

                if (
                    x_test_out_2 is not None
                    and y_opt_2_new.min(axis=0).max() <= 1
                    and y_opt_2_new.max(axis=0).min() >= 0
                ):
                    k_2.append(int(file_2.split("-")[-1].split(".")[0]))

                if (
                    x_test_out_3 is not None
                    and y_opt_3_new.min(axis=0).max() <= 1
                    and y_opt_3_new.max(axis=0).min() >= 0
                ):
                    k_3.append(int(file_2.split("-")[-1].split(".")[0]))

        index_2.append(k_2)
        index_3.append(k_3)

    intersection = set(index_2[0])

    for sublist in index_2[1:]:
        intersection = intersection.intersection(sublist)

    index_2 = list(intersection)

    intersection = set(index_3[0])

    for sublist in index_3[1:]:
        intersection = intersection.intersection(sublist)

    index_3 = list(intersection)

    return index_2, index_3


def post_opt_mae(CSV_DIR, results, results_conv, index_2, index_3, rerun=False):
    """
    Process optimization results and calculate mean absolute error.
    """
    name = [
        "base (1.0)",
        "base (0.1)",
        "pi (1.0)",
        "pi (0.1)",
        "pi+ (1.0)",
        "pi+ (0.1)",
    ]

    csv_file = os.path.join(CSV_DIR, "opt_mae.csv")

    if os.path.exists(csv_file) and not rerun:
        df = pd.read_csv(csv_file, index_col=None)
    else:

        data = []

        for i in [0, 2, 4, 1, 3, 5]:
            yr_true_2 = []
            yr_pred_2_ml = []

            yr_true_2_conv = []
            yr_pred_2_conv = []

            yr_true_3 = []
            yr_pred_3_ml = []

            yr_true_3_conv = []
            yr_pred_3_conv = []

            count_2 = 0
            count_3 = 0

            for k, result in tqdm(enumerate(results), total=len(results)):
                if result is not None:
                    (
                        _,
                        file_2,
                        _,
                        _,
                        yr_true_2_out,
                        yr_pred_2_ml_out,
                        y_opt_2_new,
                        _,
                        yr_true_3_out,
                        yr_pred_3_ml_out,
                        y_opt_3_new,
                    ) = result[i]

                    (
                        _,
                        _,
                        _,
                        _,
                        yr_true_2_out_conv,
                        _,
                        y_opt_2_new_conv,
                        _,
                        yr_true_3_out_conv,
                        _,
                        y_opt_3_new_conv,
                    ) = results_conv[k][i]

                    idx = int(file_2.split("-")[-1].split(".")[0])

                    if idx in index_2:
                        yr_true_2.append(yr_true_2_out)
                        yr_pred_2_ml.append(yr_pred_2_ml_out)

                        yr_true_2_conv.append(yr_true_2_out_conv)
                        yr_pred_2_conv.append(y_opt_2_new_conv)
                        count_2 += 1

                    if idx in index_3:

                        yr_true_3.append(yr_true_3_out)
                        yr_pred_3_ml.append(yr_pred_3_ml_out)

                        yr_true_3_conv.append(yr_true_3_out_conv)
                        yr_pred_3_conv.append(y_opt_3_new_conv)
                        count_3 += 1

            yr_true_2 = np.concatenate(yr_true_2)
            yr_pred_2_ml = np.concatenate(yr_pred_2_ml)

            yr_true_2_conv = np.concatenate(yr_true_2_conv)
            yr_pred_2_conv = np.concatenate(yr_pred_2_conv)

            yr_true_3 = np.concatenate(yr_true_3)
            yr_pred_3_ml = np.concatenate(yr_pred_3_ml)

            yr_true_3_conv = np.concatenate(yr_true_3_conv)
            yr_pred_3_conv = np.concatenate(yr_pred_3_conv)

            mae_ml_2 = np.abs(yr_true_2 - yr_pred_2_ml).mean(axis=1)
            mae_conv_2 = np.abs(yr_true_2_conv - yr_pred_2_conv).mean(axis=1)

            mae_ml_3 = np.abs(yr_true_3 - yr_pred_3_ml).mean(axis=1)
            mae_conv_3 = np.abs(yr_true_3_conv - yr_pred_3_conv).mean(axis=1)

            data.append(
                {
                    "name": name[i],
                    "phase": "2 phase",
                    "ml/opt": "ml",
                    "mean": mae_ml_2.mean(),
                    "std_error": mae_ml_2.std() / len(mae_ml_2) ** 0.5,
                }
            )

            data.append(
                {
                    "name": name[i],
                    "phase": "2 phase",
                    "ml/opt": "opt",
                    "mean": mae_conv_2.mean(),
                    "std_error": mae_conv_2.std() / len(mae_conv_2) ** 0.5,
                }
            )

            data.append(
                {
                    "name": name[i],
                    "phase": "3 phase",
                    "ml/opt": "ml",
                    "mean": mae_ml_3.mean(),
                    "std_error": mae_ml_3.std() / len(mae_ml_3) ** 0.5,
                }
            )

            data.append(
                {
                    "name": name[i],
                    "phase": "3 phase",
                    "ml/opt": "opt",
                    "mean": mae_conv_3.mean(),
                    "std_error": mae_conv_3.std() / len(mae_conv_3) ** 0.5,
                }
            )

        df = pd.DataFrame(data)

        df.to_csv(csv_file, index=None)
    return df


def post_opt_success(CSV_DIR, results, index_2, index_3, rerun=False):
    """
    Process optimization results and calculate success rate.
    """
    name = [
        "base (1.0)",
        "base (0.1)",
        "pi (1.0)",
        "pi (0.1)",
        "pi+ (1.0)",
        "pi+ (0.1)",
    ]

    csv_file = os.path.join(CSV_DIR, "opt_success.csv")

    if os.path.exists(csv_file) and not rerun:
        df = pd.read_csv(csv_file, index_col=None)
    else:
        performance = []

        for i in [0, 2, 4, 1, 3, 5]:
            time = []
            success_2_opt = []
            success_3_opt = []

            for k, result in enumerate(results):
                if result is not None:
                    (
                        time_out,
                        file_2,
                        _,
                        x_test_out_2,
                        _,
                        _,
                        y_opt_2_new,
                        x_test_out_3,
                        _,
                        _,
                        y_opt_3_new,
                    ) = result[i]

                    idx = int(file_2.split("-")[-1].split(".")[0])

                    if idx in index_2:
                        success_2 = []
                        for j in range(len(y_opt_2_new)):
                            success_2.append(dis2(y_opt_2_new[j], x_test_out_2) / 2)
                        success_2_opt.extend(success_2)

                        time.append(time_out)

                    if idx in index_3:
                        success_3 = []
                        for j in range(len(y_opt_3_new)):
                            success_3.append(dis3(y_opt_3_new[j], x_test_out_3) / 3)
                        success_3_opt.extend(success_3)

            time = np.array(time)
            success_2_opt = np.array(success_2_opt)
            success_3_opt = np.array(success_3_opt)

            performance.append(
                {
                    "name": name[i],
                    "time_mean": time.mean(),
                    "time_se": time.std() / len(time) ** 0.5,
                    "2_phase_success_mean": success_2_opt.mean(),
                    "2_phase_success_se": success_2_opt.std()
                    / len(success_2_opt) ** 0.5,
                    "3_phase_success_mean": success_3_opt.mean(),
                    "3_phase_success_se": success_3_opt.std()
                    / len(success_3_opt) ** 0.5,
                }
            )
        df = pd.DataFrame(performance)
        df.to_csv(csv_file, index=None)

    return df
