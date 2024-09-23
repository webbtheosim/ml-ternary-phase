import os
import pickle
import glob
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, r2_score, mean_absolute_error

# Output vector indices
# 0, 1, 2 are phi_a^alpha, phi_b^alpha, phi_c^alpha (c depends on a and b, not considered)
# 3, 4, 5 are phi_a^beta, phi_b^beta, phi_c^beta
# 6, 7, 8 are phi_a^gamma, phi_b^gamma, phi_c^gamma
# 9, 10, 11 are w^alpha, w^beta, w^gamma

def load_metrics(
    res_dir, train_res_dir, reload=True, losses=["base", "softbase", "softpir"]
):
    """
    Load test metrics from result files.
    """
    res_pkl = os.path.join(res_dir, "result_test_metrics.pickle")

    if os.path.exists(res_pkl) and not reload:
        with open(res_pkl, "rb") as handle:
            f1_res = pickle.load(handle)
            r2_res = pickle.load(handle)
            mae_res = pickle.load(handle)
    else:
        r2_res, f1_res, mae_res = [], [], []

        for loss in losses:
            for mask in [0, 1]:
                r2_scores, f1_scores, mae_scores = [], [], []

                for ratio in tqdm(
                    [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                    total=12,
                ):
                    try:
                        train_pkl = os.path.join(
                            train_res_dir,
                            f"MLPHASEOUTER-*-1-*-*-*-{mask}-{ratio}-*-{loss}.pickle",
                        )

                        files = sorted(glob.glob(train_pkl))
                        r2_tests, f1_tests, mae_tests = [], [], []

                        for file in files:
                            with open(file, "rb") as handle:
                                yc_test = pickle.load(handle)
                                yc_pred = pickle.load(handle)
                                yr_test = pickle.load(handle)
                                yr_pred = pickle.load(handle)

                            # 0, 1, 2 are phi_a^alpha, phi_b^alpha, phi_c^alpha (c depends on a and b, not considered)
                            # 3, 4, 5 are phi_a^beta, phi_b^beta, phi_c^beta
                            # 6, 7, 8 are phi_a^gamma, phi_b^gamma, phi_c^gamma
                            # 9, 10, 11 are w^alpha, w^beta, w^gamma
                            
                            idx = [0, 1, 3, 4, 6, 7, 9, 10, 11]

                            f1_test = f1_score(
                                yc_test.argmax(axis=1),
                                yc_pred.argmax(axis=1),
                                average="micro",
                            )
                            r2_test = r2_score(
                                yr_test[:, idx].ravel(), yr_pred[:, idx].ravel()
                            )
                            mae_test = mean_absolute_error(
                                yr_test[:, idx].ravel(), yr_pred[:, idx].ravel()
                            )

                            r2_tests.append(r2_test)
                            f1_tests.append(f1_test)
                            mae_tests.append(mae_test)
                            
                    except Exception as e:
                        print(f"Error: {e}")
                        continue

                    r2_scores.append([np.mean(r2_tests), np.std(r2_tests)])
                    f1_scores.append([np.mean(f1_tests), np.std(f1_tests)])
                    mae_scores.append([np.mean(mae_tests), np.std(mae_tests)])

                r2_res.append(np.array(r2_scores))
                f1_res.append(np.array(f1_scores))
                mae_res.append(np.array(mae_scores))

        with open(res_pkl, "wb") as handle:
            pickle.dump(f1_res, handle)
            pickle.dump(r2_res, handle)
            pickle.dump(mae_res, handle)

    return np.array(f1_res), np.array(r2_res), np.array(mae_res)


def load_test_metrics_by_phase(
    res_dir, train_res_dir, reload=True, losses=["base", "softbase", "softpir"]
):
    """
    Load test metrics from result files.
    """
    res_pkl = os.path.join(res_dir, "result_test_metrics_by_phase.pickle")

    if os.path.exists(res_pkl) and not reload:
        with open(res_pkl, "rb") as handle:
            f1_res = pickle.load(handle)
            r2_res = pickle.load(handle)
            mae_res = pickle.load(handle)
    else:
        r2_res, f1_res, mae_res = [], [], []

        for loss in losses:
            for mask in [0]:
                r2_scores_all, f1_scores_all, mae_scores_all = [], [], []
                for phase in [0, 1, 2]:
                    r2_scores, f1_scores, mae_scores = [], [], []
                    for ratio in [1.0]:

                        train_pkl = os.path.join(
                            train_res_dir,
                            f"MLPHASEOUTER-*-1-*-*-*-{mask}-{ratio}-*-{loss}.pickle",
                        )
                        print(train_pkl)
                        files = sorted(glob.glob(train_pkl))
                        r2_tests, f1_tests, mae_tests = [], [], []

                        for file in files:
                            with open(file, "rb") as handle:
                                yc_test = pickle.load(handle)
                                yc_pred = pickle.load(handle)
                                yr_test = pickle.load(handle)
                                yr_pred = pickle.load(handle)

                            # 0, 1, 2 are phi_a^alpha, phi_b^alpha, phi_c^alpha (c depends on a and b, not considered)
                            # 3, 4, 5 are phi_a^beta, phi_b^beta, phi_c^beta
                            # 6, 7, 8 are phi_a^gamma, phi_b^gamma, phi_c^gamma
                            # 9, 10, 11 are w^alpha, w^beta, w^gamma
                            
                            idx = [0, 1, 3, 4, 6, 7, 9, 10, 11]

                            phase_idx = np.where(yc_test.argmax(axis=1) == phase)[0]

                            f1_test = f1_score(
                                yc_test[phase_idx].argmax(axis=1),
                                yc_pred[phase_idx].argmax(axis=1),
                                average="micro",
                            )
                            r2_test = r2_score(
                                yr_test[phase_idx, :][:, idx].ravel(),
                                yr_pred[phase_idx, :][:, idx].ravel(),
                            )
                            mae_test = mean_absolute_error(
                                yr_test[phase_idx, :][:, idx].ravel(),
                                yr_pred[phase_idx, :][:, idx].ravel(),
                            )

                            r2_tests.append(r2_test)
                            f1_tests.append(f1_test)
                            mae_tests.append(mae_test)

                        r2_scores.append([np.mean(r2_tests), np.std(r2_tests)])
                        f1_scores.append([np.mean(f1_tests), np.std(f1_tests)])
                        mae_scores.append([np.mean(mae_tests), np.std(mae_tests)])
                    r2_scores_all.append(r2_scores)
                    f1_scores_all.append(f1_scores)
                    mae_scores_all.append(mae_scores)

                r2_res.append(np.array(r2_scores_all))
                f1_res.append(np.array(f1_scores_all))
                mae_res.append(np.array(mae_scores_all))

        with open(res_pkl, "wb") as handle:
            pickle.dump(f1_res, handle)
            pickle.dump(r2_res, handle)
            pickle.dump(mae_res, handle)

    return np.array(f1_res), np.array(r2_res), np.array(mae_res)
