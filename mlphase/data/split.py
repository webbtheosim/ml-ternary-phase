import os
import numpy as np
from timeit import default_timer as timer
from sklearn.model_selection import StratifiedKFold
from mlphase.data import load_data


def split_data(fold, sub_fold, n_folds=5, sample_ratio=0.1, random_seed=42, DATA_DIR="/scratch/gpfs/sj0161/mlphase/data/"):
    t0 = timer()
    # load data
    file = os.path.join(DATA_DIR, "data.pickle")
    x, yc, yr, phase_idx, n_phase, phase_type = load_data(file)

    # outer 5-fold CV based on whole phase diagrams (phase_idx)
    # and phase_type
    uni_p_id = np.unique(phase_idx)

    skf = StratifiedKFold(n_splits=n_folds, random_state=random_seed, shuffle=True)

    for fold_temp, (train_idx, test_idx) in enumerate(skf.split(uni_p_id, phase_type)):

        if fold == fold_temp + 1:
            print(f"# Fold: {fold}")
            uni_p_id_train = uni_p_id[train_idx]
            uni_p_id_test = uni_p_id[test_idx]
            phase_type_train = phase_type[train_idx]

    mask_test = np.isin(phase_idx, uni_p_id_test)
    test_idx = np.where(mask_test)[0]

    x_test = x[test_idx]
    yr_test = yr[test_idx]
    yc_test = yc[test_idx]
    phase_idx_test = phase_idx[test_idx]

    # inner f-fold
    skf2 = StratifiedKFold(n_splits=n_folds, random_state=random_seed, shuffle=True)
    for fold_temp, (train_idx, val_idx) in enumerate(
        skf2.split(uni_p_id_train, phase_type_train)
    ):
        # fold_temp = int(fold_temp)
        # sub_fold = int(sub_fold)
        if sub_fold == fold_temp + 1:
            print(f"# Sub fold: {sub_fold}")
            uni_p_id_train_new = uni_p_id_train[train_idx]
            uni_p_id_val_new = uni_p_id_train[val_idx]

    mask_train = np.isin(phase_idx, uni_p_id_train_new)
    train_idx = np.where(mask_train)[0]

    mask_val = np.isin(phase_idx, uni_p_id_val_new)
    val_idx = np.where(mask_val)[0]

    x_train = x[train_idx]  # input dim 8
    yr_train = yr[train_idx]  # regression label dim 9
    yc_train = yc[train_idx]  # classification label dim 3
    phase_idx_train = phase_idx[train_idx]

    x_val = x[val_idx]
    yr_val = yr[val_idx]
    yc_val = yc[val_idx]
    phase_idx_val = phase_idx[val_idx]

    x_train = np.random.RandomState(0).permutation(x_train)
    x_val = np.random.RandomState(0).permutation(x_val)

    yr_train = np.random.RandomState(0).permutation(yr_train)
    yr_val = np.random.RandomState(0).permutation(yr_val)

    yc_train = np.random.RandomState(0).permutation(yc_train)
    yc_val = np.random.RandomState(0).permutation(yc_val)

    phase_idx_train = np.random.RandomState(0).permutation(phase_idx_train)
    phase_idx_val = np.random.RandomState(0).permutation(phase_idx_val)

    idx_sample = np.random.RandomState(seed=random_seed).choice(
        np.arange(len(x_train)), size=int(sample_ratio * len(x_train)), replace=False
    )

    x_train = x_train[idx_sample]

    yr_train = yr_train[idx_sample]

    yc_train = yc_train[idx_sample]

    phase_idx_train = phase_idx_train[idx_sample]

    t1 = timer()
    print(f"# Data loading in {t1-t0:0.4f} sec ...")
    # print(f"# x train:   {x_train.shape}")
    # print(f"# y_r train: {yr_train.shape}")
    # print(f"# y_c train: {yc_train.shape}")
    # print(f"# x val:     {x_val.shape}")
    # print(f"# y_r val:   {yr_val.shape}")
    # print(f"# y_c val:   {yc_val.shape}")
    # print(f"# x test:    {x_test.shape}")
    # print(f"# y_r test:  {yr_test.shape}")
    # print(f"# y_c test:  {yc_test.shape}")

    return (
        x_train,
        x_val,
        x_test,
        yr_train,
        yr_val,
        yr_test,
        yc_train,
        yc_val,
        yc_test,
        phase_idx_train,
        phase_idx_val,
        phase_idx_test,
    )
