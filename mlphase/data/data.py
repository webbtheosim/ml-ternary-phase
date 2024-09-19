import os
import torch
import pickle
import numpy as np

from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, x, yc, yr):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.yc = torch.tensor(yc, dtype=torch.float32)
        self.yr = torch.tensor(yr, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.yc[idx], self.yr[idx]


def load_data(file):
    file_clean = file.replace("data.pickle", "data_clean.pickle")

    if os.path.exists(file_clean):
        with open(file_clean, "rb") as handle:
            (x, y, y_cp, idx, n_phase, diag_phase) = pickle.load(handle)

    else:
        with open(file, "rb") as handle:
            data = pickle.load(handle)

        keys = [
            "vs",
            "vc",
            "vp",
            "chi_sc",
            "chi_ps",
            "chi_pc",
            "phi_s",
            "phi_p",
            "label0",
            "label1",
            "label2",
            "phi_s1",
            "phi_p1",
            "phi_s2",
            "phi_p2",
            "phi_s3",
            "phi_p3",
            "w1",
            "w2",
            "w3",
            "idx",
        ]
        keys = np.array(keys)

        x = data[:, :8]
        y = data[:, 8:11]
        y_cp = data[:, [11, 12, 17, 13, 14, 18, 15, 16, 19]]
        idx = data[:, -1]
        n_phase = data[:, 8:11].argmax(axis=1) + 1

        diag_phase = []
        for u in np.unique(idx):
            rows = np.where(idx == u)[0]
            diag_phase.append(n_phase[rows].max())

        diag_phase = np.array(diag_phase)

        with open(file_clean, "wb") as handle:
            pickle.dump((x, y, y_cp, idx, n_phase, diag_phase), handle)

    print(f"# Input shape: {x.shape}")
    print(f"# Class shape: {y.shape}")
    print(f"# Comp  shape: {y_cp.shape}")

    return x, y, y_cp, idx, n_phase, diag_phase
