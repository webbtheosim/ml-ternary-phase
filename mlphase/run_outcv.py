#############################################################################
# This script performs the outer loop of five-fold CV for final prediction.
# for hyperparameter tuning.
#
# Models:
# - Base model: ChainLinear ("base")
# - PI model: ChainSoftmax with non-physics-informed losses ("softbase")
# - PI+ model: ChainSoftmax with physics-informed losses ("softpir")
#############################################################################
import os
import glob
import pickle
import random
import itertools
from timeit import default_timer as timer
from sklearn import metrics

import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from mlphase.data import CustomDataset, split_data
from mlphase.model import (
    EarlyStopping,
    ChainLinear,
    ChainSoftmax,
    train_cls_reg,
    test_cls_reg,
    wu_loss,
    pif_loss,
    fill_prob_tensor,
)

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

print(f"# DEVICE: {DEVICE}")

# Define paths for data, models, and pickle files
DATA_DIR = "/scratch/gpfs/sj0161/mlphase/data/"
MODEL_PATH = "/scratch/gpfs/sj0161/mlphase/model/"
PICKLE_INNER_PATH = "/scratch/gpfs/sj0161/mlphase/pickle_inner/"
PICKLE_PATH = "/scratch/gpfs/sj0161/mlphase/pickle/"

class Args:
    def __init__(self):
        self.fold = 1
        self.sub_fold = 1
        self.epoch = 500
        self.batchsize = 10000
        self.learningrate = 0.001
        self.mask = 0
        self.sample_ratio = 0.1
        self.dim = 256
        self.loss = "softbase"
        self.verbose = 1

def main(args):
    if os.path.exists(PICKLE_INNER_PATH):
        batchsize = "*"
        epoch = "*"
        learningrate = "*"
        sample_ratio = "*"
        dim = "*"

        f1_vals_all = []
        r2_vals_all = []

        # Loop through sub-folds to gather results
        for sub_fold in range(1, 6):
            hyper_name = f"MLPHASE-{args.fold}-{sub_fold}-{batchsize}-{epoch}-{learningrate}-{args.mask}-{sample_ratio}-{dim}-{args.loss}"
            pickle_file = os.path.join(PICKLE_INNER_PATH, hyper_name + ".pickle")
            files = sorted(glob.glob(pickle_file))

            f1_vals = np.zeros(len(files))
            r2_vals = np.zeros(len(files))

            # Load F1 and R2 values from pickle files
            for i, file in enumerate(files):
                with open(file, "rb") as handle:
                    f1_val = pickle.load(handle)
                    r2_val = pickle.load(handle)

                f1_vals[i] = f1_val
                r2_vals[i] = r2_val

            f1_vals_all.append(f1_vals)
            r2_vals_all.append(r2_vals)

        # Calculate mean F1 and R2 values
        f1_vals_all = np.array(f1_vals_all).mean(axis=0)
        r2_vals_all = np.array(r2_vals_all).mean(axis=0)

        # Find the best hyperparameters based on R2 and F1 scores
        best_idx = np.argmax(r2_vals_all + f1_vals_all)
        best_file = os.path.basename(files[best_idx])

        print(f"# BEST Hyper: {best_file}")
        print(
            f"# BEST Score: F1 {f1_vals_all[best_idx]:0.4f} R2 {r2_vals_all[best_idx]:0.4f}"
        )

        # Update args with the best hyperparameters
        args.batchsize = int(best_file.split("-")[3])
        args.learningrate = float(best_file.split("-")[5])
        args.dim = int(best_file.split("-")[8])

    # Define the hyperparameter name and other settings
    hyper_name = f"MLPHASEOUTER-{args.fold}-{args.sub_fold}-{args.batchsize}-{args.epoch}-{args.learningrate}-{args.mask}-{args.sample_ratio}-{args.dim}-{args.loss}"
    args.hyper_name = hyper_name
    args.random_seed = 2024

    model_file = os.path.join(MODEL_PATH, f"{hyper_name}.pt")
    pickle_file = os.path.join(PICKLE_PATH, hyper_name + ".pickle")

    print(f"# {hyper_name} started")

    if not os.path.exists(pickle_file):
        print(f"# {hyper_name} started")

        # Split the data into training, validation, and test sets
        data = split_data(
            fold=args.fold,
            sub_fold=args.sub_fold,
            n_folds=5,
            sample_ratio=args.sample_ratio,
            random_seed=args.random_seed,
            DATA_DIR=DATA_DIR,
        )

        (
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
        ) = data

        # Fill probability tensors
        yr_train = fill_prob_tensor(yr_train, np.zeros((yr_train.shape[0], 12)))
        yr_val = fill_prob_tensor(yr_val, np.zeros((yr_val.shape[0], 12)))
        yr_test = fill_prob_tensor(yr_test, np.zeros((yr_test.shape[0], 12)))

        # Create custom datasets and data loaders
        data_train = CustomDataset(x_train, yc_train, yr_train)
        data_val = CustomDataset(x_val, yc_val, yr_val)
        data_test = CustomDataset(x_test, yc_test, yr_test)

        train_loader = DataLoader(data_train, batch_size=args.batchsize, shuffle=True)
        val_loader = DataLoader(data_val, batch_size=args.batchsize, shuffle=False)
        test_loader = DataLoader(data_test, batch_size=args.batchsize, shuffle=False)

        # Initialize the model based on the loss type
        if "soft" in args.loss:
            model = ChainSoftmax(DEVICE, mask=args.mask, dim=args.dim).to(DEVICE)
        else:
            model = ChainLinear(DEVICE, mask=args.mask, dim=args.dim).to(DEVICE)

        cls_criterion = nn.CrossEntropyLoss()

        # Set the regression criterion based on the loss type
        loss_config = {
            "base": (wu_loss, 1),
            "softbase": (wu_loss, 1),
            "softpir": (pif_loss, 4),
        }

        reg_criterion, reg_criterion_size = loss_config.get(args.loss, (wu_loss, 1))

        optimizer = optim.Adam(model.parameters(), lr=args.learningrate)
        scheduler = ReduceLROnPlateau(optimizer, "min", patience=100)

        cls_train_losses = []
        reg_train_losses = []

        cls_val_losses = []
        reg_val_losses = []

        t0 = timer()
        early_stopping = EarlyStopping(patience=500, verbose=args.verbose)

        for epoch in range(args.epoch):
            cls_train_loss, reg_train_loss = train_cls_reg(
                model,
                train_loader,
                optimizer,
                cls_criterion,
                reg_criterion,
                reg_criterion_size,
                DEVICE,
            )
            cls_val_loss, reg_val_loss = test_cls_reg(
                model,
                val_loader,
                cls_criterion,
                reg_criterion,
                reg_criterion_size,
                DEVICE,
            )

            cls_train_losses.append(cls_train_loss)
            reg_train_losses.append(reg_train_loss)

            cls_val_losses.append(cls_val_loss)
            reg_val_losses.append(reg_val_loss)

            cls_train_loss_str = f"{cls_train_loss.item():0.5f}"
            reg_train_loss_str = " ".join(f"{loss:.5f}" for loss in reg_train_loss)
            cls_val_loss_str = f"{cls_val_loss.item():0.5f}"
            reg_val_loss_str = " ".join(f"{loss:.5f}" for loss in reg_val_loss)

            if args.verbose == 1:
                print(
                    f"Epoch {epoch + 1:>3}/{args.epoch}, loss: {cls_train_loss_str} | {reg_train_loss_str} val loss: {cls_val_loss_str} | {reg_val_loss_str}"
                )

            early_stopping(
                torch.sum(cls_val_losses[-1]) + torch.sum(reg_val_losses[-1]),
                model=model,
                path=MODEL_PATH,
                name=hyper_name,
                epoch=epoch,
            )

            scheduler.step(
                torch.sum(cls_val_losses[-1]) + torch.sum(reg_val_losses[-1])
            )

            if early_stopping.early_stop:
                print("Early stopping")
                break

        t1 = timer()

        model.load_state_dict(torch.load(model_file, map_location=torch.device("cpu")))

        cls_val_loss, reg_val_loss = test_cls_reg(
            model, val_loader, cls_criterion, reg_criterion, reg_criterion_size, DEVICE
        )
        print(f"# Val loss: {torch.sum(cls_val_loss) + torch.sum(reg_val_loss):.4f}")

        yc_pred_val, yr_pred_val = model(torch.tensor(x_val, dtype=torch.float32))
        yc_pred_val = yc_pred_val.detach().cpu().numpy()
        yr_pred_val = yr_pred_val.detach().cpu().numpy()

        cls_test_loss, reg_test_loss = test_cls_reg(
            model, test_loader, cls_criterion, reg_criterion, reg_criterion_size, DEVICE
        )
        print(f"# Test loss: {torch.sum(cls_test_loss) + torch.sum(reg_test_loss):.4f}")

        yc_pred, yr_pred = model(torch.tensor(x_test, dtype=torch.float32))
        yc_pred = yc_pred.detach().cpu().numpy()
        yr_pred = yr_pred.detach().cpu().numpy()

        f1_val = metrics.f1_score(
            yc_val.argmax(axis=1), yc_pred_val.argmax(axis=1), average="micro"
        )
        f1_test = metrics.f1_score(
            yc_test.argmax(axis=1), yc_pred.argmax(axis=1), average="micro"
        )

        # 0, 1, 2 are phi_a^alpha, phi_b^alpha, phi_c^alpha (c depends on a and b, not considered)
        # 3, 4, 5 are phi_a^beta, phi_b^beta, phi_c^beta
        # 6, 7, 8 are phi_a^gamma, phi_b^gamma, phi_c^gamma
        # 9, 10, 11 are w^alpha, w^beta, w^gamma

        valid_idx = [0, 1, 3, 4, 6, 7, 9, 10, 11]

        r2_val = metrics.r2_score(
            yr_val[:, valid_idx].ravel(), yr_pred_val[:, valid_idx].ravel()
        )
        r2_test = metrics.r2_score(
            yr_test[:, valid_idx].ravel(), yr_pred[:, valid_idx].ravel()
        )

        # Print final scores
        print("#" * 26)
        print(f"# F1 val:  {f1_val:0.4f}")
        print(f"# F1 test: {f1_test:0.4f}")
        print(f"# R2 val:  {r2_val:0.4f}")
        print(f"# R2 test: {r2_test:0.4f}")

        with open(pickle_file, "wb") as handle:
            pickle.dump(yc_test, handle)
            pickle.dump(yc_pred, handle)
            pickle.dump(yr_test, handle)
            pickle.dump(yr_pred, handle)
            pickle.dump(x_test, handle)
            pickle.dump(phase_idx_test, handle)
            pickle.dump(yc_val, handle)
            pickle.dump(yc_pred_val, handle)
            pickle.dump(yr_val, handle)
            pickle.dump(yr_pred_val, handle)
            pickle.dump(t1 - t0, handle)


def job_array(idx, max_idx):
    """Function to generate job array for parameter combinations"""
    folds = [1, 2, 3, 4, 5]
    masks = [0, 1]
    losses = ["base", "softbase", "softpir"]

    sample_ratios = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    combs = itertools.product(folds, masks, losses, sample_ratios)
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
        fold, mask, loss, sample_ratio = comb

        args = Args()
        args.fold = fold
        args.mask = mask
        args.loss = loss
        args.sample_ratio = sample_ratio

        main(args)
