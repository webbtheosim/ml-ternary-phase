#############################################################################
# This script performs the inner loop of five-fold cross-validation (CV)
# for hyperparameter tuning.
#
# Models:
# - Base model: ChainLinear ("base")
# - PI model: ChainSoftmax with non-physics-informed losses ("softbase")
# - PI+ model: ChainSoftmax with physics-informed losses ("softpir")
#############################################################################

import os
import pickle
import random
import itertools
from sklearn import metrics

import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader

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

# Define directories for data, models, and pickle files
DATA_DIR = "/scratch/gpfs/sj0161/mlphase/data/"
MODEL_PATH = "/scratch/gpfs/sj0161/mlphase/model_inner/"
PICKLE_INNER_PATH = "/scratch/gpfs/sj0161/mlphase/pickle_inner/"

class Args:
    def __init__(self):
        self.fold = 1
        self.sub_fold = 1
        self.epoch = 200
        self.batchsize = 10000
        self.learningrate = 0.001
        self.mask = 1
        self.sample_ratio = 0.1
        self.dim = 256
        self.loss = "softbase"
        self.verbose = 1

def main(args):
    # Create a unique name for the hyperparameter combination
    hyper_name = f"MLPHASE-{args.fold}-{args.sub_fold}-{args.batchsize}-{args.epoch}-{args.learningrate}-{args.mask}-{args.sample_ratio}-{args.dim}-{args.loss}"
    args.hyper_name = hyper_name

    # Define file paths for model and pickle files
    model_file = os.path.join(MODEL_PATH, f"{hyper_name}.pt")
    pickle_file = os.path.join(PICKLE_INNER_PATH, hyper_name + ".pickle")

    print(f"# {hyper_name} started")

    # Check if the pickle file already exists
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

        # Fill probability tensors for regression targets
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

        # Define classification criterion
        cls_criterion = nn.CrossEntropyLoss()

        # Define regression criterion and its size based on the loss type
        loss_config = {
            "base": (wu_loss, 1),
            "softbase": (wu_loss, 1),
            "softpir": (pif_loss, 4),
        }

        reg_criterion, reg_criterion_size = loss_config.get(args.loss, (wu_loss, 1))

        optimizer = optim.Adam(model.parameters(), lr=args.learningrate)

        cls_train_losses = []
        reg_train_losses = []

        cls_val_losses = []
        reg_val_losses = []

        early_stopping = EarlyStopping(patience=200, verbose=1)

        # Training loop
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

            if args.verbose == 1:
                print(
                    f"Epoch {epoch+1:>3}/{args.epoch}, loss: "
                    f"{cls_train_loss.item():0.3f} | {reg_train_loss[0]:.3f} "
                    f"val loss: {cls_val_loss.item():0.3f} | {reg_val_loss[0]:.3f}"
                )

            early_stopping(
                torch.sum(cls_val_loss) + torch.sum(reg_val_loss),
                model=model,
                path=MODEL_PATH,
                name=hyper_name,
                epoch=epoch,
            )
            if early_stopping.early_stop:
                print("Early stopping")
                break

        # Load the best model
        model.load_state_dict(torch.load(model_file, map_location=torch.device("cpu")))

        # Make predictions on validation and test sets
        yc_pred_val, yr_pred_val = model(torch.tensor(x_val, dtype=torch.float32))
        yc_pred_val = yc_pred_val.detach().cpu().numpy()
        yr_pred_val = yr_pred_val.detach().cpu().numpy()

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

        print("#" * 26)
        print(f"# F1 val:  {f1_val:0.4f}")
        print(f"# F1 test: {f1_test:0.4f}")
        print(f"# R2 val:  {r2_val:0.4f}")
        print(f"# R2 test: {r2_test:0.4f}")

        with open(pickle_file, "wb") as handle:
            pickle.dump(f1_val, handle)
            pickle.dump(r2_val, handle)


def job_array(idx, max_idx):
    """Function to generate job array for hyperparameter combinations"""
    folds = [1, 2, 3, 4, 5]
    sub_folds = [1, 2, 3, 4, 5]
    batch_sizes = [5000, 10000, 20000]
    learning_rates = [0.001, 0.005, 0.01]
    masks = [0, 1]
    losses = ["base", "softbase", "softpir"]
    dims = [64, 128, 256]

    hyperparameters = itertools.product(
        folds, sub_folds, batch_sizes, learning_rates, masks, losses, dims
    )

    combinations = list(hyperparameters)
    random.seed(0)
    random.shuffle(combinations)

    # Determine the range of combinations for the current job index
    size = len(combinations) // max_idx
    start = idx * size
    end = min((idx + 1) * size, len(combinations))

    return combinations[start:end]


if __name__ == "__main__":
    idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
    max_idx = int(os.environ["SLURM_ARRAY_TASK_MAX"]) + 1
    combs = job_array(idx, max_idx)

    for comb in combs:
        fold, sub_fold, batch_size, learning_rate, mask, loss, dim = comb

        args = Args()
        args.fold = fold
        args.sub_fold = sub_fold
        args.batchsize = batch_size
        args.learningrate = learning_rate
        args.mask = mask
        args.loss = loss
        args.dim = dim

        main(args)
