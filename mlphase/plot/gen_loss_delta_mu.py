import os
import glob
import pickle
import numpy as np
from scipy.stats import gaussian_kde
import proplot as pplt
import matplotlib.pyplot as plt

# Output vector indices
# 0, 1, 2 are phi_a^alpha, phi_b^alpha, phi_c^alpha (c depends on a and b, not considered)
# 3, 4, 5 are phi_a^beta, phi_b^beta, phi_c^beta
# 6, 7, 8 are phi_a^gamma, phi_b^gamma, phi_c^gamma
# 9, 10, 11 are w^alpha, w^beta, w^gamma

def mu_compute_(x_in, vol_tensor):
    EPS = 1e-10
    phi_A = np.clip(vol_tensor[:, 0], EPS, None)
    phi_B = np.clip(vol_tensor[:, 1], EPS, None)
    phi_C = 1 - phi_A - phi_B

    mu_A = (
        np.log(phi_A)
        + 1
        - phi_A
        - x_in[:, 0] / np.clip(x_in[:, 2], EPS, None) * phi_B
        - x_in[:, 0] / np.clip(x_in[:, 1], EPS, None) * phi_C
        + x_in[:, 0]
        * (
            phi_B**2 * x_in[:, 4]
            + phi_C**2 * x_in[:, 3]
            + phi_B * phi_C * (x_in[:, 4] + x_in[:, 3] - x_in[:, 5])
        )
    )

    mu_B = (
        np.log(phi_B)
        + 1
        - phi_B
        - x_in[:, 2] / np.clip(x_in[:, 0], EPS, None) * phi_A
        - x_in[:, 2] / np.clip(x_in[:, 1], EPS, None) * phi_C
        + x_in[:, 2]
        * (
            phi_A**2 * x_in[:, 4]
            + phi_C**2 * x_in[:, 5]
            + phi_A * phi_C * (x_in[:, 4] + x_in[:, 5] - x_in[:, 3])
        )
    )

    mu_C = (
        np.log(np.clip(phi_C, EPS, None))
        + 1
        - np.clip(phi_C, EPS, None)
        - x_in[:, 1] / np.clip(x_in[:, 0], EPS, None) * phi_A
        - x_in[:, 1] / np.clip(x_in[:, 2], EPS, None) * phi_B
        + x_in[:, 1]
        * (
            phi_B**2 * x_in[:, 5]
            + phi_A**2 * x_in[:, 3]
            + phi_A * phi_B * (x_in[:, 5] + x_in[:, 3] - x_in[:, 4])
        )
    )

    return mu_A, mu_B, mu_C


def mu_loss_fn_(x_in, routputs, coutputs, which="all"):
    max_indices = coutputs
    mask2 = max_indices == 1
    mask3 = max_indices == 2
    losses = np.zeros_like(x_in[:, 0])

    if np.any(mask2):
        indices = np.where(mask2)[0]
        mu_A1, mu_B1, mu_C1 = mu_compute_(x_in[indices], routputs[indices, 0:2])
        mu_A2, mu_B2, mu_C2 = mu_compute_(x_in[indices], routputs[indices, 3:5])

        if which == "A":
            losses[indices] = (mu_A1 - mu_A2) ** 2
        elif which == "B":
            losses[indices] = (mu_B1 - mu_B2) ** 2
        elif which == "C":
            losses[indices] = (mu_C1 - mu_C2) ** 2
        elif which == "all":
            losses[indices] = (
                (mu_A1 - mu_A2) ** 2 + (mu_B1 - mu_B2) ** 2 + (mu_C1 - mu_C2) ** 2
            )

        row_sum1 = np.sum(routputs[indices, 0:2], axis=1)
        row_sum2 = np.sum(routputs[indices, 3:5], axis=1)

        invalid_indices = np.logical_or(row_sum1 > 1, row_sum2 > 1)
        losses[indices[invalid_indices]] = np.inf

    if np.any(mask3):
        indices = np.where(mask3)[0]
        mu_A1, mu_B1, mu_C1 = mu_compute_(x_in[indices], routputs[indices, 0:2])
        mu_A2, mu_B2, mu_C2 = mu_compute_(x_in[indices], routputs[indices, 3:5])
        mu_A3, mu_B3, mu_C3 = mu_compute_(x_in[indices], routputs[indices, 6:8])

        if which == "A":
            losses[indices] = (
                (mu_A1 - mu_A2) ** 2 + (mu_A1 - mu_A3) ** 2 + (mu_A2 - mu_A3) ** 2
            ) / 3
        elif which == "B":
            losses[indices] = (
                (mu_B1 - mu_B2) ** 2 + (mu_B1 - mu_B3) ** 2 + (mu_B2 - mu_B3) ** 2
            ) / 3
        elif which == "C":
            losses[indices] = (
                (mu_C1 - mu_C2) ** 2 + (mu_C1 - mu_C3) ** 2 + (mu_C2 - mu_C3) ** 2
            ) / 3
        elif which == "all":
            losses[indices] = (
                (mu_A1 - mu_A2) ** 2
                + (mu_B1 - mu_B2) ** 2
                + (mu_A1 - mu_A3) ** 2
                + (mu_B1 - mu_B3) ** 2
                + (mu_A2 - mu_A3) ** 2
                + (mu_B2 - mu_B3) ** 2
                + (mu_C1 - mu_C2) ** 2
                + (mu_C1 - mu_C3) ** 2
                + (mu_C2 - mu_C3) ** 2
            ) / 9

        row_sum1 = np.sum(routputs[indices, 0:2], axis=1)
        row_sum2 = np.sum(routputs[indices, 3:5], axis=1)
        row_sum3 = np.sum(routputs[indices, 6:8], axis=1)

        invalid_indices = np.logical_or(
            row_sum1 > 1, np.logical_or(row_sum2 > 1, row_sum3 > 1)
        )
        losses[indices[invalid_indices]] = np.inf

    non_zero_losses = losses[losses != 0]
    return non_zero_losses


def extract_loss_delta_mu(RESULT_DIR, model, which="all"):
    mask = 0
    sample_ratio = 1.0

    loss = []

    for i in range(1, 6):
        try:
            file_pattern = os.path.join(
                RESULT_DIR,
                f"MLPHASEOUTER-{i}-1-*-*-*-{mask}-{sample_ratio}-*-{model}.pickle",
            )

            file = glob.glob(file_pattern)[0]

            with open(file, "rb") as handle:
                yc_test = pickle.load(handle)
                for _ in range(2):
                    _ = pickle.load(handle)
                yr_pred = pickle.load(handle)
                x_test = pickle.load(handle)

            yr_pred_new = np.copy(yr_pred)
            yr_pred = yr_pred_new
            loss.append(
                mu_loss_fn_(x_test, yr_pred, yc_test.argmax(axis=1), which=which)
            )
        except:
            continue

    loss = np.concatenate(loss)
    return loss


def gen_loss_delta_mu(RESULT_DIR, PLOT_DIR, COLORS, which="all", format="png"):
    x_vals = np.linspace(-6, 2, 100)
    bins = np.linspace(-6, 2, 50)
    alpha = 0.2

    loss_base = extract_loss_delta_mu(RESULT_DIR, "base", which=which)
    loss_pi = extract_loss_delta_mu(RESULT_DIR, "softbase", which=which)
    loss_pir = extract_loss_delta_mu(RESULT_DIR, "softpir", which=which)

    f_base = np.where(loss_base < np.inf)[0].shape[0] / len(loss_base)
    f_pi = np.where(loss_pi < np.inf)[0].shape[0] / len(loss_pi)
    f_pir = np.where(loss_pir < np.inf)[0].shape[0] / len(loss_pir)

    loss_base = loss_base[loss_base < np.inf]
    loss_pi = loss_pi[loss_pi < np.inf]
    loss_pir = loss_pir[loss_pir < np.inf]

    kde_base_func = gaussian_kde(np.log10(loss_base), bw_method=0.2)
    kde_base = kde_base_func(x_vals)

    kde_pi_func = gaussian_kde(np.log10(loss_pi), bw_method=0.2)
    kde_pi = kde_pi_func(x_vals)

    kde_pir_func = gaussian_kde(np.log10(loss_pir), bw_method=0.2)
    kde_pir = kde_pir_func(x_vals)

    # plot
    fig, ax = pplt.subplots()

    data = np.log10(loss_base)
    hist, bin_edges = np.histogram(data, bins=bins, density=True)
    ax.bar(
        bin_edges[:-1],
        hist * f_base,
        width=1,
        align="edge",
        alpha=alpha,
        color=COLORS[0],
    )

    data = np.log10(loss_pi)
    hist, bin_edges = np.histogram(data, bins=bins, density=True)
    plt.bar(
        bin_edges[:-1],
        hist * f_pi,
        width=1,
        align="edge",
        alpha=alpha,
        color=COLORS[1],
    )

    data = np.log10(loss_pir)
    hist, bin_edges = np.histogram(data, bins=bins, density=True)
    plt.bar(
        bin_edges[:-1],
        hist * f_pir,
        width=1,
        align="edge",
        alpha=alpha,
        color="forestgreen",
    )

    ax.plot(x_vals, kde_base, color=COLORS[0], lw=2, label="Base")
    ax.plot(x_vals, kde_pi, color=COLORS[1], lw=2, label="PI")
    ax.plot(x_vals, kde_pir, color="forestgreen", lw=2, label="PI+")

    # Mean lines
    mean_base = np.mean(np.log10(loss_base))
    mean_pi = np.mean(np.log10(loss_pi))
    mean_pir = np.mean(np.log10(loss_pir))

    ax.axvline(mean_base, color=COLORS[0], linestyle="--", linewidth=1)
    ax.axvline(mean_pi, color=COLORS[1], linestyle="--", linewidth=1)
    ax.axvline(mean_pir, color="forestgreen", linestyle="--", linewidth=1)

    xlabel = r"log$_{10}$ $\Delta \mathit{\mu}$"

    ax.legend(ncol=1, prop={"size": 11})

    ax.format(
        xlabel=xlabel,
        ylabel="Density",
        xlabelsize=13,
        ylabelsize=13,
        xticklabelsize=12,
        yticklabelsize=12,
        xlim=[-6, 2],
        ylim=[0, 0.8],
        xticks=[-8, -6, -4, -2, 0, 2],
        yticks=[0, 0.2, 0.4, 0.6, 0.8],
        grid="off",
    )

    ax.xaxis.set_tick_params(
        labelbottom=True, labeltop=False, top=True, bottom=True, which="both"
    )
    ax.yaxis.set_tick_params(
        labelleft=True, labelright=False, left=True, right=True, which="both"
    )

    ax.tick_params(axis="both", which="both", width=1)
    for spine in ax.spines.values():
        spine.set_linewidth(1)

    out_file = os.path.join(PLOT_DIR, f"loss_delta_mu.{format}")

    print(out_file + " saved ...")

    fig.savefig(out_file, dpi=600, transparent=True, bbox_inches="tight")
