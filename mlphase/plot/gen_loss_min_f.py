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


def mu_loss_fn_(x_in, routputs):
    losses = np.zeros_like(x_in[:, 0])

    mu_a1, mu_b1, mu_c1 = mu_compute_(x_in, routputs[:, 0:3])
    mu_a2, mu_b2, mu_c2 = mu_compute_(x_in, routputs[:, 3:6])
    mu_a3, mu_b3, mu_c3 = mu_compute_(x_in, routputs[:, 6:9])
    losses[:] = (
        mu_a1 * routputs[:, 0] * routputs[:, -3]
        + mu_b1 * routputs[:, 1] * routputs[:, -3]
        + mu_c1 * routputs[:, 2] * routputs[:, -3]
        + mu_a2 * routputs[:, 3] * routputs[:, -2]
        + mu_b2 * routputs[:, 4] * routputs[:, -2]
        + mu_c2 * routputs[:, 5] * routputs[:, -2]
        + mu_a3 * routputs[:, 6] * routputs[:, -1]
        + mu_b3 * routputs[:, 7] * routputs[:, -1]
        + mu_c3 * routputs[:, 8] * routputs[:, -1]
    )

    return losses


def extract_loss_min_f(RESULT_DIR, model):
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
                for _ in range(3):
                    _ = pickle.load(handle)
                yr_pred = pickle.load(handle)
                x_test = pickle.load(handle)

            yr_pred_new = np.copy(yr_pred)
            yr_pred = yr_pred_new
            loss.append(mu_loss_fn_(x_test, yr_pred))
        except:
            continue

    loss = np.concatenate(loss)
    return loss


def gen_loss_min_f(RESULT_DIR, PLOT_DIR, COLORS, format="png"):
    x_vals = np.linspace(-0.8, 0.2, 100)
    bins = np.linspace(-0.8, 0.2, 50)
    alpha = 0.2

    loss_pi = extract_loss_min_f(RESULT_DIR, "softbase")
    loss_pir = extract_loss_min_f(RESULT_DIR, "softpir")

    f_pi = np.where(loss_pi < np.inf)[0].shape[0] / len(loss_pi)
    f_pir = np.where(loss_pir < np.inf)[0].shape[0] / len(loss_pir)

    loss_pi = loss_pi[loss_pi < np.inf]
    loss_pir = loss_pir[loss_pir < np.inf]

    kde_pi_func = gaussian_kde(loss_pi, bw_method=0.2)
    kde_pi = kde_pi_func(x_vals)

    kde_pir_func = gaussian_kde(loss_pir, bw_method=0.2)
    kde_pir = kde_pir_func(x_vals)

    # plot
    fig, ax = pplt.subplots()

    data = loss_pi
    hist, bin_edges = np.histogram(data, bins=bins, density=True)
    plt.bar(
        bin_edges[:-1],
        hist * f_pi,
        width=1,
        align="edge",
        alpha=alpha,
        color=COLORS[1],
    )

    data = loss_pir
    hist, bin_edges = np.histogram(data, bins=bins, density=True)
    plt.bar(
        bin_edges[:-1],
        hist * f_pir,
        width=1,
        align="edge",
        alpha=alpha,
        color="forestgreen",
    )

    ax.plot(x_vals, kde_pi * f_pi, color=COLORS[1], lw=2, label="PI")
    ax.plot(x_vals, kde_pir * f_pir, color="forestgreen", lw=2, label="PI+")

    # Mean lines
    mean_pi = np.mean(loss_pi)
    mean_pir = np.mean(loss_pir)

    ax.axvline(mean_pi, color=COLORS[1], linestyle="--", linewidth=1)
    ax.axvline(mean_pir, color="forestgreen", linestyle="--", linewidth=1)

    xlabel = r"$\mathcal{L}_\mathrm{F}$"

    ax.legend(ncol=1, prop={"size": 11}, loc="upper right")

    ax.format(
        xlabel=xlabel,
        ylabel="Density",
        xlabelsize=13,
        ylabelsize=13,
        xticklabelsize=12,
        yticklabelsize=12,
        xlim=[-0.8, 0.2],
        ylim=[0, 5],
        xticks=[-0.8, -0.6, -0.4, -0.2, 0, 0.2],
        yticks=[0, 1, 2, 3, 4, 5],
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

    out_file = os.path.join(PLOT_DIR, f"loss_min_f.{format}")

    print(out_file + " saved ...")

    fig.savefig(out_file, dpi=600, transparent=True, bbox_inches="tight")
