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

def split_loss_(x_in, routputs):
    final_split = (
        routputs[:, 0:2] * routputs[:, [-3]]
        + routputs[:, 3:5] * routputs[:, [-2]]
        + routputs[:, 6:8] * routputs[:, [-1]]
    )
    split_loss = (x_in[:, -2:] - final_split) ** 2
    return split_loss.mean(axis=1)


def extract_loss_split(RESULT_DIR, model):
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
                    pickle.load(handle)
                yr_pred = pickle.load(handle)
                x_test = pickle.load(handle)

            loss.append(split_loss_(x_test, yr_pred))
        except:
            continue

    loss = np.concatenate(loss)
    return loss


def gen_loss_split(RESULT_DIR, PLOT_DIR, COLORS, format="png"):
    x_vals = np.linspace(-8, 2, 100)
    bins = np.linspace(-8, 2, 50)
    alpha = 0.2

    loss_base = extract_loss_split(RESULT_DIR, "base")
    loss_pi = extract_loss_split(RESULT_DIR, "softbase")
    loss_pir = extract_loss_split(RESULT_DIR, "softpir")

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
        hist,
        width=1,
        align="edge",
        alpha=alpha,
        color=COLORS[0],
    )

    data = np.log10(loss_pi)
    hist, bin_edges = np.histogram(data, bins=bins, density=True)
    plt.bar(
        bin_edges[:-1],
        hist,
        width=1,
        align="edge",
        alpha=alpha,
        color=COLORS[1],
    )

    data = np.log10(loss_pir)
    hist, bin_edges = np.histogram(data, bins=bins, density=True)
    plt.bar(
        bin_edges[:-1],
        hist,
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

    ax.legend(ncol=1, prop={"size": 11})

    ax.format(
        xlabel=r"log$_{10}$ $\mathcal{L}_\mathrm{Split}$",
        ylabel="Density",
        xlabelsize=13,
        ylabelsize=13,
        xticklabelsize=12,
        yticklabelsize=12,
        xlim=[-8, 2],
        ylim=[0.0, 0.8],
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

    out_file = os.path.join(PLOT_DIR, f"loss_split.{format}")

    print(out_file + " saved ...")

    fig.savefig(out_file, dpi=600, transparent=True, bbox_inches="tight")
