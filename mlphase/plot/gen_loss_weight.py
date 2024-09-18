import os
import glob
import pickle
import numpy as np
from scipy.stats import gaussian_kde
import proplot as pplt

# Output vector indices
# 0, 1, 2 are phi_a^alpha, phi_b^alpha, phi_c^alpha (c depends on a and b, not considered)
# 3, 4, 5 are phi_a^beta, phi_b^beta, phi_c^beta
# 6, 7, 8 are phi_a^gamma, phi_b^gamma, phi_c^gamma
# 9, 10, 11 are w^alpha, w^beta, w^gamma

def wts_loss_(routputs):
    sum_wts = routputs[:, -3] + routputs[:, -2] + routputs[:, -1]
    wts_loss = (1 - sum_wts) ** 2
    return wts_loss


def extract_loss_wts(RESULT_DIR, model):
    mask = 0
    sample_ratio = 1.0

    loss = []

    for i in range(1, 6):
        file_pattern = os.path.join(
            RESULT_DIR,
            f"MLPHASEOUTER-{i}-1-*-*-*-{mask}-{sample_ratio}-*-{model}.pickle",
        )
        file = glob.glob(file_pattern)[0]

        with open(file, "rb") as handle:
            for _ in range(3):
                pickle.load(handle)
            yr_pred = pickle.load(handle)

        loss.append(wts_loss_(yr_pred))

    loss = np.concatenate(loss)

    return loss


def gen_loss_weight(RESULT_DIR, PLOT_DIR, COLORS, format="png"):
    x_vals = np.linspace(-12, 0, 100)
    bins = np.linspace(-12, 0, 50)
    alpha = 0.2

    loss_base = extract_loss_wts(RESULT_DIR, "base")

    f_base = np.where(loss_base != 0)[0].shape[0] / len(loss_base)
    loss_base = loss_base[loss_base != 0]

    kde_base_func = gaussian_kde(np.log10(loss_base), bw_method=0.2)
    kde_base = kde_base_func(x_vals)

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

    ax.plot(x_vals, kde_base * f_base, color=COLORS[0], lw=2, label="Base")

    # Mean lines
    mean_base = np.mean(np.log10(loss_base))

    ax.axvline(mean_base, color=COLORS[0], linestyle="--", linewidth=1)

    ax.legend(ncol=1, prop={"size": 11})

    ax.format(
        xlabel=r"log$_{10}$ $\mathcal{L}_\mathrm{Weight}$",
        ylabel="Density",
        xlabelsize=13,
        ylabelsize=13,
        xticklabelsize=12,
        yticklabelsize=12,
        xlim=[-12, 0],
        ylim=[0.0, 0.4],
        xticks=[-12, -8, -4, 0],
        yticks=[0, 0.1, 0.2, 0.3, 0.4],
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

    out_file = os.path.join(PLOT_DIR, f"loss_weight.{format}")

    print(out_file + " saved ...")

    fig.savefig(out_file, dpi=600, transparent=True, bbox_inches="tight")
