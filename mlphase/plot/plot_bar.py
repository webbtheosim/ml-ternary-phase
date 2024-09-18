import os
import proplot as pplt
from mlphase.plot import sat
from matplotlib.ticker import NullLocator

# Output vector indices
# 0, 1, 2 are phi_a^alpha, phi_b^alpha, phi_c^alpha (c depends on a and b, not considered)
# 3, 4, 5 are phi_a^beta, phi_b^beta, phi_c^beta
# 6, 7, 8 are phi_a^gamma, phi_b^gamma, phi_c^gamma
# 9, 10, 11 are w^alpha, w^beta, w^gamma

def plot_bar(plot_dir, colors, res_f1, res_r2, ylims=[0.8, 1.0], fmt="png"):
    """Generate bar plot of phase classification F1 and composition prediction R2."""
    fig, ax = pplt.subplots(refwidth=2.53, refheight=2.53)

    ax.bar(
        [-3],
        res_f1[[0], -1, 0],
        yerr=res_f1[[0], -1, 1],
        width=4,
        color=colors[1],
        label="Base No Mask",
        edgecolor="k",
    )

    ax.bar(
        [3],
        res_f1[[1], -1, 0],
        yerr=res_f1[[1], -1, 1],
        width=4,
        color=sat(colors[1], 0.7),
        label="Base Mask",
        edgecolor="k",
    )

    ax.bar(
        [-1],
        res_f1[[2], -1, 0],
        yerr=res_f1[[2], -1, 1],
        width=4,
        color=colors[0],
        label="PI No Mask",
        edgecolor="k",
    )

    ax.bar(
        [5],
        res_f1[[3], -1, 0],
        yerr=res_f1[[3], -1, 1],
        width=4,
        color=sat(colors[0], 0.7),
        label="PI Mask",
        edgecolor="k",
    )

    ax.bar(
        [1],
        res_f1[[4], -1, 0],
        yerr=res_f1[[4], -1, 1],
        width=4,
        color=colors[2],
        label="PI+ No Mask",
        edgecolor="k",
    )

    ax.bar(
        [7],
        res_f1[[5], -1, 0],
        yerr=res_f1[[5], -1, 1],
        width=4,
        color=sat(colors[2], 0.7),
        label="PI+ Mask",
        edgecolor="k",
    )

    ax.bar(
        [13, 15, 17, 19, 21, 23],
        res_r2[[0, 2, 4, 1, 3, 5], -1, 0],
        yerr=res_r2[[0, 2, 4, 1, 3, 5], -1, 1],
        width=1,
        color=[
            colors[1],
            colors[0],
            colors[2],
            sat(colors[1], 0.7),
            sat(colors[0], 0.7),
            sat(colors[2], 0.7),
        ],
        edgecolor="k",
    )

    ax.legend(ncol=1, prop={"size": 9.5}, facecolor="white", facealpha=1.0)

    ax.format(
        ylabel="Score",
        xlabelsize=12,
        ylabelsize=12,
        xticklabelsize=11,
        yticklabelsize=11,
        ylim=ylims,
        grid="off",
        yticks=[0.8, 0.85, 0.9, 0.95, 1.0],
        xticks=[2, 18],
        xticklabels=["$\mathit{F}_1$", "$\mathit{R}^2$"],
        xrotation=0,
    )

    ax.tick_params(axis="both", which="both", width=1)
    for spine in ax.spines.values():
        spine.set_linewidth(1)

    ax.xaxis.set_tick_params(
        labelbottom=True, labeltop=False, top=True, bottom=True, which="both"
    )
    ax.yaxis.set_tick_params(
        labelleft=True, labelright=False, left=True, right=True, which="both"
    )
    ax.xaxis.set_minor_locator(NullLocator())

    out_file = os.path.join(plot_dir, f"bar_plot.{fmt}")

    fig.savefig(out_file, dpi=600, transparent=True)
