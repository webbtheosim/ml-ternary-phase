import os
import proplot as pplt
import numpy as np
from mlphase.plot import sat

# Output vector indices
# 0, 1, 2 are phi_a^alpha, phi_b^alpha, phi_c^alpha (c depends on a and b, not considered)
# 3, 4, 5 are phi_a^beta, phi_b^beta, phi_c^beta
# 6, 7, 8 are phi_a^gamma, phi_b^gamma, phi_c^gamma
# 9, 10, 11 are w^alpha, w^beta, w^gamma

def plot_train_size(
    plot_dir,
    colors,
    res_f1,
    res_r2,
    plot_std=False,
    labels=["Base No Mask", "Base Mask", "PI No Mask", "PI Mask"],
    ylims=[[0.93, 0.98], [0.87, 0.95]],
    out_name="metric",
    single_index=None,
    fmt="png",
):
    """Plot training size impact on model performance."""
    fig, ax = pplt.subplots(refwidth=2.53, refheight=2.53, ncols=2, sharey=False)

    train_size = np.array(
        [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    )

    markers = ["o", "o", "s", "s", "^", "^"]
    color_set = [
        colors[1],
        sat(colors[1], 0.6),
        colors[0],
        sat(colors[0], 0.6),
        colors[2],
        sat(colors[2], 0.6),
    ]

    if res_f1.shape[-1] == 2:
        for i, label in enumerate(labels):
            err_f1 = res_f1[i, ..., 1] if plot_std else np.zeros_like(res_f1[i, ..., 1])
            err_r2 = res_r2[i, ..., 1] if plot_std else np.zeros_like(res_r2[i, ..., 1])
            if single_index is None or i == single_index:
                ax[0].errorbar(
                    train_size,
                    res_f1[i, ..., 0],
                    err_f1,
                    linestyle="--",
                    label=label,
                    marker=markers[i],
                    c=color_set[i],
                    lw=1,
                )
                ax[1].errorbar(
                    train_size,
                    res_r2[i, ..., 0],
                    err_r2,
                    linestyle="--",
                    label=label,
                    marker=markers[i],
                    c=color_set[i],
                    lw=1,
                )
    else:

        for i, label in enumerate(labels):
            if single_index is None or i == single_index:
                ax[0].errorbar(
                    train_size,
                    res_f1[i],
                    linestyle="--",
                    label=label,
                    marker=markers[i],
                    c=color_set[i],
                    markeredgecolor="k",
                    markersize=8,
                )
                ax[1].errorbar(
                    train_size,
                    res_r2[i],
                    linestyle="--",
                    label=label,
                    marker=markers[i],
                    c=color_set[i],
                    markeredgecolor="k",
                    markersize=8,
                )

    ax[0].legend(ncol=1, prop={"size": 9.5})
    ax[0].format(
        xlabel="Training Set Proportion",
        ylabel=r"$\mathit{F}_1$",
        xlabelsize=12,
        ylabelsize=12,
        xticklabelsize=11,
        yticklabelsize=11,
        xlim=[-0.1, 1.1],
        xticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        ylim=ylims[0],
        grid="off",
        xrotation=0,
    )

    ax[1].format(
        xlabel="Training Set Proportion",
        ylabel=r"$\mathit{R}^2$",
        xlabelsize=12,
        ylabelsize=12,
        xticklabelsize=11,
        yticklabelsize=11,
        xticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        xlim=[-0.1, 1.1],
        ylim=ylims[1],
        grid="off",
        xrotation=0,
    )

    for axis in ax:
        axis.xaxis.set_tick_params(
            labelbottom=True, labeltop=False, top=True, bottom=True, which="both"
        )
        axis.yaxis.set_tick_params(
            labelleft=True, labelright=False, left=True, right=True, which="both"
        )

        axis.tick_params(axis="both", which="both", width=1)
        for spine in axis.spines.values():
            spine.set_linewidth(1)

    out_file = os.path.join(plot_dir, f"{out_name}.{fmt}")
    fig.savefig(out_file, dpi=600, transparent=True)
