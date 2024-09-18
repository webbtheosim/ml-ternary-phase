import os
import glob
import pickle
import proplot as pplt
import numpy as np

# Output vector indices
# 0, 1, 2 are phi_a^alpha, phi_b^alpha, phi_c^alpha (c depends on a and b, not considered)
# 3, 4, 5 are phi_a^beta, phi_b^beta, phi_c^beta
# 6, 7, 8 are phi_a^gamma, phi_b^gamma, phi_c^gamma
# 9, 10, 11 are w^alpha, w^beta, w^gamma

def gen_parity_plot(
    TRAIN_RESULT_DIR,
    PLOT_DIR,
    COLORS,
    mask=0,
    loss="pi",
    sample_ratio="1.0",
    format="png",
    avg_three_phase=False,
):
    """
    Parity plot of composition prediction.
    """
    yr_tests = []
    yr_preds = []

    for i in range(1, 6):

        filename = os.path.join(
            TRAIN_RESULT_DIR,
            f"MLPHASEOUTER-{i}-1-*-*-*-{mask}-{sample_ratio}-*-{loss}.pickle",
        )

        file = glob.glob(filename)[0]

        with open(file, "rb") as handle:
            yc_test = pickle.load(handle)
            yc_pred = pickle.load(handle)
            yr_test = pickle.load(handle)
            yr_pred = pickle.load(handle)
            x_test = pickle.load(handle)
            phase_idx_test = pickle.load(handle)

        yr_pred_new = np.copy(yr_pred)
        u_phase_idx_test = np.unique(phase_idx_test)

        if avg_three_phase:
            for _, u in enumerate(u_phase_idx_test):
                idx = np.where(phase_idx_test == u)[0]

                if yc_test[idx].argmax(axis=1).max() == 2:
                    idx = np.where(
                        (phase_idx_test == u) & (yc_pred.argmax(axis=1) == 2)
                    )[0]

                    if len(idx) > 0:
                        for j in [0, 1, 3, 4, 6, 7]:
                            yr_pred_new[idx, j] = yr_pred_new[idx, j].mean()

        yr_pred = yr_pred_new

        yr_preds.append(yr_pred)
        yr_tests.append(yr_test)

    # plot
    fig, ax = pplt.subplots(
        nrows=3,
        ncols=3,
        refwidth=2.8 / 3 * 2,
        refheight=2.8 / 3 * 2.5 / 3,
        wspace=1.5,
        hspace=1.5,
    )

    labels = [
        r"$\mathit{\phi}_{A}^{\alpha}$",
        r"$\mathit{\phi}_{B}^{\alpha}$",
        r"$\mathit{w}^{\alpha}$",
        r"$\mathit{\phi}_{A}^{\beta}$",
        r"$\mathit{\phi}_{B}^{\beta}$",
        r"$\mathit{w}^{\beta}$",
        r"$\mathit{\phi}_{A}^{\gamma}$",
        r"$\mathit{\phi}_{B}^{\gamma}$",
        r"$\mathit{w}^{\gamma}$",
    ]

    plot_order = [0, 1, -3, 3, 4, -2, 6, 7, -1]

    for i in range(9):
        for j in range(5):
            ax[i].scatter(
                yr_tests[j][:, plot_order[i]],
                yr_preds[j][:, plot_order[i]],
                s=0.0001,
                alpha=0.1,
                color=COLORS[1],
            )
        ax[i].plot([-1, 2], [-1, 2], "k--", lw=1)

    for i in range(9):
        ax[i].text(0.8, 0.2, labels[i], size=14, transform=ax[i].transAxes)

    for i in range(9):
        ax[i].format(
            xlabel="True Equilibrium Composition",
            ylabel="Predicted Equilibrium Composition",
            xlabelsize=12,
            ylabelsize=12,
            xticklabelsize=12,
            yticklabelsize=12,
            xlim=[-0.05, 1.05],
            ylim=[-0.05, 1.05],
            xticks=[0, 0.5, 1.0],
            yticks=[0, 0.5, 1.0],
            grid="off",
        )
        ax[i].xaxis.set_tick_params(
            labelbottom=True, labeltop=False, top=True, bottom=True, which="both"
        )
        ax[i].yaxis.set_tick_params(
            labelleft=True, labelright=False, left=True, right=True, which="both"
        )

        ax[i].tick_params(axis="both", which="both", width=1)
        for spine in ax[i].spines.values():
            spine.set_linewidth(1)

    if avg_three_phase:
        out_file = os.path.join(
            PLOT_DIR, f"parity_{mask}_{loss}_{sample_ratio}.{format}"
        )
    else:
        out_file = os.path.join(
            PLOT_DIR, f"parity_{mask}_{loss}_{sample_ratio}_no_avg.{format}"
        )

    fig.savefig(out_file, dpi=600, transparent=False, bbox_inches="tight")
