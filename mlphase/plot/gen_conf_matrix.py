import os
import pickle
import proplot as pplt
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
from matplotlib.ticker import NullLocator

# Output vector indices
# 0, 1, 2 are phi_a^alpha, phi_b^alpha, phi_c^alpha (c depends on a and b, not considered)
# 3, 4, 5 are phi_a^beta, phi_b^beta, phi_c^beta
# 6, 7, 8 are phi_a^gamma, phi_b^gamma, phi_c^gamma
# 9, 10, 11 are w^alpha, w^beta, w^gamma

def gen_conf_matrix(PLOT_DIR, COLORS, files, format="png"):
    """
    Confusion matrix of phase classfication.
    """
    yc_tests = []
    yc_preds = []

    for file in files:
        with open(file, "rb") as handle:
            yc_test = pickle.load(handle)
            yc_pred = pickle.load(handle)

        if len(yc_test.shape) == 2:
            yc_test = yc_test.argmax(axis=1)

        if len(yc_pred.shape) == 2:
            yc_pred = yc_pred.argmax(axis=1)

        yc_tests.append(yc_test)
        yc_preds.append(yc_pred)

    yc_tests = np.concatenate(yc_tests)
    yc_preds = np.concatenate(yc_preds)

    cm1 = confusion_matrix(yc_tests, yc_preds, normalize="true").T
    cm2 = confusion_matrix(yc_tests, yc_preds).T

    print(f"F1: {f1_score(yc_tests, yc_preds, average='micro'):0.4f}\n")

    fig, ax = pplt.subplots(refheight=2.8, refwidth=2.8, tight=True)

    img = ax.imshow((cm1 - cm1.min()) / (cm1.max() - cm1.min()), cmap="Blues")

    for i in range(cm1.shape[0]):
        for j in range(cm1.shape[1]):
            if cm1[i, j] > 0.5:
                color = "w"
            else:
                color = "k"
            ax.text(
                j,
                i,
                f"{cm1[i, j]*100:.2f}%\n({cm2[i, j]:.0f})",
                ha="center",
                va="center",
                color=color,
                size=12,
            )

    # Add colorbar
    cbar = ax.colorbar(img, loc="r", label="Frequency", labelsize=12, ticklabelsize=11)
    cbar.ax.tick_params(axis="both", which="both", width=1)

    # Customize the colorbar boundary (spines)
    for spine in cbar.ax.spines.values():
        spine.set_linewidth(1)

    ax.grid()

    ax.tick_params(axis="both", which="both", width=1)
    for spine in ax.spines.values():
        spine.set_linewidth(1)

    ax.format(
        xlabel="True Number of Equilibrium Phases",
        ylabel="Predicted Number of Equilibrium Phases",  # Equil. is the CAS standard abbrivation
        xlabelsize=12,
        ylabelsize=12,
        xticklabelsize=12,
        yticklabelsize=12,
        xticks=[0, 1, 2],
        xticklabels=["One", "Two", "Three"],
        yticks=[0, 1, 2],
        yticklabels=["One", "Two", "Three"],
        yrotation=90,
    )

    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())

    out_name = os.path.basename(file).split(".pickle")[0]

    out_file = os.path.join(PLOT_DIR, f"conf_{out_name}.{format}")

    fig.savefig(out_file, dpi=600, transparent=True, bbox_inches="tight")
