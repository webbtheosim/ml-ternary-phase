import os
import proplot as pplt
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from mlphase.plot import sat
import ternary
import mpltern

# Output vector indices
# 0, 1, 2 are phi_a^alpha, phi_b^alpha, phi_c^alpha (c depends on a and b, not considered)
# 3, 4, 5 are phi_a^beta, phi_b^beta, phi_c^beta
# 6, 7, 8 are phi_a^gamma, phi_b^gamma, phi_c^gamma
# 9, 10, 11 are w^alpha, w^beta, w^gamma

def gen_ternary_scatter(
    file,
    x_test,
    yc_test,
    yc_pred,
    yr_test,
    PLOT_DIR,
    COLORS,
    three_phase=True,
    format="png",
):

    idx = np.where(yc_test == 1)[0]

    yr_test_temp = yr_test[idx]

    phi_A1_test = yr_test_temp[:, 0]
    phi_B1_test = yr_test_temp[:, 1]
    phi_C1_test = 1 - phi_A1_test - phi_B1_test

    phi_A2_test = yr_test_temp[:, 3]
    phi_B2_test = yr_test_temp[:, 4]
    phi_C2_test = 1 - phi_A2_test - phi_B2_test

    tie_mid = np.array(
        [
            (phi_A1_test + phi_A2_test) / 2,
            (phi_B1_test + phi_B2_test) / 2,
            (phi_C1_test + phi_C2_test) / 2,
        ]
    ).T

    idx_1_true = np.where(yc_pred == 0)[0]
    idx_2_true = np.where(yc_pred == 1)[0]
    idx_3_true = np.where(yc_pred == 2)[0]

    # kmean
    km = KMeans(n_clusters=100, random_state=0, n_init="auto")
    km.fit(tie_mid)

    # 3 phase triangle
    if three_phase:
        idx_3 = np.where(yc_test == 2)[0]

        yr_test_temp_3 = yr_test[idx_3]

        phi_A1_test_3 = yr_test_temp_3[:, 0]
        phi_B1_test_3 = yr_test_temp_3[:, 1]
        phi_C1_test_3 = 1 - phi_A1_test_3 - phi_B1_test_3

        phi_A2_test_3 = yr_test_temp_3[:, 3]
        phi_B2_test_3 = yr_test_temp_3[:, 4]
        phi_C2_test_3 = 1 - phi_A2_test_3 - phi_B2_test_3

        phi_A3_test_3 = yr_test_temp_3[:, 6]
        phi_B3_test_3 = yr_test_temp_3[:, 7]
        phi_C3_test_3 = 1 - phi_A3_test_3 - phi_B3_test_3

        tri_1_true = np.array(
            [phi_A1_test_3.mean(), phi_A2_test_3.mean(), phi_A3_test_3.mean()]
        )
        tri_2_true = np.array(
            [phi_B1_test_3.mean(), phi_B2_test_3.mean(), phi_B3_test_3.mean()]
        )
        tri_3_true = np.array(
            [phi_C1_test_3.mean(), phi_C2_test_3.mean(), phi_C3_test_3.mean()]
        )

    fig = plt.figure(figsize=(3.3, 3.3))
    ax = fig.add_subplot(projection="ternary")

    ax.fill([0, 0, 1], [1, 0, 0], [0, 1, 0], zorder=0, color=sat(COLORS[3], 0.5))

    for i in range(100):
        idx_i = np.where(km.labels_ == i)[0]
        for j in range(200):
            idx_j = np.random.RandomState(j).permutation(idx_i)[::20]
            for k, idx_k in enumerate(idx_j):
                k_old = idx_i[k]
                ax.fill(
                    [phi_A1_test[k_old], phi_A2_test[idx_k], phi_A1_test[idx_k]],
                    [phi_B1_test[k_old], phi_B2_test[idx_k], phi_B1_test[idx_k]],
                    [phi_C1_test[k_old], phi_C2_test[idx_k], phi_C1_test[idx_k]],
                    zorder=0,
                    c=sat(COLORS[1], 0.5),
                    lw=3,
                )

    if three_phase:
        ax.fill(tri_1_true, tri_2_true, tri_3_true, zorder=0, c=sat(COLORS[0], 0.5))

    ax.scatter(
        x_test[idx_1_true, -2],
        x_test[idx_1_true, -1],
        1 - x_test[idx_1_true, -2] - x_test[idx_1_true, -1],
        s=0.5,
        c="k",
    )

    ax.scatter(
        x_test[idx_2_true, -2],
        x_test[idx_2_true, -1],
        1 - x_test[idx_2_true, -2] - x_test[idx_2_true, -1],
        s=0.5,
        c="b",
    )

    ax.scatter(
        x_test[idx_3_true, -2],
        x_test[idx_3_true, -1],
        1 - x_test[idx_3_true, -2] - x_test[idx_3_true, -1],
        s=0.5,
        c="r",
    )

    ax.set_llabel(r"$\mathit{\phi}_{\mathrm{B}}$", fontsize=13)
    ax.set_rlabel(r"$\mathit{\phi}_{\mathrm{C}}$", fontsize=13)
    ax.set_tlabel(r"$\mathit{\phi}_{\mathrm{A}}$", fontsize=13)

    ax.taxis.set_ticks_position("tick2")
    ax.laxis.set_ticks_position("tick2")
    ax.raxis.set_ticks_position("tick2")

    ax.tick_params(axis="r", labelsize=13)
    ax.tick_params(axis="l", labelsize=13)
    ax.tick_params(axis="t", labelsize=13)

    out_names = os.path.basename(file).split(".pickle")[0].split("-")
    out_name = (
        f"{out_names[1]}-{out_names[2]}-{out_names[7]}-{out_names[9]}-{out_names[-1]}"
    )
    out_file = os.path.join(PLOT_DIR, f"{out_name}_cls.{format}")
    print(out_file + " saved ...")

    # plt.tight_layout()

    fig.savefig(out_file, dpi=300, transparent=True, bbox_inches="tight")
