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

def gen_ternary_boundary_post(
    file,
    yc_test,
    yr_test,
    phase_2_pred,
    phase_3_pred,
    PLOT_DIR,
    COLORS,
    three_phase,
    format="png",
    avg_three_phase=False,
):
    """
    Post-ML optimized ternary phase-coexistence curves.
    """

    idx = np.where(yc_test == 1)[0]

    yr_test_temp = yr_test[idx]

    phi_A1_pred = phase_2_pred[:, 0]
    phi_B1_pred = phase_2_pred[:, 1]
    phi_C1_pred = 1 - phi_A1_pred - phi_B1_pred

    phi_A2_pred = phase_2_pred[:, 2]
    phi_B2_pred = phase_2_pred[:, 3]
    phi_C2_pred = 1 - phi_A2_pred - phi_B2_pred

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

    # kmean
    km = KMeans(n_clusters=100, random_state=0, n_init="auto")
    km.fit(tie_mid)

    # 3 phase triangle
    if three_phase:
        phi_A1_pred_3 = phase_3_pred[:, 0]
        phi_B1_pred_3 = phase_3_pred[:, 1]
        phi_C1_pred_3 = 1 - phi_A1_pred_3 - phi_B1_pred_3

        phi_A2_pred_3 = phase_3_pred[:, 2]
        phi_B2_pred_3 = phase_3_pred[:, 3]
        phi_C2_pred_3 = 1 - phi_A2_pred_3 - phi_B2_pred_3

        phi_A3_pred_3 = phase_3_pred[:, 4]
        phi_B3_pred_3 = phase_3_pred[:, 5]
        phi_C3_pred_3 = 1 - phi_A3_pred_3 - phi_B3_pred_3

        if avg_three_phase:
            tri_1 = np.array(
                [phi_A1_pred_3.mean(), phi_A2_pred_3.mean(), phi_A3_pred_3.mean()]
            )
            tri_2 = np.array(
                [phi_B1_pred_3.mean(), phi_B2_pred_3.mean(), phi_B3_pred_3.mean()]
            )
            tri_3 = np.array(
                [phi_C1_pred_3.mean(), phi_C2_pred_3.mean(), phi_C3_pred_3.mean()]
            )
        else:
            tri_1 = np.array([phi_A1_pred_3, phi_A2_pred_3, phi_A3_pred_3])
            tri_2 = np.array([phi_B1_pred_3, phi_B2_pred_3, phi_B3_pred_3])
            tri_3 = np.array([phi_C1_pred_3, phi_C2_pred_3, phi_C3_pred_3])

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

        if avg_three_phase:
            ax.plot(
                tri_1[[0, 1, 2, 0]],
                tri_2[[0, 1, 2, 0]],
                tri_3[[0, 1, 2, 0]],
                c=COLORS[0],
                lw=2,
            )
        else:
            # print(tri_1.shape) # (3, n_points)
            for t in range(tri_1.shape[1]):
                # print(tri_1[..., t])
                a_1 = np.array([tri_1[..., t][0], tri_2[..., t][0], tri_3[..., t][0]])
                b_1 = np.array([tri_1[..., t][1], tri_2[..., t][1], tri_3[..., t][1]])
                c_1 = np.array([tri_1[..., t][2], tri_2[..., t][2], tri_3[..., t][2]])

                dist_12 = np.linalg.norm(a_1 - b_1)
                dist_13 = np.linalg.norm(a_1 - c_1)
                dist_23 = np.linalg.norm(b_1 - c_1)

                if min([dist_12, dist_13, dist_23]) > 1e-3:
                    ax.plot(
                        tri_1[..., t][[0, 1, 2, 0]],
                        tri_2[..., t][[0, 1, 2, 0]],
                        tri_3[..., t][[0, 1, 2, 0]],
                        c=COLORS[0],
                        lw=2,
                        linestyle="--",
                    )
                    ax.scatter(
                        tri_1[..., t][[0, 1, 2, 0]],
                        tri_2[..., t][[0, 1, 2, 0]],
                        tri_3[..., t][[0, 1, 2, 0]],
                        c=COLORS[0],
                        s=5,
                    )

    ax.scatter(
        phi_A1_pred,
        phi_B1_pred,
        phi_C1_pred,
        c=COLORS[1],
        s=5,
    )
    ax.scatter(
        phi_A2_pred,
        phi_B2_pred,
        phi_C2_pred,
        c="#ED7A12",
        s=5,
    )

    k = np.argsort(phi_A1_pred)[int(len(phi_A1_pred) / 6)]

    ax.plot(
        [phi_A1_pred[k], phi_A2_pred[k]],
        [phi_B1_pred[k], phi_B2_pred[k]],
        [phi_C1_pred[k], phi_C2_pred[k]],
        "--",
        c="#E2E01D",
        lw=2,
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
    out_file = os.path.join(PLOT_DIR, f"{out_name}_post.{format}")
    print(out_file + " saved ...")

    fig.savefig(out_file, dpi=300, transparent=True, bbox_inches="tight")
