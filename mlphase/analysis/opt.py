import autograd.numpy as anp


def safe_log(x):
    if x <= 0:
        return -1e10
    else:
        return anp.log(x)


def mu_compute_scp(X, y1, y2):
    phi_s = y1
    phi_p = y2

    mu_s = (
        safe_log(phi_s)
        + 1
        - phi_s
        - X[0] / X[2] * phi_p
        - X[0] / X[1] * (1 - phi_s - phi_p)
        + X[0]
        * (
            phi_p**2 * X[4]
            + (1 - phi_s - phi_p) ** 2 * X[3]
            + phi_p * (1 - phi_s - phi_p) * (X[4] + X[3] - X[5])
        )
    )

    mu_p = (
        safe_log(phi_p)
        + 1
        - phi_p
        - X[2] / X[0] * phi_s
        - X[2] / X[1] * (1 - phi_s - phi_p)
        + X[2]
        * (
            phi_s**2 * X[4]
            + (1 - phi_s - phi_p) ** 2 * X[5]
            + phi_s * (1 - phi_s - phi_p) * (X[4] + X[5] - X[3])
        )
    )

    mu_c = (
        safe_log(1 - phi_s - phi_p)
        + 1
        - (1 - phi_s - phi_p)
        - X[1] / X[0] * phi_s
        - X[1] / X[2] * phi_p
        + X[1]
        * (phi_p**2 * X[5] + phi_s**2 * X[3] + phi_s * phi_p * (X[5] + X[3] - X[4]))
    )

    return mu_s, mu_p, mu_c


def stability_condition(phi_s, phi_p, vs, vc, vp, chi_ps, chi_pc, chi_sc):
    return (1 / (vp * phi_p) + 1 / (vc * (1 - phi_s - phi_p)) - 2 * chi_pc) * (
        1 / (vs * phi_s) + 1 / (vc * (1 - phi_s - phi_p)) - 2 * chi_sc
    ) - (1 / (vc * (1 - phi_s - phi_p)) + chi_ps - chi_pc - chi_sc) ** 2


def angle_2phase(params, X):
    s1, p1, s2, p2 = params
    c1 = 1 - s1 - p1
    c2 = 1 - s2 - p2

    mi = anp.array([X[-2], 1 - X[-2] - X[-1], X[-1]])

    v1 = anp.array([s1, c1, p1])
    v2 = anp.array([s2, c2, p2])

    v1_m = v1 - mi
    v2_m = v2 - mi

    norm_v1_m = v1_m / anp.linalg.norm(v1_m)
    norm_v2_m = v2_m / anp.linalg.norm(v2_m)

    cos_angle = anp.dot(norm_v1_m, norm_v2_m)

    return cos_angle


def min_3phase(params, X):
    y1, y2, y3, y4, y5, y6 = params
    s1, p1, c1 = mu_compute_scp(X, y1, y2)
    s2, p2, c2 = mu_compute_scp(X, y3, y4)
    s3, p3, c3 = mu_compute_scp(X, y5, y6)
    return (
        (s1 - s2) ** 2
        + (p1 - p2) ** 2
        + (c1 - c2) ** 2
        + (s1 - s3) ** 2
        + (p1 - p3) ** 2
        + (c1 - c3) ** 2
        + (s2 - s3) ** 2
        + (p2 - p3) ** 2
        + (c2 - c3) ** 2
    )


def min_2phase(params, X):
    y1, y2, y3, y4 = params

    s1, p1, c1 = mu_compute_scp(X, y1, y2)
    s2, p2, c2 = mu_compute_scp(X, y3, y4)

    return (s1 - s2) ** 2 + (p1 - p2) ** 2 + (c1 - c2) ** 2 + angle_2phase(params, X)
