import torch
import torch.nn as nn

# Output vector indices
# 0, 1, 2 are phi_a^alpha, phi_b^alpha, phi_c^alpha (c depends on a and b, not considered)
# 3, 4, 5 are phi_a^beta, phi_b^beta, phi_c^beta
# 6, 7, 8 are phi_a^gamma, phi_b^gamma, phi_c^gamma
# 9, 10, 11 are w^alpha, w^beta, w^gamma

def mu_compute(X, vol_tensor):
    """
    Calculates chemical potentials (mu_a, mu_b, mu_c)
    using Flory-Huggins theory.
    """

    EPS = 1e-7  # Small value to prevent division by zero

    phi_A = torch.clamp(vol_tensor[:, 0], min=EPS)
    phi_B = torch.clamp(vol_tensor[:, 1], min=EPS)
    phi_C = torch.clamp(vol_tensor[:, 2], min=EPS)

    # Calculate chemical potential of component A (mu_a)
    mu_a = (
        torch.log(phi_A)
        + 1
        - phi_A
        - (X[:, 0] / torch.clamp(X[:, 2], min=EPS)) * phi_B
        - (X[:, 0] / torch.clamp(X[:, 1], min=EPS)) * phi_C
        + X[:, 0]
        * (
            phi_B**2 * X[:, 4]
            + phi_C**2 * X[:, 3]
            + phi_B * phi_C * (X[:, 4] + X[:, 3] - X[:, 5])
        )
    )

    # Calculate chemical potential of component B (mu_b)
    mu_b = (
        torch.log(phi_B)
        + 1
        - phi_B
        - (X[:, 2] / torch.clamp(X[:, 0], min=EPS)) * phi_A
        - (X[:, 2] / torch.clamp(X[:, 1], min=EPS)) * phi_C
        + X[:, 2]
        * (
            phi_A**2 * X[:, 4]
            + phi_C**2 * X[:, 5]
            + phi_A * phi_C * (X[:, 4] + X[:, 5] - X[:, 3])
        )
    )

    # Calculate chemical potential of component C (mu_c)
    mu_c = (
        torch.log(phi_C)
        + 1
        - phi_C
        - (X[:, 1] / torch.clamp(X[:, 0], min=EPS)) * phi_A
        - (X[:, 1] / torch.clamp(X[:, 2], min=EPS)) * phi_B
        + X[:, 1]
        * (
            phi_B**2 * X[:, 5]
            + phi_A**2 * X[:, 3]
            + phi_A * phi_B * (X[:, 5] + X[:, 3] - X[:, 4])
        )
    )

    return mu_a[0], mu_b[0], mu_c[0]


def energy_min(x_in, r_outputs):
    """
    The energy minimization loss based on predicted composition
    and chemical potentials.
    """

    # Calculate chemical potentials for each component and each phase
    mu_a1, mu_b1, mu_c1 = mu_compute(x_in, r_outputs[:, 0:3])
    mu_a2, mu_b2, mu_c2 = mu_compute(x_in, r_outputs[:, 3:6])
    mu_a3, mu_b3, mu_c3 = mu_compute(x_in, r_outputs[:, 6:9])

    # Calculate total free energy of the equilibrium system
    losses = (
        mu_a1 * r_outputs[:, 0] * r_outputs[:, -3]
        + mu_b1 * r_outputs[:, 1] * r_outputs[:, -3]
        + mu_c1 * r_outputs[:, 2] * r_outputs[:, -3]
        + mu_a2 * r_outputs[:, 3] * r_outputs[:, -2]
        + mu_b2 * r_outputs[:, 4] * r_outputs[:, -2]
        + mu_c2 * r_outputs[:, 5] * r_outputs[:, -2]
        + mu_a3 * r_outputs[:, 6] * r_outputs[:, -1]
        + mu_b3 * r_outputs[:, 7] * r_outputs[:, -1]
        + mu_c3 * r_outputs[:, 8] * r_outputs[:, -1]
    )

    return torch.mean(losses)


def mu_loss_fn(x_input, r_outputs, c_outputs):
    """
    The chemical potential difference loss based on
    predicted composition and classification output.
    """

    # Get indices of maximum class probabilities
    max_indices = torch.argmax(c_outputs, dim=1)
    mask2 = max_indices == 1
    mask3 = max_indices == 2
    losses = torch.zeros_like(x_input[:, 0])

    # Case 1: Two valid phases
    if mask2.any():
        mu_a1, mu_b1, mu_c1 = mu_compute(x_input[mask2], r_outputs[mask2, 0:3])
        mu_a2, mu_b2, mu_c2 = mu_compute(x_input[mask2], r_outputs[mask2, 3:6])
        losses[mask2] = (
            torch.log1p((mu_a1 - mu_a2).pow(2)) / 3
            + torch.log1p((mu_b1 - mu_b2).pow(2)) / 3
            + torch.log1p((mu_c1 - mu_c2).pow(2)) / 3
        )

    # Case 2: Three valid phases
    if mask3.any():
        mu_a1, mu_b1, mu_c1 = mu_compute(x_input[mask3], r_outputs[mask3, 0:3])
        mu_a2, mu_b2, mu_c2 = mu_compute(x_input[mask3], r_outputs[mask3, 3:6])
        mu_a3, mu_b3, mu_c3 = mu_compute(x_input[mask3], r_outputs[mask3, 6:9])

        losses[mask3] = (
            torch.log1p((mu_a1 - mu_a2).pow(2)) / 9
            + torch.log1p((mu_a1 - mu_a3).pow(2)) / 9
            + torch.log1p((mu_a2 - mu_a3).pow(2)) / 9
            + torch.log1p((mu_b1 - mu_b2).pow(2)) / 9
            + torch.log1p((mu_b1 - mu_b3).pow(2)) / 9
            + torch.log1p((mu_b2 - mu_b3).pow(2)) / 9
            + torch.log1p((mu_c1 - mu_c2).pow(2)) / 9
            + torch.log1p((mu_c1 - mu_c3).pow(2)) / 9
            + torch.log1p((mu_c2 - mu_c3).pow(2)) / 9
        )

    # Calculate the mean of non-zero losses, ensuring zero loss when no valid comparisons exist
    return (
        torch.mean(losses[losses != 0])
        if losses.nonzero().size(0) > 0
        else torch.tensor(0.0, device=r_outputs.device)
    )


def split_loss_fn(x_input, r_outputs):
    """
    The mean squared error between the true split and the predicted split.
    """
    final_split = (
        r_outputs[:, 0:2] * r_outputs[:, -3].unsqueeze(1)
        + r_outputs[:, 3:5] * r_outputs[:, -2].unsqueeze(1)
        + r_outputs[:, 6:8] * r_outputs[:, -1].unsqueeze(1)
    )
    split_loss = torch.mean(torch.sum((x_input[:, -2:] - final_split) ** 2, dim=1))

    return split_loss


def wu_loss(r_outputs, c_outputs, r_labels, x_input):
    """
    L1 regression loss (baseline).
    """
    return [
        nn.L1Loss()(
            r_outputs[:, [0, 1, 3, 4, 6, 7, 9, 10, 11]],
            r_labels[:, [0, 1, 3, 4, 6, 7, 9, 10, 11]],
        )
    ]


def pif_loss(r_outputs, c_outputs, r_labels, x_input):
    """
    Augmented physics-informed loss with free energy minimization.
    """
    w_comp = 1.0  # Weight for component loss
    w_f = 0.001  # Weight for free energy loss
    w_mu = 0.01  # Weight for chemical potential difference loss
    w_split = 0.01  # Weight for split loss

    comp_loss = nn.L1Loss()(
        r_outputs[:, [0, 1, 3, 4, 6, 7, 9, 10, 11]],
        r_labels[:, [0, 1, 3, 4, 6, 7, 9, 10, 11]],
    )

    f_loss = energy_min(x_input, r_outputs)

    delta_mu_loss = mu_loss_fn(x_input, r_outputs, c_outputs)

    spl_loss = split_loss_fn(x_input, r_outputs)

    return [
        w_f * f_loss,
        w_comp * comp_loss,
        w_mu * delta_mu_loss,
        w_split * spl_loss,
    ]
