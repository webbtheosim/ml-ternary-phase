import numpy as np


def fill_prob_tensor(yr, yr_prob):
    """
    Converts raw composition representation to probabilities.
    """
    for i in range(0, 9, 3):
        yr_prob[:, i : i + 2] = yr[:, i : i + 2]
        zero_sum_rows = np.sum(yr[:, i : i + 2], axis=1) == 0
        # equal probability to find A, B and C
        yr_prob[zero_sum_rows, i : i + 2] = 1 / 3
        yr_prob[:, i + 2] = 1 - yr_prob[:, i : i + 2].sum(axis=1)

    yr_prob[:, 9] = yr[:, 2]
    yr_prob[:, 10] = yr[:, 5]
    yr_prob[:, 11] = yr[:, 8]  
    
    # Output vector indices
    # 0, 1, 2 are phi_a^alpha, phi_b^alpha, phi_c^alpha (c depends on a and b, not considered)
    # 3, 4, 5 are phi_a^beta, phi_b^beta, phi_c^beta
    # 6, 7, 8 are phi_a^gamma, phi_b^gamma, phi_c^gamma
    # 9, 10, 11 are w^alpha, w^beta, w^gamma
    return yr_prob
