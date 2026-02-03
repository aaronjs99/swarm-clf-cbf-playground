# scripts/controllers/base_qp.py
import numpy as np
from cvxopt import matrix, solvers


def solve_qp_safe(P, q, G, b, a_max):
    solvers.options["show_progress"] = False
    n = P.shape[0]

    # We add a slack variable 'delta' to the optimization:
    # min 0.5 * u'Pu + q'u + 1e6 * delta^2
    # s.t. G u <= b + delta
    #      -a_max <= u <= a_max

    # Augmenting P and q for the slack variable
    P_aug = np.zeros((n + 1, n + 1))
    P_aug[:n, :n] = P
    P_aug[n, n] = 1e8  # Massive penalty for safety violations

    q_aug = np.zeros(n + 1)
    q_aug[:n] = q

    # Building constraints: G_aug * [u; delta] <= b_aug
    # 1. Safety: G*u - delta <= b
    G_safety = np.hstack([G, -np.ones((G.shape[0], 1))])

    # 2. Box Constraints: u <= a_max, -u <= a_max, -delta <= 0
    G_box = np.zeros((2 * n + 1, n + 1))
    for i in range(n):
        G_box[i, i] = 1.0  # u_i <= a_max
        G_box[i + n, i] = -1.0  # -u_i <= a_max
    G_box[-1, -1] = -1.0  # -delta <= 0

    G_total = np.vstack([G_safety, G_box])

    b_total = np.concatenate([b, np.ones(n) * a_max, np.ones(n) * a_max, [0.0]])

    try:
        sol = solvers.qp(matrix(P_aug), matrix(q_aug), matrix(G_total), matrix(b_total))
        if sol["status"] in ["optimal", "unknown"]:
            return np.array(sol["x"]).flatten()[:n]
    except Exception:
        pass
    return np.zeros(n)  # Absolute fallback: Kill thrust
