import numpy as np
from cvxopt import matrix, solvers


def solve_qp_safe(P, q, G, b, a_max, slack_weight=1.0e4, fallback_zero_on_fail=True):
    solvers.options["show_progress"] = False
    n = P.shape[0]

    P_aug = np.zeros((n + 1, n + 1))
    P_aug[:n, :n] = P
    P_aug[n, n] = float(slack_weight)

    q_aug = np.zeros(n + 1)
    q_aug[:n] = q

    G = np.asarray(G, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)

    if G.ndim != 2 or G.shape[1] != n:
        raise ValueError(f"G must be (m,n) with n={n}, got {G.shape}")

    # Safety: G*u - delta <= b
    G_safety = np.hstack([G, -np.ones((G.shape[0], 1))])

    # Box constraints on u and delta >= 0:
    # u_i <= a_max
    # -u_i <= a_max
    # -delta <= 0
    G_box = np.zeros((2 * n + 1, n + 1))
    for i in range(n):
        G_box[i, i] = 1.0
        G_box[i + n, i] = -1.0
    G_box[-1, -1] = -1.0

    G_total = np.vstack([G_safety, G_box])
    b_total = np.concatenate([b, np.ones(n) * a_max, np.ones(n) * a_max, [0.0]])

    try:
        sol = solvers.qp(matrix(P_aug), matrix(q_aug), matrix(G_total), matrix(b_total))
        if sol["status"] in ["optimal", "unknown"]:
            x = np.array(sol["x"]).flatten()
            u = x[:n]
            delta = float(x[n])
            return u, delta
    except Exception:
        pass

    # Always return something
    if fallback_zero_on_fail:
        return np.zeros(n), 0.0
    return np.zeros(n), 0.0
