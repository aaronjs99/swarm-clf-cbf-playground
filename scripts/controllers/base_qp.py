import numpy as np
from cvxopt import matrix, solvers


def solve_qp_safe(P, q, G, b):
    solvers.options["show_progress"] = False
    try:
        # Convert to CVXOPT format
        P_opt = matrix(P.astype(float))
        q_opt = matrix(q.astype(float))
        G_opt = matrix(G.astype(float))
        b_opt = matrix(b.astype(float))

        sol = solvers.qp(P_opt, q_opt, G_opt, b_opt)
        if sol["status"] == "optimal":
            return np.array(sol["x"]).flatten()
    except Exception:
        pass
    return None
