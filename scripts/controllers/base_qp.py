import numpy as np
from cvxopt import matrix, solvers


def solve_qp_safe(P, q, G, b):
    solvers.options["show_progress"] = False
    try:
        sol = solvers.qp(matrix(P), matrix(q), matrix(G), matrix(b))
        if sol["status"] == "optimal":
            return np.array(sol["x"]).flatten()
        return None
    except:
        return None
