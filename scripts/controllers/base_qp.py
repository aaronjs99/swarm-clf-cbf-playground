import numpy as np
from cvxopt import matrix, solvers


def solve_qp_safe(P, q, G, b, a_max, slack_weight=None, fallback_zero_on_fail=True):
    """
    QP with optional softening of inequality constraints via nonnegative slacks.

    Minimize: 1/2 x^T Pbar x + qbar^T x
      where x = [u; s], s >= 0

    Subject to:
      G u - s <= b     (relax each inequality)
      -s <= 0          (i.e. s >= 0)

    Returns:
      u, delta where delta is max slack (proxy for violation pressure)
    """
    solvers.options["show_progress"] = False

    G = np.asarray(G, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)

    m = G.shape[0]
    dim_u = len(q)

    use_slack = (slack_weight is not None) and (slack_weight > 0) and (m > 0)

    if not use_slack:
        # Original hard-constraint solve
        try:
            sol = solvers.qp(
                matrix(P.astype(float)),
                matrix(q.astype(float)),
                matrix(G.astype(float)),
                matrix(b.astype(float)),
            )
            if sol["status"] == "optimal":
                u = np.array(sol["x"]).flatten()
                return u, 0.0
        except Exception:
            pass

        if fallback_zero_on_fail:
            return np.zeros(dim_u), 1.0
        return np.zeros(dim_u), 1.0

    # Build augmented problem in x = [u; s]
    # Pbar
    Pbar = np.zeros((dim_u + m, dim_u + m), dtype=float)
    Pbar[:dim_u, :dim_u] = P
    Pbar[dim_u:, dim_u:] = float(slack_weight) * np.eye(m)

    # qbar
    qbar = np.zeros(dim_u + m, dtype=float)
    qbar[:dim_u] = q

    # Inequalities:
    # (1) G u - I s <= b
    G1 = np.hstack([G, -np.eye(m)])
    b1 = b

    # (2) -s <= 0  (i.e. s >= 0)
    G2 = np.hstack([np.zeros((m, dim_u)), -np.eye(m)])
    b2 = np.zeros(m)

    Gbar = np.vstack([G1, G2])
    bbar = np.concatenate([b1, b2])

    try:
        sol = solvers.qp(matrix(Pbar), matrix(qbar), matrix(Gbar), matrix(bbar))
        if sol["status"] == "optimal":
            x = np.array(sol["x"]).flatten()
            u = x[:dim_u]
            s = x[dim_u:]
            delta = float(np.max(s)) if s.size else 0.0
            return u, delta
    except Exception:
        pass

    if fallback_zero_on_fail:
        # Better-than-zero fallback: at least saturate toward nominal direction
        # (keeps you from "coasting into death" at high speed)
        u_fallback = -q  # since q = -u_nom in your caller
        n = np.linalg.norm(u_fallback)
        if n > 1e-9:
            u_fallback = (u_fallback / n) * min(a_max, n)
        return u_fallback, 1.0

    return np.zeros(dim_u), 1.0
