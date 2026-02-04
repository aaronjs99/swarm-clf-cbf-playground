import numpy as np


def _as3(x):
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size == 2:
        return np.array([x[0], x[1], 0.0], dtype=float)
    if x.size == 3:
        return x
    raise ValueError("Expected a 2D or 3D vector.")


class ECBFRelativeDegree2:
    """
    Exponential Control Barrier Function (ECBF) for relative-degree-2 barriers.

    Enforces:
        h_ddot + k1 * h_dot + k0 * h >= 0

    Returned in QP form:
        G u <= b
    """

    def __init__(self, k1: float, k0: float):
        self.k1 = float(k1)
        self.k0 = float(k0)

    def constraint_sphere_distance_squared(
        self,
        x,
        v,
        other_pos,
        other_vel,
        d_safe: float,
    ):
        """
        h = ||r||^2 - d_safe^2
        r = x - other_pos
        v_rel = v - other_vel

        For a double integrator for THIS agent, assume other accel is 0,
        so:
          h_dot  = 2 r^T v_rel
          h_ddot = 2 v_rel^T v_rel + 2 r^T u
        """
        x = _as3(x)
        v = _as3(v)
        other_pos = _as3(other_pos)
        other_vel = _as3(other_vel)

        r = x - other_pos
        v_rel = v - other_vel

        h = float(np.dot(r, r) - d_safe * d_safe)
        h_dot = float(2.0 * np.dot(r, v_rel))
        h_ddot_no_u = float(2.0 * np.dot(v_rel, v_rel))

        # h_ddot + k1 h_dot + k0 h >= 0
        # 2 r^T u >= -(h_ddot_no_u + k1 h_dot + k0 h)
        # -2 r^T u <= h_ddot_no_u + k1 h_dot + k0 h
        G_row = -2.0 * r
        b_val = h_ddot_no_u + self.k1 * h_dot + self.k0 * h

        return G_row, float(b_val), {"h": h, "h_dot": h_dot}

    def constraint_halfspace_wall(
        self,
        x,
        v,
        wall_normal,
        wall_point,
        offset: float,
    ):
        """
        Wall as half-space:
          h = n^T (x - p) - offset

        For double integrator:
          h_dot  = n^T v
          h_ddot = n^T u
        """
        x = _as3(x)
        v = _as3(v)
        n = _as3(wall_normal)
        p = _as3(wall_point)

        h = float(np.dot(n, x - p) - offset)
        h_dot = float(np.dot(n, v))

        # n^T u + k1 h_dot + k0 h >= 0
        # -n^T u <= k1 h_dot + k0 h
        G_row = -n
        b_val = self.k1 * h_dot + self.k0 * h

        return G_row, float(b_val), {"h": h, "h_dot": h_dot}
