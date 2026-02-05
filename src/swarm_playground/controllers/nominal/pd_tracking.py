import numpy as np
from utils.geometry import as3


class PDTrackingStrategy:
    """
    Plain PD tracking for a double integrator:

        u = k_p (goal - x) - k_d v

    Exposes components so higher-level policies can fade only the P term
    (useful when you add actuator lag / drag and still want convergence).
    """

    def __init__(self, cfg, a_max: float):
        self.kp = float(cfg.get("controller.nominal.pd.kp", 5.0))
        self.kd = float(cfg.get("controller.nominal.pd.kd", 3.0))
        self.a_max = float(a_max)

    def compute_components(self, x, v, goal, is_2d: bool):
        x = as3(x)
        v = as3(v)
        goal = as3(goal)

        e = goal - x
        u_p = self.kp * e
        u_d = -self.kd * v
        u = u_p + u_d

        # Saturate total command (not each component separately)
        n = float(np.linalg.norm(u))
        if n > self.a_max:
            u = (u / (n + 1e-12)) * self.a_max

        if is_2d:
            u_p[2] = 0.0
            u_d[2] = 0.0
            u[2] = 0.0

        return u_p, u_d, u

    def compute(self, x, v, goal, is_2d: bool) -> np.ndarray:
        _, _, u = self.compute_components(x, v, goal, is_2d=is_2d)
        return u
