import numpy as np
from utils.geometry import as3


class CLFProgressStrategy:
    """
    Minimal "CLF-like" progress strategy.

    In many CLF-QP stacks, the unconstrained minimizer of a quadratic CLF objective
    looks like PD tracking for a double integrator. This class provides that same
    nominal in a compact form.

    It does not return CLF constraints. It only returns a nominal acceleration.

    If you later want a true CLF-QP layer:
      - define V(x, v) = [e; v]^T P [e; v] with P from a CARE/LQR design
      - compute LfV and LgV for the double integrator
      - add: LgV u <= -LfV - gamma V + delta
    """

    def __init__(self, cfg, a_max: float = 30.0):
        self.kp = float(cfg.get("controller.nominal.pd.kp", 5.0))
        self.kd = float(cfg.get("controller.nominal.pd.kd", 3.0))
        self.a_max = float(a_max)

    def compute_control(self, x, v, goal, is_2d: bool) -> np.ndarray:
        x = as3(x)
        v = as3(v)
        goal = as3(goal)

        e = goal - x
        u = self.kp * e - self.kd * v

        n = float(np.linalg.norm(u))
        if n > self.a_max:
            u = (u / (n + 1e-12)) * self.a_max

        if is_2d:
            u[2] = 0.0

        return u
