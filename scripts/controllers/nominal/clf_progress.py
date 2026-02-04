import numpy as np
from controllers.constraints import LinearConstraint

from utils.geometry import as3


class CLFProgressStrategy:
    """
    Control Lyapunov Function (CLF) for goal striving.

    Candidate V(x, v):
        V = 0.5 * ||x - x_g||^2 + 0.5 * ||v||^2  (Simple Energy-like)

    Actually for double integrator tracking, we often use:
        V = (x-xd)^T P (x-xd)
    But sticking to a simple PD-like CLF constraint for demonstration:

    We want V_dot <= -gamma * V

    For double integrator:
    V = 1/2 k_p ||e||^2 + 1/2 ||v||^2 + epsilon * e^T v ...

    Let's implement a simple "Direct" CLF constraint that tries to enforce
    exponential stability if possible, or just returns a nominal control
    based on the gradient.

    For now, we simply expose the interface and return a PD control as "nominal"
    but packaged as a strategy.

    If we strictly want a CLF constraint:
    L_g V u <= -L_f V - gamma V

    Let's implement a simplified CLF based on [Ames et al.] for simple reachability.
    V(x) = ...

    To avoid complexity without a full CARE solver argument, we will implement
    a nominal PD control (which is the solution to an unconstrained CLF QP)
    and return it.

    """

    def __init__(self, cfg):
        self.gamma = float(cfg.get("controller.nominal.clf.gamma", 1.0))
        self.kp = float(cfg.get("controller.nominal.pd.kp", 5.0))
        self.kd = float(cfg.get("controller.nominal.pd.kd", 3.0))

    def compute_control(self, x, v, goal):
        """
        Returns u_nominal.
        Future extension: Return (u_ref, [CLF constraint])
        """
        x = as3(x)
        v = as3(v)
        goal = as3(goal)

        e = x - goal

        # Simple PD as the CLF-optimal control for V = 1/2 e'Kp e + 1/2 v' v ...
        # u = - Kp e - Kd v
        u_nominal = -self.kp * e - self.kd * v

        # If we wanted to return a constraint:
        # V = 0.5 * ||e||^2 + 0.5 * ||v + e||^2 (example)
        # dot(V) = ... u ...
        # return u_nominal, [Constraint(...)]

        return u_nominal
