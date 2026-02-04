import numpy as np
from utils.config import get as dget
from controllers.nominal.pd_tracking import PDTrackingStrategy
from utils.geometry import as3


def _smoothstep01(t: float) -> float:
    t = max(0.0, min(1.0, float(t)))
    return t * t * (3.0 - 2.0 * t)


class NominalPolicy:
    """
    Minimal nominal controller for debugging and reliable convergence.

    Key change for realism:
      - When close to the goal, fade only the position term (P), not the damping term (D).
        This prevents "stuck near goal" limit cycles when you add actuator lag, drag,
        speed limits, or noise.
    """

    def __init__(self, cfg, a_max: float):
        self.cfg = cfg
        self.a_max = float(a_max)

        self.enabled = bool(dget(cfg, "controller.nominal.enabled", True))
        self.clip_fraction = float(dget(cfg, "controller.nominal.clip_fraction", 0.8))

        self.fade_dist = float(dget(cfg, "controller.nominal.goal_fade_dist", 0.6))
        self.stop_dist = float(dget(cfg, "controller.nominal.pd.stop_dist", 0.25))
        self.stop_speed = float(dget(cfg, "controller.nominal.pd.stop_speed", 0.2))

        # Optional: add a tiny deadzone to stop hunting due to noise
        self.goal_deadzone = float(dget(cfg, "controller.nominal.goal_deadzone", 0.02))

        self.pd = PDTrackingStrategy(cfg, a_max=self.a_max)

    def compute(self, x, v, goal, agent_idx, threats, is_2d: bool) -> np.ndarray:
        if not self.enabled:
            return np.zeros(3)

        x = as3(x)
        v = as3(v)
        goal = as3(goal)

        vec = (goal - x)[: (2 if is_2d else 3)]
        dist_goal = float(np.linalg.norm(vec))
        speed = float(np.linalg.norm(v[: (2 if is_2d else 3)]))

        # Absolute deadzone: if you're extremely close, just damp velocity
        # (helps with actuator lag + numerical jitter)
        if dist_goal <= self.goal_deadzone:
            u = -2.0 * v
            if is_2d:
                u[2] = 0.0
            return np.clip(u, -self.clip_fraction * self.a_max, self.clip_fraction * self.a_max)

        # Parking brake: if you are basically there and already slow, stop.
        if dist_goal <= self.stop_dist and speed <= self.stop_speed:
            return np.zeros(3)

        # Base PD components
        u_p, u_d, _ = self.pd.compute_components(x, v, goal, is_2d=is_2d)

        # Smooth fade near goal to avoid oscillation / chattering.
        # Important: fade only the P term. Keep damping alive.
        if self.fade_dist > self.stop_dist:
            t = (dist_goal - self.stop_dist) / (self.fade_dist - self.stop_dist)
            fade = _smoothstep01(t)
            fade_min = float(dget(self.cfg, "controller.nominal.pd.fade_min", 0.15))
            fade_eff = max(fade_min, fade)
            u = fade_eff * u_p + u_d
        else:
            u = u_p + u_d

        # Final clip (debug friendly, keeps nominal bounded)
        clip = max(1e-9, self.clip_fraction * self.a_max)
        u = np.clip(u, -clip, clip)

        if is_2d:
            u[2] = 0.0

        return u
