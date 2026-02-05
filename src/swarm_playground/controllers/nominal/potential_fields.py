import numpy as np
from utils.geometry import as3


class PotentialFieldStrategy:
    """
    Computes repulsion vectors from obstacles (spheres and walls) based on a
    reciprocal potential field: U = 1/dist - 1/trigger.
    """

    def __init__(self, cfg):
        self.rep_enabled = bool(cfg.get("controller.nominal.repulsion.enabled", True))
        self.rep_range_extra = float(
            cfg.get("controller.nominal.repulsion.range_extra", 2.5)
        )
        self.rep_gain_2d = float(cfg.get("controller.nominal.repulsion.gain_2d", 12.0))
        self.rep_gain_3d = float(cfg.get("controller.nominal.repulsion.gain_3d", 10.0))
        self.agent_radius = float(cfg.get("controller.cbf.agent_radius", 0.6))

    def compute_repulsion(self, x, threats, is_2d):
        """
        Calculates the summation of repulsion vectors from all visible threats.
        """
        if not self.rep_enabled:
            return np.zeros(3)

        u_repulse = np.zeros(3)
        for o in threats:
            kind = o.get("kind", "sphere")

            if kind in ["sphere", "agent"]:
                rel = x - as3(o["pos"])
                dist = np.linalg.norm(rel) + 1e-9
                trigger = float(o.get("r", self.agent_radius)) + self.rep_range_extra
                if dist >= trigger:
                    continue

                # Gradient of the reciprocal potential
                strength = (1.0 / dist - 1.0 / trigger) * (1.0 / (dist**2))
                direction = rel / dist

            elif kind == "wall":
                # Distance to half-plane: n^T(x - p)
                dist = float(np.dot(as3(o["normal"]), x - as3(o["pos"]))) + 1e-9
                trigger = self.rep_range_extra  # No radius for walls
                if dist >= trigger:
                    continue

                strength = (1.0 / dist - 1.0 / trigger) * (1.0 / (dist**2))
                direction = as3(o["normal"])
            else:
                continue

            gain = self.rep_gain_3d if not is_2d else self.rep_gain_2d
            u_repulse += gain * strength * direction

        return u_repulse
