import numpy as np
from utils.config import get as dget
from utils.geometry import as3
from controllers.constraints import LinearConstraint
from controllers.cbf.ecbf import ECBFRelativeDegree2
from controllers.safety_filter import BrakingSafetyFilter


class ConstraintBuilder:
    def __init__(self, cfg, manager, a_max):
        self.cfg = cfg
        self.manager = manager
        self.a_max = a_max

        # CBF
        k1 = float(dget(cfg, "controller.cbf.k1", 10.0))
        k2 = float(dget(cfg, "controller.cbf.k2", 10.0))
        self.ecbf = ECBFRelativeDegree2(k1=k1, k0=k2)

        self.buffer_obstacles = float(
            dget(cfg, "controller.cbf.buffer_obstacles", 0.65)
        )
        self.buffer_agents = float(dget(cfg, "controller.cbf.buffer_agents", 0.65))
        self.agent_radius = float(dget(cfg, "controller.cbf.agent_radius", 0.6))

        # Safety Filter
        self.brake_filter = BrakingSafetyFilter(
            enabled=bool(dget(cfg, "controller.braking.enabled", False)),
            margin=float(dget(cfg, "controller.braking.margin", 0.0)),
            activate_h=float(dget(cfg, "controller.braking.activate_h", 0.0)),
            gain=float(dget(cfg, "controller.braking.gain", 2.0)),
            clamp_v_close=float(dget(cfg, "controller.braking.clamp_v_close", 50.0)),
        )

        self.last_brake_active = False

    def build_threats(self, obstacles, agent_idx, all_agents):
        threats = list(obstacles)
        for i, a in enumerate(all_agents):
            if i == agent_idx:
                continue
            threats.append(
                {
                    "pos": as3(a["pos"]),
                    "r": self.agent_radius,
                    "vel": as3(a["vel"]),
                    "kind": "agent",
                    "id": i,
                }
            )
        return threats

    def add_constraints(self, x, v, threats, is_2d):
        self.manager.clear()

        # 1. Limits (Hard)
        self._add_limit_constraints(is_2d)

        # 2. Obstacle & Agent Safety
        self.last_brake_active = False
        for o in threats:
            self._add_safety_constraint(x, v, o)

    def _add_limit_constraints(self, is_2d):
        for i in range(3):
            limit = 0.0 if (is_2d and i == 2) else self.a_max
            # u_i <= limit
            row_p = np.zeros(3)
            row_p[i] = 1.0
            self.manager.add_constraint(
                LinearConstraint(G=row_p, b=np.array([limit]), hard=True)
            )
            # -u_i <= limit
            row_n = np.zeros(3)
            row_n[i] = -1.0
            self.manager.add_constraint(
                LinearConstraint(G=row_n, b=np.array([limit]), hard=True)
            )

    def _add_safety_constraint(self, x, v, o):
        kind = o.get("kind", "sphere")

        if kind in ["sphere", "agent"]:
            r_other = float(o.get("r", self.agent_radius))
            margin = self.buffer_agents if kind == "agent" else self.buffer_obstacles
            d_safe = self.agent_radius + r_other + margin

            # ECBF
            c_ecbf, _ = self.ecbf.constraint_sphere_distance_squared(
                x, v, o["pos"], o.get("vel", np.zeros(3)), d_safe
            )
            self.manager.add_constraint(c_ecbf)

            # Braking
            c_brake, info = self.brake_filter.get_constraint(
                x, v, o["pos"], o.get("vel", np.zeros(3)), d_safe, self.a_max
            )
            if c_brake:
                self.manager.add_constraint(c_brake)
                self.last_brake_active = True

        elif kind == "wall":
            n = as3(o["normal"])
            p = as3(o["pos"])
            offset = self.agent_radius + self.buffer_obstacles

            c_wall, _ = self.ecbf.constraint_halfspace_wall(x, v, n, p, offset)
            self.manager.add_constraint(c_wall)
