import numpy as np

from controllers.admm_engine import ADMMSolver
from controllers.base_qp import solve_qp_safe
from controllers.state_manager import SwarmStateManager
from controllers.network import SwarmNetwork
from utils.geometry import project_to_tangent_plane_3d, get_tangent_2d


def _dget(d, path, default=None):
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


class SwarmController:
    def __init__(self, cfg, a_max=15.0):
        self.cfg = cfg

        # --- CBF params ---
        self.k1 = float(_dget(cfg, "controller.cbf.k1", 10.0))
        self.k2 = float(_dget(cfg, "controller.cbf.k2", 10.0))
        self.buffer_obstacles = float(
            _dget(cfg, "controller.cbf.buffer_obstacles", 0.65)
        )
        self.buffer_agents = float(_dget(cfg, "controller.cbf.buffer_agents", 0.65))
        self.agent_radius = float(_dget(cfg, "controller.cbf.agent_radius", 0.6))

        # --- Limits ---
        self.a_max = float(a_max)
        self.nom_clip_fraction = float(_dget(cfg, "limits.nom_clip_fraction", 0.8))

        # --- QP ---
        self.slack_weight = float(_dget(cfg, "controller.qp.slack_weight", 1.0e4))
        self.fallback_zero_on_fail = bool(
            _dget(cfg, "controller.qp.fallback_zero_on_fail", True)
        )

        # --- ADMM ---
        self.use_admm = bool(_dget(cfg, "controller.admm.enabled", True))
        rho = float(_dget(cfg, "controller.admm.rho", 0.1))
        iters = int(_dget(cfg, "controller.admm.iters", 2))
        self.solver = ADMMSolver(rho=rho, num_iters=iters)
        self.state_manager = SwarmStateManager()

        # --- Connectivity ---
        self.connectivity_enabled = bool(
            _dget(cfg, "controller.connectivity.enabled", True)
        )
        k_neighbors = int(_dget(cfg, "controller.connectivity.k_neighbors", 3))
        self.network = SwarmNetwork(k_neighbors=k_neighbors)
        self.conn_gain = float(_dget(cfg, "controller.connectivity.gain", 5.0))
        self.l2_threshold = float(
            _dget(cfg, "controller.connectivity.l2_threshold", 0.5)
        )
        self.inject_conn_into_nominal = bool(
            _dget(cfg, "controller.connectivity.inject_into_nominal", False)
        )

        # --- Nominal ---
        self.nom_enabled = bool(_dget(cfg, "controller.nominal.enabled", True))
        self.v_damp_far = float(_dget(cfg, "controller.nominal.v_damp_far", 2.0))
        self.pd_switch_dist = float(
            _dget(cfg, "controller.nominal.pd.switch_dist", 0.1)
        )
        self.pd_kp = float(_dget(cfg, "controller.nominal.pd.kp", 5.0))
        self.pd_kd = float(_dget(cfg, "controller.nominal.pd.kd", 3.0))

        self.rep_enabled = bool(
            _dget(cfg, "controller.nominal.repulsion.enabled", True)
        )
        self.rep_range_extra = float(
            _dget(cfg, "controller.nominal.repulsion.range_extra", 2.5)
        )
        self.rep_gain_2d = float(
            _dget(cfg, "controller.nominal.repulsion.gain_2d", 12.0)
        )
        self.rep_gain_3d = float(
            _dget(cfg, "controller.nominal.repulsion.gain_3d", 10.0)
        )
        self.rep_tangent_bias = bool(
            _dget(cfg, "controller.nominal.repulsion.tangent_bias", True)
        )

        # --- Debug/state ---
        self.last_l2 = 0.0
        self.last_delta = 0.0

    def compute_control(self, x, v, goal, obstacles, agent_idx, all_agents):
        is_2d = abs(x[2]) < 1e-5 and abs(goal[2]) < 1e-5

        if agent_idx == 0 and self.connectivity_enabled:
            self.network.update_topology(all_agents)

        threats = self._build_threats(obstacles, agent_idx, all_agents)

        # connectivity term
        u_conn = np.zeros(3)
        if self.connectivity_enabled and len(all_agents) > 1:
            conn_grad, l2 = self.network.get_connectivity_gradient(
                agent_idx, all_agents
            )
            if agent_idx == 0:
                self.last_l2 = l2
            if l2 < self.l2_threshold:
                u_conn = self.conn_gain * conn_grad

        # nominal
        u_nom = self._get_nominal_control(x, v, goal, threats, is_2d)
        if self.inject_conn_into_nominal:
            u_nom = u_nom + u_conn

        clip = self.nom_clip_fraction * self.a_max
        u_nom = np.clip(u_nom, -clip, clip)

        # local QP
        P = np.eye(3)
        q = -u_nom
        G, b = self._get_constraints(x, v, threats, is_2d)

        if self.use_admm:
            u_opt, delta = self.solver.step(
                agent_idx,
                u_nom,
                P,
                q,
                np.asarray(G),
                np.asarray(b),
                self.state_manager,
                self.network,
                self.a_max,
                slack_weight=self.slack_weight,
                fallback_zero_on_fail=self.fallback_zero_on_fail,
            )
        else:
            u_opt, delta = solve_qp_safe(
                P,
                q,
                np.asarray(G),
                np.asarray(b),
                self.a_max,
                slack_weight=self.slack_weight,
                fallback_zero_on_fail=self.fallback_zero_on_fail,
            )

        self.last_delta = float(delta)

        if is_2d:
            u_opt[2] = 0.0
        return u_opt

    def _build_threats(self, obstacles, agent_idx, all_agents):
        threats = list(obstacles)
        for i, a in enumerate(all_agents):
            if i == agent_idx:
                continue
            threats.append(
                {
                    "pos": a["pos"],
                    "r": self.agent_radius,
                    "vel": a["vel"],
                    "kind": "agent",
                    "id": i,
                }
            )
        return threats

    def _get_nominal_control(self, x, v, goal, threats, is_2d):
        rel_goal = goal - x
        dist_goal = np.linalg.norm(rel_goal)

        # Goal tracking (PD + Far-field push)
        if dist_goal > self.pd_switch_dist:
            u_goal = (rel_goal / (dist_goal + 1e-9)) * self.a_max - self.v_damp_far * v
        else:
            u_goal = self.pd_kp * rel_goal - self.pd_kd * v

        if not (self.nom_enabled and self.rep_enabled):
            return np.clip(u_goal, -self.a_max, self.a_max)

        # Repulsion potential field
        u_repulse = np.zeros(3)
        for o in threats:
            kind = o.get("kind", "sphere")

            if kind in ["sphere", "agent"]:
                rel = x - o["pos"]
                dist = np.linalg.norm(rel) + 1e-9
                trigger = o["r"] + self.rep_range_extra
                if dist >= trigger:
                    continue

                # Gradient of the reciprocal potential
                strength = (1.0 / dist - 1.0 / trigger) * (1.0 / (dist**2))
                direction = rel / dist

            elif kind == "wall":
                # Distance to half-plane: n^T(x - p)
                dist = np.dot(o["normal"], x - o["pos"]) + 1e-9
                trigger = self.rep_range_extra  # No radius for walls
                if dist >= trigger:
                    continue

                strength = (1.0 / dist - 1.0 / trigger) * (1.0 / (dist**2))
                direction = o["normal"]  # Push along the wall normal

            # Apply gain based on dimensionality
            gain = self.rep_gain_3d if not is_2d else self.rep_gain_2d
            u_repulse += gain * strength * direction

        u_nom = u_goal + u_repulse
        norm = np.linalg.norm(u_nom)
        return (u_nom / norm) * self.a_max if norm > self.a_max else u_nom

    def _get_constraints(self, x, v, threats, is_2d):
        G, b = [], []

        for o in threats:
            kind = o.get("kind", "sphere")
            buffer = self.buffer_agents if kind == "agent" else self.buffer_obstacles

            if kind in ["sphere", "agent"]:
                # Standard Sphere CBF
                rel = x - o["pos"]
                rel_vel = v - o["vel"]
                h = np.linalg.norm(rel) ** 2 - (o["r"] + buffer) ** 2
                h_dot = 2.0 * np.dot(rel, rel_vel)

                grad_h = -2.0 * rel
                # k1*h_dot + k2*h + ||rel_vel||^2 part is specific to double integrator safety
                # G*u <= b
                G.append(grad_h)
                b.append(self.k1 * h_dot + self.k2 * h + 2.0 * np.dot(rel_vel, rel_vel))

            elif kind == "wall":
                # Half-plane CBF: n^T(x - p) - buffer >= 0
                n = o["normal"]  # Unit vector pointing into safe space
                p = o["pos"]  # Point on the wall

                h = np.dot(n, x - p) - buffer
                h_dot = np.dot(n, v)

                G.append(-n)  # Since G*u <= b, and we want n*u >= ...
                b.append(self.k1 * h_dot + self.k2 * h)

        # accel bounds (including z=0 in 2D)
        for i in range(3):
            r_p, r_n = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
            r_p[i], r_n[i] = 1.0, -1.0
            limit = 0.0 if (is_2d and i == 2) else self.a_max
            G.extend([r_p, r_n])
            b.extend([limit, limit])

        return G, b
