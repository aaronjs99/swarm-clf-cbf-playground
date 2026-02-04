import numpy as np

from controllers.admm_engine import ADMMSolver
from controllers.base_qp import solve_qp_safe
from controllers.state_manager import SwarmStateManager
from controllers.network import SwarmNetwork

from controllers.cbf.ecbf import ECBFRelativeDegree2
from controllers.safety_filter import BrakingSafetyFilter
from controllers.mpc_layer import SamplingMPCPlanner


def _dget(d, path, default=None):
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _as3(x):
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size == 2:
        return np.array([x[0], x[1], 0.0], dtype=float)
    if x.size == 3:
        return x
    raise ValueError("Expected a 2D or 3D vector.")


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

        # ECBF builder (relative degree 2)
        # We map your existing (k1, k2) to (k1, k0) in the standard form:
        #   h_ddot + k1 h_dot + k0 h >= 0
        self.ecbf = ECBFRelativeDegree2(k1=self.k1, k0=self.k2)

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

        # --- Braking-distance viability filter ---
        self.brake_filter = BrakingSafetyFilter(
            enabled=bool(_dget(cfg, "controller.braking.enabled", False)),
            margin=float(_dget(cfg, "controller.braking.margin", 0.0)),
            activate_h=float(_dget(cfg, "controller.braking.activate_h", 0.0)),
            gain=float(_dget(cfg, "controller.braking.gain", 2.0)),
            clamp_v_close=float(_dget(cfg, "controller.braking.clamp_v_close", 50.0)),
        )

        # --- MPC planning layer ---
        self.mpc_stride = int(_dget(cfg, "controller.mpc.stride", 10))
        self._mpc_counters = {}  # per-agent counter
        self._u_mpc_cache = {}

        self.mpc = SamplingMPCPlanner(
            enabled=bool(_dget(cfg, "controller.mpc.enabled", False)),
            horizon_steps=int(_dget(cfg, "controller.mpc.horizon_steps", 20)),
            dt=float(
                _dget(cfg, "controller.mpc.dt", float(_dget(cfg, "sim.dt", 0.02)))
            ),
            num_samples=int(_dget(cfg, "controller.mpc.num_samples", 64)),
            goal_weight=float(_dget(cfg, "controller.mpc.goal_weight", 1.0)),
            effort_weight=float(_dget(cfg, "controller.mpc.effort_weight", 0.02)),
            collision_weight=float(_dget(cfg, "controller.mpc.collision_weight", 50.0)),
            rng_seed=int(_dget(cfg, "controller.mpc.rng_seed", 0)),
        )

        # --- Debug/state ---
        self.last_l2 = 0.0
        self.last_delta = 0.0
        self.last_brake_active = False

    def compute_control(self, x, v, goal, obstacles, agent_idx, all_agents):
        x = _as3(x)
        v = _as3(v)
        goal = _as3(goal)

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

        # nominal (possibly MPC planned)
        u_nom = self._get_nominal_control_with_mpc(
            x, v, goal, threats, is_2d, agent_idx
        )

        if self.inject_conn_into_nominal:
            u_nom = u_nom + u_conn

        clip = self.nom_clip_fraction * self.a_max
        u_nom = np.clip(u_nom, -clip, clip)

        # local QP
        P = np.eye(3)
        q = -u_nom

        G_soft, b_soft, G_hard, b_hard = self._get_constraints(x, v, threats, is_2d)

        if self.use_admm:
            # ADMMSolver needs updating too, but assuming it just forwards for now
            # We'll need to update admm_engine.py next.
            u_opt, delta = self.solver.step(
                agent_idx,
                u_nom,
                P,
                q,
                np.asarray(G_soft),
                np.asarray(b_soft),
                self.state_manager,
                self.network,
                self.a_max,
                slack_weight=self.slack_weight,
                fallback_zero_on_fail=self.fallback_zero_on_fail,
                G_hard=np.asarray(G_hard),
                b_hard=np.asarray(b_hard),
            )
        else:
            u_opt, delta = solve_qp_safe(
                P,
                q,
                np.asarray(G_soft),
                np.asarray(b_soft),
                self.a_max,
                slack_weight=self.slack_weight,
                fallback_zero_on_fail=self.fallback_zero_on_fail,
                G_hard=np.asarray(G_hard),
                b_hard=np.asarray(b_hard),
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
                    "pos": _as3(a["pos"]),
                    "r": self.agent_radius,
                    "vel": _as3(a["vel"]),
                    "kind": "agent",
                    "id": i,
                }
            )
        return threats

    def _get_nominal_control_with_mpc(self, x, v, goal, threats, is_2d, agent_idx):
        # Base nominal (your existing PD + far-field + repulsion)
        u_base = self._get_nominal_control(x, v, goal, threats, is_2d, agent_idx)

        if not self.mpc.enabled:
            return u_base

        # Compute MPC reference occasionally, cache per-agent
        if agent_idx not in self._mpc_counters:
            self._mpc_counters[agent_idx] = 0

        self._mpc_counters[agent_idx] += 1
        cnt = self._mpc_counters[agent_idx]

        do_plan = (cnt % max(1, self.mpc_stride)) == 0
        if do_plan or (agent_idx not in self._u_mpc_cache):

            def d_safe_fn(o):
                kind = o.get("kind", "sphere")
                if kind == "agent":
                    r_other = float(o.get("r", self.agent_radius))
                    return self.agent_radius + r_other + self.buffer_agents
                if kind == "sphere":
                    r_other = float(o.get("r", 0.0))
                    return self.agent_radius + r_other + self.buffer_obstacles
                if kind == "wall":
                    # offset for wall barrier
                    return self.agent_radius + self.buffer_obstacles
                return self.agent_radius + self.buffer_obstacles

            u_mpc = self.mpc.plan(
                x=x,
                v=v,
                goal=goal,
                threats=threats,
                a_max=self.a_max,
                is_2d=is_2d,
                d_safe_fn=d_safe_fn,
            )
            if u_mpc is None:
                u_mpc = u_base
            self._u_mpc_cache[agent_idx] = u_mpc

        # Blend: MPC provides a longer-horizon “direction”, base provides local shaping.
        blend = float(_dget(self.cfg, "controller.mpc.blend", 0.7))
        blend = max(0.0, min(1.0, blend))
        u_ref = blend * self._u_mpc_cache[agent_idx] + (1.0 - blend) * u_base
        return np.clip(u_ref, -self.a_max, self.a_max)

    def _get_nominal_control(self, x, v, goal, threats, is_2d, agent_idx=0):
        rel_goal = goal - x
        dist_goal = np.linalg.norm(rel_goal)

        # Goal tracking (PD + Far-field push)
        if dist_goal > self.pd_switch_dist:
            u_goal = (rel_goal / (dist_goal + 1e-9)) * self.a_max - self.v_damp_far * v
        else:
            u_goal = self.pd_kp * rel_goal - self.pd_kd * v

        # Symmetry breaking: small rotation based on ID
        # Adds a tiny tangential push relative to the goal vector
        # cross(z, rel_goal) is roughly tangent in 2D plane
        if dist_goal > 0.01:
            z_axis = np.array([0, 0, 1.0])
            tangent = np.cross(z_axis, rel_goal)
            tangent = tangent / np.linalg.norm(tangent)
            # Alternating bias based on ID to encourage passing on different sides
            bias_dir = 1.0 if (agent_idx % 2 == 0) else -1.0
            u_sym = 0.5 * bias_dir * tangent  # Small nudge of 0.5 m/s^2
            u_goal += u_sym

        if not (self.nom_enabled and self.rep_enabled):
            return np.clip(u_goal, -self.a_max, self.a_max)

        # Repulsion potential field
        u_repulse = np.zeros(3)
        for o in threats:
            kind = o.get("kind", "sphere")

            if kind in ["sphere", "agent"]:
                rel = x - _as3(o["pos"])
                dist = np.linalg.norm(rel) + 1e-9
                trigger = float(o.get("r", self.agent_radius)) + self.rep_range_extra
                if dist >= trigger:
                    continue

                # Gradient of the reciprocal potential
                strength = (1.0 / dist - 1.0 / trigger) * (1.0 / (dist**2))
                direction = rel / dist

            elif kind == "wall":
                # Distance to half-plane: n^T(x - p)
                dist = float(np.dot(_as3(o["normal"]), x - _as3(o["pos"]))) + 1e-9
                trigger = self.rep_range_extra  # No radius for walls
                if dist >= trigger:
                    continue

                strength = (1.0 / dist - 1.0 / trigger) * (1.0 / (dist**2))
                direction = _as3(o["normal"])

            gain = self.rep_gain_3d if not is_2d else self.rep_gain_2d
            u_repulse += gain * strength * direction

        u_nom = u_goal + u_repulse
        norm = np.linalg.norm(u_nom)
        return (u_nom / norm) * self.a_max if norm > self.a_max else u_nom

    def _get_constraints(self, x, v, threats, is_2d):
        G_soft, b_soft = [], []
        G_hard, b_hard = [], []

        # Track whether braking constraints were active (debug)
        brake_any = False

        for o in threats:
            kind = o.get("kind", "sphere")

            if kind in ["sphere", "agent"]:
                r_other = float(o.get("r", self.agent_radius))
                margin = (
                    self.buffer_agents if kind == "agent" else self.buffer_obstacles
                )
                d_safe = self.agent_radius + r_other + margin

                G_row, b_val, _dbg = self.ecbf.constraint_sphere_distance_squared(
                    x=x,
                    v=v,
                    other_pos=o["pos"],
                    other_vel=o.get("vel", np.zeros(3)),
                    d_safe=d_safe,
                )
                G_soft.append(G_row.tolist())
                b_soft.append(float(b_val))

                # Braking-distance viability constraint (optional, activates only in braking zone)
                # We typically treat braking safety as a soft constraint too,
                # as it is a heuristic safety filter.
                info = self.brake_filter.maybe_add_constraint(
                    G_list=G_soft,
                    b_list=b_soft,
                    x=x,
                    v=v,
                    other_pos=o["pos"],
                    other_vel=o.get("vel", np.zeros(3)),
                    d_safe=d_safe,
                    a_max=self.a_max,
                )
                brake_any = brake_any or bool(info.get("active", False))

            elif kind == "wall":
                n = _as3(o["normal"])
                p = _as3(o["pos"])
                offset = self.agent_radius + self.buffer_obstacles

                G_row, b_val, _dbg = self.ecbf.constraint_halfspace_wall(
                    x=x,
                    v=v,
                    wall_normal=n,
                    wall_point=p,
                    offset=offset,
                )
                G_soft.append(G_row.tolist())
                b_soft.append(float(b_val))

        self.last_brake_active = bool(brake_any)

        # Acceleration bounds are inherently hard constraints
        for i in range(3):
            limit = 0.0 if (is_2d and i == 2) else self.a_max

            row_p = [0.0, 0.0, 0.0]
            row_p[i] = 1.0
            G_hard.append(row_p)
            b_hard.append(float(limit))

            row_n = [0.0, 0.0, 0.0]
            row_n[i] = -1.0
            G_hard.append(row_n)
            b_hard.append(float(limit))

        return (
            np.asarray(G_soft, dtype=float),
            np.asarray(b_soft, dtype=float),
            np.asarray(G_hard, dtype=float),
            np.asarray(b_hard, dtype=float),
        )
