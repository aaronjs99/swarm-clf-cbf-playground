import numpy as np

from controllers.solvers.admm_engine import ADMMSolver
from controllers.solvers.base_qp import solve_qp_safe
from controllers.solvers.interface import ProblemTranslator
from controllers.state_manager import SwarmStateManager
from controllers.network import SwarmNetwork

from controllers.cbf.ecbf import ECBFRelativeDegree2
from controllers.safety_filter import BrakingSafetyFilter
from controllers.mpc_layer import SamplingMPCPlanner

from controllers.nominal.pd_tracking import PDTrackingStrategy
from controllers.nominal.potential_fields import PotentialFieldStrategy
from controllers.nominal.clf_progress import CLFProgressStrategy
from controllers.constraints.manager import ConstraintManager
from controllers.constraints import LinearConstraint

from utils.config import get as dget
from utils.geometry import as3


class SwarmController:
    """
    Central controller class that coordinates:
     1. Nominal Progress Strategies (PD, Potential Fields, MPC)
     2. Constraints (Limits, CBFs, Safety Filters)
     3. Optimization Solvers (QP, ADMM)

    It serves as the main entry point for the simulation engine to fallback to
    when computing control inputs for agents.
    """

    def __init__(self, cfg, a_max=15.0):
        self.cfg = cfg
        self.a_max = float(a_max)

        # --- Progress / Nominal Strategies ---
        self.pd_strategy = PDTrackingStrategy(cfg)
        self.pot_field_strategy = PotentialFieldStrategy(cfg)
        self.clf_strategy = CLFProgressStrategy(cfg)

        self.nom_enabled = bool(dget(cfg, "controller.nominal.enabled", True))
        self.nom_clip_fraction = float(dget(cfg, "limits.nom_clip_fraction", 0.8))

        # --- Constraints ---
        self.constraint_manager = ConstraintManager()

        # CBF
        self.k1 = float(dget(cfg, "controller.cbf.k1", 10.0))
        self.k2 = float(dget(cfg, "controller.cbf.k2", 10.0))
        self.ecbf = ECBFRelativeDegree2(k1=self.k1, k0=self.k2)

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

        # --- Solver & Translator ---
        self.translator = ProblemTranslator(cfg)
        self.use_admm = bool(dget(cfg, "controller.admm.enabled", True))
        rho = float(dget(cfg, "controller.admm.rho", 0.1))
        iters = int(dget(cfg, "controller.admm.iters", 2))
        self.solver = ADMMSolver(rho=rho, num_iters=iters)

        self.state_manager = SwarmStateManager()

        # --- Connectivity ---
        self.connectivity_enabled = bool(
            dget(cfg, "controller.connectivity.enabled", True)
        )
        k_neighbors = int(dget(cfg, "controller.connectivity.k_neighbors", 3))
        self.network = SwarmNetwork(k_neighbors=k_neighbors)
        self.conn_gain = float(dget(cfg, "controller.connectivity.gain", 5.0))
        self.l2_threshold = float(
            dget(cfg, "controller.connectivity.l2_threshold", 0.5)
        )
        self.inject_conn_into_nominal = bool(
            dget(cfg, "controller.connectivity.inject_into_nominal", False)
        )

        # --- MPC ---
        self.mpc_stride = int(dget(cfg, "controller.mpc.stride", 10))
        self._mpc_counters = {}
        self._u_mpc_cache = {}

        self.mpc = SamplingMPCPlanner(
            enabled=bool(dget(cfg, "controller.mpc.enabled", False)),
            horizon_steps=int(dget(cfg, "controller.mpc.horizon_steps", 20)),
            dt=float(dget(cfg, "controller.mpc.dt", float(dget(cfg, "sim.dt", 0.02)))),
            num_samples=int(dget(cfg, "controller.mpc.num_samples", 64)),
            goal_weight=float(dget(cfg, "controller.mpc.goal_weight", 1.0)),
            effort_weight=float(dget(cfg, "controller.mpc.effort_weight", 0.02)),
            collision_weight=float(dget(cfg, "controller.mpc.collision_weight", 50.0)),
            rng_seed=int(dget(cfg, "controller.mpc.rng_seed", 0)),
        )

        # --- Debug ---
        self.last_l2 = 0.0
        self.last_delta = 0.0
        self.last_brake_active = False

    def compute_control(self, x, v, goal, obstacles, agent_idx, all_agents):
        """
        Computes the control input (acceleration) for a single agent.

        Steps:
         1. Nominal Control: Determine desired direction (PD, Potential Field, MPC).
         2. Constraints: Gather safety, limits, and other constraints.
         3. Solve: Use QP or ADMM to find closes u to u_nominal satisfying constraints.
        """
        x = as3(x)
        v = as3(v)
        goal = as3(goal)
        is_2d = abs(x[2]) < 1e-5 and abs(goal[2]) < 1e-5

        if agent_idx == 0 and self.connectivity_enabled:
            self.network.update_topology(all_agents)

        threats = self._build_threats(obstacles, agent_idx, all_agents)

        # 1. Nominal / Progress

        # Base Nominal (PD + Symmetry + Repulsion)
        u_base = self.pd_strategy.compute_control(x, v, goal)

        # Symmetry breaking
        rel_goal = goal - x
        dist_goal = np.linalg.norm(rel_goal)
        if dist_goal > 0.01:
            z_axis = np.array([0, 0, 1.0])
            tangent = np.cross(z_axis, rel_goal)
            norm_t = np.linalg.norm(tangent)
            if norm_t > 1e-6:
                tangent = tangent / norm_t
                bias_dir = 1.0 if (agent_idx % 2 == 0) else -1.0
                u_sym = 0.5 * bias_dir * tangent
                u_base += u_sym

        # Repulsion
        if self.nom_enabled:
            u_rep = self.pot_field_strategy.compute_repulsion(x, threats, is_2d)
            u_base += u_rep

        # MPC Blending
        u_nom = self._blend_mpc(x, v, goal, threats, is_2d, agent_idx, u_base)

        # Connectivity
        if self.connectivity_enabled and len(all_agents) > 1:
            conn_grad, l2 = self.network.get_connectivity_gradient(
                agent_idx, all_agents
            )
            if agent_idx == 0:
                self.last_l2 = l2
            if l2 < self.l2_threshold:
                u_conn = self.conn_gain * conn_grad
                if self.inject_conn_into_nominal:
                    u_nom += u_conn

        # Clip Nominal
        clip = self.nom_clip_fraction * self.a_max
        u_nom = np.clip(u_nom, -clip, clip)

        # 2. Constraints
        self.constraint_manager.clear()

        # Limits (Hard)
        for i in range(3):
            limit = 0.0 if (is_2d and i == 2) else self.a_max
            # u_i <= limit
            row_p = np.zeros(3)
            row_p[i] = 1.0
            self.constraint_manager.add_constraint(
                LinearConstraint(G=row_p, b=np.array([limit]), hard=True)
            )

            # -u_i <= limit
            row_n = np.zeros(3)
            row_n[i] = -1.0
            self.constraint_manager.add_constraint(
                LinearConstraint(G=row_n, b=np.array([limit]), hard=True)
            )

        # Obstacle/Agent Safety (ECBF + Braking)
        brake_any = False
        for o in threats:
            kind = o.get("kind", "sphere")
            if kind in ["sphere", "agent"]:
                r_other = float(o.get("r", self.agent_radius))
                margin = (
                    self.buffer_agents if kind == "agent" else self.buffer_obstacles
                )
                d_safe = self.agent_radius + r_other + margin

                # ECBF
                c_ecbf, _ = self.ecbf.constraint_sphere_distance_squared(
                    x, v, o["pos"], o.get("vel", np.zeros(3)), d_safe
                )
                self.constraint_manager.add_constraint(c_ecbf)

                # Braking
                c_brake, info = self.brake_filter.get_constraint(
                    x, v, o["pos"], o.get("vel", np.zeros(3)), d_safe, self.a_max
                )
                if c_brake:
                    self.constraint_manager.add_constraint(c_brake)
                    brake_any = True

            elif kind == "wall":
                n = as3(o["normal"])
                p = as3(o["pos"])
                offset = self.agent_radius + self.buffer_obstacles

                c_wall, _ = self.ecbf.constraint_halfspace_wall(x, v, n, p, offset)
                self.constraint_manager.add_constraint(c_wall)

        self.last_brake_active = brake_any

        # 3. Solver
        qp_prob = self.translator.translate(
            u_nom, self.constraint_manager.get_combined_constraints()
        )

        if self.use_admm:
            u_opt, delta = self.solver.step(
                agent_id=agent_idx,
                u_nom=u_nom,
                P=qp_prob["P"],
                q=qp_prob["q"],
                G=qp_prob["G"],
                b=qp_prob["b"],
                state_manager=self.state_manager,
                network=self.network,
                a_max=self.a_max,
                slack_weight=qp_prob["slack_weight"],
                fallback_zero_on_fail=qp_prob["fallback_zero_on_fail"],
                G_hard=qp_prob["G_hard"],
                b_hard=qp_prob["b_hard"],
            )
        else:
            u_opt, delta = solve_qp_safe(
                P=qp_prob["P"],
                q=qp_prob["q"],
                G=qp_prob["G"],
                b=qp_prob["b"],
                a_max=self.a_max,
                slack_weight=qp_prob["slack_weight"],
                fallback_zero_on_fail=qp_prob["fallback_zero_on_fail"],
                G_hard=qp_prob["G_hard"],
                b_hard=qp_prob["b_hard"],
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
                    "pos": as3(a["pos"]),
                    "r": self.agent_radius,
                    "vel": as3(a["vel"]),
                    "kind": "agent",
                    "id": i,
                }
            )
        return threats

    def _blend_mpc(self, x, v, goal, threats, is_2d, agent_idx, u_base):
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

        # Blend
        blend = float(dget(self.cfg, "controller.mpc.blend", 0.7))
        blend = max(0.0, min(1.0, blend))
        u_ref = blend * self._u_mpc_cache[agent_idx] + (1.0 - blend) * u_base
        return np.clip(u_ref, -self.a_max, self.a_max)
