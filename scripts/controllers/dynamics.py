import numpy as np

from controllers.state_manager import SwarmStateManager
from controllers.constraints.manager import ConstraintManager

from controllers.policies.nominal import NominalPolicy
from controllers.policies.connectivity import ConnectivityPolicy
from controllers.constraints.builders import ConstraintBuilder
from controllers.solvers.wrapper import SolverWrapper

from controllers.mpc_layer import SamplingMPCPlanner

from utils.config import get as dget
from utils.geometry import as3


class SwarmController:
    """
    Central controller class that coordinates:
     1. Nominal Progress Strategies (NominalPolicy)
     2. Constraints (ConstraintBuilder)
     3. Optimization Solvers (SolverWrapper)

    Adds an optional MPC planning hint that blends into the nominal.
    """

    def __init__(self, cfg, a_max=15.0):
        self.cfg = cfg
        self.a_max = float(a_max)

        # 1. Policies
        self.nominal_policy = NominalPolicy(cfg, self.a_max)
        self.connectivity_policy = ConnectivityPolicy(cfg)

        # 2. Constraints
        self.constraint_manager = ConstraintManager(u_dim=3)
        self.constraint_builder = ConstraintBuilder(
            cfg, self.constraint_manager, self.a_max
        )

        # 3. Solvers
        self.solver_wrapper = SolverWrapper(cfg, self.a_max)
        self.state_manager = SwarmStateManager()

        # 4. MPC planning layer (optional)
        mpc_enabled = bool(dget(cfg, "controller.mpc.enabled", False))
        horizon_steps = int(dget(cfg, "controller.mpc.horizon_steps", 25))
        num_samples = int(dget(cfg, "controller.mpc.num_samples", 32))
        goal_w = float(dget(cfg, "controller.mpc.goal_weight", 5.0))
        effort_w = float(dget(cfg, "controller.mpc.effort_weight", 0.02))
        col_w = float(dget(cfg, "controller.mpc.collision_weight", 30.0))
        rng_seed = int(dget(cfg, "controller.mpc.rng_seed", 0))

        dt_cfg = dget(cfg, "controller.mpc.dt", None)
        dt_use = (
            float(dt_cfg) if dt_cfg is not None else float(dget(cfg, "sim.dt", 0.02))
        )

        self.mpc_stride = int(dget(cfg, "controller.mpc.stride", 15))
        self.mpc_blend = float(dget(cfg, "controller.mpc.blend", 0.2))
        self.mpc_disable_dist = float(dget(cfg, "controller.mpc.disable_dist", 0.5))

        self.mpc = SamplingMPCPlanner(
            enabled=mpc_enabled,
            horizon_steps=horizon_steps,
            dt=dt_use,
            num_samples=num_samples,
            goal_weight=goal_w,
            effort_weight=effort_w,
            collision_weight=col_w,
            rng_seed=rng_seed,
        )

        # Cache planned control at a stride to keep things cheap and stable
        self._tick = 0
        self._mpc_cache = {}  # agent_idx -> np.ndarray(3,)

    def _d_safe_for_threat(self, threat):
        """
        Match the safety distance logic used by ConstraintBuilder so MPC's
        collision cost "agrees" with your CBF buffers.
        """
        agent_radius = float(dget(self.cfg, "controller.cbf.agent_radius", 0.05))
        buf_obs = float(dget(self.cfg, "controller.cbf.buffer_obstacles", 0.10))
        buf_agents = float(dget(self.cfg, "controller.cbf.buffer_agents", 0.05))

        kind = threat.get("kind", "sphere")
        if kind == "wall":
            # For walls, the "distance" inside MPC is n^T(x - p) and offset is safety margin
            return agent_radius + buf_obs

        r_other = float(threat.get("r", agent_radius))
        margin = buf_agents if kind == "agent" else buf_obs
        return agent_radius + r_other + margin

    def _should_disable_mpc(self, x, threats):
        """
        If something is already quite close, skip MPC planning for that tick.
        This keeps MPC from sampling nonsense in near-contact situations.
        """
        if not threats:
            return False

        x = as3(x)
        best = float("inf")
        for o in threats:
            kind = o.get("kind", "sphere")
            if kind in ["sphere", "agent"]:
                d = float(np.linalg.norm(x - as3(o["pos"])))
                best = min(best, d)
            elif kind == "wall":
                n = as3(o["normal"])
                p = as3(o["pos"])
                d = float(np.dot(n, x - p))
                best = min(best, d)

        return best < self.mpc_disable_dist

    def compute_control(self, x, v, goal, obstacles, agent_idx, all_agents):
        """
        Computes the control input (acceleration) for a single agent.
        """
        x = as3(x)
        v = as3(v)
        goal = as3(goal)
        is_2d = abs(x[2]) < 1e-5 and abs(goal[2]) < 1e-5

        g_2d = float(dget(self.cfg, "sim.dynamics.gravity_2d", 0.0))
        g_3d = float(dget(self.cfg, "sim.dynamics.gravity_3d", 9.8))
        g = g_2d if is_2d else g_3d
        u_grav_comp = np.array([0.0, 0.0, g], dtype=float)  # cancels -z gravity

        # Treat "agent 0 call" as the start of a new control tick.
        if agent_idx == 0:
            self._tick += 1

        # 0. Topology Update
        if agent_idx == 0:
            self.connectivity_policy.update_topology(all_agents)

        threats = self.constraint_builder.build_threats(
            obstacles, agent_idx, all_agents
        )

        # 1. Nominal Control
        u_nom = self.nominal_policy.compute(x, v, goal, agent_idx, threats, is_2d)

        # 1b. MPC planning hint (optional, cached at a stride)
        if self.mpc.enabled and (not self._should_disable_mpc(x, threats)):
            do_replan = (self.mpc_stride <= 1) or (self._tick % self.mpc_stride == 0)
            if do_replan or (agent_idx not in self._mpc_cache):
                u_plan = self.mpc.plan(
                    x=x,
                    v=v,
                    goal=goal,
                    threats=threats,
                    a_max=self.a_max,
                    is_2d=is_2d,
                    d_safe_fn=self._d_safe_for_threat,
                )
                if u_plan is not None:
                    self._mpc_cache[agent_idx] = u_plan

            if agent_idx in self._mpc_cache:
                u_plan = self._mpc_cache[agent_idx]
                b = max(0.0, min(1.0, self.mpc_blend))
                u_nom = (1.0 - b) * u_nom + b * u_plan

        # 2. Connectivity Injection
        u_conn = self.connectivity_policy.compute_contribution(agent_idx, all_agents)
        if self.connectivity_policy.inject_into_nominal:
            u_nom += u_conn

        # 3. Constraints
        self.constraint_builder.add_constraints(x, v, threats, is_2d)

        # 4. Solvers
        u_opt, delta = self.solver_wrapper.solve(
            u_nom,
            self.constraint_manager.get_combined_constraints(),
            agent_idx,
            self.state_manager,
            self.connectivity_policy.network,
            mpc_solver=self.mpc,  # NEW: Pass DMPC object
            x=x,  # NEW: Pass State
            v=v,  # NEW: Pass Velocity
            goal=goal,  # NEW: Pass Goal
            threats=threats,  # NEW: Pass Threats
            is_2d=is_2d,  # NEW: Pass 2D flag
        )

        if is_2d:
            u_opt[2] = 0.0
        else:
            u_opt = u_opt + u_grav_comp

        return u_opt

    @property
    def last_l2(self):
        return self.connectivity_policy.last_l2

    @property
    def last_delta(self):
        return self.solver_wrapper.last_delta
