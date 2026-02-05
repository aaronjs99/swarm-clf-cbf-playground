import numpy as np
import copy
from utils.geometry import as3
from utils.stepper import physics_step
from world.environment import init_agents_and_goals, get_obstacles
from controllers.dynamics import SwarmController


class SwarmRLWrapper:
    """
    Gym-like environment wrapper for the Swarm CLF-CBF simulation.
    Targeting single-agent RL training first (agent 0).
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.dim = 3 if cfg["viz"]["dim"] == "3d" else 2
        self.dt = float(cfg["sim"]["dt"])
        self.sub_steps = int(
            cfg["sim"]["substeps"]["live"]
        )  # Use live steps for training speed

        # Override config for RL if needed
        # We want the RL agent to control the acceleration directly
        self.a_max = float(
            cfg["controller"]["dim_defaults" if self.dim == 2 else "dim_defaults"][
                "a_max_2d" if self.dim == 2 else "a_max_3d"
            ]
        )

        # Controller for Safety Layer (CBF)
        # We will use this to "shield" the RL action
        self.ctrl = SwarmController(cfg, a_max=self.a_max)

        # State
        self.agents = []
        self.goals_init = []
        self.obs = []
        self.t = 0.0

        # RL Config
        self.k_nearest_obs = 5
        self.observation_dim = 9 + self.k_nearest_obs * 4
        self.action_dim = 3

    def reset(self):
        n = int(self.cfg["agents"]["num_agents"])

        # Init World
        agents_init, goals_init = init_agents_and_goals(self.cfg, self.dim, n)
        self.obs = get_obstacles(self.cfg, agents_init, goals_init, self.dim)
        self.goals_init = goals_init

        # Flight config (3D waypoint profile)
        flight_cfg = (self.cfg.get("world", {}) or {}).get("flight", {}) or {}
        z_cruise = float(flight_cfg.get("z_cruise", 6.0))

        self.agents = []
        for i, p in enumerate(agents_init):
            g = goals_init[i].copy()

            a = {
                "pos": p.copy(),
                "goal": g.copy(),
                "vel": np.zeros(3),
                "acc": np.zeros(3),
                "path": [],
            }

            # 3D waypoint profile: takeoff -> cruise -> land
            if self.dim == 3:
                mid = 0.5 * (p + g)
                mid[2] = z_cruise
                a["goal_seq"] = [mid.copy(), g.copy()]
                a["goal_idx"] = 0
                a["goal"] = a["goal_seq"][0].copy()

            self.agents.append(a)

        self.t = 0.0

        # We also need to reset the controller internal state if any
        # SwarmController doesn't have much state except policies (connectivity)
        # But for single agent training, connectivity might not matter yet.

        return self._get_obs(0)

    def step(self, action_rl):
        """
        Args:
            action_rl: np.array (3,) normalized or raw acceleration?
                       Let's assume it's raw acceleration in [-a_max, a_max]
        """
        # We are training Agent 0
        agent_idx = 0
        a = self.agents[agent_idx]

        # 1. Nominal "RL" Action
        u_rl = np.asarray(action_rl, dtype=float)
        u_rl = np.clip(u_rl, -self.a_max, self.a_max)
        if self.dim == 2:
            u_rl[2] = 0.0

        # 2. Safety Layer (CBF Shield)
        # We hijack the usual flow:
        # Instead of NominalPolicy, we treat u_rl as the nominal u.

        # Topology Update (usually done by controller at idx 0)
        self.ctrl.connectivity_policy.update_topology(self.agents)

        # Build Threats
        threats = self.ctrl.constraint_builder.build_threats(
            self.obs, agent_idx, self.agents
        )

        # Build Constraints
        # Note: We do NOT use nominal_policy.compute() here. We use u_rl.
        self.ctrl.constraint_builder.add_constraints(
            a["pos"], a["vel"], threats, self.dim == 2
        )

        # Solve CBF-QP
        # solve(u_nom, constraints, ...)
        u_safe, delta = self.ctrl.solver_wrapper.solve(
            u_rl,
            self.ctrl.constraint_manager.get_combined_constraints(),
            agent_idx,
            self.ctrl.state_manager,
            self.ctrl.connectivity_policy.network,
            mpc_solver=getattr(self.ctrl, "mpc", None),  # Safe access if mpc not init
            x=a["pos"],
            v=a["vel"],
            goal=a["goal"],
            threats=threats,
            is_2d=(self.dim == 2),
        )

        # 3. Physics Step
        # The physics_step function iterates over ALL agents.
        # But we only computed control for Agent 0 manually.
        # For other agents, we should let them behave normally (Nominal Policy or Static).
        # To use `physics_step`, we need to hook into `ctrl.compute_control`?
        # Or simpler: We just override the control inside physics step?
        # `physics_step` calls `ctrl.compute_control` for each agent.

        # Helper Controller to inject RL action for Agent 0 while keeping others nominal
        class RLOverrideController:
            def __init__(self, real_ctrl, u_override_0):
                self.real_ctrl = real_ctrl
                self.u_0 = u_override_0
                self.last_l2 = real_ctrl.last_l2
                self.last_delta = real_ctrl.last_delta

            def compute_control(self, x, v, goal, obstacles, i, all_agents):
                if i == 0:
                    return self.u_0
                return self.real_ctrl.compute_control(
                    x, v, goal, obstacles, i, all_agents
                )

        override_ctrl = RLOverrideController(self.ctrl, u_safe)

        # Run Physics
        w_cfg = self.cfg["world"]["obstacles"].get("walls", {})
        dyn_cfg = self.cfg["world"]["obstacles"].get("dynamic", {}) or {}
        agent_dyn_cfg = self.cfg["sim"].get("dynamics", {}) or {}

        physics_step(
            agents=self.agents,
            obs=self.obs,
            ctrl=override_ctrl,
            dt=self.dt,
            dim=self.dim,
            w_cfg=w_cfg,
            dyn_cfg=dyn_cfg,
            sub_steps=self.sub_steps,
            agent_radius=float(self.cfg["controller"]["cbf"]["agent_radius"]),
            agent_dyn_cfg=agent_dyn_cfg,
            flight_cfg=(self.cfg.get("world", {}) or {}).get("flight", {}),
            logger=None,  # No logging during training for speed?
            t_start=self.t,
        )

        self.t += self.dt * self.sub_steps

        # 4. Compute Reward & Obs
        obs = self._get_obs(0)

        reward, info = self._compute_reward(self.agents[0], u_rl, u_safe, delta)

        # Check Termination
        done = self._is_done(self.agents[0])

        return obs, reward, done, info

    def _get_obs(self, idx):
        a = self.agents[idx]
        x = a["pos"]
        v = a["vel"]
        g = a["goal"]

        # Relative Goal
        rel_g = g - x
        dist_g = np.linalg.norm(rel_g)
        dir_g = rel_g / (dist_g + 1e-6)

        # Nearest Obstacles
        # For simplicity, let's just observe relative position of K nearest threats
        # We can use the constraint builder's logic or simple distance

        # Brute force distance
        threats = []
        for o in self.obs:
            p_o = np.asarray(o["pos"])
            rel_o = p_o - x
            dist_o = np.linalg.norm(rel_o)
            radius = float(o.get("r", 0.5)) if o.get("kind") != "wall" else 0.0
            # Rough "distance to surface"
            d_surf = dist_o - radius
            threats.append((d_surf, rel_o, radius))

        threats.sort(key=lambda t: t[0])
        k = self.k_nearest_obs

        # Normalized Obs
        obs_vec = []
        # Agent State (6)
        obs_vec.extend(x)
        obs_vec.extend(v)
        # Goal Rel (3)
        obs_vec.extend(rel_g)

        # Obstacles (K * 4)
        for i in range(k):
            if i < len(threats):
                d, rel, r = threats[i]
                obs_vec.extend(rel)
                obs_vec.append(r)
            else:
                # Padding
                obs_vec.extend([0.0, 0.0, 0.0, 0.0])

        return np.array(obs_vec, dtype=np.float32)

    def _compute_reward(self, agent, u_rl, u_safe, delta):
        # 1. Goal Progress
        dist_goal = np.linalg.norm(agent["pos"] - agent["goal"])
        r_goal = -dist_goal

        # 2. Safety Intervention Penalty
        # If u_safe is very different from u_rl, we penalize
        # intervene_diff = np.linalg.norm(u_safe - u_rl)
        # r_intervene = -1.0 * intervene_diff

        # Better: Uses the solver's delta (slack)?
        # Or just the L2 norm of the modification
        modification = np.linalg.norm(u_safe - u_rl)
        r_mod = -0.5 * (modification**2)

        # 3. Collision Penalty (Ultimate Fail)
        r_coll = 0.0
        for o in self.obs:
            p_o = np.asarray(o["pos"])
            dist = np.linalg.norm(self.agents[0]["pos"] - p_o)
            radius = float(o.get("r", 0.5)) if o.get("kind") != "wall" else 0.0
            if dist < radius + float(self.cfg["controller"]["cbf"]["agent_radius"]):
                r_coll = -1000.0
                break

        # 4. Sparse Goal Reward
        r_success = 0.0
        if dist_goal < 0.2:
            r_success = 100.0

        # 5. Energy Penalty
        r_energy = -0.01 * np.linalg.norm(u_safe) ** 2

        total = r_goal + r_mod + r_coll + r_success + r_energy
        return total, {"dist_goal": dist_goal, "mod": modification}

    def _is_done(self, agent):
        dist_goal = np.linalg.norm(agent["pos"] - agent["goal"])
        if dist_goal < 0.2:
            return True
        # Timeout?
        return False
