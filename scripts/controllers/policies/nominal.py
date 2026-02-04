import numpy as np
from utils.config import get as dget
from controllers.nominal.pd_tracking import PDTrackingStrategy
from controllers.nominal.potential_fields import PotentialFieldStrategy
from controllers.mpc_layer import SamplingMPCPlanner


class NominalPolicy:
    def __init__(self, cfg, a_max):
        self.cfg = cfg
        self.a_max = a_max
        self.pd = PDTrackingStrategy(cfg)
        self.pot_field = PotentialFieldStrategy(cfg)

        self.enabled = bool(dget(cfg, "controller.nominal.enabled", True))
        self.clip_fraction = float(dget(cfg, "limits.nom_clip_fraction", 0.8))

        # MPC
        self.mpc_stride = int(dget(cfg, "controller.mpc.stride", 10))
        self._mpc_counters = {}
        self._u_mpc_cache = {}

        dt_default = float(dget(cfg, "sim.dt", 0.02))
        self.mpc = SamplingMPCPlanner(
            enabled=bool(dget(cfg, "controller.mpc.enabled", False)),
            horizon_steps=int(dget(cfg, "controller.mpc.horizon_steps", 20)),
            dt=float(dget(cfg, "controller.mpc.dt", dt_default)),
            num_samples=int(dget(cfg, "controller.mpc.num_samples", 64)),
            goal_weight=float(dget(cfg, "controller.mpc.goal_weight", 1.0)),
            effort_weight=float(dget(cfg, "controller.mpc.effort_weight", 0.02)),
            collision_weight=float(dget(cfg, "controller.mpc.collision_weight", 50.0)),
            rng_seed=int(dget(cfg, "controller.mpc.rng_seed", 0)),
        )

    def compute(self, x, v, goal, agent_idx, threats, is_2d):
        # 1. Base (PD)
        u_base = self.pd.compute_control(x, v, goal)

        # 2. Symmetry Breaking
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

        # 3. Repulsion
        if self.enabled:
            u_rep = self.pot_field.compute_repulsion(x, threats, is_2d)
            u_base += u_rep

        # 4. MPC Blending
        u_nom = self._blend_mpc(x, v, goal, threats, is_2d, agent_idx, u_base)

        # 5. Clipping
        clip = self.clip_fraction * self.a_max
        return np.clip(u_nom, -clip, clip)

    def _blend_mpc(self, x, v, goal, threats, is_2d, agent_idx, u_base):
        if not self.mpc.enabled:
            return u_base

        if agent_idx not in self._mpc_counters:
            self._mpc_counters[agent_idx] = 0

        self._mpc_counters[agent_idx] += 1
        cnt = self._mpc_counters[agent_idx]

        do_plan = (cnt % max(1, self.mpc_stride)) == 0
        if do_plan or (agent_idx not in self._u_mpc_cache):
            # Define d_safe for MPC (simplified)
            def d_safe_fn(o):
                kind = o.get("kind", "sphere")
                # Fallback values if not in threat dict, though threats usually come from ConstraintBuilder
                r = float(o.get("r", 0.0))
                # Note: We don't have direct access to buffer configs here unless we parse them
                # or pass them in. For now, using hardcoded small buffers or reading from cfg again
                # Ideally, this function should be passed in or configured.
                # Let's read from cfg to be safe, though optimized would be better.
                buf_obs = float(dget(self.cfg, "controller.cbf.buffer_obstacles", 0.1))
                buf_agn = float(dget(self.cfg, "controller.cbf.buffer_agents", 0.05))
                agent_r = float(dget(self.cfg, "controller.cbf.agent_radius", 0.05))

                margin = buf_agn if kind == "agent" else buf_obs
                return agent_r + r + margin

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

        blend = float(dget(self.cfg, "controller.mpc.blend", 0.7))
        blend = max(0.0, min(1.0, blend))
        u_ref = blend * self._u_mpc_cache[agent_idx] + (1.0 - blend) * u_base
        return np.clip(u_ref, -self.a_max, self.a_max)
