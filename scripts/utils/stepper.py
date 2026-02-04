import numpy as np
from world.physics import (
    resolve_sphere_sphere_collisions,
    bounce_sphere_off_walls_inplace,
    agents_enforce_world_bounds,
)


def _clip_speed(v: np.ndarray, v_max: float, dim: int) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    if v_max <= 0.0:
        return v
    if dim == 2:
        s = float(np.linalg.norm(v[:2]))
        if s > v_max:
            v[:2] = (v[:2] / (s + 1e-12)) * v_max
        v[2] = 0.0
        return v
    s = float(np.linalg.norm(v))
    if s > v_max:
        v = (v / (s + 1e-12)) * v_max
    return v


def physics_step(
    agents,
    obs,
    ctrl,
    dt,
    dim,
    w_cfg,
    dyn_cfg,
    sub_steps,
    agent_radius,
    # new: realism dynamics for agents (not obstacles)
    agent_dyn_cfg=None,
    logger=None,
    t_start=0.0,
    log_params=None,
):
    """
    Performs `sub_steps` of physics integration.

    Args:
        agents: List of agent state dicts.
        obs: List of obstacle dicts.
        ctrl: Controller instance.
        dt: Time step.
        dim: Dimension (2 or 3).
        w_cfg: Wall configuration.
        dyn_cfg: Obstacle dynamics configuration.
        sub_steps: Number of physics steps per visual frame.
        agent_radius: Radius of agents.
        agent_dyn_cfg: Agent dynamics realism config (accel lag, drag, vmax, noise).
        logger: Optional SafetyLogger instance.
        t_start: Start time for this batch of steps.
        log_params: Dictionary of params for logging (buffers, etc.)

    Returns:
        swarm_pos_accum: Accumulator of agent positions for centroid calculation.
    """
    swarm_pos_accum = np.zeros(3)
    rest_obs = float(dyn_cfg.get("restitution", 1.0))
    rest_agents = float(dyn_cfg.get("restitution_agents", 0.0))

    agent_dyn_cfg = agent_dyn_cfg or {}
    tau = float(agent_dyn_cfg.get("accel_time_constant", 0.0))
    drag = float(agent_dyn_cfg.get("linear_drag", 0.0))
    noise_std = float(agent_dyn_cfg.get("accel_noise_std", 0.0))

    v_max_2d = float(agent_dyn_cfg.get("v_max_2d", 0.0))
    v_max_3d = float(agent_dyn_cfg.get("v_max_3d", 0.0))
    v_max = v_max_2d if dim == 2 else v_max_3d

    # Stable RNG for process noise
    rng = np.random.default_rng(12345)

    for s in range(sub_steps):
        # 1) Update obstacle positions
        for o in obs:
            if o.get("kind") == "sphere":
                o["pos"] += o["vel"] * dt
                if w_cfg.get("enabled", False):
                    bounce_sphere_off_walls_inplace(o, w_cfg, dim)
            else:
                o["pos"] += o["vel"] * dt

        # 1b) Sphere-sphere collisions for obstacles
        resolve_sphere_sphere_collisions(obs, dim=dim, restitution=rest_obs)

        # 2) Agent control + integrate
        curr_states = [{"pos": a["pos"], "vel": a["vel"]} for a in agents]

        for i, a in enumerate(agents):
            a.setdefault("acc", np.zeros(3))

            acc_cmd = ctrl.compute_control(
                a["pos"], a["vel"], a["goal"], obs, i, curr_states
            )
            acc_cmd = np.asarray(acc_cmd, dtype=float).reshape(3)

            if dim == 2:
                acc_cmd[2] = 0.0

            # (A) Actuator lag: track applied acceleration state a["acc"]
            if tau > 1e-9:
                a["acc"] = a["acc"] + (dt / tau) * (acc_cmd - a["acc"])
                acc_applied = a["acc"].copy()
            else:
                a["acc"] = acc_cmd.copy()
                acc_applied = acc_cmd

            # (B) Optional process noise on applied acceleration
            if noise_std > 0.0:
                n = rng.normal(0.0, noise_std, size=3)
                if dim == 2:
                    n[2] = 0.0
                acc_applied = acc_applied + n

            g_2d = float(agent_dyn_cfg.get("gravity_2d", 0.0))
            g_3d = float(agent_dyn_cfg.get("gravity_3d", 9.8))
            g = g_2d if dim == 2 else g_3d
            g_vec = np.array([0.0, 0.0, -g], dtype=float)

            # (C) Velocity integration with linear drag
            # v_dot = a_applied - drag * v
            a["vel"] = a["vel"] + (acc_applied + g_vec - drag * a["vel"]) * dt

            # (D) Speed limit
            a["vel"] = _clip_speed(a["vel"], v_max=v_max, dim=dim)

            # (E) Position integration
            a["pos"] = a["pos"] + a["vel"] * dt

            if dim == 2:
                a["pos"][2] = 0.0
                a["vel"][2] = 0.0
                a["acc"][2] = 0.0

            a["path"].append(a["pos"].copy())
            swarm_pos_accum += a["pos"]

        # 2b) HARD world bounds for agents
        agents_enforce_world_bounds(
            agents=agents,
            w_cfg=w_cfg,
            dim=dim,
            agent_radius=agent_radius,
            restitution=rest_agents,
        )

        # 3) Log safety
        if logger and log_params:
            logger.log(
                t_start + s * dt,
                agents,
                obs,
                buffer_obs=log_params["buffer_obs"],
                buffer_agents=log_params["buffer_agents"],
                agent_radius=agent_radius,
                l2=float(getattr(ctrl, "last_l2", 0.0)),
                delta=float(getattr(ctrl, "last_delta", 0.0)),
            )

    return swarm_pos_accum
