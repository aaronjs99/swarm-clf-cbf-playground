import numpy as np
from world.physics import (
    resolve_sphere_sphere_collisions,
    bounce_sphere_off_walls_inplace,
    agents_enforce_world_bounds,
)


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
        dyn_cfg: Dynamic obstacle configuration.
        sub_steps: Number of physics steps per visual frame.
        agent_radius: Radius of agents.
        logger: Optional SafetyLogger instance.
        t_start: Start time for this batch of steps.
        log_params: Dictionary of params for logging (buffers, etc.)

    Returns:
        swarm_pos_accum: Accumulator of agent positions for centroid calculation.
    """
    swarm_pos_accum = np.zeros(3)
    rest_obs = float(dyn_cfg.get("restitution", 1.0))
    rest_agents = float(dyn_cfg.get("restitution_agents", 0.0))

    for s in range(sub_steps):
        # 1) Update obstacle positions
        for o in obs:
            if o.get("kind") == "sphere":
                o["pos"] += o["vel"] * dt
                if w_cfg.get("enabled", False):
                    bounce_sphere_off_walls_inplace(o, w_cfg, dim)
            else:
                o["pos"] += o["vel"] * dt

        # 1b) Sphere-sphere collisions
        resolve_sphere_sphere_collisions(obs, dim=dim, restitution=rest_obs)

        # 2) Agent control + integrate
        curr_states = [{"pos": a["pos"], "vel": a["vel"]} for a in agents]

        for i, a in enumerate(agents):
            acc = ctrl.compute_control(
                a["pos"], a["vel"], a["goal"], obs, i, curr_states
            )
            if dim == 2:
                acc[2] = 0.0

            a["vel"] += acc * dt
            a["pos"] += a["vel"] * dt

            if dim == 2:
                a["pos"][2] = 0.0
                a["vel"][2] = 0.0

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
