import os
from itertools import count
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

from world.environment import get_obstacles, init_agents_and_goals
from utils.logger import SafetyLogger
from utils.stepper import physics_step
from utils.animator import SwarmAnimator


def run_simulation(cfg, ctrl):
    """
    Main entry point for running the simulation.
    Orchestrates the environment init, physics stepping (via stepper),
    and visualization (via animator).
    """
    dim = 2 if cfg["viz"]["dim"] == "2d" else 3
    n = int(cfg["agents"]["num_agents"])

    # Ensure 3D walls have z-bounds if using a 2D config adapted for 3D
    if dim == 3:
        w = cfg["world"]["obstacles"].get("walls")
        if w and w.get("enabled"):
            w.setdefault("z_min", 0.0)
            w.setdefault("z_max", 10.0)

    # Init World
    agents_init, goals_init = init_agents_and_goals(cfg, dim, n)
    obs = get_obstacles(cfg, agents_init, goals_init, dim)

    # Config Extract
    sim_cfg = cfg["sim"]
    term_cfg = sim_cfg["termination"]
    cbf_cfg = cfg["controller"]["cbf"]

    dt = float(sim_cfg["dt"])
    record = bool(cfg["viz"]["record"])
    sub_steps = int(sim_cfg["substeps"]["record"] if record else sim_cfg["substeps"]["live"])

    goal_delta = float(term_cfg["goal_delta"])
    settle_seconds = float(term_cfg["settle_seconds"])
    max_frames = int(term_cfg["max_frames"])
    frames_after_reach = int(np.ceil(settle_seconds / (dt * sub_steps)))

    # Dynamics realism config
    dyn_cfg = sim_cfg.get("dynamics", {}) or {}

    # Flight config (3D waypoint profile)
    flight_cfg = (cfg.get("world", {}) or {}).get("flight", {}) or {}
    z_cruise = float(flight_cfg.get("z_cruise", 6.0))

    # Init State
    agents = []
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
        if dim == 3:
            mid = 0.5 * (p + g)
            mid[2] = z_cruise
            a["goal_seq"] = [mid.copy(), g.copy()]
            a["goal_idx"] = 0
            a["goal"] = a["goal_seq"][0].copy()

        agents.append(a)

    logger = SafetyLogger(len(agents))

    # Matplotlib Setup
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d" if dim == 3 else None)

    if dim == 2:
        ax.set_aspect("equal")
        walls_cfg = cfg["world"]["obstacles"].get("walls", {})
        if walls_cfg.get("enabled", False):
            ax.set_xlim(float(walls_cfg["x_min"]) - 2, float(walls_cfg["x_max"]) + 2)
            ax.set_ylim(float(walls_cfg["y_min"]) - 2, float(walls_cfg["y_max"]) + 2)
    else:
        walls_cfg = cfg["world"]["obstacles"].get("walls", {})
        if walls_cfg.get("enabled", False):
            w = walls_cfg
            xx, yy = np.meshgrid([w["x_min"], w["x_max"]], [w["y_min"], w["y_max"]])
            ax.plot_surface(xx, yy, np.full_like(xx, w.get("z_min", 0.0)), alpha=0.05, color="k")
            ax.plot_surface(xx, yy, np.full_like(xx, w.get("z_max", 10.0)), alpha=0.05, color="k")

        # A nicer default 3D view
        ax.view_init(elev=18, azim=-55)

    # Animator
    animator = SwarmAnimator(
        ax,
        agents,
        goals_init,
        obs,
        dim,
        agent_radius=float(cbf_cfg["agent_radius"]),
        buffer_agents=float(cbf_cfg["buffer_agents"]),
        buffer_obs=float(cbf_cfg["buffer_obstacles"]),
    )

    # Loop State
    reached_frame = None
    done = False

    def all_agents_within_delta():
        for a in agents:
            d = np.linalg.norm(a["pos"][:dim] - goals_init[int(np.clip(len(a.get("path", [])) >= 0, 0, 1)) * 0][:dim]) if False else np.linalg.norm(a["pos"][:dim] - a.get("goal_seq", [a["goal"]])[-1][:dim])
            if d > goal_delta:
                return False
        return True

    def frame_gen():
        for k in count(0):
            if done or k >= max_frames:
                break
            yield k

    def update(frame):
        nonlocal reached_frame, done

        swarm_pos_accum = physics_step(
            agents=agents,
            obs=obs,
            ctrl=ctrl,
            dt=dt,
            dim=dim,
            w_cfg=cfg["world"]["obstacles"].get("walls", {}),
            dyn_cfg=cfg["world"]["obstacles"].get("dynamic", {}) or {},
            agent_dyn_cfg=dyn_cfg,
            flight_cfg=flight_cfg,
            sub_steps=sub_steps,
            agent_radius=float(cbf_cfg["agent_radius"]),
            logger=logger,
            t_start=frame * sub_steps * dt,
            log_params={
                "buffer_obs": float(cbf_cfg["buffer_obstacles"]),
                "buffer_agents": float(cbf_cfg["buffer_agents"]),
            },
        )

        if reached_frame is None and all_agents_within_delta():
            reached_frame = frame

        if reached_frame is not None and frame >= reached_frame + frames_after_reach:
            done = True
            if not record:
                if hasattr(ani, "event_source") and ani.event_source is not None:
                    ani.event_source.stop()
                plt.close(fig)

        c_p = swarm_pos_accum / (len(agents) * sub_steps)
        return animator.update(agents, obs, c_p, frame)

    interval_ms = int(sim_cfg["interval_ms"])
    ani = FuncAnimation(
        fig,
        update,
        frames=frame_gen(),
        interval=interval_ms,
        blit=False,
        cache_frame_data=False,
    )

    if getattr(ani, "event_source", None) is None and not record:
        print("[warn] Matplotlib backend cannot animate. Falling back to record mode.")
        record = True

    if record:
        os.makedirs("media", exist_ok=True)
        ani.save(
            f"media/swarm_{dim}d.mp4",
            writer=FFMpegWriter(fps=30),
            dpi=150,
        )
    else:
        plt.show(block=True)

    if bool(cfg["viz"]["safety_plot"]):
        logger.plot(f"{dim}D Swarm Certificate")
