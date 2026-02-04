import os
from itertools import count

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter


class SafetyLogger:
    """
    Logs safety metrics (CBF values), connectivity, and solver slack for post-simulation analysis.
    """

    def __init__(self, num_agents):
        self.h_buf_history = [[] for _ in range(num_agents)]
        self.h_phys_history = [[] for _ in range(num_agents)]
        self.l2_history = []
        self.delta_history = []
        self.t_history = []

    def log(
        self, t, agents, obstacles, buffer_obs, buffer_agents, agent_radius, l2, delta
    ):
        self.t_history.append(t)
        self.l2_history.append(l2)
        self.delta_history.append(delta)

        for i, a in enumerate(agents):
            h_buf_vals, h_phys_vals = [], []
            for o in obstacles:
                kind = o.get("kind", "sphere")
                if kind == "sphere":
                    safety_dist = agent_radius + o["r"] + buffer_obs
                    h_buf_vals.append(
                        np.linalg.norm(a["pos"] - o["pos"]) ** 2 - safety_dist**2
                    )
                elif kind == "wall":
                    h_val = np.dot(o["normal"], a["pos"] - o["pos"])
                    h_buf_vals.append(h_val - (agent_radius + buffer_obs))

            for j, other in enumerate(agents):
                if i == j:
                    continue
                d_sq = np.linalg.norm(a["pos"] - other["pos"]) ** 2
                h_buf_vals.append(d_sq - (agent_radius + buffer_agents) ** 2)
                h_phys_vals.append(d_sq - agent_radius**2)

            self.h_buf_history[i].append(min(h_buf_vals) if h_buf_vals else 10.0)
            self.h_phys_history[i].append(min(h_phys_vals) if h_phys_vals else 10.0)

    def plot(self, title):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

        for i in range(len(self.h_buf_history)):
            ax1.plot(self.t_history, self.h_buf_history[i], label=f"Agent {i}")
        ax1.axhline(y=0, color="k", linestyle="--", label="Safety Boundary")
        ax1.set_ylabel("Safety h(x)")
        ax1.set_title(title)
        ax1.grid(True)
        ax1.legend(loc="upper right")

        ax2.plot(self.t_history, self.l2_history, linewidth=2, label="lambda_2")
        ax2.axhline(y=0, color="r", linestyle="-", alpha=0.5, label="Disconnect (0.0)")
        ax2.axhline(
            y=0.5, color="orange", linestyle="--", alpha=0.5, label="Warn (0.5)"
        )
        ax2.set_ylabel("Connectivity")
        ax2.grid(True)
        ax2.legend(loc="upper right")

        ax3.plot(
            self.t_history, self.delta_history, linewidth=2, label="QP slack delta"
        )
        ax3.set_ylabel("Slack")
        ax3.set_xlabel("Time [s]")
        ax3.grid(True)
        ax3.legend(loc="upper right")

        plt.tight_layout()
        plt.show(block=True)


from world.physics import (
    resolve_sphere_sphere_collisions,
    bounce_sphere_off_walls_inplace,
    agents_enforce_world_bounds,
)
from world.environment import get_obstacles, init_agents_and_goals


def run_simulation(cfg, ctrl):
    """
    Main entry point for running the simulation.
    Initializes environment and starts the execution loop.
    """
    dim = 2 if cfg["viz"]["dim"] == "2d" else 3
    n = int(cfg["agents"]["num_agents"])

    agents_init, goals_init = init_agents_and_goals(cfg, dim, n)
    obs = get_obstacles(cfg, agents_init, goals_init, dim)

    _execute_sim(cfg, ctrl, obs, agents_init, goals_init, dim)


def _execute_sim(cfg, ctrl, obs, agents_init, goals_init, dim):
    sim_cfg = cfg["sim"]
    term_cfg = sim_cfg["termination"]
    cbf_cfg = cfg["controller"]["cbf"]

    dt = float(sim_cfg["dt"])
    record = bool(cfg["viz"]["record"])
    # mode = str(cfg["viz"]["mode"]) # Removed

    sub_steps = int(
        sim_cfg["substeps"]["record"] if record else sim_cfg["substeps"]["live"]
    )

    goal_delta = float(term_cfg["goal_delta"])
    settle_seconds = float(term_cfg["settle_seconds"])
    max_frames = int(term_cfg["max_frames"])
    frames_after_reach = int(np.ceil(settle_seconds / (dt * sub_steps)))

    agents = [
        {"pos": p.copy(), "goal": goals_init[i].copy(), "vel": np.zeros(3), "path": []}
        for i, p in enumerate(agents_init)
    ]
    logger = SafetyLogger(len(agents))

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d" if dim == 3 else None)
    if dim == 2:
        ax.set_aspect("equal")
        walls_cfg = cfg["world"]["obstacles"].get("walls_2d", {})
        if walls_cfg.get("enabled", False):
            ax.set_xlim(float(walls_cfg["x_min"]) - 2, float(walls_cfg["x_max"]) + 2)
            ax.set_ylim(float(walls_cfg["y_min"]) - 2, float(walls_cfg["y_max"]) + 2)
    else:
        if cfg["world"]["obstacles"].get("walls_3d", {}).get("enabled", False):
            w = cfg["world"]["obstacles"]["walls_3d"]
            xx, yy = np.meshgrid([w["x_min"], w["x_max"]], [w["y_min"], w["y_max"]])
            ax.plot_surface(xx, yy, np.full_like(xx, w["z_min"]), alpha=0.05, color="k")
            ax.plot_surface(xx, yy, np.full_like(xx, w["z_max"]), alpha=0.05, color="k")

    lines = [
        ax.plot([], [], [] if dim == 3 else "-", lw=2, label=f"A{i}")[0]
        for i in range(len(agents))
    ]
    if dim == 3:
        (centroid_plot,) = ax.plot([0], [0], [0], "kx", markersize=10, label="Centroid")
    else:
        (centroid_plot,) = ax.plot([0], [0], "kx", markersize=10, label="Centroid")

    for i, g in enumerate(goals_init):
        color = lines[i].get_color()
        if dim == 2:
            ax.scatter(g[0], g[1], marker="*", s=150, c=color, alpha=0.5)
        else:
            ax.scatter(g[0], g[1], g[2], marker="*", s=150, c=color, alpha=0.5)

    agent_body_patches = []
    agent_halo_patches = []
    agent_radius = float(cbf_cfg["agent_radius"])
    buffer_obs = float(cbf_cfg["buffer_obstacles"])
    buffer_agents = float(cbf_cfg["buffer_agents"])

    if dim == 2:
        for i, a in enumerate(agents):
            color = lines[i].get_color()

            body = plt.Circle(a["pos"][:2], agent_radius, color=color, alpha=0.9)
            ax.add_patch(body)
            agent_body_patches.append(body)

            halo = plt.Circle(
                a["pos"][:2],
                agent_radius + buffer_agents,
                fill=False,
                linewidth=2,
                color=color,
                alpha=0.35,
            )
            ax.add_patch(halo)
            agent_halo_patches.append(halo)
    else:
        for i, a in enumerate(agents):
            agent_body_patches.append(None)
            agent_halo_patches.append(None)

    if dim == 2:
        obs_vis = []
        obs_halos = []

        for o in obs:
            if o.get("kind", "sphere") == "sphere":
                patch = plt.Circle(o["pos"][:2], o["r"], alpha=0.20, color="r")
                ax.add_patch(patch)
                obs_vis.append(patch)

                halo = plt.Circle(
                    o["pos"][:2],
                    o["r"] + buffer_obs,
                    fill=False,
                    linewidth=2,
                    color="r",
                    alpha=0.25,
                )
                ax.add_patch(halo)
                obs_halos.append(halo)

            elif o.get("kind") == "wall":
                n, p = o["normal"], o["pos"]
                t_vec = np.array([-n[1], n[0], 0])
                p1, p2 = p + t_vec * 20, p - t_vec * 20
                (line,) = ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "r--", alpha=0.4)
                obs_vis.append(line)
    else:
        obs_vis = [
            ax.plot_wireframe(
                np.zeros((2, 2)),
                np.zeros((2, 2)),
                np.zeros((2, 2)),
                alpha=0.1,
            )
            for _ in obs
        ]

    reached_frame = None
    done = False

    def all_agents_within_delta():
        for a in agents:
            d = np.linalg.norm(a["pos"][:dim] - a["goal"][:dim])
            if d > goal_delta:
                return False
        return True

    def update(frame):
        nonlocal obs_vis, reached_frame, done
        swarm_pos_accum = np.zeros(3)

        w_cfg = cfg["world"]["obstacles"].get(
            "walls_2d" if dim == 2 else "walls_3d", {}
        )

        dyn_cfg = cfg["world"]["obstacles"].get("dynamic", {}) or {}
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

            # 2b) HARD world bounds for agents (prevents discrete-time wall tunneling)
            agents_enforce_world_bounds(
                agents=agents,
                w_cfg=w_cfg,
                dim=dim,
                agent_radius=agent_radius,
                restitution=rest_agents,
            )

            # 3) Log safety
            logger.log(
                (frame * sub_steps + s) * dt,
                agents,
                obs,
                buffer_obs=float(cbf_cfg["buffer_obstacles"]),
                buffer_agents=float(cbf_cfg["buffer_agents"]),
                agent_radius=float(cbf_cfg["agent_radius"]),
                l2=float(getattr(ctrl, "last_l2", 0.0)),
                delta=float(getattr(ctrl, "last_delta", 0.0)),
            )

        if reached_frame is None and all_agents_within_delta():
            reached_frame = frame

        if reached_frame is not None and frame >= reached_frame + frames_after_reach:
            done = True
            if not record:
                if hasattr(ani, "event_source") and ani.event_source is not None:
                    ani.event_source.stop()
                plt.close(fig)

        # Update trajectories
        for i, a in enumerate(agents):
            h = np.array(a["path"])
            lines[i].set_data(h[:, 0], h[:, 1])
            if dim == 3:
                lines[i].set_3d_properties(h[:, 2])

        c_p = swarm_pos_accum / (len(agents) * sub_steps)
        centroid_plot.set_data([c_p[0]], [c_p[1]])
        if dim == 3:
            centroid_plot.set_3d_properties([c_p[2]])

        # Update agent visuals
        if dim == 2:
            for i, a in enumerate(agents):
                agent_body_patches[i].center = a["pos"][:2]
                agent_halo_patches[i].center = a["pos"][:2]
        elif frame % 5 == 0:
            u_m, v_m = np.mgrid[0 : 2 * np.pi : 10j, 0 : np.pi : 10j]
            for i, a in enumerate(agents):
                color = lines[i].get_color()

                if agent_body_patches[i] is not None:
                    agent_body_patches[i].remove()
                agent_body_patches[i] = ax.plot_wireframe(
                    a["pos"][0] + agent_radius * np.cos(u_m) * np.sin(v_m),
                    a["pos"][1] + agent_radius * np.sin(u_m) * np.sin(v_m),
                    a["pos"][2] + agent_radius * np.cos(v_m),
                    color=color,
                    alpha=0.8,
                )

                if agent_halo_patches[i] is not None:
                    agent_halo_patches[i].remove()
                r_halo = agent_radius + buffer_agents
                agent_halo_patches[i] = ax.plot_wireframe(
                    a["pos"][0] + r_halo * np.cos(u_m) * np.sin(v_m),
                    a["pos"][1] + r_halo * np.sin(u_m) * np.sin(v_m),
                    a["pos"][2] + r_halo * np.cos(v_m),
                    color=color,
                    alpha=0.15,
                )

        # Update obstacle visuals
        if dim == 2:
            sphere_idx = 0
            for i, o in enumerate(obs):
                if o.get("kind") == "sphere":
                    if hasattr(obs_vis[i], "center"):
                        obs_vis[i].center = o["pos"][:2]
                    obs_halos[sphere_idx].center = o["pos"][:2]
                    sphere_idx += 1
                elif o.get("kind") == "wall":
                    if hasattr(obs_vis[i], "set_data"):
                        n, p = o["normal"], o["pos"]
                        t_vec = np.array([-n[1], n[0], 0])
                        p1, p2 = p + t_vec * 20, p - t_vec * 20
                        obs_vis[i].set_data([p1[0], p2[0]], [p1[1], p2[1]])
        elif frame % 5 == 0:
            for i, o in enumerate(obs):
                if o.get("kind") == "sphere":
                    obs_vis[i].remove()
                    u_m, v_m = np.mgrid[0 : 2 * np.pi : 10j, 0 : np.pi : 10j]
                    obs_vis[i] = ax.plot_wireframe(
                        o["pos"][0] + o["r"] * np.cos(u_m) * np.sin(v_m),
                        o["pos"][1] + o["r"] * np.sin(u_m) * np.sin(v_m),
                        o["pos"][2] + o["r"] * np.cos(v_m),
                        alpha=0.1,
                        color="r",
                    )

        return lines + [centroid_plot]

    def frame_gen():
        for k in count(0):
            if done or k >= max_frames:
                break
            yield k

    interval_ms = int(sim_cfg["interval_ms"])

    if True:  # Always animate
        ani = FuncAnimation(
            fig,
            update,
            frames=frame_gen(),
            interval=interval_ms,
            blit=False,
            cache_frame_data=False,
        )

        if getattr(ani, "event_source", None) is None and not record:
            print(
                "[warn] Matplotlib backend cannot animate (no timer). Falling back to record mode."
            )
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
