import os
from itertools import count

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter


class SafetyLogger:
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
                    d_sq = np.linalg.norm(a["pos"] - o["pos"]) ** 2
                    h_buf_vals.append(d_sq - (o["r"] + buffer_obs) ** 2)
                    h_phys_vals.append(d_sq - o["r"] ** 2)
                elif kind == "wall":
                    # n^T(x - p) - buffer
                    h_val = np.dot(o["normal"], a["pos"] - o["pos"])
                    h_buf_vals.append(h_val - buffer_obs)
                    h_phys_vals.append(h_val)

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


def _coerce_int_or_none(x):
    if x is None:
        return None
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, float):
        return int(x)
    if isinstance(x, str) and x.strip() == "":
        return None
    return int(x)


def get_obstacles(cfg, agents_init, goals_init, dim):
    obs_cfg = cfg["world"]["obstacles"]
    np.random.seed(int(cfg.get("seed", 42)))
    obs = []

    # 1. Instantiate the Box (4 Walls)
    if dim == 2 and obs_cfg.get("walls_2d", {}).get("enabled", False):
        w = obs_cfg["walls_2d"]
        # Normals point INWARD to safe space
        obs.append(
            {
                "kind": "wall",
                "pos": np.array([0, float(w["y_min"]), 0]),
                "normal": np.array([0, 1.0, 0]),
                "vel": np.zeros(3),
            }
        )
        obs.append(
            {
                "kind": "wall",
                "pos": np.array([0, float(w["y_max"]), 0]),
                "normal": np.array([0, -1.0, 0]),
                "vel": np.zeros(3),
            }
        )
        obs.append(
            {
                "kind": "wall",
                "pos": np.array([float(w["x_min"]), 0, 0]),
                "normal": np.array([1.0, 0, 0]),
                "vel": np.zeros(3),
            }
        )
        obs.append(
            {
                "kind": "wall",
                "pos": np.array([float(w["x_max"]), 0, 0]),
                "normal": np.array([-1.0, 0, 0]),
                "vel": np.zeros(3),
            }
        )

        # Spawn spheres inside the walls
        margin = 1.0
        box_min = np.array([w["x_min"] + margin, w["y_min"] + margin, 0.0])
        box_max = np.array([w["x_max"] - margin, w["y_max"] - margin, 0.0])
    else:
        box_key = "spawn_box_2d" if dim == 2 else "spawn_box_3d"
        box_min = np.array(obs_cfg[box_key]["min"], dtype=float)
        box_max = np.array(obs_cfg[box_key]["max"], dtype=float)

    # 2. Sphere Generation with initial velocity
    num = int(
        obs_cfg.get("num_override")
        or (obs_cfg["num_obs_2d"] if dim == 2 else obs_cfg["num_obs_3d"])
    )
    r_lo, r_hi = obs_cfg["radius_range"]
    v_lo, v_hi = obs_cfg["dynamic"]["vel_range"]

    while len(obs) < num:
        pos = np.random.uniform(box_min, box_max)
        if dim == 2:
            pos[2] = 0.0
        r = float(np.random.uniform(r_lo, r_hi))
        if any(np.linalg.norm(pos - p) < (r + 0.8) for p in agents_init + goals_init):
            continue

        vel = np.random.uniform(v_lo, v_hi, 3)
        if dim == 2:
            vel[2] = 0.0
        obs.append({"pos": pos, "r": r, "vel": vel, "kind": "sphere"})

    return obs


def _init_agents_and_goals(cfg, dim, n):
    np.random.seed(int(cfg.get("seed", 42)))

    if dim == 2:
        init_cfg = cfg["agents"]["init"]
        goal_cfg = cfg["agents"]["goals"]
        walls_cfg = cfg["world"]["obstacles"].get("walls_2d", {})

        # Determine y-centering based on walls if they exist
        if walls_cfg.get("enabled", False):
            y_mid = (walls_cfg["y_max"] + walls_cfg["y_min"]) / 2.0
            # Start lanes so the whole swarm is centered vertically
            y_start = y_mid - ((n - 1) * init_cfg["lanes"]["y_spacing"]) / 2.0
        else:
            y_start = 0.0

        if init_cfg["mode_2d"] == "circle":
            center = np.array(init_cfg["circle"]["center"], dtype=float)
            radius = float(init_cfg["circle"]["radius"])
            agents_init = [
                center
                + np.array(
                    [
                        radius * np.cos((2 * np.pi / n) * i),
                        radius * np.sin((2 * np.pi / n) * i),
                        0.0,
                    ]
                )
                for i in range(n)
            ]
        else:
            lanes = init_cfg["lanes"]
            y_spacing = float(lanes["y_spacing"])
            agents_init = [
                np.array(
                    [
                        np.random.uniform(lanes["x_min"], lanes["x_max"]),
                        y_start + i * y_spacing,
                        0.0,
                    ]
                )
                for i in range(n)
            ]

        if goal_cfg["mode_2d"] == "mirror":
            center = np.array(init_cfg["circle"]["center"], dtype=float)
            goals_init = [
                np.array(
                    [
                        center[0] - (p[0] - center[0]),
                        center[1] - (p[1] - center[1]),
                        0.0,
                    ]
                )
                for p in agents_init
            ]
        else:
            lanes = goal_cfg["lanes"]
            y_spacing = float(lanes["y_spacing"])
            reverse = bool(lanes.get("reverse_order", True))

            # Recalculate y_start for goals to ensure they are also centered
            y_start_goal = y_mid - ((n - 1) * y_spacing) / 2.0

            ys = (
                [(n - 1 - i) * y_spacing + y_start_goal for i in range(n)]
                if reverse
                else [i * y_spacing + y_start_goal for i in range(n)]
            )
            goals_init = [
                np.array(
                    [np.random.uniform(lanes["x_min"], lanes["x_max"]), ys[i], 0.0]
                )
                for i in range(n)
            ]

        return agents_init, goals_init

    agents_init = [np.array([0.0, i * 1.5, i * 1.5]) for i in range(n)]
    goals_init = [np.array([10.0, 10.0 - i * 1.5, 10.0 - i * 1.5]) for i in range(n)]
    return agents_init, goals_init


def run_simulation(cfg, ctrl):
    dim = 2 if cfg["viz"]["dim"] == "2d" else 3
    n = int(cfg["agents"]["num_agents"])

    agents_init, goals_init = _init_agents_and_goals(cfg, dim, n)
    obs = get_obstacles(cfg, agents_init, goals_init, dim)

    _execute_sim(cfg, ctrl, obs, agents_init, goals_init, dim)


def _execute_sim(cfg, ctrl, obs, agents_init, goals_init, dim):
    sim_cfg = cfg["sim"]
    term_cfg = sim_cfg["termination"]
    cbf_cfg = cfg["controller"]["cbf"]

    dt = float(sim_cfg["dt"])
    record = bool(cfg["viz"]["record"])
    mode = str(cfg["viz"]["mode"])

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
        ax.set_aspect("equal")  # Prevents circles from looking like ellipses

        # Frame the box based on config if walls are enabled
        walls_cfg = cfg["world"]["obstacles"].get("walls_2d", {})
        if walls_cfg.get("enabled", False):
            # Pad the limits by 2 units so the walls aren't touching the edge
            ax.set_xlim(float(walls_cfg["x_min"]) - 2, float(walls_cfg["x_max"]) + 2)
            ax.set_ylim(float(walls_cfg["y_min"]) - 2, float(walls_cfg["y_max"]) + 2)

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

    if dim == 2:
        obs_vis = []
        for o in obs:
            if o.get("kind", "sphere") == "sphere":
                patch = plt.Circle(o["pos"][:2], o["r"], alpha=0.15, color="r")
                ax.add_patch(patch)
                obs_vis.append(patch)
            elif o.get("kind") == "wall":
                # Draw a line for the wall boundary
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
        swarm_pos_accum = np.zeros(3)  # Fix for UnboundLocalError
        w_cfg = cfg["world"]["obstacles"].get("walls_2d", {})

        for s in range(sub_steps):
            # 1. Update Obstacles + BOUNCE Physics
            for o in obs:
                if o.get("kind") == "sphere":
                    o["pos"] += o["vel"] * dt

                    # Bounce off X walls (Left/Right)
                    if o["pos"][0] - o["r"] < w_cfg["x_min"]:
                        o["vel"][0] *= -1
                        o["pos"][0] = w_cfg["x_min"] + o["r"]
                    elif o["pos"][0] + o["r"] > w_cfg["x_max"]:
                        o["vel"][0] *= -1
                        o["pos"][0] = w_cfg["x_max"] - o["r"]

                    # Bounce off Y walls (Bottom/Top)
                    if o["pos"][1] - o["r"] < w_cfg["y_min"]:
                        o["vel"][1] *= -1
                        o["pos"][1] = w_cfg["y_min"] + o["r"]
                    elif o["pos"][1] + o["r"] > w_cfg["y_max"]:
                        o["vel"][1] *= -1
                        o["pos"][1] = w_cfg["y_max"] - o["r"]
                else:
                    o["pos"] += o["vel"] * dt

            # 2. Agent Control Loop
            curr_states = [{"pos": a["pos"], "vel": a["vel"]} for a in agents]

            for i, a in enumerate(agents):
                acc = ctrl.compute_control(
                    a["pos"], a["vel"], a["goal"], obs, i, curr_states
                )
                if dim == 2:
                    acc[2] = 0.0
                a["vel"] += acc * dt
                a["pos"] += a["vel"] * dt
                a["path"].append(a["pos"].copy())
                swarm_pos_accum += a["pos"]  # Accumulate positions

            # 3. Log Safety
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
                # Only stop if animation has an event source
                if hasattr(ani, "event_source") and ani.event_source is not None:
                    ani.event_source.stop()
                plt.close(fig)

        for i, a in enumerate(agents):
            h = np.array(a["path"])
            lines[i].set_data(h[:, 0], h[:, 1])
            if dim == 3:
                lines[i].set_3d_properties(h[:, 2])

        c_p = swarm_pos_accum / len(agents)
        centroid_plot.set_data([c_p[0]], [c_p[1]])
        if dim == 3:
            centroid_plot.set_3d_properties([c_p[2]])

        for i, o in enumerate(obs):
            if dim == 2:
                if hasattr(obs_vis[i], "center"):  # Sphere
                    obs_vis[i].center = o["pos"][:2]
                elif hasattr(obs_vis[i], "set_data"):  # Wall/Line
                    n, p = o["normal"], o["pos"]
                    t_vec = np.array([-n[1], n[0], 0])
                    p1, p2 = p + t_vec * 20, p - t_vec * 20
                    obs_vis[i].set_data([p1[0], p2[0]], [p1[1], p2[1]])
            elif frame % 5 == 0:
                obs_vis[i].remove()
                u_m, v_m = np.mgrid[0 : 2 * np.pi : 10j, 0 : np.pi : 10j]
                obs_vis[i] = ax.plot_wireframe(
                    o["pos"][0] + o["r"] * np.cos(u_m) * np.sin(v_m),
                    o["pos"][1] + o["r"] * np.sin(u_m) * np.sin(v_m),
                    o["pos"][2] + o["r"] * np.cos(v_m),
                    alpha=0.1,
                )

        return lines + [centroid_plot]

    def frame_gen():
        for k in count(0):
            if done or k >= max_frames:
                break
            yield k

    interval_ms = int(sim_cfg["interval_ms"])

    if mode == "still":
        # Run a single update and show the resulting figure
        update(0)
        plt.show(block=True)
    else:
        ani = FuncAnimation(
            fig,
            update,
            frames=frame_gen(),
            interval=interval_ms,
            blit=False,
            cache_frame_data=False,
        )

        # If backend cannot create timers, event_source ends up None.
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
                savefig_kwargs={"dpi": 150},
            )
        else:
            plt.show(block=True)

    if bool(cfg["viz"]["safety_plot"]):
        logger.plot(f"{dim}D Swarm Certificate")
