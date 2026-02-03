import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.animation import FuncAnimation, FFMpegWriter


class SafetyLogger:
    def __init__(self, num_agents):
        self.h_buf_history = [[] for _ in range(num_agents)]
        self.h_phys_history = [[] for _ in range(num_agents)]
        self.t_history = []

    def log(self, t, agents, obstacles, buffer):
        self.t_history.append(t)
        for i, a in enumerate(agents):
            h_buf_vals, h_phys_vals = [], []
            for o in obstacles:
                d_sq = np.linalg.norm(a["pos"] - o["pos"]) ** 2
                h_buf_vals.append(d_sq - (o["r"] + buffer) ** 2)
                h_phys_vals.append(d_sq - o["r"] ** 2)
            for j, other in enumerate(agents):
                if i != j:
                    d_sq = np.linalg.norm(a["pos"] - other["pos"]) ** 2
                    h_buf_vals.append(d_sq - (0.6 + buffer) ** 2)
                    h_phys_vals.append(d_sq - 0.6**2)
            self.h_buf_history[i].append(min(h_buf_vals) if h_buf_vals else 10.0)
            self.h_phys_history[i].append(min(h_phys_vals) if h_phys_vals else 10.0)

    def plot(self, title):
        plt.figure(figsize=(10, 6))
        for i in range(len(self.h_buf_history)):
            plt.plot(self.t_history, self.h_buf_history[i], label=f"Agent {i}")
        plt.axhline(y=0, color="k", linestyle="--", label="Safety Boundary")
        min_phys = np.array(self.h_phys_history).min(axis=0)
        plt.fill_between(
            self.t_history,
            -1,
            0,
            where=(min_phys < -0.01),
            color="red",
            alpha=0.3,
            label="Actual Collision",
        )
        plt.ylabel("h(x)")
        plt.xlabel("Time [s]")
        plt.title(title)
        plt.grid(True)
        plt.legend(loc="lower right")
        plt.show()


def get_obstacles(dynamic, num, agents_init, goals_init, d_idx):
    np.random.seed(42)
    obs = []
    clear_zones = agents_init + goals_init
    while len(obs) < num:
        pos = np.random.uniform(0.5, 9.5, 3)
        if d_idx == 2:
            pos[2] = 0.0
        r = np.random.uniform(0.4, 0.6)
        if any(np.linalg.norm(pos - p) < (r + 1.2) for p in clear_zones):
            continue
        vel = np.random.uniform(-0.2, 0.2, 3)
        if d_idx == 2:
            vel[2] = 0.0
        obs.append({"pos": pos, "r": r, "vel": vel if dynamic else np.zeros(3)})
    return obs


def run_simulation(config):
    d_idx = 2 if config["dim"] == "2d" else 3
    n = config["num_agents"]
    if d_idx == 2:
        agents_init = [
            np.array(
                [
                    3.0 + 2.8 * np.cos((2 * np.pi / n) * i),
                    3.0 + 2.8 * np.sin((2 * np.pi / n) * i),
                    0.0,
                ]
            )
            for i in range(n)
        ]
        goals_init = [
            np.array([3.0 - (p[0] - 3.0), 3.0 - (p[1] - 3.0), 0.0]) for p in agents_init
        ]
    else:
        agents_init = [np.array([0.0, i * 1.5, i * 1.5]) for i in range(n)]
        goals_init = [
            np.array([10.0, 10.0 - i * 1.5, 10.0 - i * 1.5]) for i in range(n)
        ]

    obs = get_obstacles(
        config["dynamic"], config["num_obs"], agents_init, goals_init, d_idx
    )
    _execute_sim(config, obs, agents_init, goals_init, d_idx)


def _execute_sim(config, obs, agents_init, goals_init, dim):
    n, ctrl, dt = config["num_agents"], config["ctrl"], 0.02
    agents = [
        {"pos": p.copy(), "goal": goals_init[i].copy(), "vel": np.zeros(3), "path": []}
        for i, p in enumerate(agents_init)
    ]
    logger = SafetyLogger(n)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d" if dim == 3 else None)
    if dim == 3:
        lines = [ax.plot([], [], [], "-", lw=2, label=f"A{i}")[0] for i in range(n)]
        (centroid_plot,) = ax.plot([0], [0], [0], "kx", markersize=10, label="Centroid")
    else:
        lines = [ax.plot([], [], "-", lw=2, label=f"A{i}")[0] for i in range(n)]
        (centroid_plot,) = ax.plot([0], [0], "kx", markersize=10, label="Centroid")

    # Plot Goals with matching agent colors
    for i, g in enumerate(goals_init):
        color = lines[i].get_color()
        if dim == 2:
            ax.scatter(g[0], g[1], marker="*", s=150, c=color, alpha=0.5)
        else:
            ax.scatter(g[0], g[1], g[2], marker="*", s=150, c=color, alpha=0.5)

    if dim == 2:
        obs_vis = [plt.Circle(o["pos"][:2], o["r"], color="r", alpha=0.15) for o in obs]
        for c in obs_vis:
            ax.add_patch(c)
        ax.set_xlim(-1, 7)
        ax.set_ylim(-1, 7)
    else:
        obs_vis = [
            ax.plot_wireframe(
                np.zeros((2, 2)),
                np.zeros((2, 2)),
                np.zeros((2, 2)),
                color="r",
                alpha=0.1,
            )
            for _ in obs
        ]
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_zlim(0, 10)
    ax.legend()

    def update(frame):
        for i, o in enumerate(obs):
            o["pos"] += o["vel"] * dt
            if dim == 2:
                obs_vis[i].center = o["pos"][:2]
            elif frame % 5 == 0:
                obs_vis[i].remove()
                u_m, v_m = np.mgrid[0 : 2 * np.pi : 10j, 0 : np.pi : 10j]
                obs_vis[i] = ax.plot_wireframe(
                    o["pos"][0] + o["r"] * np.cos(u_m) * np.sin(v_m),
                    o["pos"][1] + o["r"] * np.sin(u_m) * np.sin(v_m),
                    o["pos"][2] + o["r"] * np.cos(v_m),
                    color="r",
                    alpha=0.1,
                )

        curr_states = [{"pos": a["pos"], "vel": a["vel"]} for a in agents]
        swarm_pos = np.zeros(3)
        for i, a in enumerate(agents):
            acc = ctrl.compute_control(
                a["pos"], a["vel"], a["goal"], obs, i, curr_states
            )
            if dim == 2:
                acc[2] = 0.0  # Force mathematical 2D plane
            a["vel"] += acc * dt
            a["pos"] += a["vel"] * dt
            a["path"].append(a["pos"].copy())
            h = np.array(a["path"])
            lines[i].set_data(h[:, 0], h[:, 1])
            if dim == 3:
                lines[i].set_3d_properties(h[:, 2])
            swarm_pos += a["pos"]

        c_p = swarm_pos / n
        centroid_plot.set_data([c_p[0]], [c_p[1]])
        if dim == 3:
            centroid_plot.set_3d_properties([c_p[2]])
        logger.log(frame * dt, agents, obs, getattr(ctrl, "safe_buffer", 0.0))
        return lines + [centroid_plot]

    ani = FuncAnimation(fig, update, frames=600, interval=30, blit=False)
    if config["record"]:
        os.makedirs("media", exist_ok=True)
        ani.save(f"media/swarm_{dim}d.mp4", writer=FFMpegWriter(fps=30))
    else:
        plt.show()
    if config["safety_plot"]:
        logger.plot(f"{dim}D Swarm Certificate")
