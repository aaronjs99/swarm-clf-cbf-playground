import numpy as np
import matplotlib.pyplot as plt


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
