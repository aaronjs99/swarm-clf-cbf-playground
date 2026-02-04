import numpy as np
import matplotlib.pyplot as plt


class SwarmAnimator:
    def __init__(
        self, ax, agents, goals, obs, dim, agent_radius, buffer_agents, buffer_obs
    ):
        self.ax = ax
        self.dim = dim
        self.agent_radius = agent_radius
        self.buffer_agents = buffer_agents
        self.buffer_obs = buffer_obs

        # Init Lines
        self.lines = [
            ax.plot([], [], [] if dim == 3 else "-", lw=2, label=f"A{i}")[0]
            for i in range(len(agents))
        ]

        # Init Centroid
        if dim == 3:
            (self.centroid_plot,) = ax.plot(
                [0], [0], [0], "kx", markersize=10, label="Centroid"
            )
        else:
            (self.centroid_plot,) = ax.plot(
                [0], [0], "kx", markersize=10, label="Centroid"
            )

        # Init Goals
        for i, g in enumerate(goals):
            color = self.lines[i].get_color()
            if dim == 2:
                ax.scatter(g[0], g[1], marker="*", s=150, c=color, alpha=0.5)
            else:
                ax.scatter(g[0], g[1], g[2], marker="*", s=150, c=color, alpha=0.5)

        # Init Bodies & Halos
        self.agent_body_patches = []
        self.agent_halo_patches = []

        if dim == 2:
            for i, a in enumerate(agents):
                color = self.lines[i].get_color()
                body = plt.Circle(a["pos"][:2], agent_radius, color=color, alpha=0.9)
                ax.add_patch(body)
                self.agent_body_patches.append(body)

                halo = plt.Circle(
                    a["pos"][:2],
                    agent_radius + buffer_agents,
                    fill=False,
                    linewidth=2,
                    color=color,
                    alpha=0.35,
                )
                ax.add_patch(halo)
                self.agent_halo_patches.append(halo)
        else:
            for i in range(len(agents)):
                self.agent_body_patches.append(None)
                self.agent_halo_patches.append(None)

        # Init Obstacles
        self.obs_vis = []
        self.obs_halos = []

        if dim == 2:
            for o in obs:
                if o.get("kind", "sphere") == "sphere":
                    patch = plt.Circle(o["pos"][:2], o["r"], alpha=0.20, color="r")
                    ax.add_patch(patch)
                    self.obs_vis.append(patch)

                    halo = plt.Circle(
                        o["pos"][:2],
                        o["r"] + buffer_obs,
                        fill=False,
                        linewidth=2,
                        color="r",
                        alpha=0.25,
                    )
                    ax.add_patch(halo)
                    self.obs_halos.append(halo)

                elif o.get("kind") == "wall":
                    n, p = o["normal"], o["pos"]
                    t_vec = np.array([-n[1], n[0], 0])
                    p1, p2 = p + t_vec * 20, p - t_vec * 20
                    (line,) = ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "r--", alpha=0.4)
                    self.obs_vis.append(line)
        else:
            self.obs_vis = [
                ax.plot_wireframe(
                    np.zeros((2, 2)),
                    np.zeros((2, 2)),
                    np.zeros((2, 2)),
                    alpha=0.1,
                )
                for _ in obs
            ]

    def update(self, agents, obs, swarm_center, frame):
        # Update trajectories
        for i, a in enumerate(agents):
            h = np.array(a["path"])
            self.lines[i].set_data(h[:, 0], h[:, 1])
            if self.dim == 3:
                self.lines[i].set_3d_properties(h[:, 2])

        # Update centroid
        self.centroid_plot.set_data([swarm_center[0]], [swarm_center[1]])
        if self.dim == 3:
            self.centroid_plot.set_3d_properties([swarm_center[2]])

        # Update Agents
        if self.dim == 2:
            for i, a in enumerate(agents):
                self.agent_body_patches[i].center = a["pos"][:2]
                self.agent_halo_patches[i].center = a["pos"][:2]
        elif frame % 5 == 0:
            u_m, v_m = np.mgrid[0 : 2 * np.pi : 10j, 0 : np.pi : 10j]
            for i, a in enumerate(agents):
                color = self.lines[i].get_color()
                if self.agent_body_patches[i] is not None:
                    self.agent_body_patches[i].remove()

                self.agent_body_patches[i] = self.ax.plot_wireframe(
                    a["pos"][0] + self.agent_radius * np.cos(u_m) * np.sin(v_m),
                    a["pos"][1] + self.agent_radius * np.sin(u_m) * np.sin(v_m),
                    a["pos"][2] + self.agent_radius * np.cos(v_m),
                    color=color,
                    alpha=0.8,
                )

                if self.agent_halo_patches[i] is not None:
                    self.agent_halo_patches[i].remove()
                r_halo = self.agent_radius + self.buffer_agents
                self.agent_halo_patches[i] = self.ax.plot_wireframe(
                    a["pos"][0] + r_halo * np.cos(u_m) * np.sin(v_m),
                    a["pos"][1] + r_halo * np.sin(u_m) * np.sin(v_m),
                    a["pos"][2] + r_halo * np.cos(v_m),
                    color=color,
                    alpha=0.15,
                )

        # Update Obstacles
        if self.dim == 2:
            sphere_idx = 0
            for i, o in enumerate(obs):
                if o.get("kind") == "sphere":
                    if hasattr(self.obs_vis[i], "center"):
                        self.obs_vis[i].center = o["pos"][:2]
                    self.obs_halos[sphere_idx].center = o["pos"][:2]
                    sphere_idx += 1
                elif o.get("kind") == "wall":
                    if hasattr(self.obs_vis[i], "set_data"):
                        pass  # Static walls don't need updates currently
        elif frame % 5 == 0:
            for i, o in enumerate(obs):
                if o.get("kind") == "sphere":
                    self.obs_vis[i].remove()
                    u_m, v_m = np.mgrid[0 : 2 * np.pi : 10j, 0 : np.pi : 10j]
                    self.obs_vis[i] = self.ax.plot_wireframe(
                        o["pos"][0] + o["r"] * np.cos(u_m) * np.sin(v_m),
                        o["pos"][1] + o["r"] * np.sin(u_m) * np.sin(v_m),
                        o["pos"][2] + o["r"] * np.cos(v_m),
                        alpha=0.1,
                        color="r",
                    )

        return self.lines + [self.centroid_plot]
