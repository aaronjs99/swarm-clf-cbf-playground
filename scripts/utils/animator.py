import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def _cuboid_faces(center, size):
    cx, cy, cz = [float(v) for v in center]
    lx, ly, lz = [float(v) for v in size]
    dx, dy, dz = lx / 2.0, ly / 2.0, lz / 2.0

    x0, x1 = cx - dx, cx + dx
    y0, y1 = cy - dy, cy + dy
    z0, z1 = cz - dz, cz + dz

    v = np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ],
        dtype=float,
    )

    faces = [
        [v[0], v[1], v[2], v[3]],  # bottom
        [v[4], v[5], v[6], v[7]],  # top
        [v[0], v[1], v[5], v[4]],  # side
        [v[1], v[2], v[6], v[5]],  # side
        [v[2], v[3], v[7], v[6]],  # side
        [v[3], v[0], v[4], v[7]],  # side
    ]
    return faces


def _ellipsoid_mesh(center, radii, nu=14, nv=10):
    cx, cy, cz = [float(v) for v in center]
    a, b, c = [float(r) for r in radii]

    u = np.linspace(0, 2 * np.pi, nu)
    v = np.linspace(0, np.pi, nv)
    uu, vv = np.meshgrid(u, v)

    x = cx + a * np.cos(uu) * np.sin(vv)
    y = cy + b * np.sin(uu) * np.sin(vv)
    z = cz + c * np.cos(vv)
    return x, y, z


def _cylinder_mesh(center_xy, z0, z1, radius, nu=14):
    cx, cy = [float(v) for v in center_xy]
    z0 = float(z0)
    z1 = float(z1)
    r = float(radius)

    u = np.linspace(0, 2 * np.pi, nu)
    uu, zz = np.meshgrid(u, np.array([z0, z1]))
    x = cx + r * np.cos(uu)
    y = cy + r * np.sin(uu)
    z = zz
    return x, y, z


class SwarmAnimator:
    def __init__(
        self, ax, agents, goals, obs, dim, agent_radius, buffer_agents, buffer_obs
    ):
        self.ax = ax
        self.dim = dim
        self.agent_radius = float(agent_radius)
        self.buffer_agents = float(buffer_agents)
        self.buffer_obs = float(buffer_obs)

        # Init Lines
        self.lines = [
            ax.plot([], [], [] if dim == 3 else "-", lw=2, label=f"A{i}")[0]
            for i in range(len(agents))
        ]

        # Init Centroid
        if dim == 3:
            (self.centroid_plot,) = ax.plot([0], [0], [0], "kx", markersize=10, label="Centroid")
        else:
            (self.centroid_plot,) = ax.plot([0], [0], "kx", markersize=10, label="Centroid")

        # Init Goals
        for i, g in enumerate(goals):
            color = self.lines[i].get_color()
            if dim == 2:
                ax.scatter(g[0], g[1], marker="*", s=150, c=color, alpha=0.5)
            else:
                ax.scatter(g[0], g[1], g[2], marker="*", s=150, c=color, alpha=0.5)

        # Init Agent Bodies
        self.agent_body_patches = []
        self.agent_halo_patches = []

        if dim == 2:
            for i, a in enumerate(agents):
                color = self.lines[i].get_color()
                body = plt.Circle(a["pos"][:2], self.agent_radius, color=color, alpha=0.9)
                ax.add_patch(body)
                self.agent_body_patches.append(body)

                halo = plt.Circle(
                    a["pos"][:2],
                    self.agent_radius + self.buffer_agents,
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

        # Obstacles visualization
        self.obs_artists = []
        self._init_obstacles_3d(obs) if dim == 3 else self._init_obstacles_2d(obs)

        # Nice axes labels in 3D
        if dim == 3:
            self.ax.set_xlabel("x")
            self.ax.set_ylabel("y")
            self.ax.set_zlabel("z")

    def _init_obstacles_2d(self, obs):
        self.obs_vis = []
        self.obs_halos = []

        for o in obs:
            if o.get("kind", "sphere") == "sphere":
                patch = plt.Circle(o["pos"][:2], o["r"], alpha=0.20, color="r")
                self.ax.add_patch(patch)
                self.obs_vis.append(patch)

                halo = plt.Circle(
                    o["pos"][:2],
                    o["r"] + self.buffer_obs,
                    fill=False,
                    linewidth=2,
                    color="r",
                    alpha=0.25,
                )
                self.ax.add_patch(halo)
                self.obs_halos.append(halo)

            elif o.get("kind") == "wall":
                n, p = o["normal"], o["pos"]
                t_vec = np.array([-n[1], n[0], 0])
                p1, p2 = p + t_vec * 20, p - t_vec * 20
                (line,) = self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "r--", alpha=0.4)
                self.obs_vis.append(line)

    def _init_obstacles_3d(self, obs):
        """
        Draw static 3D obstacles once. Avoids redrawing every frame.
        """
        for o in obs:
            if o.get("kind") == "wall":
                continue

            render = o.get("render", "sphere")

            if render == "car":
                col = (o.get("colors") or {}).get("body", "#64748b")
                size = np.asarray((o.get("car") or {}).get("size", [1.8, 0.9, 0.45]), dtype=float).reshape(3)
                faces = _cuboid_faces(o["pos"], size)
                poly = Poly3DCollection(faces, alpha=0.70, facecolor=col, edgecolor="none")
                self.ax.add_collection3d(poly)
                self.obs_artists.append(poly)

            elif render == "saucer":
                col = (o.get("colors") or {}).get("body", "#94a3b8")
                radii = np.asarray((o.get("saucer") or {}).get("radii", [1.0, 1.0, 0.15]), dtype=float).reshape(3)
                x, y, z = _ellipsoid_mesh(o["pos"], radii, nu=16, nv=10)
                surf = self.ax.plot_surface(x, y, z, alpha=0.55, linewidth=0, color=col)
                self.obs_artists.append(surf)

            elif render == "tree":
                colors = o.get("colors") or {}
                trunk_col = colors.get("trunk", "#8b5a2b")
                canopy_col = colors.get("canopy", "#16a34a")
                td = o.get("tree") or {}
                trunk_r = float(td.get("trunk_radius", 0.12))
                trunk_h = float(td.get("trunk_height", 3.2))
                canopy_r = float(td.get("canopy_radius", 0.9))
                z_base = float(td.get("z_base", 0.0))

                # trunk center in xy is tree center xy
                x0, y0 = float(o["pos"][0]), float(o["pos"][1])

                # cylinder trunk
                xt, yt, zt = _cylinder_mesh((x0, y0), z_base, z_base + trunk_h, trunk_r, nu=14)
                trunk = self.ax.plot_surface(xt, yt, zt, alpha=0.75, linewidth=0, color=trunk_col)
                self.obs_artists.append(trunk)

                # canopy
                canopy_center = np.array([x0, y0, z_base + trunk_h + 0.35 * canopy_r], dtype=float)
                xr, yr, zr = _ellipsoid_mesh(canopy_center, (canopy_r, canopy_r, 0.8 * canopy_r), nu=14, nv=10)
                canopy = self.ax.plot_surface(xr, yr, zr, alpha=0.55, linewidth=0, color=canopy_col)
                self.obs_artists.append(canopy)

            else:
                # default: sphere-like obstacle
                r = float(o.get("r", 0.5))
                x, y, z = _ellipsoid_mesh(o["pos"], (r, r, r), nu=14, nv=10)
                surf = self.ax.plot_surface(x, y, z, alpha=0.15, linewidth=0)
                self.obs_artists.append(surf)

    def _redraw_obstacles_3d(self, obs):
        # Remove old artists
        for art in self.obs_artists:
            try:
                art.remove()
            except Exception:
                pass
        self.obs_artists = []
        self._init_obstacles_3d(obs)

    def update(self, agents, obs, swarm_center, frame):
        # Update trajectories
        for i, a in enumerate(agents):
            h = np.array(a["path"])
            if h.size == 0:
                continue
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
        else:
            # Lightweight agent wireframes, update occasionally
            if frame % 4 == 0:
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
                        alpha=0.12,
                    )

        # Update Obstacles (2D needs per-frame updates)
        if self.dim == 2:
            vis_i = 0
            halo_i = 0
            for o in obs:
                kind = o.get("kind", "sphere")
                if kind == "sphere":
                    # Move filled obstacle
                    self.obs_vis[vis_i].center = o["pos"][:2]
                    vis_i += 1

                    # Move halo
                    self.obs_halos[halo_i].center = o["pos"][:2]
                    halo_i += 1
                elif kind == "wall":
                    # Walls are static, but keep index consistent if you stored them in obs_vis
                    vis_i += 1
        else:
            # 3D obstacles may move. Redraw occasionally to keep it cheap.
            if frame % 2 == 0:
                self._redraw_obstacles_3d(obs)

        return self.lines + [self.centroid_plot]
