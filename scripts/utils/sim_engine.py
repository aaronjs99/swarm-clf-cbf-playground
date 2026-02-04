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
                # Ensure the logger accounts for the agent's radius
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


def _sphere_mass(r: float, dim: int) -> float:
    # Proportional to area (2D) or volume (3D). Constants cancel in impulse math.
    if dim == 2:
        return max(1e-9, r * r)
    return max(1e-9, r * r * r)


def _resolve_sphere_sphere_collisions(obs, dim: int, restitution: float = 1.0):
    # Only sphere-sphere collisions (ignore walls)
    spheres = [o for o in obs if o.get("kind") == "sphere"]
    n = len(spheres)
    if n <= 1:
        return

    e = float(restitution)

    for i in range(n):
        a = spheres[i]
        for j in range(i + 1, n):
            b = spheres[j]

            pa = a["pos"]
            pb = b["pos"]
            ra = float(a["r"])
            rb = float(b["r"])

            delta = pb - pa
            dist = float(np.linalg.norm(delta))
            min_dist = ra + rb

            if dist <= 1e-12:
                # Degenerate: random tiny separation to define a normal
                n_hat = np.array([1.0, 0.0, 0.0])
                dist = 1e-12
            else:
                n_hat = delta / dist

            # No overlap, no collision
            if dist >= min_dist:
                continue

            va = a["vel"]
            vb = b["vel"]

            # Separate positions (positional correction) to eliminate overlap
            penetration = min_dist - dist

            ma = _sphere_mass(ra, dim)
            mb = _sphere_mass(rb, dim)
            inv_ma = 1.0 / ma
            inv_mb = 1.0 / mb
            inv_sum = inv_ma + inv_mb

            # Push apart along normal, weighted by inverse mass (lighter moves more)
            pa -= n_hat * penetration * (inv_ma / inv_sum)
            pb += n_hat * penetration * (inv_mb / inv_sum)

            # Relative velocity along normal
            v_rel = vb - va
            v_rel_n = float(np.dot(v_rel, n_hat))

            # If separating already, do not apply impulse
            if v_rel_n >= 0.0:
                continue

            # Impulse magnitude
            j_imp = -(1.0 + e) * v_rel_n / inv_sum

            impulse = j_imp * n_hat
            va -= impulse * inv_ma
            vb += impulse * inv_mb

            # Re-assign (in case references are not in-place)
            a["pos"] = pa
            b["pos"] = pb
            a["vel"] = va
            b["vel"] = vb


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

    # 1. 2D/3D Wall Generation
    w_key = "walls_2d" if dim == 2 else "walls_3d"
    if obs_cfg.get(w_key, {}).get("enabled", False):
        w = obs_cfg[w_key]
        # X and Y walls (Common to both)
        obs.append(
            {
                "kind": "wall",
                "pos": np.array([0, w["y_min"], 0]),
                "normal": np.array([0, 1.0, 0]),
                "vel": np.zeros(3),
            }
        )
        obs.append(
            {
                "kind": "wall",
                "pos": np.array([0, w["y_max"], 0]),
                "normal": np.array([0, -1.0, 0]),
                "vel": np.zeros(3),
            }
        )
        obs.append(
            {
                "kind": "wall",
                "pos": np.array([w["x_min"], 0, 0]),
                "normal": np.array([1.0, 0, 0]),
                "vel": np.zeros(3),
            }
        )
        obs.append(
            {
                "kind": "wall",
                "pos": np.array([w["x_max"], 0, 0]),
                "normal": np.array([-1.0, 0, 0]),
                "vel": np.zeros(3),
            }
        )

        if dim == 3:
            # Z walls for Cuboid
            obs.append(
                {
                    "kind": "wall",
                    "pos": np.array([0, 0, w["z_min"]]),
                    "normal": np.array([0, 0, 1.0]),
                    "vel": np.zeros(3),
                }
            )
            obs.append(
                {
                    "kind": "wall",
                    "pos": np.array([0, 0, w["z_max"]]),
                    "normal": np.array([0, 0, -1.0]),
                    "vel": np.zeros(3),
                }
            )

        # Auto-calculate Spawn Box inside walls
        margin = 1.2
        box_min = np.array(
            [w["x_min"] + margin, w["y_min"] + margin, w.get("z_min", 0) + margin]
        )
        box_max = np.array(
            [w["x_max"] - margin, w["y_max"] - margin, w.get("z_max", 10) - margin]
        )
    else:
        box_min = np.array(obs_cfg["spawn_box_3d"]["min"])
        box_max = np.array(obs_cfg["spawn_box_3d"]["max"])

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
    w_key = "walls_2d" if dim == 2 else "walls_3d"
    w = cfg["world"]["obstacles"].get(w_key, {})

    # Calculate centering
    y_mid = (w["y_max"] + w["y_min"]) / 2.0 if w.get("enabled") else 0.0
    z_mid = (w["z_max"] + w["z_min"]) / 2.0 if (dim == 3 and w.get("enabled")) else 0.0

    y_spacing = cfg["agents"]["init"]["lanes"]["y_spacing"]
    y_start = y_mid - ((n - 1) * y_spacing) / 2.0

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
    else:
        # Centered 3D Lane Initialization
        agents_init = [
            np.array([w["x_min"] + 1.0, y_start + i * y_spacing, z_mid])
            for i in range(n)
        ]
        goals_init = [
            np.array([w["x_max"] - 1.0, y_start + (n - 1 - i) * y_spacing, z_mid])
            for i in range(n)
        ]

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
    else:
        # 3D Wall Visualization (Rendering the Floor/Ceiling)
        if cfg["world"]["obstacles"].get("walls_3d", {}).get("enabled", False):
            w = cfg["world"]["obstacles"]["walls_3d"]
            # Render a faint grid or plane for the floor (z_min)
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

            # Body: radius = agent_radius
            body = plt.Circle(a["pos"][:2], agent_radius, color=color, alpha=0.9)
            ax.add_patch(body)
            agent_body_patches.append(body)

            # Halo for agent-agent buffer: radius = agent_radius + buffer_agents
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
        # 3D: Initialize placeholders for wireframes
        for i, a in enumerate(agents):
            agent_body_patches.append(None)
            agent_halo_patches.append(None)

    if dim == 2:
        obs_vis = []
        obs_halos = []
        sphere_obs = [o for o in obs if o.get("kind") == "sphere"]

        for o in obs:
            if o.get("kind", "sphere") == "sphere":
                # Physical obstacle
                patch = plt.Circle(o["pos"][:2], o["r"], alpha=0.20, color="r")
                ax.add_patch(patch)
                obs_vis.append(patch)

                # Halo: obstacle buffer (thickness buffer_obs)
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
        w_cfg = cfg["world"]["obstacles"].get(
            "walls_2d" if dim == 2 else "walls_3d", {}
        )
        rest = float(
            cfg["world"]["obstacles"].get("dynamic", {}).get("restitution", 1.0)
        )

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

                    if dim == 3:
                        if o["pos"][2] - o["r"] < w_cfg["z_min"]:
                            o["vel"][2] *= -1
                            o["pos"][2] = w_cfg["z_min"] + o["r"]
                        elif o["pos"][2] + o["r"] > w_cfg["z_max"]:
                            o["vel"][2] *= -1
                            o["pos"][2] = w_cfg["z_max"] - o["r"]
                else:
                    o["pos"] += o["vel"] * dt

            # After updating sphere positions + wall bounces for all obstacles:
            _resolve_sphere_sphere_collisions(obs, dim=dim, restitution=rest)

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

        c_p = swarm_pos_accum / (len(agents) * sub_steps)
        centroid_plot.set_data([c_p[0]], [c_p[1]])
        if dim == 3:
            centroid_plot.set_3d_properties([c_p[2]])

        if dim == 2:
            for i, a in enumerate(agents):
                agent_body_patches[i].center = a["pos"][:2]
                agent_halo_patches[i].center = a["pos"][:2]

        elif frame % 5 == 0:
            # 3D Agent Updates (Wireframes are heavy, update less often)
            u_m, v_m = np.mgrid[0 : 2 * np.pi : 10j, 0 : np.pi : 10j]
            for i, a in enumerate(agents):
                color = lines[i].get_color()

                # Update Body
                if agent_body_patches[i] is not None:
                    agent_body_patches[i].remove()
                agent_body_patches[i] = ax.plot_wireframe(
                    a["pos"][0] + agent_radius * np.cos(u_m) * np.sin(v_m),
                    a["pos"][1] + agent_radius * np.sin(u_m) * np.sin(v_m),
                    a["pos"][2] + agent_radius * np.cos(v_m),
                    color=color,
                    alpha=0.8,
                )

                # Update Halo
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

        for i, o in enumerate(obs):
            if dim == 2:
                if hasattr(obs_vis[i], "center"):  # Sphere
                    # Note: We need to update halos as well, but `obs_vis` mixed spheres and walls.
                    # The cleanest way is to iterate over spheres separately or use the index mapping logic.
                    # Given `obs_vis` order matches `obs` order:
                    obs_vis[i].center = o["pos"][:2]

                    # Update halo if this is a sphere.
                    # We need to find the corresponding halo index.
                    # Simpler strategy: Iterate over spheres separately using the list we built earlier:
                    pass
                elif hasattr(obs_vis[i], "set_data"):  # Wall/Line
                    n, p = o["normal"], o["pos"]
                    t_vec = np.array([-n[1], n[0], 0])
                    p1, p2 = p + t_vec * 20, p - t_vec * 20
                    obs_vis[i].set_data([p1[0], p2[0]], [p1[1], p2[1]])

        if dim == 2:
            k = 0
            for o in obs:
                if o.get("kind") == "sphere":
                    obs_vis[k].center = o["pos"][:2]
                    obs_halos[k].center = o["pos"][:2]
                    k += 1
                elif o.get("kind") == "wall":
                    # wall update handled above via direct set_data on obs_vis[index]
                    # BUT wait - obs_vis has mixed types.
                    # Correction: obs_vis contains ALL obstacles in order.
                    # obs_halos ONLY contains sphere halos.
                    # So we need to be careful with indices.
                    pass

            # Re-do the loop correctly based on the plan
            # "maintain a separate list only for spheres" -> we did: sphere_obs = [...]
            # But inside update loop we iterate 'obs'.

            # Correct logic:
            sphere_idx = 0
            for i, o in enumerate(obs):
                if o.get("kind") == "sphere":
                    if hasattr(obs_vis[i], "center"):
                        obs_vis[i].center = o["pos"][:2]

                    obs_halos[sphere_idx].center = o["pos"][:2]
                    sphere_idx += 1

        elif frame % 5 == 0:
            for i, o in enumerate(obs):
                if o.get("kind") == "sphere":
                    obs_vis[i].remove()
                    u_m, v_m = np.mgrid[0 : 2 * np.pi : 10j, 0 : np.pi : 10j]

                    # Physical Obstacle
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
