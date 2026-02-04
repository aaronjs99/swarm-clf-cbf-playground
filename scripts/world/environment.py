import numpy as np


def get_obstacles(cfg, agents_init, goals_init, dim):
    """
    Generates obstacles based on configuration.
    Supports walls and random spherical obstacles with collision checks.
    """
    obs_cfg = cfg["world"]["obstacles"]
    seed = cfg.get("sim", {}).get("seed", 42)
    np.random.seed(int(seed))
    obs = []

    # Handle walls (support 'walls' generic key or legacy 'walls_2d'/'walls_3d')
    w = obs_cfg.get("walls")
    if not w:
        w_key = "walls_2d" if dim == 2 else "walls_3d"
        w = obs_cfg.get(w_key, {})

    if w.get("enabled", False):
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
            obs.append(
                {
                    "kind": "wall",
                    "pos": np.array([0, 0, w.get("z_min", 0.0)]),
                    "normal": np.array([0, 0, 1.0]),
                    "vel": np.zeros(3),
                }
            )
            obs.append(
                {
                    "kind": "wall",
                    "pos": np.array([0, 0, w.get("z_max", 10.0)]),
                    "normal": np.array([0, 0, -1.0]),
                    "vel": np.zeros(3),
                }
            )

        margin = 1.2
        box_min = np.array(
            [w["x_min"] + margin, w["y_min"] + margin, w.get("z_min", 0) + margin]
        )
        box_max = np.array(
            [w["x_max"] - margin, w["y_max"] - margin, w.get("z_max", 10) - margin]
        )
    else:
        # Fallback to spawn box if no walls
        box_key = (
            "spawn_box"
            if "spawn_box" in obs_cfg
            else ("spawn_box_2d" if dim == 2 else "spawn_box_3d")
        )
        box_min = np.array(obs_cfg[box_key]["min"])
        box_max = np.array(obs_cfg[box_key]["max"])

    num = int(
        obs_cfg.get("num_override")
        or obs_cfg.get("num")
        or (obs_cfg["num_obs_2d"] if dim == 2 else obs_cfg["num_obs_3d"])
    )
    r_lo, r_hi = obs_cfg["radius_range"]
    v_lo, v_hi = obs_cfg["dynamic"]["vel_range"]

    while len(obs) < num:
        pos = np.random.uniform(box_min, box_max)
        if dim == 2:
            pos[2] = 0.0
        r = float(np.random.uniform(r_lo, r_hi))

        # Check overlap
        if any(np.linalg.norm(pos - p) < (r + 0.8) for p in agents_init + goals_init):
            continue

        vel = np.random.uniform(v_lo, v_hi, 3)
        if dim == 2:
            vel[2] = 0.0
        obs.append({"pos": pos, "r": r, "vel": vel, "kind": "sphere"})

    return obs


def init_agents_and_goals(cfg, dim, n):
    """
    Initializes agent start positions and goal positions.
    Supports 'circle', 'lanes', and 'mirror' modes for 2D.
    """
    seed = cfg.get("sim", {}).get("seed", 42)
    np.random.seed(int(seed))

    # Walls logic
    w = cfg["world"]["obstacles"].get("walls")
    if not w:
        w_key = "walls_2d" if dim == 2 else "walls_3d"
        w = cfg["world"]["obstacles"].get(w_key, {})

    y_mid = (w["y_max"] + w["y_min"]) / 2.0 if w.get("enabled") else 0.0
    z_mid = (
        (w.get("z_max", 10.0) + w.get("z_min", 0.0)) / 2.0
        if (dim == 3 and w.get("enabled"))
        else 0.0
    )

    init_cfg = cfg["agents"]["init"]
    goal_cfg = cfg["agents"]["goals"]

    if dim == 2:
        mode_init = init_cfg.get("mode") or init_cfg["mode_2d"]
        mode_goal = goal_cfg.get("mode") or goal_cfg["mode_2d"]

        y_spacing = init_cfg.get("lanes", {}).get("y_spacing", 0.8)
        y_start = y_mid - ((n - 1) * y_spacing) / 2.0

        if mode_init == "circle":
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
        else:  # lanes
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

        if mode_goal == "mirror":
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
        else:  # lanes
            lanes = goal_cfg["lanes"]
            y_spacing = float(lanes["y_spacing"])
            reverse = bool(lanes.get("reverse_order", True))
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
        # 3D Simple init
        # This part was hardcoded in original, now we might want to check for 'box' mode
        # but defaulting to fixed positions relative to walls is fine for now to match behavior
        y_spacing = 0.8  # Default or read from config if we had 3d config
        y_start = y_mid - ((n - 1) * y_spacing) / 2.0

        agents_init = [
            np.array([w["x_min"] + 1.0, y_start + i * y_spacing, z_mid])
            for i in range(n)
        ]
        goals_init = [
            np.array([w["x_max"] - 1.0, y_start + (n - 1 - i) * y_spacing, z_mid])
            for i in range(n)
        ]

    return agents_init, goals_init
