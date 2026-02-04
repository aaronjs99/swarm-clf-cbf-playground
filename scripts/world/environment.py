import numpy as np


def get_obstacles(cfg, agents_init, goals_init, dim):
    obs_cfg = cfg["world"]["obstacles"]
    np.random.seed(int(cfg.get("seed", 42)))
    obs = []

    w_key = "walls_2d" if dim == 2 else "walls_3d"
    if obs_cfg.get(w_key, {}).get("enabled", False):
        w = obs_cfg[w_key]
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


def init_agents_and_goals(cfg, dim, n):
    np.random.seed(int(cfg.get("seed", 42)))
    w_key = "walls_2d" if dim == 2 else "walls_3d"
    w = cfg["world"]["obstacles"].get(w_key, {})

    y_mid = (w["y_max"] + w["y_min"]) / 2.0 if w.get("enabled") else 0.0
    z_mid = (w["z_max"] + w["z_min"]) / 2.0 if (dim == 3 and w.get("enabled")) else 0.0

    y_spacing = cfg["agents"]["init"]["lanes"]["y_spacing"]
    y_start = y_mid - ((n - 1) * y_spacing) / 2.0

    if dim == 2:
        init_cfg = cfg["agents"]["init"]
        goal_cfg = cfg["agents"]["goals"]
        walls_cfg = cfg["world"]["obstacles"].get("walls_2d", {})

        if walls_cfg.get("enabled", False):
            y_mid = (walls_cfg["y_max"] + walls_cfg["y_min"]) / 2.0
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
        agents_init = [
            np.array([w["x_min"] + 1.0, y_start + i * y_spacing, z_mid])
            for i in range(n)
        ]
        goals_init = [
            np.array([w["x_max"] - 1.0, y_start + (n - 1 - i) * y_spacing, z_mid])
            for i in range(n)
        ]

    return agents_init, goals_init
