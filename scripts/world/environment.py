import numpy as np


def _rand_in_box(rng, box_min, box_max) -> np.ndarray:
    return rng.uniform(box_min, box_max)


def _nonoverlap_ok(pos, r, points, min_clearance: float) -> bool:
    for p in points:
        if float(np.linalg.norm(pos - p)) < (float(r) + float(min_clearance)):
            return False
    return True


def _add_walls(obs, w, dim: int):
    obs.append(
        {
            "kind": "wall",
            "pos": np.array([0.0, w["y_min"], 0.0]),
            "normal": np.array([0.0, 1.0, 0.0]),
            "vel": np.zeros(3),
        }
    )
    obs.append(
        {
            "kind": "wall",
            "pos": np.array([0.0, w["y_max"], 0.0]),
            "normal": np.array([0.0, -1.0, 0.0]),
            "vel": np.zeros(3),
        }
    )
    obs.append(
        {
            "kind": "wall",
            "pos": np.array([w["x_min"], 0.0, 0.0]),
            "normal": np.array([1.0, 0.0, 0.0]),
            "vel": np.zeros(3),
        }
    )
    obs.append(
        {
            "kind": "wall",
            "pos": np.array([w["x_max"], 0.0, 0.0]),
            "normal": np.array([-1.0, 0.0, 0.0]),
            "vel": np.zeros(3),
        }
    )

    if dim == 3:
        obs.append(
            {
                "kind": "wall",
                "pos": np.array([0.0, 0.0, w.get("z_min", 0.0)]),
                "normal": np.array([0.0, 0.0, 1.0]),
                "vel": np.zeros(3),
            }
        )
        obs.append(
            {
                "kind": "wall",
                "pos": np.array([0.0, 0.0, w.get("z_max", 10.0)]),
                "normal": np.array([0.0, 0.0, -1.0]),
                "vel": np.zeros(3),
            }
        )


def _get_wall_cfg(cfg, dim: int):
    obs_cfg = cfg["world"]["obstacles"]
    w = obs_cfg.get("walls")
    if not w:
        w_key = "walls_2d" if dim == 2 else "walls_3d"
        w = obs_cfg.get(w_key, {})
    return w or {}


def _get_spawn_box(cfg, dim: int):
    obs_cfg = cfg["world"]["obstacles"]
    box_key = (
        "spawn_box"
        if "spawn_box" in obs_cfg
        else ("spawn_box_2d" if dim == 2 else "spawn_box_3d")
    )
    box_min = np.array(obs_cfg[box_key]["min"], dtype=float)
    box_max = np.array(obs_cfg[box_key]["max"], dtype=float)
    if dim == 2:
        box_min[2] = 0.0
        box_max[2] = 0.0
    return box_min, box_max


def _scene3d_obstacles(cfg, agents_init, goals_init):
    """
    Builds a 3D scene:
      - static trees (rendered as trunk+crown)
      - static cars on the ground (rendered as cuboids)
      - static saucers in the sky (rendered as flattened ellipsoids)

    For safety math, all of these are still treated as spheres with radius `r`.
    """

    def _pick_from(palette, idx):
        return palette[int(idx) % len(palette)]

    car_palette = [
        "#3b82f6",
        "#ef4444",
        "#f59e0b",
        "#10b981",
        "#a855f7",
    ]  # blue, red, amber, green, purple
    saucer_palette = ["#64748b", "#7c3aed", "#0ea5e9"]  # slate, violet, sky
    trunk_palette = ["#8b5a2b", "#6b4226", "#7a4a2a"]  # browns
    leaf_palette = ["#16a34a", "#15803d", "#22c55e", "#4ade80"]  # greens

    obs_cfg = cfg["world"]["obstacles"]
    scene = (obs_cfg.get("scene3d") or {}) if isinstance(obs_cfg, dict) else {}
    enabled = bool(scene.get("enabled", True))

    seed = int(cfg.get("sim", {}).get("seed", 42))
    rng = np.random.default_rng(seed)

    w = _get_wall_cfg(cfg, dim=3)
    has_walls = bool(w.get("enabled", False))

    if has_walls:
        margin = 1.2
        box_min = np.array(
            [w["x_min"] + margin, w["y_min"] + margin, w.get("z_min", 0.0) + margin],
            dtype=float,
        )
        box_max = np.array(
            [w["x_max"] - margin, w["y_max"] - margin, w.get("z_max", 10.0) - margin],
            dtype=float,
        )
    else:
        box_min, box_max = _get_spawn_box(cfg, dim=3)

    # Keep trees near the middle corridor so drones must go over them.
    y_mid = (w["y_max"] + w["y_min"]) / 2.0 if has_walls else 0.0
    x_mid = (w["x_max"] + w["x_min"]) / 2.0 if has_walls else 0.0

    points_to_avoid = [p for p in agents_init] + [g for g in goals_init]

    obs = []
    if not enabled:
        return obs

    trees_cfg = scene.get("trees", {}) or {}
    cars_cfg = scene.get("cars", {}) or {}
    saucers_cfg = scene.get("saucers", {}) or {}

    n_trees = int(trees_cfg.get("num", 12))
    n_cars = int(cars_cfg.get("num", 3))
    n_saucers = int(saucers_cfg.get("num", 3))

    trunk_r_lo, trunk_r_hi = trees_cfg.get("trunk_radius_range") or [0.10, 0.18]
    trunk_h_lo, trunk_h_hi = trees_cfg.get("trunk_height_range") or [2.5, 4.0]
    canopy_lo, canopy_hi = trees_cfg.get("canopy_radius_range") or [0.7, 1.1]
    tree_margin = float(trees_cfg.get("safety_margin", 0.25))

    size_rng = (
        (cars_cfg.get("size_range") or {})
        if isinstance(cars_cfg.get("size_range"), dict)
        else {}
    )
    car_min = np.array(size_rng.get("min", [1.4, 0.7, 0.35]), dtype=float)
    car_max = np.array(size_rng.get("max", [2.2, 1.0, 0.55]), dtype=float)
    car_margin = float(cars_cfg.get("safety_margin", 0.35))

    zc_lo, zc_hi = saucers_cfg.get("center_z_range") or [7.0, 9.0]
    rad_rng = (
        (saucers_cfg.get("radii_range") or {})
        if isinstance(saucers_cfg.get("radii_range"), dict)
        else {}
    )
    sau_min = np.array(rad_rng.get("min", [0.8, 0.8, 0.12]), dtype=float)
    sau_max = np.array(rad_rng.get("max", [1.4, 1.4, 0.22]), dtype=float)
    sau_margin = float(saucers_cfg.get("safety_margin", 0.35))

    def place_near_midline(z_value: float, x_spread: float, y_spread: float):
        x = float(rng.normal(loc=x_mid, scale=x_spread))
        y = float(rng.normal(loc=y_mid, scale=y_spread))
        x = float(np.clip(x, box_min[0], box_max[0]))
        y = float(np.clip(y, box_min[1], box_max[1]))
        return np.array([x, y, float(z_value)], dtype=float)

    # Trees: put them in a corridor-ish band around x_mid so they form a barrier.
    for _ in range(n_trees):
        trunk_r = float(rng.uniform(trunk_r_lo, trunk_r_hi))
        trunk_h = float(rng.uniform(trunk_h_lo, trunk_h_hi))
        canopy_r = float(rng.uniform(canopy_lo, canopy_hi))

        # Render parameters
        z_base = float(w.get("z_min", 0.0)) if has_walls else 0.0
        z_top = z_base + trunk_h

        # Safety sphere radius roughly covers canopy
        r_safe = canopy_r + tree_margin

        for _try in range(200):
            pos = place_near_midline(
                z_value=z_top - 0.3 * canopy_r, x_spread=1.2, y_spread=2.0
            )
            # keep trees on the ground
            pos[2] = max(pos[2], z_base + 0.5 * canopy_r)
            if _nonoverlap_ok(pos, r_safe, points_to_avoid, min_clearance=0.8):
                break
        else:
            continue

        points_to_avoid.append(pos.copy())

        tree_idx = len([o for o in obs if o.get("render") == "tree"])
        trunk_c = _pick_from(trunk_palette, tree_idx)
        leaf_c = _pick_from(leaf_palette, tree_idx)

        obs.append(
            {
                "kind": "sphere",
                "render": "tree",
                "pos": pos,
                "r": r_safe,
                "vel": np.zeros(3),
                "colors": {"trunk": trunk_c, "canopy": leaf_c},
                "tree": {
                    "trunk_radius": trunk_r,
                    "trunk_height": trunk_h,
                    "canopy_radius": canopy_r,
                    "z_base": z_base,
                },
            }
        )

    # Cars: parked on the ground
    for _ in range(n_cars):
        size = rng.uniform(car_min, car_max)
        lx, ly, lz = float(size[0]), float(size[1]), float(size[2])

        # Bounding sphere radius for safety
        r_safe = 0.5 * float(np.linalg.norm([lx, ly, lz])) + car_margin

        z_base = float(w.get("z_min", 0.0)) if has_walls else 0.0
        z_center = z_base + 0.3 * lz

        for _try in range(200):
            pos = _rand_in_box(rng, box_min, box_max)
            pos[2] = z_center
            # Prefer cars not exactly on the tree midline, so it's visually interesting
            pos[0] = float(
                np.clip(pos[0] + rng.uniform(-1.5, 1.5), box_min[0], box_max[0])
            )
            if _nonoverlap_ok(pos, r_safe, points_to_avoid, min_clearance=0.8):
                break
        else:
            continue

        points_to_avoid.append(pos.copy())

        dyn = obs_cfg.get("dynamic") or {}
        dyn_on = bool(dyn.get("enabled", False))

        vel = np.zeros(3)
        if dyn_on:
            # slow drifting cars on ground
            vx = float(rng.uniform(-0.45, 0.45))
            vy = float(rng.uniform(-0.45, 0.45))
            vel = np.array([vx, vy, 0.0], dtype=float)

        car_idx = len([o for o in obs if o.get("render") == "car"])
        car_c = _pick_from(car_palette, car_idx)

        obs.append(
            {
                "kind": "sphere",
                "render": "car",
                "pos": pos,
                "r": r_safe,
                "vel": vel,
                "colors": {"body": car_c},
                "car": {"size": np.array([lx, ly, lz], dtype=float), "z_base": z_base},
            }
        )

    # Saucers: in the sky
    for _ in range(n_saucers):
        radii = rng.uniform(sau_min, sau_max)
        a, b, c = float(radii[0]), float(radii[1]), float(radii[2])
        r_safe = float(max(a, b)) + sau_margin

        for _try in range(200):
            pos = _rand_in_box(rng, box_min, box_max)
            pos[2] = float(rng.uniform(zc_lo, zc_hi))
            if _nonoverlap_ok(pos, r_safe, points_to_avoid, min_clearance=0.8):
                break
        else:
            continue

        points_to_avoid.append(pos.copy())

        dyn = obs_cfg.get("dynamic") or {}
        dyn_on = bool(dyn.get("enabled", False))

        vel = np.zeros(3)
        if dyn_on:
            # faster saucers in xy, keep z constant for now
            vx = float(rng.uniform(-0.8, 0.8))
            vy = float(rng.uniform(-0.8, 0.8))
            vel = np.array([vx, vy, 0.0], dtype=float)

        sau_idx = len([o for o in obs if o.get("render") == "saucer"])
        sau_c = _pick_from(saucer_palette, sau_idx)

        obs.append(
            {
                "kind": "sphere",
                "render": "saucer",
                "pos": pos,
                "r": r_safe,
                "vel": vel,
                "colors": {"body": sau_c},
                "saucer": {"radii": np.array([a, b, c], dtype=float)},
            }
        )

    return obs


def get_obstacles(cfg, agents_init, goals_init, dim):
    """
    Generates obstacles based on configuration.

    2D: spheres + optional walls (existing behavior).
    3D: optional walls + a "scene" consisting of trees, cars, and saucers.
        All scene objects are treated as spheres for safety, but have richer rendering.
    """
    obs_cfg = cfg["world"]["obstacles"]
    seed = int(cfg.get("sim", {}).get("seed", 42))
    rng = np.random.default_rng(seed)

    obs = []

    # Walls
    w = _get_wall_cfg(cfg, dim=dim)
    if w.get("enabled", False):
        _add_walls(obs, w, dim=dim)

    if dim == 3:
        obs.extend(_scene3d_obstacles(cfg, agents_init, goals_init))
        return obs

    # 2D fallback: original random spheres
    if w.get("enabled", False):
        margin = 1.2
        box_min = np.array([w["x_min"] + margin, w["y_min"] + margin, 0.0], dtype=float)
        box_max = np.array([w["x_max"] - margin, w["y_max"] - margin, 0.0], dtype=float)
    else:
        box_min, box_max = _get_spawn_box(cfg, dim=2)

    num = int(obs_cfg.get("num_override") or obs_cfg.get("num", 8))
    r_lo, r_hi = obs_cfg["radius_range"]
    v_lo, v_hi = obs_cfg["dynamic"]["vel_range"]

    points_to_avoid = [p for p in agents_init] + [g for g in goals_init]

    while len([o for o in obs if o.get("kind") == "sphere"]) < num:
        pos = rng.uniform(box_min, box_max)
        pos[2] = 0.0
        r = float(rng.uniform(r_lo, r_hi))

        if not _nonoverlap_ok(pos, r, points_to_avoid, min_clearance=0.8):
            continue

        vel = rng.uniform(v_lo, v_hi, 3)
        vel[2] = 0.0

        obs.append({"pos": pos, "r": r, "vel": vel, "kind": "sphere"})
        points_to_avoid.append(pos.copy())

    return obs


def init_agents_and_goals(cfg, dim, n):
    """
    Initializes agent start positions and goal positions.

    2D: existing lane/circle modes.
    3D: start near z_start, goal at z_goal, with x from left wall to right wall.
    """
    seed = int(cfg.get("sim", {}).get("seed", 42))
    rng = np.random.default_rng(seed)

    w = _get_wall_cfg(cfg, dim=dim)

    has_walls = bool(w.get("enabled", False))
    y_mid = (w["y_max"] + w["y_min"]) / 2.0 if has_walls else 0.0

    init_cfg = cfg["agents"]["init"]
    goal_cfg = cfg["agents"]["goals"]

    if dim == 2:
        mode_init = init_cfg.get("mode") or init_cfg.get("mode_2d", "lanes")
        mode_goal = goal_cfg.get("mode") or goal_cfg.get("mode_2d", "lanes")

        y_spacing = float(init_cfg.get("lanes", {}).get("y_spacing", 0.8))
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
                    ],
                    dtype=float,
                )
                for i in range(n)
            ]
        else:
            lanes = init_cfg["lanes"]
            y_spacing = float(lanes["y_spacing"])
            agents_init = [
                np.array(
                    [
                        rng.uniform(lanes["x_min"], lanes["x_max"]),
                        y_start + i * y_spacing,
                        0.0,
                    ],
                    dtype=float,
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
                    ],
                    dtype=float,
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
                    [rng.uniform(lanes["x_min"], lanes["x_max"]), ys[i], 0.0],
                    dtype=float,
                )
                for i in range(n)
            ]

        return agents_init, goals_init

    # 3D: start near ground, land near ground on the far side
    flight = (cfg.get("world", {}) or {}).get("flight", {}) or {}
    z_start = float(flight.get("z_start", 0.5))
    z_goal = float(flight.get("z_goal", 0.0))

    # Lane-ish spacing in y for readability
    y_spacing = 0.8
    y_start = y_mid - ((n - 1) * y_spacing) / 2.0

    if has_walls:
        x0 = float(w["x_min"] + 1.0)
        x1 = float(w["x_max"] - 1.0)
    else:
        x0 = -4.0
        x1 = 4.0

    agents_init = [
        np.array([x0, y_start + i * y_spacing, z_start], dtype=float) for i in range(n)
    ]
    goals_init = [
        np.array([x1, y_start + (n - 1 - i) * y_spacing, z_goal], dtype=float)
        for i in range(n)
    ]

    return agents_init, goals_init
