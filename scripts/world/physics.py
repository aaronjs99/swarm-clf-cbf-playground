import numpy as np


def _sphere_mass(r: float, dim: int) -> float:
    # Proportional to area (2D) or volume (3D). Constants cancel in impulse math.
    if dim == 2:
        return max(1e-9, r * r)
    return max(1e-9, r * r * r)


def resolve_sphere_sphere_collisions(obs, dim: int, restitution: float = 1.0):
    """
    Detects and resolves overlaps between spherical obstacles/agents using impulse response.
    Updates positions and velocities in-place.
    """
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
                n_hat = np.array([1.0, 0.0, 0.0])
                dist = 1e-12
            else:
                n_hat = delta / dist

            if dist >= min_dist:
                continue

            va = a["vel"]
            vb = b["vel"]

            penetration = min_dist - dist

            ma = _sphere_mass(ra, dim)
            mb = _sphere_mass(rb, dim)
            inv_ma = 1.0 / ma
            inv_mb = 1.0 / mb
            inv_sum = inv_ma + inv_mb

            pa -= n_hat * penetration * (inv_ma / inv_sum)
            pb += n_hat * penetration * (inv_mb / inv_sum)

            v_rel = vb - va
            v_rel_n = float(np.dot(v_rel, n_hat))

            if v_rel_n >= 0.0:
                continue

            j_imp = -(1.0 + e) * v_rel_n / inv_sum
            impulse = j_imp * n_hat
            va -= impulse * inv_ma
            vb += impulse * inv_mb

            a["pos"] = pa
            b["pos"] = pb
            a["vel"] = va
            b["vel"] = vb


def bounce_sphere_off_walls_inplace(o, w_cfg, dim: int):
    # Sphere obstacle bounce against axis-aligned walls.
    # Uses the obstacle's radius.
    r = float(o["r"])

    # X
    if o["pos"][0] - r < w_cfg["x_min"]:
        o["vel"][0] *= -1
        o["pos"][0] = w_cfg["x_min"] + r
    elif o["pos"][0] + r > w_cfg["x_max"]:
        o["vel"][0] *= -1
        o["pos"][0] = w_cfg["x_max"] - r

    # Y
    if o["pos"][1] - r < w_cfg["y_min"]:
        o["vel"][1] *= -1
        o["pos"][1] = w_cfg["y_min"] + r
    elif o["pos"][1] + r > w_cfg["y_max"]:
        o["vel"][1] *= -1
        o["pos"][1] = w_cfg["y_max"] - r

    # Z (3D only)
    if dim == 3:
        if o["pos"][2] - r < w_cfg["z_min"]:
            o["vel"][2] *= -1
            o["pos"][2] = w_cfg["z_min"] + r
        elif o["pos"][2] + r > w_cfg["z_max"]:
            o["vel"][2] *= -1
            o["pos"][2] = w_cfg["z_max"] - r


def agents_enforce_world_bounds(
    agents,
    w_cfg,
    dim: int,
    agent_radius: float,
    restitution: float = 0.0,
):
    """
    Hard simulator-side world constraint for agents.

    Why: CBFs are continuous-time; our sim is discrete-time with optional slack.
    So an agent can step across a wall between controller updates. This prevents
    "teleporting" by enforcing that positions remain inside the wall box.

    Behavior:
      - clamp position to [min+R, max-R]
      - optionally bounce velocity component with restitution in [0, 1]
        * 0.0 => "stick" (zero out that velocity component on impact)
        * 1.0 => perfectly elastic reflection
    """
    if not (isinstance(w_cfg, dict) and w_cfg.get("enabled", False)):
        return

    e = float(restitution)
    e = max(0.0, min(1.0, e))
    R = float(agent_radius)

    x_min = float(w_cfg["x_min"]) + R
    x_max = float(w_cfg["x_max"]) - R
    y_min = float(w_cfg["y_min"]) + R
    y_max = float(w_cfg["y_max"]) - R

    if dim == 3:
        z_min = float(w_cfg["z_min"]) + R
        z_max = float(w_cfg["z_max"]) - R
    else:
        z_min = 0.0
        z_max = 0.0

    for a in agents:
        p = a["pos"]
        v = a["vel"]

        # X
        if p[0] < x_min:
            p[0] = x_min
            v[0] = -e * v[0]
        elif p[0] > x_max:
            p[0] = x_max
            v[0] = -e * v[0]

        # Y
        if p[1] < y_min:
            p[1] = y_min
            v[1] = -e * v[1]
        elif p[1] > y_max:
            p[1] = y_max
            v[1] = -e * v[1]

        # Z
        if dim == 3:
            if p[2] < z_min:
                p[2] = z_min
                v[2] = -e * v[2]
            elif p[2] > z_max:
                p[2] = z_max
                v[2] = -e * v[2]
        else:
            p[2] = 0.0
            v[2] = 0.0

        a["pos"] = p
        a["vel"] = v
