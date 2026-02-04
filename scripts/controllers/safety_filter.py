import numpy as np


def _as3(x):
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size == 2:
        return np.array([x[0], x[1], 0.0], dtype=float)
    if x.size == 3:
        return x
    raise ValueError("Expected a 2D or 3D vector.")


class BrakingSafetyFilter:
    """
    Braking-distance awareness / viability filter.

    It does two things:
      1) Computes a braking-aware barrier value:
            h_brake = (dist_to_safe) - v_close^2/(2 a_max) - margin
      2) If in the braking zone (h_brake < activate_h), adds an inequality constraint
         that forces acceleration to push away along the line of sight.

    Constraint added (when active):
        e^T u >= gain * v_close
    which becomes:
        (-e)^T u <= -gain * v_close
    """

    def __init__(
        self,
        enabled: bool = True,
        margin: float = 0.0,
        activate_h: float = 0.0,
        gain: float = 2.0,
        clamp_v_close: float = 50.0,
    ):
        self.enabled = bool(enabled)
        self.margin = float(margin)
        self.activate_h = float(activate_h)
        self.gain = float(gain)
        self.clamp_v_close = float(clamp_v_close)

    def compute_braking_h(
        self, x, v, other_pos, other_vel, d_safe: float, a_max: float
    ):
        x = _as3(x)
        v = _as3(v)
        other_pos = _as3(other_pos)
        other_vel = _as3(other_vel)

        r = x - other_pos
        dist = float(np.linalg.norm(r)) + 1e-9
        e = r / dist

        v_rel = v - other_vel
        closing = float(-np.dot(e, v_rel))  # positive when approaching
        v_close = max(0.0, min(self.clamp_v_close, closing))

        dist_to_safe = dist - float(d_safe)
        brake_term = (v_close * v_close) / (2.0 * float(a_max) + 1e-9)

        h_brake = dist_to_safe - brake_term - self.margin
        return float(h_brake), e, float(v_close), float(dist_to_safe)

    def maybe_add_constraint(
        self,
        G_list,
        b_list,
        x,
        v,
        other_pos,
        other_vel,
        d_safe: float,
        a_max: float,
    ):
        """
        Mutates G_list, b_list in-place.
        Returns a dict with debug info.
        """
        info = {
            "active": False,
            "h_brake": None,
            "v_close": None,
            "dist_to_safe": None,
        }

        if not self.enabled:
            return info

        h_brake, e, v_close, dist_to_safe = self.compute_braking_h(
            x, v, other_pos, other_vel, d_safe, a_max
        )
        info["h_brake"] = h_brake
        info["v_close"] = v_close
        info["dist_to_safe"] = dist_to_safe

        # Only activate when actually closing and in the braking zone.
        if (v_close > 1e-6) and (h_brake < self.activate_h):
            # e^T u >= gain * v_close
            # (-e)^T u <= -gain * v_close
            G_list.append((-e).tolist())
            b_list.append(float(-self.gain * v_close))
            info["active"] = True

        return info
