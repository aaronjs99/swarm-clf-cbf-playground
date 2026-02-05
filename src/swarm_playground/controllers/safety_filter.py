import numpy as np
from controllers.constraints import LinearConstraint


from utils.geometry import as3


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
        x = as3(x)
        v = as3(v)
        other_pos = as3(other_pos)
        other_vel = as3(other_vel)

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

    def get_constraint(
        self,
        x,
        v,
        other_pos,
        other_vel,
        d_safe: float,
        a_max: float,
    ):
        """
        Returns a LinearConstraint if active, else None.
        Returns (constraint, info_dict).
        """
        info = {
            "active": False,
            "h_brake": None,
            "v_close": None,
            "dist_to_safe": None,
        }

        if not self.enabled:
            return None, info

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

            c = LinearConstraint(
                G=np.atleast_2d(-e),
                b=np.atleast_1d(-self.gain * v_close),
                hard=False,  # Braking filter often treated as soft or high-priority soft
            )
            info["active"] = True
            return c, info

        return None, info
