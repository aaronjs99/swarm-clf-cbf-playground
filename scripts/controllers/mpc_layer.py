import numpy as np


def _as3(x):
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size == 2:
        return np.array([x[0], x[1], 0.0], dtype=float)
    if x.size == 3:
        return x
    raise ValueError("Expected a 2D or 3D vector.")


class SamplingMPCPlanner:
    """
    Minimal sampling-based MPC (shooting method).

    It proposes a constant acceleration over a short horizon and picks the best
    according to a cost that trades off:
      - goal proximity at horizon
      - control effort
      - predicted collisions (soft penalties)

    This is meant to be a planning layer. The CBF-QP still enforces hard safety.
    """

    def __init__(
        self,
        enabled: bool = False,
        horizon_steps: int = 20,
        dt: float = 0.02,
        num_samples: int = 64,
        goal_weight: float = 1.0,
        effort_weight: float = 0.02,
        collision_weight: float = 50.0,
        rng_seed: int = 0,
    ):
        self.enabled = bool(enabled)
        self.H = int(horizon_steps)
        self.dt = float(dt)
        self.N = int(num_samples)
        self.goal_w = float(goal_weight)
        self.effort_w = float(effort_weight)
        self.collision_w = float(collision_weight)

        self._rng = np.random.default_rng(int(rng_seed))

    def _rollout(self, x0, v0, u, is_2d: bool):
        x = x0.copy()
        v = v0.copy()
        for _ in range(self.H):
            v = v + u * self.dt
            x = x + v * self.dt
            if is_2d:
                x[2] = 0.0
                v[2] = 0.0
        return x, v

    def _collision_penalty(self, x, threats, d_safe_fn):
        # Soft, smooth-ish penalty if inside safety distance
        pen = 0.0
        for o in threats:
            kind = o.get("kind", "sphere")
            if kind in ["sphere", "agent"]:
                d_safe = float(d_safe_fn(o))
                rel = x - o["pos"]
                dist = float(np.linalg.norm(rel)) + 1e-9
                gap = dist - d_safe
                if gap < 0.0:
                    pen += gap * gap
            elif kind == "wall":
                n = np.asarray(o["normal"], dtype=float).reshape(-1)
                p = np.asarray(o["pos"], dtype=float).reshape(-1)
                dist = float(np.dot(n, x - p))
                # For walls, d_safe_fn(o) returns offset
                offset = float(d_safe_fn(o))
                gap = dist - offset
                if gap < 0.0:
                    pen += gap * gap
        return pen

    def plan(self, x, v, goal, threats, a_max: float, is_2d: bool, d_safe_fn):
        if not self.enabled:
            return None

        x0 = _as3(x)
        v0 = _as3(v)
        goal = _as3(goal)

        # 1) Threat Filtering: Keep only nearest K or proximity-gated
        # This reduces the inner loop cost significantly.
        filtered_threats = []
        if threats:
            # Simple distance sorting
            # (optimization: could use spatial hash or kd-tree if N is huge,
            # but for N<100 sorting is fine)
            def dist_sq(o):
                return np.sum((o["pos"] - x0) ** 2)

            # Keep massive walls always? Or just treat them as threats.
            # Here we just sort everything by distance.
            sorted_threats = sorted(threats, key=dist_sq)

            # Keep closest 10
            K = 10
            filtered_threats = sorted_threats[:K]

            # Or use distance gate:
            # filtered_threats = [o for o in threats if np.linalg.norm(o["pos"] - x0) < 3.0]

        # 2) Smart Sampling
        candidates = []

        # Helper to add candidate
        def add_u(u_vec):
            n = np.linalg.norm(u_vec)
            if n > a_max:
                u_vec = (u_vec / n) * a_max
            if is_2d:
                u_vec[2] = 0.0
            candidates.append(u_vec)

        # A) Goal direction (braking-aware)
        rel_goal = goal - x0
        ng = np.linalg.norm(rel_goal) + 1e-9
        u_goal = (rel_goal / ng) * float(a_max) - 2.0 * v0
        add_u(u_goal)

        # B) Lateral directions (perpendicular to goal)
        # Helpful for sidestepping
        if ng > 0.1:
            if is_2d:
                # 2D cross product with Z
                z_axis = np.array([0, 0, 1.0])
                lat = np.cross(rel_goal, z_axis)
                lat = (lat / np.linalg.norm(lat)) * a_max
                add_u(lat)
                add_u(-lat)
            else:
                # 3D arbitrary lateral
                # Just pick a couple of random orthagonals or cross with velocity
                pass  # simplicity

        # C) Evasion (away from nearest threat)
        if filtered_threats:
            nearest = filtered_threats[0]
            rel_threat = x0 - nearest["pos"]
            nt = np.linalg.norm(rel_threat)
            if nt > 1e-9:
                u_evade = (rel_threat / nt) * a_max
                add_u(u_evade)

        # D) Random samples to fill the rest
        # We need self.N total candidates
        # Create (N - current_count) random ones
        needed = self.N - len(candidates)
        if needed > 0:
            for _ in range(needed):
                u = self._rng.uniform(-a_max, a_max, size=3)
                if is_2d:
                    u[2] = 0.0
                candidates.append(u)

        # 3) Evaluation
        best_u = None
        best_cost = float("inf")

        for u in candidates:
            # Rollout and cost accumulation
            curr_x = x0.copy()
            curr_v = v0.copy()
            col_cost_accum = 0.0

            # Integrate over horizon
            for _ in range(self.H):
                curr_v = curr_v + u * self.dt
                curr_x = curr_x + curr_v * self.dt
                if is_2d:
                    curr_x[2] = 0.0
                    curr_v[2] = 0.0

                # Check collision at this step against FILTERED threats
                col_cost_accum += self._collision_penalty(
                    curr_x, filtered_threats, d_safe_fn
                )

            goal_cost = self.goal_w * float(np.dot(curr_x - goal, curr_x - goal))
            effort_cost = self.effort_w * float(np.dot(u, u))
            total_col_cost = self.collision_w * col_cost_accum

            cost = goal_cost + effort_cost + total_col_cost
            if cost < best_cost:
                best_cost = cost
                best_u = u

        if best_u is None:
            return None
        best_u = np.asarray(best_u, dtype=float)
        best_u = np.clip(best_u, -a_max, a_max)
        if is_2d:
            best_u[2] = 0.0
        return best_u
