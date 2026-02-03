import numpy as np
from controllers.base_qp import solve_qp_safe
from utils.geometry import project_to_tangent_plane_3d

class SwarmController:
    def __init__(self, a_max=15.0, k1=10.0, k2=10.0, p_slack=1e6):
        # 1. FIX: Lower p_slack for numerical stability (1e6 is plenty)
        # 2. FIX: Increased gains (k1, k2) for a "snappier" safety response
        self.a_max, self.k1, self.k2, self.p_slack = a_max, k1, k2, p_slack
        self.v_limit_base, self.v_limit_max = 3.0, 8.0
        self.safe_buffer = 0.45 # Slightly increased buffer

    def compute_control(self, x, v, goal, obstacles, agent_idx, all_agents):
        # 3. FIX: More robust 2D check using tolerance
        is_2d = abs(x[2]) < 1e-5 and abs(goal[2]) < 1e-5

        threats = obstacles + [
            {"pos": a["pos"], "r": 0.6, "vel": a["vel"]}
            for i, a in enumerate(all_agents) if i != agent_idx
        ]

        # Nominal Control
        u_goal = -1.5 * (x - goal) - 3.0 * v
        u_repulse = np.zeros(3)
        for o in threats:
            rel = x - o["pos"]
            dist = np.linalg.norm(rel)
            if dist < (o["r"] + 2.5): # Early intervention
                if is_2d:
                    from utils.geometry import get_tangent_2d
                    tangent = get_tangent_2d(rel[:2], u_goal[:2])
                    # Higher repulsion magnitude
                    u_repulse[:2] += 12.0 * (1.0/dist - 1.0/(o["r"]+2.5)) * (1.0/dist**2) * tangent
                else:
                    u_repulse += 10.0 * (1.0/dist - 1.0/(o["r"]+2.5)) * (1.0/dist**2) * project_to_tangent_plane_3d(rel, u_goal)

        u_nom = u_goal + u_repulse
        if np.linalg.norm(u_nom) > self.a_max:
            u_nom = (u_nom / np.linalg.norm(u_nom)) * self.a_max

        # 4. QP Setup with stable weights
        P = np.diag([1.0, 1.0, 1.0, self.p_slack, self.p_slack * 10])
        q = np.array([-u_nom[0], -u_nom[1], -u_nom[2], 0.0, 0.0])

        G, b = [], []
        # Goal seeking (CLF)
        G.append([(x[i]-goal[i]) + v[i] for i in range(3)] + [-1.0, 0.0])
        b.append(-0.5 * np.linalg.norm(x - goal)**2)

        # Safety (CBF)
        for o in threats:
            rel, rel_vel = x - o["pos"], v - o["vel"]
            # 5. FIX: Incorporate safe_buffer into the CBF itself
            r_eff = o["r"] + self.safe_buffer
            h = np.linalg.norm(rel)**2 - r_eff**2
            h_dot = 2 * np.dot(rel, rel_vel)
            
            # Constraint: 2*rel^T*a >= -k1*h_dot - k2*h - 2*|rel_vel|^2
            G.append([-2*rel[0], -2*rel[1], -2*rel[2], 0.0, -1.0])
            b.append(self.k1*h_dot + self.k2*h + 2*np.linalg.norm(rel_vel)**2)

        # Box Constraints & Dimensionality
        for i in range(3):
            r_p, r_n = [0.0]*5, [0.0]*5
            r_p[i], r_n[i] = 1.0, -1.0
            if is_2d and i == 2:
                G.extend([r_p, r_n]); b.extend([0.0, 0.0])
            else:
                G.extend([r_p, r_n]); b.extend([self.a_max, self.a_max])

        res = solve_qp_safe(P, q, np.array(G, dtype=float), np.array(b, dtype=float))
        
        # 6. Safety check: If solver fails, apply max braking away from nearest threat
        if res is None:
            if threats:
                nearest = min(threats, key=lambda o: np.linalg.norm(x - o["pos"]))
                repel_dir = (x - nearest["pos"]) / (np.linalg.norm(x - nearest["pos"]) + 1e-6)
                return repel_dir * self.a_max
            return np.zeros(3)
            
        return res[:3]