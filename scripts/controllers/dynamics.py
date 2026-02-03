import numpy as np
from controllers.admm_engine import ADMMSolver
from controllers.state_manager import SwarmStateManager
from controllers.network import SwarmNetwork
from utils.geometry import project_to_tangent_plane_3d, get_tangent_2d


class SwarmController:
    def __init__(self, a_max=15.0, k1=10.0, k2=10.0):
        self.a_max, self.k1, self.k2 = a_max, k1, k2
        self.safe_buffer = 0.65
        self.solver = ADMMSolver(rho=0.1, num_iters=2)
        self.state_manager = SwarmStateManager()
        self.network = SwarmNetwork(k_neighbors=3)
        self.last_l2 = 0.0

    def compute_control(self, x, v, goal, obstacles, agent_idx, all_agents):
        if agent_idx == 0:
            self.network.update_topology(all_agents)

        is_2d = abs(x[2]) < 1e-5 and abs(goal[2]) < 1e-5

        # 1. Independent threats based on local perception
        threats = obstacles + [
            {"pos": a["pos"], "r": 0.6, "vel": a["vel"], "id": i}
            for i, a in enumerate(all_agents)
            if i != agent_idx
        ]

        # 1. Connectivity Maintenance Force
        # We treat this as part of the 'nominal' desire,
        # ensuring the agent doesn't pick a path that breaks the swarm.
        conn_grad, l2 = self.network.get_connectivity_gradient(agent_idx, all_agents)
        if agent_idx == 0:
            self.last_l2 = l2
        # Gain for connectivity (tune as needed)
        k_conn = 5.0 if l2 < 0.5 else 0.0
        u_conn = k_conn * conn_grad

        # 2. Get standard nominal control and add connectivity bias
        u_nom_orig = self._get_nominal_control(x, v, goal, threats, is_2d)
        u_nom = np.clip(u_nom_orig, -0.8 * self.a_max, 0.8 * self.a_max)

        # 3. Local QP Matrices
        P = np.eye(3)
        q = -u_nom
        G, b = self._get_constraints(x, v, threats, is_2d)

        # 3. Decentralized ADMM Negotiation
        u_opt = self.solver.step(
            agent_idx,
            u_nom,
            P,
            q,
            np.array(G),
            np.array(b),
            self.state_manager,
            self.network,
            self.a_max,
        )

        return u_opt

    def _get_nominal_control(self, x, v, goal, threats, is_2d):
        # 1. High-Thrust Goal Tracking
        # Instead of proportional distance, use the unit vector to the goal
        rel_goal = goal - x
        dist_goal = np.linalg.norm(rel_goal)

        if dist_goal > 0.1:
            # Push at max acceleration in the direction of the goal
            u_goal = (rel_goal / dist_goal) * self.a_max - 2.0 * v
        else:
            # Switch back to PD for fine positioning once very close
            u_goal = 5.0 * rel_goal - 3.0 * v

        u_repulse = np.zeros(3)
        for o in threats:
            rel = x - o["pos"]
            dist = np.linalg.norm(rel)
            if dist < (o["r"] + 2.5):
                if is_2d:
                    u_repulse[:2] += (
                        12.0
                        * (1.0 / dist - 1.0 / (o["r"] + 2.5))
                        * (1.0 / dist**2)
                        * get_tangent_2d(rel[:2], u_goal[:2])
                    )
                else:
                    u_repulse += (
                        10.0
                        * (1.0 / dist - 1.0 / (o["r"] + 2.5))
                        * (1.0 / dist**2)
                        * project_to_tangent_plane_3d(rel, u_goal)
                    )

        u_nom = u_goal + u_repulse
        norm = np.linalg.norm(u_nom)
        return (u_nom / norm) * self.a_max if norm > self.a_max else u_nom

    def _get_constraints(self, x, v, threats, is_2d):
        G, b = [], []
        for o in threats:
            rel, rel_vel = x - o["pos"], v - o["vel"]
            h = np.linalg.norm(rel) ** 2 - (o["r"] + self.safe_buffer) ** 2
            h_dot = 2 * np.dot(rel, rel_vel)
            G.append([-2 * rel[0], -2 * rel[1], -2 * rel[2]])
            b.append(self.k1 * h_dot + self.k2 * h + 2 * np.linalg.norm(rel_vel) ** 2)

        for i in range(3):
            r_p, r_n = [0.0] * 3, [0.0] * 3
            r_p[i], r_n[i] = 1.0, -1.0
            limit = 0.0 if (is_2d and i == 2) else self.a_max
            G.extend([r_p, r_n])
            b.extend([limit, limit])
        return G, b
