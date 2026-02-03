import numpy as np
from controllers.base_qp import solve_qp_safe


class ADMMSolver:
    def __init__(self, rho=0.1, num_iters=2):
        self.rho = rho
        self.num_iters = num_iters

    def step(self, agent_id, u_nom, P, q, G, b, state_manager, network, a_max):
        dim = len(u_nom)
        state_manager.initialize_agent(agent_id, dim)

        u_local = state_manager.get_local_u(agent_id)
        neighbors = network.get_neighbors(agent_id)

        for _ in range(self.num_iters):
            # 1. x-update: Prioritize individual goal (q) over global consensus
            z = state_manager.get_consensus(agent_id)
            y = state_manager.get_dual(agent_id)

            P_aug = P + self.rho * np.eye(dim)
            q_aug = q - self.rho * (z - y)

            sol = solve_qp_safe(P_aug, q_aug, G, b, a_max)
            if sol is not None:
                u_local = sol

            # 2. z-update: Coordinate on safety agreement, not identical paths
            # In a distributed QP, z represents the shared consensus variable
            neighbor_u = [state_manager.get_local_u(nid) for nid in neighbors]
            all_u = [u_local] + neighbor_u
            z_new = np.mean(all_u, axis=0)

            state_manager.update_local_u(agent_id, u_local)
            state_manager.update_consensus(agent_id, z_new)

            # 3. y-update: Dual step for constraint satisfaction
            y_new = y + (u_local - z_new)
            state_manager.update_dual(agent_id, y_new)

        return u_local
