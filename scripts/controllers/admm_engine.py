import numpy as np
from controllers.base_qp import solve_qp_safe


class ADMMSolver:
    def __init__(self, rho=0.1, num_iters=2):
        self.rho = float(rho)
        self.num_iters = int(num_iters)

    def step(
        self,
        agent_id,
        u_nom,
        P,
        q,
        G,
        b,
        state_manager,
        network,
        a_max,
        slack_weight,
        fallback_zero_on_fail=True,
    ):
        dim = len(u_nom)
        state_manager.initialize_agent(agent_id, dim)

        u_local = state_manager.get_local_u(agent_id)
        neighbors = network.get_neighbors(agent_id)

        last_delta = 0.0

        for _ in range(self.num_iters):
            z = state_manager.get_consensus(agent_id)
            # consensus
            y = state_manager.get_dual(agent_id)  # dual

            P_aug = P + self.rho * np.eye(dim)
            q_aug = q - self.rho * (z - y)

            u_candidate, delta = solve_qp_safe(
                P_aug,
                q_aug,
                G,
                b,
                a_max,
                slack_weight=slack_weight,
                fallback_zero_on_fail=fallback_zero_on_fail,
            )
            u_local = u_candidate
            last_delta = delta

            neighbor_u = [state_manager.get_local_u(nid) for nid in neighbors]
            all_u = [u_local] + neighbor_u
            z_new = np.mean(all_u, axis=0)

            state_manager.update_local_u(agent_id, u_local)
            state_manager.update_consensus(agent_id, z_new)

            y_new = y + (u_local - z_new)
            state_manager.update_dual(agent_id, y_new)

        return u_local, last_delta
