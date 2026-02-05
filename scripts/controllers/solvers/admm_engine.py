import numpy as np
from .base_qp import solve_qp_safe


class ADMMSolver:
    def __init__(self, rho=0.1, num_iters=2):
        self.rho = float(rho)
        self.num_iters = int(num_iters)

    def _integrate_path(self, x0, v0, U, dt):
        """
        Integrates U to get position trajectory.
        U: (H, 3), x0: (3,), v0: (3,)
        Returns P: (H, 3)
        """
        H = len(U)
        P = np.zeros((H, 3))
        
        curr_x = x0.copy()
        curr_v = v0.copy()
        
        for k in range(H):
            # Double integrator
            # x_{k+1} = x_k + v_k dt + 0.5 a dt^2
            # v_{k+1} = v_k + a dt
            u = U[k]
            next_x = curr_x + curr_v * dt + 0.5 * u * (dt**2)
            next_v = curr_v + u * dt
            
            P[k] = next_x
            
            curr_x = next_x
            curr_v = next_v
            
        return P

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
        G_hard=None,
        b_hard=None,
        mpc_solver=None,  # NEW: DMPC optimizer instance
        x=None,          # NEW: Current State
        v=None,          # NEW: Current Vel
        goal=None        # NEW: Goal
    ):
        """
        Executes ADMM step. 
        If 'mpc_solver' is provided, performs Trajectory DMPC.
        Otherwise, performs legacy single-step QP.
        """
        
        # -------------------------------------------------------------
        # Path A: DMPC (Trajectory Optimization)
        # -------------------------------------------------------------
        if mpc_solver is not None and mpc_solver.enabled:
             # 1. Initialize State Manager for Trajectory (if simplified int init was used before)
             # We assume state_manager handles shape updates or is initialized correctly.
             # H x 3
             H = mpc_solver.H
             state_manager.initialize_agent(agent_id, (H, 3))
             
             # neighbors
             neighbors = network.get_neighbors(agent_id)
             
             last_delta = 0.0 # Metric for convergence?
             u_local_traj = None
             
             for _ in range(self.num_iters):
                 # Retrieve Consensus vars
                 z_traj = state_manager.get_consensus(agent_id)
                 y_traj = state_manager.get_dual(agent_id)
                 
                 # Retrieve Neighbors' PREDICTED PATHS (Positions)
                 # Note: neighbors might not have initialized yet if asynchronous?
                 # Assume sync or handle missing.
                 neigh_paths = []
                 for nid in neighbors:
                     path = state_manager.get_predicted_path(nid)
                     if path is not None:
                         # Ensure path length matches? Or truncate/pad?
                         neigh_paths.append(path)
                         
                 # Local Solve (Trajectory)
                 # We ignore u_nom, P, q (single step)
                 u_local_traj = mpc_solver.solve_trajectory(
                     x, v, goal, z_traj, y_traj, neigh_paths, a_max, is_2d=False
                 )
                 
                 # Update Local U
                 state_manager.update_local_u(agent_id, u_local_traj)
                 
                 # Update Predicted Path (for neighbors to see)
                 # Integrate u_local_traj
                 pred_path = self._integrate_path(x[:3], v[:3], u_local_traj, mpc_solver.dt)
                 state_manager.update_predicted_path(agent_id, pred_path)
                 
                 # Consensus Update (ADMM)
                 # Z_new = mean( U_local + U_neighbors_consensus? )
                 # In standard ADMM for swarm, neighbors also have copies of MY variables?
                 # Or is this consensus on "What should I do"? 
                 # Usually DMPC Consensus is: we agree on a trajectory for ME? Or we agree on joint?
                 # Prompt: "rho ||Ui - Zi + Yi|| ... Zi is the consensus trajectory from neighbors".
                 # This usually means "What neighbors think I should do" or "A shared reference".
                 # Simple Consensus: Z_i = mean(U_i, U_neighbors_copies_of_i).
                 # But here neighbors optimise THEIR Own U.
                 # The prompt says: "Agents broadcast entire predicted path... step 3 Collisions updated".
                 # That describes Distributed MPC with Obstacle Avoidance (Collision Constraints).
                 # The "ADMM Consensus" term usually implies agreement on a shared variable.
                 # If we only share collision constraints, maybe Z_i is just U_i? 
                 # "Z_i is the consensus trajectory from neighbors" -> Maybe existing code tracks "What neighbors think I should do"?
                 # EXISTING admm_engine: 
                 # z_new = np.mean(all_u, axis=0) where all_u = [u_local] + [neighbor_u]
                 # This implies "We all try to match the SAME vector". 
                 # This is FLOCKING/CONSENSUS ADMM.
                 # "Step 1: Each agent solves local MPC... Step 3: Collision Constraints".
                 # The ADMM consensus implementation might be "Velocity Consensus" (Flocking) extended to "Trajectory Consensus" (Flocking in time)?
                 # Yes, Prompt: "Currently your ADMM ... shares single acc vector u. For DMPC ... share matrices".
                 # So we are implementing Consensus DMPC (Flocking).
                 
                 neighbor_trajs = [] # Neighbor's U
                 for nid in neighbors:
                     nu = state_manager.get_local_u(nid)
                     if nu is not None and nu.shape == u_local_traj.shape:
                         neighbor_trajs.append(nu)
                 
                 all_trajs = [u_local_traj] + neighbor_trajs
                 z_new = np.mean(all_trajs, axis=0)
                 
                 state_manager.update_consensus(agent_id, z_new)
                 
                 # Dual Update
                 # Y_new = Y + (U - Z)
                 y_new = y_traj + (u_local_traj - z_new)
                 state_manager.update_dual(agent_id, y_new)
                 
                 # Delta?
                 last_delta = np.linalg.norm(u_local_traj - z_new)
                 
             # Return first step
             return u_local_traj[0], last_delta

        # -------------------------------------------------------------
        # Path B: Legacy Single-Step ADMM
        # -------------------------------------------------------------
        
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
                G_hard=G_hard,
                b_hard=b_hard,
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

