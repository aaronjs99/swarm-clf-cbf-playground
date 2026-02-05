import numpy as np
import scipy.sparse as sparse
import osqp

from utils.geometry import as3


class DMPCOptimizer:
    """
    Optimization-based MPC using OSQP for Distributed MPC (ADMM).

    Decision Variables:
      U = [u_0, u_1, ..., u_{H-1}]  (Flattened)

    State Dynamics (Double Integrator):
      x_{k+1} = x_k + v_k * dt + 0.5 * u_k * dt^2  (approx or symplectic)
      v_{k+1} = v_k + u_k * dt

    For the QP formulation, we can either:
      1. condense (optimize only U, substitute X) -> Dense Hessian, smaller dim
      2. sparse (optimize X, U) -> Sparse Hessian, larger dim

    Given H ~ 20-30, condensing is often faster/simpler for small state dims (3D position/velocity).
    Let's use Condensing for simplicity and direct mapping to the DMPC cost function provided.

    Cost Function:
      J = sum_{k=0}^{H-1} ( ||x_{k+1} - goal||^2 * w_goal
                          + ||u_k||^2 * w_effort
                          + ||u_k - (z_k - y_k)||^2 * rho/2 )
       (Note: The ADMM penalty is strictly on U usually, or X and U. The prompt says U.)

    Constraints:
      - |u| <= a_max (Box)
      - Collision Soft/Hard constraints? Prompt says "Collision Constraints are updated based on neighbors".
        Linearized half-planes: n^T (x_k - p_obs) >= d_safe
    """

    def __init__(
        self,
        enabled: bool = True,
        horizon_steps: int = 20,
        dt: float = 0.02,
        num_samples: int = 0,  # Unused, kept for signature compat
        goal_weight: float = 1.0,
        effort_weight: float = 0.02,
        collision_weight: float = 50.0,
        rng_seed: int = 0,
        rho: float = 1.0,  # ADMM penalty parameter
    ):
        self.enabled = bool(enabled)
        self.H = int(horizon_steps)
        self.dt = float(dt)
        self.goal_w = float(goal_weight)
        self.effort_w = float(effort_weight)
        self.col_w = float(collision_weight)
        self.rho = float(rho)

        # Linear Dynamics Matrices for single step: x_{k+1} = A x_k + B u_k
        # State: [px, py, pz, vx, vy, vz]
        self.nx = 6
        self.nu = 3

        # Symplectic Euler:
        # v_{k+1} = v_k + u_k * dt
        # x_{k+1} = x_k + v_{k+1} * dt = x_k + v_k * dt + u_k * dt^2

        self.A = np.eye(self.nx)
        self.A[0:3, 3:6] = np.eye(3) * self.dt

        self.B = np.zeros((self.nx, self.nu))
        self.B[0:3, :] = np.eye(3) * (self.dt**2)
        self.B[3:6, :] = np.eye(3) * self.dt

        # Precompute Prediction Matrices (Condensing)
        # X = Mx * x0 + Mu * U
        # X is stacked [x_1; ...; x_H] (Length H*nx)
        # U is stacked [u_0; ...; u_{H-1}] (Length H*nu)

        self.Mx = np.zeros((self.H * self.nx, self.nx))
        self.Mu = np.zeros((self.H * self.nx, self.H * self.nu))

        A_pow = np.eye(self.nx)
        for k in range(self.H):
            # Row k corresponds to step k+1
            # x_{k+1} = A^{k+1} x0 + sum_{j=0}^k A^{k-j} B u_j

            # Fill Mx
            A_pow = A_pow @ self.A
            self.Mx[k * self.nx : (k + 1) * self.nx, :] = A_pow

            # Fill Mu
            # The term for u_j in x_{k+1} is A^{k-j} B
            # We need to fill block (k, j)
            for j in range(k + 1):
                # Power of A is (k-j)
                # We can compute A^p * B efficiently?
                # Actually, A^p B is constant.
                # Let's just recompute for clarity or cache if slow.
                # A is simple enough.

                # A^0 B = B
                # A^1 B = A B

                # Careful with k and j indices.
                # x_{k+1} depends on u_0...u_k.
                p = k - j
                # Compute A^p * B
                # Since A is sparse/structured, we can do this cheaper, but H is small.
                Ap = np.linalg.matrix_power(self.A, p)
                coeff = Ap @ self.B
                self.Mu[
                    k * self.nx : (k + 1) * self.nx, j * self.nu : (j + 1) * self.nu
                ] = coeff

        # Pre-slice Position-only matrices for faster QP building
        # We only care about [px, py, pz] (first 3 components of each 6-dim state)
        pos_indices = []
        for k in range(self.H):
            pos_indices.extend([k * self.nx, k * self.nx + 1, k * self.nx + 2])
        self.pos_indices = np.array(pos_indices)

        self.Mu_p = sparse.csc_matrix(self.Mu[self.pos_indices, :])
        self.Mx_p = sparse.csc_matrix(self.Mx[self.pos_indices, :])
        
        # Pre-compute fixed part of Hessian for performance
        # P_track_base = 2 * goal_w * Mu_p^T Mu_p
        self.P_track_base = 2.0 * self.goal_w * (self.Mu_p.T @ self.Mu_p)

    def _build_qp(
        self,
        x0,
        v0,
        goal,
        z_traj,
        y_traj,
        neighbors_pred,
        threats,
        a_max,
        is_2d,
        u_nom=None,
    ):
        """
        Hyper-optimized QP construction.
        """
        num_u = self.H * self.nu
        x_init = np.concatenate([x0, v0])
        
        # 1. Hessian P and Gradient q
        # Hessian P is constant for a fixed rho/effort_w
        I_nu = sparse.eye(num_u, format="csc")
        P = self.P_track_base + 2.0 * (self.effort_w + self.rho) * I_nu
        
        if u_nom is not None:
             # Add nominal bias weight to P and q
             w_nom = 20.0 
             # Note: For speed we only modify the first 3x3 block
             # We can't easily modify CSC P once built, but we can add another sparse matrix
             # However, adding matrices is slow. Let's just keep P part constant if possible.
             # If we MUST have the P_u[0:3, 0:3] modification:
             pass 

        # Gradient q
        G_ref = np.tile(goal, self.H)
        E = self.Mx_p @ x_init - G_ref
        q = 2.0 * self.goal_w * (E.T @ self.Mu_p).flatten()
        
        if z_traj is not None and y_traj is not None:
            Ref = (z_traj - y_traj).reshape(-1)
            q -= 2.0 * self.rho * Ref
            
        if u_nom is not None:
             # Nominal bias w_nom should be balanced with goal_w and rho.
             # Reduced to 2.0 to prevent aggressive initial transients.
             w_nom = 2.0 
             q[0:3] -= w_nom * u_nom

        # 2. Constraints
        # Control Limits
        l_box = -np.ones(num_u) * a_max
        u_box = np.ones(num_u) * a_max
        if is_2d:
            l_box[2::3] = 0.0
            u_box[2::3] = 0.0
            
        # Collision Constraints
        A_col_list = []
        l_col_list = []
        
        U_lin = z_traj.reshape(-1) if z_traj is not None else np.zeros(num_u)
        P_lin_vec = self.Mx_p @ x_init + self.Mu_p @ U_lin
        P_lin = P_lin_vec.reshape(self.H, 3)
        
        d_min_agent = 1.4  # Radius (0.6) + Radius (0.6) + small buffer
        d_min_obs = 1.0    # Radius (0.6) + small buffer (obstacle radius added below)
        scan_dist = 4.0    # Increased scan dist for better foresight
        
        for k in range(self.H):
            p_self = P_lin[k]
            Mu_k = self.Mu_p[3*k : 3*k+3, :]
            p_const = (self.Mx_p @ x_init)[3*k : 3*k+3]
            
            # Neighbors
            for neigh_traj in neighbors_pred:
                if k < len(neigh_traj):
                    dist = np.linalg.norm(p_self - neigh_traj[k])
                    if dist < scan_dist:
                        n = (p_self - neigh_traj[k]) / (dist + 1e-6)
                        A_col_list.append(sparse.csc_matrix(n @ Mu_k))
                        l_col_list.append(d_min_agent + n @ (neigh_traj[k] - p_const))
            
            # Threats
            for t in threats:
                if t.get("kind") == "sphere":
                    dist = np.linalg.norm(p_self - t["pos"])
                    if dist < scan_dist:
                        n = (p_self - t["pos"]) / (dist + 1e-6)
                        A_col_list.append(sparse.csc_matrix(n @ Mu_k))
                        # t.get("r", 0.1) is obstacle radius. 
                        # We need agent_radius(0.6) + obstacle_radius + buffer
                        l_col_list.append(d_min_obs + t.get("r", 0.1) + n @ (t["pos"] - p_const))
                elif t.get("kind") == "wall":
                    n = np.asarray(t["normal"])
                    A_col_list.append(sparse.csc_matrix(n @ Mu_k))
                    l_col_list.append(d_min_obs + n @ (np.asarray(t["pos"]) - p_const))

        num_col = len(A_col_list)
        if num_col > 0:
            w_slack = 1e5
            P_final = sparse.block_diag([P, 2.0 * w_slack * sparse.eye(num_col, format="csc")], format="csc")
            q_final = np.concatenate([q, np.zeros(num_col)])
            
            A_col_mat = sparse.vstack(A_col_list, format="csc")
            I_u = sparse.eye(num_u, format="csc")
            I_s = sparse.eye(num_col, format="csc")
            Z_us = sparse.csc_matrix((num_u, num_col))
            Z_su = sparse.csc_matrix((num_col, num_u))

            A_final = sparse.vstack([
                sparse.hstack([I_u, Z_us]),
                sparse.hstack([A_col_mat, I_s]),
                sparse.hstack([Z_su, I_s])
            ], format="csc")
            
            l_final = np.concatenate([l_box, l_col_list, np.zeros(num_col)])
            u_final = np.concatenate([u_box, np.inf * np.ones(num_col), np.inf * np.ones(num_col)])
        else:
            P_final = P
            q_final = q
            A_final = sparse.eye(num_u, format="csc")
            l_final = l_box
            u_final = u_box
            
        return P_final, q_final, A_final, l_final, u_final

    def solve_trajectory(
        self,
        x0,
        v0,
        goal,
        z_traj,
        y_traj,
        neighbors_pred,
        threats,
        a_max,
        is_2d,
        u_nom=None,
    ):
        """
        Solves the QP and returns the full trajectory U (H, 3).
        """
        P, q, A, l, u = self._build_qp(
            x0, v0, goal, z_traj, y_traj, neighbors_pred, threats, a_max, is_2d, u_nom
        )

        prob = osqp.OSQP()
        # Matrix conversion warning fix: ensure CSC
        prob.setup(
            P, q, A, l, u, 
            verbose=False, 
            eps_abs=1e-3, 
            eps_rel=1e-3, 
            max_iter=1000, 
            warm_start=True
        )
        res = prob.solve()

        if res.info.status != "solved" and "reliable" not in res.info.status:
            # If failed, still try to return at least the nominal input part if possible
            # But usually with slacks it should solve.
            return np.zeros((self.H, 3))

        # Extract U (first self.H * self.nu components)
        u_dim = self.H * self.nu
        return res.x[:u_dim].reshape(self.H, 3)

    def plan(self, x, v, goal, threats, a_max: float, is_2d: bool, d_safe_fn):
        """
        Legacy compatibility mode.
        """
        x = as3(x)
        v = as3(v)
        goal = as3(goal)

        neighbors_pred = []
        for t in threats:
            if t.get("kind") == "agent":
                neighbors_pred.append(np.tile(as3(t["pos"]), (self.H, 1)))

        U = self.solve_trajectory(
            x, v, goal, None, None, neighbors_pred, threats, a_max, is_2d
        )
        return U[0]


# Alias for compatibility
SamplingMPCPlanner = DMPCOptimizer
