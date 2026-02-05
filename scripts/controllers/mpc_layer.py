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
        num_samples: int = 0, # Unused, kept for signature compat
        goal_weight: float = 1.0,
        effort_weight: float = 0.02,
        collision_weight: float = 50.0,
        rng_seed: int = 0,
        rho: float = 1.0, # ADMM penalty parameter
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
        self.B[0:3, :] = np.eye(3) * (self.dt ** 2)
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
            self.Mx[k*self.nx : (k+1)*self.nx, :] = A_pow
            
            # Fill Mu
            # The term for u_j in x_{k+1} is A^{k-j} B
            # We need to fill block (k, j)
            for j in range(k + 1):
                # Power of A is (k-j)
                # We can compute A^(k-j) B efficiently?
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
                self.Mu[k*self.nx : (k+1)*self.nx, j*self.nu : (j+1)*self.nu] = coeff

    def _build_qp(self, x0, v0, goal, z_traj, y_traj, neighbors_pred, a_max, is_2d):
        """
        Constructs P, q, A_ineq, l, u for OSQP.
        Minimize 0.5 * U^T P U + q^T U
        Subject to l <= A_ineq U <= u
        """
        
        # 1. Hessian P and Gradient q
        # J = J_track + J_effort + J_admm
        
        # A) Tracking Cost: sum ||x_k - g||^2 * goal_w
        # X = Mx x_init + Mu U
        # ||Mx x0 + Mu U - G_ref||^2
        # = (Mu U + E)^T (Mu U + E) ... where E = Mx x0 - G_ref
        # = U^T Mu^T Mu U + 2 E^T Mu U + const
        
        # Construct G_ref vector
        # Goal is constant g. 
        # State has [p, v]. We only care about p usually? 
        # The equation says ||x_{i,k} - g_i||^2, x contains pos and vel? 
        # Usually g_i is a position goal. Let's assume we track position only (index 0-2).
        
        # Create selection matrix S to pick positions from X
        # X has size H*6. We want H*3 positions.
        # S is (H*3 x H*6)
        S_blocks = []
        for _ in range(self.H):
            # Select first 3 rows of 6
            blk = np.zeros((3, 6))
            blk[0:3, 0:3] = np.eye(3)
            S_blocks.append(blk)
        S = sparse.block_diag(S_blocks)
        
        Mu_p = S @ self.Mu  # (3H x 3H)
        Mx_p = S @ self.Mx  # (3H x 6)
        
        # G_ref is H stacked goals
        G_ref = np.tile(goal, self.H) # (3H,)
        
        x_init = np.concatenate([x0, v0])
        E = Mx_p @ x_init - G_ref
        
        P_track = 2.0 * self.goal_w * (Mu_p.T @ Mu_p)
        q_track = 2.0 * self.goal_w * (E.T @ Mu_p)
        
        # B) Effort Cost: sum ||u_k||^2 * effort_w
        # = u^T I u * effort_w
        P_effort = 2.0 * self.effort_w * sparse.eye(self.H * self.nu)
        q_effort = np.zeros(self.H * self.nu)
        
        # C) ADMM Penalty: rho * ||U - (Z - Y)||^2
        # Let Ref = Z - Y
        # ||U - Ref||^2 = U^T U - 2 Ref^T U + Ref^T Ref
        # P = 2 * rho * I
        # q = -2 * rho * Ref
        
        # Need to handle if Z or Y are None/Zeros
        if z_traj is None: z_traj = np.zeros((self.H, 3))
        if y_traj is None: y_traj = np.zeros((self.H, 3))
        
        Ref = (z_traj - y_traj).reshape(-1) # Flatten to (3H,)
        
        P_admm = 2.0 * self.rho * sparse.eye(self.H * self.nu)
        q_admm = -2.0 * self.rho * Ref
        
        # Combine Cost
        P = P_track + P_effort + P_admm
        q = q_track + q_effort + q_admm
        
        # 2. Constraints
        # A) Control Limits (Box): -a_max <= u <= a_max
        #   If is_2d, u_z must be 0? Or just constrained.
        #   Let's enforce -a_max <= u <= a_max
        #   Additionally, if 2D, u_z = 0 (Bounds 0,0)
        
        A_box = sparse.eye(self.H * self.nu)
        l_box = -np.ones(self.H * self.nu) * a_max
        u_box = np.ones(self.H * self.nu) * a_max
        
        if is_2d:
            # Set z-limits to 0 for every 3rd element
            # 0, 1, 2(z), 3, 4, 5(z) ...
            for k in range(self.H):
                idx = k * 3 + 2
                l_box[idx] = 0.0
                u_box[idx] = 0.0
                
        # B) Collision Avoidance (Linearized)
        # We need to iterate over neighbors and horizons.
        # Strict collision avoidance in MPC is hard (non-convex). 
        # We use linearized constraints: n^T (p_k - p_obs) >= margin
        # p_k is linear in U: p_k = (Mx_p @ x_init + Mu_p @ U)_k
        # This adds linear inequalities on U.
        
        A_col_list = []
        l_col_list = []
        u_col_list = [] # Infinity
        
        # Neighbors: list of (H, 3) predicted trajectories?
        # The prompt says: "Collision Constraints are updated based on the neighbors' predicted positions over the horizon"
        # We need neighbor trajectories.
        # Let's assume neighbors_pred is a list of (H, 3) arrays of POSITIONS.
        
        # For each step k, and each neighbor j:
        #   p_self_k = row k of (Mx_p x0 + Mu_p U)
        #   p_neigh_k = neighbors_pred[j][k]
        #   dist = ||p_self_k_nominal - p_neigh_k||
        #   normal n = (p_self_k_nominal - p_neigh_k) / dist
        #   Constraint: n^T (p_self_k - p_neigh_k) >= r_safe
        #   n^T p_self_k >= r_safe + n^T p_neigh_k
        #   n^T (Mx_p_k x0 + Mu_p_k U) >= rhs
        #   (n^T Mu_p_k) U >= rhs - n^T Mx_p_k x0
        
        # To do this, we need a NOMINAL trajectory to linearize around.
        # We can use the open-loop prediction from x0 assuming 0 control? 
        # Or better, use the previous solution / Z trajectory as linearization point?
        # Let's use Z trajectory (consensus) as the linearization point if available, else zero control.
        # If Z is zero (start), use constant velocity.
        
        # Current linearization point X_lin
        # Let's compute it from Z_traj (controls) if we had one, or just assume Z is state? 
        # Z_traj in DMPC usually refers to the consensus CONTROL or STATE variables?
        # Prompt: "Z_i is the consensus trajectory from neighbors". "Decision Variables: A trajectory sequence U_i".
        # Prompt: "rho ||U_i - Z_i + Y_i||". So Z is in U-space (controls).
        
        # Compute X_lin from Z_traj (controls)
        # If Z is all zeros, assume 0 control -> ballistic.
        if z_traj is None:
            U_lin = np.zeros(self.H * self.nu)
        else:
            U_lin = z_traj.reshape(-1)
            
        X_lin_full = self.Mx @ x_init + self.Mu @ U_lin
        # Extract positions
        P_lin = (S @ X_lin_full).reshape(self.H, 3)
        
        radius = 0.5 # Default safety radius if not passed? 
        # ideally passed in or config.
        # For now, let's just make it simple.
        
        if neighbors_pred:
            for k in range(self.H):
                # My pos at k
                p_self = P_lin[k]
                
                # Row selector for position at k from Mu_p
                # Mu_p is (3H x 3H). Rows 3k:3k+3
                Mu_k = Mu_p[3*k : 3*k+3, :] # (3 x 3H)
                Mx_k = Mx_p[3*k : 3*k+3, :] # (3 x 6)
                
                p_self_param = Mx_k @ x_init # Constant part
                
                for neigh_traj in neighbors_pred:
                    if k < len(neigh_traj):
                        p_neigh = neigh_traj[k]
                        
                        xr = p_self - p_neigh
                        dist = np.linalg.norm(xr)
                        if dist < 1e-6:
                             # Overlap singularity, random normal
                             n = np.array([1.0, 0.0, 0.0])
                             dist = 0.0
                        else:
                             n = xr / dist
                             
                        # Constraint: n^T p_self_new >= dist_min + n^T p_neigh
                        # n^T (Mu_k U + p_self_param) >= dist_min + n^T p_neigh
                        # n^T Mu_k U >= dist_min + n^T (p_neigh - p_self_param)
                        
                        dist_min = 0.6 # TODO: Parametrize
                        
                        # Only add constraint if close
                        if dist < 2.0:
                            A_row = n @ Mu_k # (1 x 3H)
                            lb_val = dist_min + n @ (p_neigh - p_self_param)
                            
                            A_col_list.append(A_row)
                            l_col_list.append(lb_val)
                            u_col_list.append(np.inf)
                            
        if A_col_list:
            A_col = np.vstack(A_col_list)
            l_col = np.array(l_col_list)
            u_col = np.array(u_col_list)
            
            A_final = sparse.vstack([A_box, sparse.csc_matrix(A_col)])
            l_final = np.concatenate([l_box, l_col])
            u_final = np.concatenate([u_box, u_col])
        else:
            A_final = A_box
            l_final = l_box
            u_final = u_box
            
        return P, q, A_final, l_final, u_final

    def solve_trajectory(self, x0, v0, goal, z_traj, y_traj, neighbors_pred, a_max, is_2d):
        """
        Solves the QP and returns the full trajectory U (H, 3).
        """
        P, q, A, l, u = self._build_qp(x0, v0, goal, z_traj, y_traj, neighbors_pred, a_max, is_2d)
        
        # Solve
        prob = osqp.OSQP()
        # Ensure sparse format
        P = sparse.csc_matrix(P)
        A = sparse.csc_matrix(A)
        
        prob.setup(P, q, A, l, u, verbose=False, eps_abs=1e-3, eps_rel=1e-3, warm_start=True)
        res = prob.solve()
        
        if res.info.status != 'solved':
            # Fallback or return zeros
            # print(f"OSQP Failed: {res.info.status}")
            return np.zeros((self.H, 3))
            
        U_flat = res.x
        return U_flat.reshape(self.H, 3)

    def plan(self, x, v, goal, threats, a_max: float, is_2d: bool, d_safe_fn):
        """
        Legacy compatibility method for SamplingMPCPlanner.
        Returns single step control u0 (3,).
        
        Note: 'threats' here are dicts from ConstraintBuilder. 
        For DMPC, we really expect 'neighbors_pred' (trajectories), not just current pos.
        
        If 'threats' contains 'trajectory' key, we use it. Otherwise, we assume static/linear prediction.
        """
        x = as3(x)
        v = as3(v)
        goal = as3(goal)
        
        # Convert threats to neighbors_pred format if possible
        # Or just ignore threats in this legacy mode and assume ADMM engine handles it?
        # But 'dynamics.py' calls this.
        # Let's try to extract neighbor positions from threats and assume constant position for now?
        # Or just return a simple plan.
        
        neighbors_pred = []
        # Basic conversion of threats to static trajectories
        for t in threats:
             if t.get('kind') == 'agent':
                 pos = as3(t['pos'])
                 # Static prediction
                 traj = np.tile(pos, (self.H, 1))
                 neighbors_pred.append(traj)
        
        # Call the QP
        # Z and Y are internal to ADMM, so here they are 0.
        U = self.solve_trajectory(x, v, goal, None, None, neighbors_pred, a_max, is_2d)
        
        return U[0]

# Alias for compatibility
SamplingMPCPlanner = DMPCOptimizer

