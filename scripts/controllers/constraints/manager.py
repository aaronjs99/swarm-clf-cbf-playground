import numpy as np
from typing import List, Tuple
from . import LinearConstraint


class ConstraintManager:
    """
    Aggregates constraints from multiple sources (CBFs, Safety Filters, Limits)
    and provides them in a unified format for the Optimization Solver.
    """

    def __init__(self, u_dim: int = 3):
        self.constraints: List[LinearConstraint] = []
        self.u_dim = u_dim

    def add_constraint(self, constraint: LinearConstraint):
        """
        Adds a single LinearConstraint to the store.
        """
        self.constraints.append(constraint)

    def get_combined_constraints(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (G_soft, b_soft, G_hard, b_hard) aggregated from all added constraints.
        """
        G_soft_list, b_soft_list = [], []
        G_hard_list, b_hard_list = [], []

        for c in self.constraints:
            if c.G.size == 0:
                continue

            # Ensure 2D for G and 1D for b
            g_mat = np.atleast_2d(c.G)
            b_vec = np.atleast_1d(c.b)

            if c.hard:
                G_hard_list.append(g_mat)
                b_hard_list.append(b_vec)
            else:
                G_soft_list.append(g_mat)
                b_soft_list.append(b_vec)

        G_soft = np.vstack(G_soft_list) if G_soft_list else np.empty((0, self.u_dim))
        b_soft = np.concatenate(b_soft_list) if b_soft_list else np.empty((0,))
        G_hard = np.vstack(G_hard_list) if G_hard_list else np.empty((0, self.u_dim))
        b_hard = np.concatenate(b_hard_list) if b_hard_list else np.empty((0,))

        return G_soft, b_soft, G_hard, b_hard

    def clear(self):
        self.constraints = []
