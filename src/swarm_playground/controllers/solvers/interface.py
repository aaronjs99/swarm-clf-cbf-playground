import numpy as np


class ProblemTranslator:
    def __init__(self, cfg):
        # QP params
        self.slack_weight = float(cfg.get("controller.qp.slack_weight", 1.0e4))
        self.fallback_zero_on_fail = bool(
            cfg.get("controller.qp.fallback_zero_on_fail", True)
        )

    def translate(self, u_nominal, constraints):
        """
        Translates nominal control and limits/constraints into QP form:
          min 1/2 u^T P u + q^T u
          s.t. G u <= b

        Args:
            u_nominal (np.ndarray): Desired acceleration (3,)
            constraints (tuple): (G_soft, b_soft, G_hard, b_hard) from ConstraintManager

        Returns:
            dict with P, q, G, b, G_hard, b_hard and other options
        """
        G_soft, b_soft, G_hard, b_hard = constraints

        # Minimize 1/2 || u - u_nominal ||^2
        # Equivalent to min 1/2 u^T I u - u_nominal^T u
        # So P = I, q = -u_nominal
        P = np.eye(3)
        q = -u_nominal

        return {
            "P": P,
            "q": q,
            "G": G_soft,
            "b": b_soft,
            "G_hard": G_hard,
            "b_hard": b_hard,
            "slack_weight": self.slack_weight,
            "fallback_zero_on_fail": self.fallback_zero_on_fail,
        }
