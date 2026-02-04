import numpy as np


class PDTrackingStrategy:
    def __init__(self, cfg):
        self.v_damp_far = float(cfg.get("controller.nominal.v_damp_far", 2.0))
        self.pd_switch_dist = float(cfg.get("controller.nominal.pd.switch_dist", 0.1))
        self.pd_kp = float(cfg.get("controller.nominal.pd.kp", 5.0))
        self.pd_kd = float(cfg.get("controller.nominal.pd.kd", 3.0))
        self.a_max = float(
            cfg.get("limits.a_max", 15.0)
        )  # Ensure this is passed or available

    def compute_control(self, x, v, goal):
        rel_goal = goal - x
        dist_goal = np.linalg.norm(rel_goal)

        # Goal tracking (PD + Far-field push)
        if dist_goal > self.pd_switch_dist:
            u_goal = (rel_goal / (dist_goal + 1e-9)) * self.a_max - self.v_damp_far * v
        else:
            u_goal = self.pd_kp * rel_goal - self.pd_kd * v

        return u_goal
