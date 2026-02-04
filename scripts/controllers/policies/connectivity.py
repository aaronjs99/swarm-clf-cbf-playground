import numpy as np
from utils.config import get as dget
from controllers.network import SwarmNetwork


class ConnectivityPolicy:
    def __init__(self, cfg):
        self.enabled = bool(dget(cfg, "controller.connectivity.enabled", True))
        k_neighbors = int(dget(cfg, "controller.connectivity.k_neighbors", 3))
        self.network = SwarmNetwork(k_neighbors=k_neighbors)

        self.gain = float(dget(cfg, "controller.connectivity.gain", 5.0))
        self.threshold = float(dget(cfg, "controller.connectivity.l2_threshold", 0.5))
        self.inject_into_nominal = bool(
            dget(cfg, "controller.connectivity.inject_into_nominal", False)
        )

        self.last_l2 = 0.0

    def update_topology(self, all_agents):
        if self.enabled:
            self.network.update_topology(all_agents)

    def compute_contribution(self, agent_idx, all_agents):
        if not self.enabled or len(all_agents) <= 1:
            return np.zeros(3)

        conn_grad, l2 = self.network.get_connectivity_gradient(agent_idx, all_agents)
        if agent_idx == 0:
            self.last_l2 = l2

        if l2 < self.threshold:
            return self.gain * conn_grad
        return np.zeros(3)
