import numpy as np


class SwarmStateManager:
    def __init__(self):
        self.consensus_states = {}
        self.dual_vars = {}
        self.local_u = {}

    def initialize_agent(self, agent_id, dim):
        if agent_id not in self.consensus_states:
            self.consensus_states[agent_id] = np.zeros(dim)
            self.dual_vars[agent_id] = np.zeros(dim)
            self.local_u[agent_id] = np.zeros(dim)

    def get_consensus(self, agent_id):
        return self.consensus_states[agent_id]

    def get_dual(self, agent_id):
        return self.dual_vars[agent_id]

    def get_local_u(self, agent_id):
        return self.local_u.get(agent_id, np.zeros(3))

    def update_consensus(self, agent_id, val):
        self.consensus_states[agent_id] = val

    def update_dual(self, agent_id, val):
        self.dual_vars[agent_id] = val

    def update_local_u(self, agent_id, val):
        self.local_u[agent_id] = val
