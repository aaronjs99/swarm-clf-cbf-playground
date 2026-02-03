import numpy as np


class SwarmNetwork:
    def __init__(self, k_neighbors=3):
        self.k = k_neighbors
        self.adj_list = {}

    def update_topology(self, agents_data):
        num_agents = len(agents_data)
        self.adj_list = {i: [] for i in range(num_agents)}
        if num_agents <= 1:
            return

        pos = np.array([a["pos"] for a in agents_data])
        for i in range(num_agents):
            dists = np.linalg.norm(pos - pos[i], axis=1)
            # Exclude self (index 0) and take k-closest
            self.adj_list[i] = np.argsort(dists)[1 : self.k + 1].tolist()

    def get_neighbors(self, agent_id):
        return self.adj_list.get(agent_id, [])
