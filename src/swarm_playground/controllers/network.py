import numpy as np


class SwarmNetwork:
    def __init__(self, k_neighbors=3):
        self.k = int(k_neighbors)
        self.adj_list = {}

    def update_topology(self, agents_data):
        num_agents = len(agents_data)
        self.adj_list = {i: [] for i in range(num_agents)}
        if num_agents <= 1:
            return

        pos = np.array([a["pos"] for a in agents_data])
        for i in range(num_agents):
            dists = np.linalg.norm(pos - pos[i], axis=1)
            self.adj_list[i] = np.argsort(dists)[1 : self.k + 1].tolist()

    def get_connectivity_gradient(self, agent_idx, agents_data):
        n = len(agents_data)
        if n <= 1:
            return np.zeros(3), 0.0

        A = np.zeros((n, n))
        for i, neighbors in self.adj_list.items():
            for j in neighbors:
                A[i, j] = 1.0
                A[j, i] = 1.0

        D = np.diag(np.sum(A, axis=1))
        L = D - A

        eigenvalues, eigenvectors = np.linalg.eigh(L)
        lambda_2 = float(eigenvalues[1])
        v2 = eigenvectors[:, 1]

        grad = np.zeros(3)
        pos_i = agents_data[agent_idx]["pos"]

        for j in self.adj_list.get(agent_idx, []):
            pos_j = agents_data[j]["pos"]
            weight = (v2[agent_idx] - v2[j]) ** 2
            grad += weight * (pos_j - pos_i)

        return grad, lambda_2

    def get_neighbors(self, agent_id):
        return self.adj_list.get(agent_id, [])
