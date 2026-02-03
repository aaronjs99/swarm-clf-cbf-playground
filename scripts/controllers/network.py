import numpy as np


class SwarmNetwork:
    def __init__(self, k_neighbors=3):
        self.k = k_neighbors
        self.adj_list = {}
        self.last_fiedler_vector = None

    def update_topology(self, agents_data):
        num_agents = len(agents_data)
        self.adj_list = {i: [] for i in range(num_agents)}
        if num_agents <= 1:
            return

        pos = np.array([a["pos"] for a in agents_data])
        for i in range(num_agents):
            dists = np.linalg.norm(pos - pos[i], axis=1)
            # Take k-closest excluding self
            self.adj_list[i] = np.argsort(dists)[1 : self.k + 1].tolist()

    def get_connectivity_gradient(self, agent_idx, agents_data):
        """Computes the gradient to increase the algebraic connectivity (lambda_2)."""
        n = len(agents_data)
        if n <= 1:
            return np.zeros(3)

        # 1. Build Adjacency and Degree matrices
        A = np.zeros((n, n))
        for i, neighbors in self.adj_list.items():
            for j in neighbors:
                A[i, j] = 1.0
                A[j, i] = 1.0  # Ensure undirected for Laplacian properties

        D = np.diag(np.sum(A, axis=1))
        L = D - A

        # 2. Get Eigenvalues and Eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        # eigenvalues are sorted; idx 0 is lambda_1=0, idx 1 is lambda_2
        lambda_2 = eigenvalues[1]
        v2 = eigenvectors[:, 1]  # The Fiedler Vector

        # 3. Gradient of lambda_2 w.r.t agent position x_i:
        # Higher v2[i] means the agent is on one "fringe" of the graph.
        # Moving toward neighbors with similar v2 values strengthens connectivity.
        grad = np.zeros(3)
        pos_i = agents_data[agent_idx]["pos"]

        for j in self.adj_list[agent_idx]:
            pos_j = agents_data[j]["pos"]
            # Weight the movement by the squared difference in Fiedler components
            weight = (v2[agent_idx] - v2[j]) ** 2
            grad += weight * (pos_j - pos_i)

        return grad, lambda_2

    def get_neighbors(self, agent_id):
        return self.adj_list.get(agent_id, [])
