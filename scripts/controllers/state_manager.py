import numpy as np


class SwarmStateManager:
    def __init__(self):
        self.consensus_states = {} # Stores Z (ADMM Consensus, e.g. Control Trajectory)
        self.dual_vars = {}        # Stores Y (Dual variables)
        self.local_u = {}          # Stores U (Local decision, Control Trajectory)
        self.predicted_paths = {}  # Stores P (Position Trajectory for Collision Avoidance)

    def initialize_agent(self, agent_id, shape_or_dim):
        """
        Initialize state storage for agent. 
        shape_or_dim can be an int (dim) or tuple (H, dim).
        """
        # Determine shape
        if isinstance(shape_or_dim, int):
            shape = (shape_or_dim,)
        else:
            shape = tuple(shape_or_dim)
            
        if agent_id not in self.consensus_states:
            self.consensus_states[agent_id] = np.zeros(shape)
            self.dual_vars[agent_id] = np.zeros(shape)
            self.local_u[agent_id] = np.zeros(shape)
            
            # For paths, if shape corresponds to controls (H, 3), path is also (H, 3) 
            # (Positions).
            # If shape is simple int (3), path might be just pos?
            # Let's just init as same shape or empty.
            if len(shape) == 2:
                 # Trajectory
                 self.predicted_paths[agent_id] = np.zeros(shape)
            else:
                 self.predicted_paths[agent_id] = np.zeros(3)

    def get_consensus(self, agent_id):
        return self.consensus_states[agent_id]

    def get_dual(self, agent_id):
        return self.dual_vars[agent_id]

    def get_local_u(self, agent_id):
        # Return whatever is stored (could be trajectory)
        return self.local_u.get(agent_id, None)

    def get_predicted_path(self, agent_id):
        return self.predicted_paths.get(agent_id, None)

    def update_consensus(self, agent_id, val):
        self.consensus_states[agent_id] = val

    def update_dual(self, agent_id, val):
        self.dual_vars[agent_id] = val

    def update_local_u(self, agent_id, val):
        self.local_u[agent_id] = val
        
    def update_predicted_path(self, agent_id, val):
        self.predicted_paths[agent_id] = val

