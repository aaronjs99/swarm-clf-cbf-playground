import numpy as np

from controllers.state_manager import SwarmStateManager
from controllers.constraints.manager import ConstraintManager

from controllers.policies.nominal import NominalPolicy
from controllers.policies.connectivity import ConnectivityPolicy
from controllers.constraints.builders import ConstraintBuilder
from controllers.solvers.wrapper import SolverWrapper

from utils.config import get as dget
from utils.geometry import as3


class SwarmController:
    """
    Central controller class that coordinates:
     1. Nominal Progress Strategies (NominalPolicy)
     2. Constraints (ConstraintBuilder)
     3. Optimization Solvers (SolverWrapper)

    Refactored to be modular.
    """

    def __init__(self, cfg, a_max=15.0):
        self.cfg = cfg
        self.a_max = float(a_max)

        # 1. Policies
        self.nominal_policy = NominalPolicy(cfg, self.a_max)
        self.connectivity_policy = ConnectivityPolicy(cfg)

        # 2. Constraints
        self.constraint_manager = ConstraintManager(u_dim=3)  # Explicit dim
        self.constraint_builder = ConstraintBuilder(
            cfg, self.constraint_manager, self.a_max
        )

        # 3. Solvers
        self.solver_wrapper = SolverWrapper(cfg, self.a_max)
        self.state_manager = SwarmStateManager()

    def compute_control(self, x, v, goal, obstacles, agent_idx, all_agents):
        """
        Computes the control input (acceleration) for a single agent.
        """
        x = as3(x)
        v = as3(v)
        goal = as3(goal)
        is_2d = abs(x[2]) < 1e-5 and abs(goal[2]) < 1e-5

        # 0. Topology Update
        if agent_idx == 0:
            self.connectivity_policy.update_topology(all_agents)

        threats = self.constraint_builder.build_threats(
            obstacles, agent_idx, all_agents
        )

        # 1. Nominal Control
        u_nom = self.nominal_policy.compute(x, v, goal, agent_idx, threats, is_2d)

        # 2. Connectivity Injection
        u_conn = self.connectivity_policy.compute_contribution(agent_idx, all_agents)
        if self.connectivity_policy.inject_into_nominal:
            u_nom += u_conn

        # 3. Constraints
        self.constraint_builder.add_constraints(x, v, threats, is_2d)

        # 4. Solvers
        u_opt, delta = self.solver_wrapper.solve(
            u_nom,
            self.constraint_manager.get_combined_constraints(),
            agent_idx,
            self.state_manager,
            self.connectivity_policy.network,
        )

        if is_2d:
            u_opt[2] = 0.0

        return u_opt

    # Properties to maintain backward compatibility with visualization/logger
    @property
    def last_l2(self):
        return self.connectivity_policy.last_l2

    @property
    def last_delta(self):
        return self.solver_wrapper.last_delta
