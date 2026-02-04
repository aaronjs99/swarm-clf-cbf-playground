from utils.config import get as dget
from controllers.solvers.admm_engine import ADMMSolver
from controllers.solvers.base_qp import solve_qp_safe
from controllers.solvers.interface import ProblemTranslator


class SolverWrapper:
    def __init__(self, cfg, a_max):
        self.cfg = cfg
        self.a_max = a_max
        self.translator = ProblemTranslator(cfg)

        self.use_admm = bool(dget(cfg, "controller.admm.enabled", True))
        rho = float(dget(cfg, "controller.admm.rho", 0.1))
        iters = int(dget(cfg, "controller.admm.iters", 2))
        self.admm = ADMMSolver(rho=rho, num_iters=iters)

        self.last_delta = 0.0

    def solve(self, u_nom, constraints, agent_idx, state_manager, network):
        qp_prob = self.translator.translate(u_nom, constraints)

        if self.use_admm:
            u_opt, delta = self.admm.step(
                agent_id=agent_idx,
                u_nom=u_nom,
                P=qp_prob["P"],
                q=qp_prob["q"],
                G=qp_prob["G"],
                b=qp_prob["b"],
                state_manager=state_manager,
                network=network,
                a_max=self.a_max,
                slack_weight=qp_prob["slack_weight"],
                fallback_zero_on_fail=qp_prob["fallback_zero_on_fail"],
                G_hard=qp_prob["G_hard"],
                b_hard=qp_prob["b_hard"],
            )
        else:
            u_opt, delta = solve_qp_safe(
                P=qp_prob["P"],
                q=qp_prob["q"],
                G=qp_prob["G"],
                b=qp_prob["b"],
                a_max=self.a_max,
                slack_weight=qp_prob["slack_weight"],
                fallback_zero_on_fail=qp_prob["fallback_zero_on_fail"],
                G_hard=qp_prob["G_hard"],
                b_hard=qp_prob["b_hard"],
            )

        self.last_delta = float(delta)
        return u_opt, delta
