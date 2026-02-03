import argparse
import sys
import os

# Ensure scripts is in path
sys.path.append(os.path.join(os.path.dirname(__file__), "scripts"))

from controllers.dynamics import SwarmController
from utils.sim_engine import run_simulation


def str2bool(v):
    if isinstance(v, bool):
        return v
    return v.lower() in ("yes", "true", "t", "y", "1")


def main():
    parser = argparse.ArgumentParser(description="Unified Swarm CLF-CBF Runner")
    parser.add_argument("--dim", type=str, choices=["2d", "3d"], default="3d")
    parser.add_argument("--mode", type=str, choices=["still", "live"], default="live")
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--num_obs", type=int, default=None)
    parser.add_argument("--num_agents", type=int, default=1)
    parser.add_argument("--safety_plot", type=str2bool, default=True)
    args = parser.parse_args()

    if args.num_obs is None:
        args.num_obs = 3 if args.dim == "2d" else 8

    # Dimension-Agnostic Swarm Controller
    ctrl = SwarmController(a_max=20.0 if args.dim == "3d" else 15.0)

    run_simulation(
        {
            "dim": args.dim,
            "mode": args.mode,
            "ctrl": ctrl,
            "dynamic": args.dynamic,
            "record": args.record,
            "num_obs": args.num_obs,
            "num_agents": args.num_agents,
            "safety_plot": args.safety_plot,
        }
    )


if __name__ == "__main__":
    main()
