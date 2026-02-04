import argparse
import os
import sys


# Choose backend BEFORE importing matplotlib anywhere.
# If you're headless (no DISPLAY), force Agg and fall back to record mode.
def _pick_backend(cli_backend, cfg_backend):
    # If user explicitly set a backend, honor it.
    if cli_backend:
        return cli_backend

    # Use config backend if present.
    if cfg_backend:
        return cfg_backend

    # Default for interactive desktops.
    return "TkAgg"


# Ensure scripts is in path
sys.path.append(os.path.join(os.path.dirname(__file__), "scripts"))

from utils.config import load_config, set_by_dotted_key


def str2bool(v):
    if isinstance(v, bool):
        return v
    return v.lower() in ("yes", "true", "t", "y", "1")


def parse_kv(pairs):
    out = {}
    for p in pairs or []:
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def coerce_scalar(s):
    sl = str(s).lower()
    if sl in ("true", "false"):
        return sl == "true"
    try:
        if "." in str(s) or "e" in sl:
            return float(s)
        return int(s)
    except Exception:
        return s


def main():
    parser = argparse.ArgumentParser(
        description="Swarm CLF-CBF Runner with YAML config"
    )
    parser.add_argument("--config", type=str, default="config/default.yaml")

    parser.add_argument("--dim", type=str, choices=["2d", "3d"], default=None)
    parser.add_argument("--mode", type=str, choices=["still", "live"], default=None)
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--num_obs", type=int, default=None)
    parser.add_argument("--num_agents", type=int, default=None)
    parser.add_argument("--safety_plot", type=str2bool, default=None)
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help="Matplotlib backend, e.g. TkAgg, Qt5Agg, Agg",
    )

    parser.add_argument(
        "--set",
        nargs="*",
        default=None,
        help="Override YAML keys: --set controller.qp.slack_weight=1e5 sim.termination.goal_delta=0.3",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    # CLI direct overrides
    if args.dim is not None:
        cfg["viz"]["dim"] = args.dim
    if args.mode is not None:
        cfg["viz"]["mode"] = args.mode
    if args.safety_plot is not None:
        cfg["viz"]["safety_plot"] = bool(args.safety_plot)
    if args.num_agents is not None:
        cfg["agents"]["num_agents"] = int(args.num_agents)

    if args.dynamic:
        cfg["world"]["obstacles"]["dynamic"]["enabled"] = True
    if args.record:
        cfg["viz"]["record"] = True
    if args.num_obs is not None:
        cfg["world"]["obstacles"]["num_override"] = int(args.num_obs)

    # dotted-key overrides
    overrides = parse_kv(args.set)
    for k, v in overrides.items():
        set_by_dotted_key(cfg, k, coerce_scalar(v))

    # Backend selection + headless handling
    cfg_backend = cfg.get("viz", {}).get("backend", None)
    chosen_backend = _pick_backend(args.backend, cfg_backend)

    # If no display, force non-interactive backend and force record mode.
    headless = (os.environ.get("DISPLAY", "") == "") and (os.name != "nt")
    if headless:
        chosen_backend = "Agg"
        if cfg["viz"]["mode"] == "live":
            cfg["viz"]["mode"] = "still"
        cfg["viz"]["record"] = True

    os.environ["MPLBACKEND"] = chosen_backend
    cfg["viz"]["backend"] = chosen_backend

    # Import after backend is set
    from controllers.dynamics import SwarmController
    from utils.sim_engine import run_simulation

    dim = cfg["viz"]["dim"]
    a_max = (
        cfg["controller"]["dim_defaults"]["a_max_3d"]
        if dim == "3d"
        else cfg["controller"]["dim_defaults"]["a_max_2d"]
    )
    ctrl = SwarmController(cfg, a_max=a_max)

    print(
        f"[run] dim={cfg['viz']['dim']} mode={cfg['viz']['mode']} "
        f"record={cfg['viz']['record']} agents={cfg['agents']['num_agents']} backend={cfg['viz']['backend']}"
    )

    run_simulation(cfg, ctrl)


if __name__ == "__main__":
    main()
