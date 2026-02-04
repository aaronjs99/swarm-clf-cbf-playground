# Swarm CLF-CBF Playground

A compact Python sandbox for **multi-agent navigation** with **Control Lyapunov Functions (CLF)** and **Control Barrier Functions (CBF)**, supporting **2D and 3D**, **static or dynamic obstacles**, optional **recording**, and a built-in **safety certificate plot**.

This repo is set up for rapid iteration: tweak the controller, add constraints, swap dynamics, and watch the swarm behave (or misbehave) in real time.

---

## Why this exists

- **CLF** pulls agents toward their goals.
- **CBF (ECBF, relative degree 2)** keeps them from hitting obstacles, walls, and each other.
- A small **QP (quadratic program)** resolves the “go fast vs do not die” conflict each timestep.
- An optional **planning layer (sampling MPC)** gives longer-horizon intent, while the **CBF-QP still enforces safety**.

The result: goal-seeking behavior with provable-style safety constraints, in a simulation harness that is easy to hack.

---

## Features

- **2D and 3D simulation** (`--dim 2d` or `--dim 3d`)
- **Swarm support** (`--num_agents N`) with inter-agent collision avoidance
- **Static obstacles, walls, and optional dynamics** (bouncing spheres inside a box)
- **Live animation** or **still render** (`--mode live|still`)
- **Optional video recording** (`--record`) to `media/`
- **Safety logging and plot** (minimum barrier values, connectivity, QP slack)
- **QP-based controller** using `cvxopt`
- Optional controller modules you can toggle in `config/default.yaml`:
  - **Sampling MPC planner** (`controller.mpc.enabled`)
  - **Connectivity maintenance** via algebraic connectivity (`controller.connectivity.enabled`)
  - **Braking-distance viability filter** (`controller.braking.enabled`)
  - Experimental **ADMM-style solve wrapper** (`controller.admm.enabled`)

---

## Quickstart

### 1. Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # Or just pip install numpy matplotlib pyyaml cvxopt
```

### 2. Run Simulations

**Default 2D Scenario (Lanes Swap):**
```bash
python run.py --config config/experiment/default.yaml
```

**3D Scenario (Box):**
```bash
python run.py --config config/experiment/3d_default.yaml --dim 3d --num_agents 4
```

**Override Configs on CLI:**
```bash
python run.py \
  --config config/experiment/default.yaml \
  --set \
    sim.termination.max_frames=500 \
    controller.cbf.buffer_obstacles=0.2
```

---

## Modular Configuration

The configuration is no longer a monolithic file. It uses a **profile-based** include system.
See `config/` directory:

- `base.yaml`: Common defaults.
- `sim/`: Physics and time steps.
- `viz/`: Visualization settings (live vs record).
- `world/`: Scenario definitions (2d lanes, 3d box).
- `controller/`: Component settings (nominal, cbf, mpc, solver).
- `experiment/`: Entry point configs that composed the above.

Example `experiment/default.yaml`:
```yaml
include:
  - ../base.yaml
  - ../sim/default.yaml
  - ../viz/live.yaml
  - ../world/2d_lanes.yaml
  - ../controller/nominal.yaml
  ...
```

---

## Modular Architecture

Refactored to support plug-and-play components:

- **Logic**: `scripts/controllers/policies/` (Nominal, Connectivity)
- **Constraints**: `scripts/controllers/constraints/` (Builder, Manager)
- **Solvers**: `scripts/controllers/solvers/` (QP, ADMM wrapper)
- **Simulation**: `scripts/utils/`
  - `stepper.py`: Pure physics stepping (no viz).
  - `animator.py`: Matplotlib visualization.
  - `logger.py`: Safety metric logging.
  - `sim_engine.py`: Orchestrator.

---

## License
See `LICENSE`.