# Swarm CLF-CBF Playground

A compact Python sandbox for **multi-agent navigation** using **Control Barrier Functions (CBF)** and a **CLF-like nominal controller**, supporting **2D and 3D**, dynamic environments, optional video recording, and a built-in safety certificate plot.

This repo is designed for rapid experimentation: tweak the controller, add constraints, swap dynamics, and observe swarm behavior in real time.

---

## Why this exists

- **Safety-First Control**: A core **CBF-QP (ECBF)** layer ensures zero collisions by projecting nominal actions into the safe set.
- **Distributed Coordination**: Implements **ADMM-based trajectory negotiation** for swarm-wide consensus without a central controller.
- **Robustness**: Supports **Adaptive GP-CBFs** to learn and compensate for unmodeled dynamics like drag and actuator lag.
- **RL Sandbox**: Easily swap nominal policies for Deep RL agents (PPO/SAC) using the built-in safety shield.

---

## Features

- **Distributed MPC (DMPC)**: Trajectory negotiation over a horizon $H$ using OSQP and ADMM.
- **Adaptive Safety**: Gaussian Process (GP) residual learning for high-fidelity safety certificates.
- **Agent Realism**: Actuator lag, linear drag, and stochastic noise modeling.
- **2D/3D Scenarios**: Pre-configured environments from lane-swaps to 3D obstacle corridors.
- **Headless Visualization**: Automated recording to `media/` using `Agg` backend for remote server training.

### Optional Controller Modules
- **DMPC Layer**: Enable via `controller.mpc.enabled` for horizon-based planning.
- **GP-CBF Shield**: Risk-aware safety margins based on predictive uncertainty.
- **Connectivity Policy**: Spectral graph methods for maintaining swarm topology.
- **PPO RL Agent**: Safe reinforcement learning with differentiable CBF layers.

---

## Quickstart

### 1. Installation

```bash
git clone [https://github.com/aaronjs99/swarm-clf-cbf-playground.git](https://github.com/aaronjs99/swarm-clf-cbf-playground.git)
cd swarm-clf-cbf-playground
python3 -m venv .venv
source .venv/bin/activate
pip install numpy matplotlib pyyaml cvxopt osqp scikit-learn torch
```

**Optional (required for video recording):**
Install `ffmpeg` so matplotlib can write MP4 files.

---

### 2. Run Simulations

**Default 2D Scenario (Lane Swap):**
```bash
python run.py --config config/experiment/default.yaml
```

**3D Scenario (Box):**
```bash
python run.py \
       --config config/experiment/3d_default.yaml \
       --dim 3d
```

**Override config values from the CLI:**
```bash
python run.py \
       --config config/experiment/default.yaml \
       --set sim.termination.max_frames=500 controller.cbf.buffer_obstacles=0.2
```

**Negotiated 3D Swarm (DMPC):**
```bash
python run.py \
       --config config/experiment/3d_default.yaml \
       --dim 3d \
       --set controller.mpc.enabled=true
```

**Train Safe RL Agent:**
```bash
python src/swarm_playground/rl/train.py --config config/experiment/default.yaml --episodes 500
```

---

## CLI Reference

```
python run.py --config <path_to_config> [options]
```

**Options:**

- `--config <path>`
  Path to the experiment YAML file. (Default: config/experiment/default.yaml)

- `--dim {2d, 3d}`
  Override the visualization and simulation dimension.

- `--num_agents <int>`
  Set the number of agents in the swarm. Overrides agents.num_agents.

- `--num_obs <int>`
  Set the number of obstacles. Overrides world.obstacles.num_override.

- `--dynamic`
  Enable dynamic movement for obstacles. Sets world.obstacles.dynamic.enabled to True.

- `--record`
  Force the simulation to save a video file to media/ instead of showing a live window. Sets viz.record to True.

- `--safety_plot {true, false, yes, no, 1, 0}`
  Toggle the post-simulation safety certificate plot (barrier values, connectivity, and slack).

- `--backend <string>`
  Select the Matplotlib backend (e.g., TkAgg, Qt5Agg, Agg). Note: If no DISPLAY environment variable is detected, the script automatically falls back to Agg and enables --record.

- `--set key=value ...`
  Inject arbitrary dotted-key overrides directly into the configuration dictionary. Supports int, float, and bool scalars.

Example:

```bash
python run.py \
       --config config/experiment/default.yaml \
       --set sim.dt=0.01 controller.mpc.enabled=true
```

---

## License

See `LICENSE`.