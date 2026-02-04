# Swarm CLF-CBF Playground

A compact Python sandbox for **multi-agent navigation** using **Control Barrier Functions (CBF)** and a **CLF-like nominal controller**, supporting **2D and 3D**, dynamic environments, optional video recording, and a built-in safety certificate plot.

This repo is designed for rapid experimentation: tweak the controller, add constraints, swap dynamics, and observe swarm behavior in real time.

---

## Why this exists

- A **CLF-like nominal controller (PD with goal fading)** drives agents toward their goals.
- **CBF (ECBF, relative degree 2)** prevents collisions with obstacles, walls, and other agents.
- A small **QP (quadratic program)** resolves the “go fast vs do not crash” conflict each timestep.
- An optional **sampling MPC layer** provides longer-horizon intent while the **CBF-QP enforces safety**.

The result is goal-directed behavior with provable-style safety constraints inside a simulation harness that is easy to extend.

---

## Features

- **2D and 3D simulation**
- **Multi-agent swarm support** with inter-agent collision avoidance
- **Dynamic obstacles** with bouncing sphere physics
- **Live animation** (default) or **record-to-video** (`--record`, writes `media/swarm_{2,3}d.mp4`)
- **Headless-safe recording**: automatically switches to `Agg` backend and records if no display is detected
- **Safety logging and certificate plots** (barrier values, connectivity, QP slack)
- **QP-based controller** powered by `cvxopt`
- **Agent dynamics realism**, including:
  - actuator lag
  - linear drag
  - velocity limits
  - optional acceleration noise
  - gravity in 3D (with controller compensation)

### Optional Controller Modules

Configured in `config/controller/*.yaml` and included via experiment configs:

- **Sampling MPC planner** → `controller.mpc.enabled`
- **Connectivity maintenance** → `controller.connectivity.enabled`
- **Braking-distance viability filter** → `controller.braking.enabled`
- **ADMM-style distributed solver (experimental)** → `controller.admm.enabled`
- **QP slack tuning** → `controller.qp.slack_weight`

---

## Quickstart

### 1. Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy matplotlib pyyaml cvxopt
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
python run.py --config config/experiment/3d_default.yaml --dim 3d
```

**Override config values from the CLI:**
```bash
python run.py   --config config/experiment/default.yaml   --set     sim.termination.max_frames=500     controller.cbf.buffer_obstacles=0.2
```

---

## CLI Reference

```
python run.py --config <experiment.yaml> [options]
```

**Options:**

- `--dim {2d,3d}`  
  Override visualization dimension.

- `--record`  
  Force video recording instead of live animation.

- `--dynamic`  
  Force dynamic obstacles on regardless of config.

- `--num_agents N`  
  Override `agents.num_agents`.

- `--num_obs N`  
  Override `world.obstacles.num_override`.

- `--safety_plot true|false`  
  Toggle the post-simulation safety certificate plot.

- `--backend {TkAgg, Qt5Agg, Agg}`  
  Select the matplotlib backend.

- `--set key=value ...`  
  Apply dotted-key overrides into the merged YAML config.

Example:

```bash
python run.py   --config config/experiment/default.yaml   --set sim.dt=0.01 controller.mpc.enabled=true
```

---

## Modular Configuration

The configuration system is fully compositional and uses YAML includes.

```
config/
├── sim/           # timestep, termination, realism dynamics
├── viz/           # live vs record backends, safety plot
├── world/         # scenario definitions (2D lanes, 3D box)
├── controller/    # nominal, CBF, MPC, connectivity, solver
└── experiment/    # entry-point configs that compose the above
```

Example `config/experiment/default.yaml`:

```yaml
include:
  - ../sim/default.yaml
  - ../viz/live.yaml
  - ../world/2d_lanes.yaml
  - ../controller/nominal.yaml
  - ../controller/cbf.yaml
  - ../controller/mpc.yaml
  - ../controller/connectivity.yaml
  - ../controller/solver.yaml
```

This structure makes it easy to create new scenarios by mixing and matching components without duplicating configuration.

---

## Modular Architecture

Designed for plug-and-play experimentation:

**Controllers**
```
scripts/controllers/
├── policies/        # nominal motion, connectivity
├── constraints/     # CBF builders and aggregation
├── solvers/         # QP + ADMM wrapper
└── mpc_layer.py     # sampling-based planner
```

**Simulation Utilities**
```
scripts/utils/
├── stepper.py       # physics integration (no visualization)
├── animator.py      # matplotlib visualization
├── logger.py        # safety metric logging
└── sim_engine.py    # orchestration
```

The separation between nominal control, constraints, and solvers allows new control strategies to be added with minimal coupling.

---

## License

See `LICENSE`.