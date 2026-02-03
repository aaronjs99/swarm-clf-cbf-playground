# Swarm CLF-CBF Playground

A compact Python sandbox for **multi-agent navigation** with **Control Lyapunov Functions (CLF)** and **Control Barrier Functions (CBF)**, supporting **2D and 3D**, **static or dynamic obstacles**, optional **recording**, and a built-in **safety certificate plot**.

This repo is set up for rapid iteration: tweak the controller, add constraints, swap dynamics, and watch the swarm behave (or misbehave) in real time.

## Why this exists

- **CLF** pulls agents toward their goals.
- **CBF** keeps them from hitting obstacles and each other.
- A small QP (quadratic program) resolves the “go fast vs do not die” conflict each timestep.

The result: goal-seeking behavior with provable-style safety constraints, in a simulation harness that is easy to hack.

---

## Features

- **2D and 3D simulation** (`--dim 2d` or `--dim 3d`)
- **Swarm support** (`--num_agents N`) with inter-agent collision avoidance
- **Static or dynamic obstacles** (`--dynamic`)
- **Live animation** or **still render** (`--mode live|still`)
- **Optional video recording** (`--record`) to `media/`
- **Safety logging and plot** of barrier values (`--safety_plot true|false`)
- **QP-based controller** using `cvxopt`

---

## Repository name suggestion

Recommended GitHub repo name (short, descriptive, not too precious):

**`swarm-clf-cbf-playground`**

Other reasonable options:

- `clf-cbf-swarm-nav`
- `safe-swarm-qps`
- `cbf-swarm-sim`
- `clf-cbf-navigation-sandbox`

---

## Quickstart

### 1) Create an environment (recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 2) Install dependencies

Option A: install from `setup.py` (editable install for development)

```bash
pip install -e .
```

Option B: install dependencies directly

```bash
pip install numpy matplotlib cvxopt
```

### 3) Run a simulation

**3D swarm with dynamic obstacles:**

```bash
python run.py --dim 3d --mode live --dynamic --num_agents 5
```

**2D swarm with defaults:**

```bash
python run.py --dim 2d --mode live --num_agents 5
```

**Record a video to `media/`:**

```bash
python run.py --dim 3d --mode live --dynamic --num_agents 5 --record
```

Disable safety plot if you only want the animation:

```bash
python run.py --dim 3d --mode live --num_agents 5 --safety_plot false
```

---

## CLI options

`run.py` exposes a single unified runner:

- `--dim {2d,3d}`: simulation dimension (default: `3d`)
- `--mode {still,live}`: visualization mode (default: `live`)
- `--dynamic`: enables obstacle motion
- `--record`: saves an MP4 in `media/` (requires ffmpeg)
- `--num_obs N`: number of obstacles (default: `3` in 2D, `8` in 3D)
- `--num_agents N`: number of agents (default: `1`)
- `--safety_plot {true,false}`: plot barrier histories at end (default: `true`)

---

## How it works (high-level)

At each timestep, for each agent:

1. Compute a **nominal acceleration** toward the goal (plus a repulsion term near threats).
2. Solve a **QP** that stays close to the nominal control while satisfying:
   - a **CLF constraint** for goal convergence
   - **CBF constraints** for obstacle and inter-agent safety
   - acceleration bounds, plus a hard 2D plane constraint when in 2D

Key files:

- `scripts/controllers/dynamics.py`  
  The CLF-CBF + QP controller (`SwarmController.compute_control`).
- `scripts/controllers/base_qp.py`  
  Robust QP wrapper (`solve_qp_safe`) using `cvxopt`.
- `scripts/utils/sim_engine.py`  
  Simulation loop, rendering, logging, and recording.

---

## Project layout

```text
.
├── run.py
├── setup.py
├── scripts/
│   ├── controllers/
│   │   ├── base_qp.py
│   │   └── dynamics.py
│   └── utils/
│       ├── geometry.py
│       ├── sim_engine.py
│       └── visualization.py
├── media/
├── figures/
└── LICENSE
```

---

## Troubleshooting

### Recording fails or produces empty video

Recording uses Matplotlib’s FFMpeg writer. Install ffmpeg:

- Ubuntu/Debian:
  ```bash
  sudo apt-get update
  sudo apt-get install ffmpeg
  ```

Then rerun with `--record`.

### QP occasionally returns `None`

`cvxopt` can fail on ill-conditioned constraints or impossible situations. The controller includes a fallback:
- If the solver fails, it applies a max-acceleration “get away from the nearest threat” behavior.

If you see frequent failures, reduce gains or reduce obstacle density:
- Try fewer obstacles (`--num_obs`)
- Lower controller aggressiveness (tune `k1`, `k2`, `a_max`, `p_slack` in `SwarmController`)

---

## Development ideas (easy wins)

- Add alternative dynamics models (double integrator vs quadrotor-like constraints).
- Add formation objectives (consensus, cohesion, separation weights).
- Add goal reassignment and task allocation.
- Log QP feasibility rates and constraint margins per timestep.
- Replace `cvxopt` with `osqp` or `qpOASES` for speed.

---

## License

See `LICENSE`.