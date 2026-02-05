# Safe Reinforcement Learning (RL-CBF)

This directory contains the implementation of Safe Reinforcement Learning using Control Barrier Functions (CBF). The agent is trained to achieve goals while staying within safe boundaries and avoiding obstacles by using a CBF-based safety filter.

## Infrastructure Overview

- `train.py`: Main script to start or resume training.
- `evaluate.py`: Script to evaluate a trained model.
- `agent.py`: Defines the RL agent architecture (SAC + CBF filter).
- `environment.py`: Custom Gymnasium environment for the multi-robot swarm.
- `rl_utils.py`: Logging and monitoring utilities.

## Training

### Start Fresh Training
To start a new training run:
```bash
python scripts/rl/train.py
```

### Training with Real-time Visualization
To monitor the training progress with plots:
```bash
python scripts/rl/train.py --plot
```

### Resume Training
To continue training from a specific timestamped directory:
```bash
python scripts/rl/train.py --resume data/rl_models/run_YYYYMMDD_HHMMSS --episodes 1000
```
*Note: Replace `run_YYYYMMDD_HHMMSS` with your actual run directory.*

## Evaluation

### Evaluate Latest Model
To evaluate the final model from a specific run:
```bash
python scripts/rl/evaluate.py --model_dir data/rl_models/run_YYYYMMDD_HHMMSS
```

### Evaluate Specific Checkpoint
To evaluate an early or intermediate checkpoint (e.g., episode 0 or 50):
```bash
python scripts/rl/evaluate.py --model_dir data/rl_models/run_YYYYMMDD_HHMMSS --checkpoint 50
```

## Logging and Artifacts
Training runs are saved in `data/rl_models/`. Each run contains:
- `models/`: Saved network weights (checkpoints).
- `logs/`: TensorBoard logs.
- `metadata.json`: Configuration and progress summary.
