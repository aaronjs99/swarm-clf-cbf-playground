import os
import argparse
import numpy as np
import torch
import sys
from datetime import datetime

# Ensure scripts is in path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.config import load_config
from rl.environment import SwarmRLWrapper
from rl.agent import PPOAgent
from rl.rl_utils import TrainingLogger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/experiment/default.yaml")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--save_interval", type=int, default=50)
    parser.add_argument("--out_dir", type=str, default="data/rl_models")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to existing model directory to resume training",
    )
    parser.add_argument("--plot", action="store_true", help="Plot at the end")
    args = parser.parse_args()

    # Handle output directory
    if args.resume:
        if not os.path.isdir(args.resume):
            print(f"Error: Resume directory {args.resume} does not exist.")
            sys.exit(1)
        args.out_dir = args.resume
        print(f"Resuming training in {args.out_dir}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out_dir = os.path.join(args.out_dir, f"run_{timestamp}")
        os.makedirs(args.out_dir, exist_ok=True)
        print(f"Starting new training run in {args.out_dir}")

    cfg = load_config(args.config)

    # Force some config settings for training speed
    cfg["viz"]["record"] = False
    cfg["viz"]["backend"] = "Agg"  # No window

    env = SwarmRLWrapper(cfg)
    logger = TrainingLogger(args.out_dir)

    print(f"Obs Dim: {env.observation_dim}, Action Dim: {env.action_dim}")

    agent = PPOAgent(env.observation_dim, env.action_dim)

    # State tracking for resume
    start_ep = 0
    # Attempt to resume from existing log file
    if os.path.exists(logger.episode_log_path):
        with open(logger.episode_log_path, "r") as f:
            start_ep = sum(1 for line in f) - 1  # minus header
            if start_ep < 0:
                start_ep = 0

    # Load existing model if available to support resume
    if start_ep > 0:
        model_path = os.path.join(args.out_dir, "ppo_model_final.pth")
        if not os.path.exists(model_path):
            model_path = os.path.join(args.out_dir, "ppo_model_best.pth")

        if os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            agent.policy.load_state_dict(
                torch.load(model_path, map_location=agent.device, weights_only=True)
            )
        else:
            print(
                f"Warning: Resuming from episode {start_ep} but no model found at {model_path}"
            )

    best_reward = -float("inf")

    def train_one_episode(ep):
        nonlocal best_reward
        obs = env.reset()
        episode_reward = 0

        for step in range(args.max_steps):
            action, log_prob, val = agent.select_action(obs)
            scaled_action = action * env.a_max
            next_obs, reward, done, info = env.step(scaled_action)

            mod = info.get("mod", 0.0)
            dist = info.get("dist_goal", 0.0)
            logger.log_step(ep, step, reward, mod, dist)

            agent.store((obs, action, log_prob, reward, done, val))

            obs = next_obs
            episode_reward += reward

            if done:
                break

        agent.update()
        logger.log_episode(ep, episode_reward, step)
        print(f"Episode {ep}: Reward = {episode_reward:.2f}, Steps = {step}")

        if ep % args.save_interval == 0:
            torch.save(agent.policy.state_dict(), f"{args.out_dir}/ppo_model_{ep}.pth")

        if episode_reward > best_reward:
            best_reward = episode_reward
            torch.save(agent.policy.state_dict(), f"{args.out_dir}/ppo_model_best.pth")

    current_ep = start_ep
    print(f"Starting/Resuming from Episode {current_ep}")

    while current_ep < args.episodes:
        try:
            train_one_episode(current_ep)
            current_ep += 1
        except KeyboardInterrupt:
            print(f"\n\nTraining Paused at Episode {current_ep}.")
            while True:
                choice = (
                    input("Options: [c]ontinue, [s]top and save, [p]lot and exit? ")
                    .lower()
                    .strip()
                )
                if choice == "c":
                    print("Resuming...")
                    # We accept that the current episode was interrupted and partial data might be in buffer/log.
                    # Ideally we clear buffer for this episode.
                    agent.buffer = []  # Clear partial trajectory
                    # logger step buffer is flushed only on episode end usually?
                    # No, log_step writes to buffer. logger.flush() is called in log_episode.
                    # So detailed log might have partial steps. That's fine.
                    # We just restart this episode index.
                    break
                elif choice == "s":
                    current_ep = args.episodes  # Force exit
                    break
                elif choice == "p":
                    args.plot = True
                    current_ep = args.episodes  # Force exit
                    break
                else:
                    print("Invalid option.")

    print("Training finished/stopped.")
    torch.save(agent.policy.state_dict(), f"{args.out_dir}/ppo_model_final.pth")
    logger.flush()

    if args.plot:
        print("Plotting results...")
        logger.plot(show=True)


if __name__ == "__main__":
    main()
