import os
import argparse
import numpy as np
import torch
import sys

# Ensure scripts is in path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.config import load_config
from rl.environment import SwarmRLWrapper
from rl.agent import PPOAgent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/experiment/default.yaml")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--save_interval", type=int, default=50)
    parser.add_argument("--out_dir", type=str, default="data/rl_models")
    args = parser.parse_args()

    cfg = load_config(args.config)
    
    # Force some config settings for training speed
    cfg["viz"]["record"] = False
    cfg["viz"]["backend"] = "Agg" # No window
    
    env = SwarmRLWrapper(cfg)
    
    print(f"Obs Dim: {env.observation_dim}, Action Dim: {env.action_dim}")
    
    agent = PPOAgent(env.observation_dim, env.action_dim)
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    best_reward = -float("inf")
    
    for ep in range(args.episodes):
        obs = env.reset()
        episode_reward = 0
        
        for step in range(args.max_steps):
            action, log_prob, val = agent.select_action(obs)
            
            # Action is normalized in agent? No, PPO output is usually gaussian.
            # We treat action as raw acceleration here, assuming network learns the range.
            # Or we can clip/scale.
            # agent.py actor uses Tanh so output is [-1, 1].
            # We need to scale to env.a_max.
            
            scaled_action = action * env.a_max
            
            next_obs, reward, done, info = env.step(scaled_action)
            
            mod = info['mod']
            dist = info['dist_goal']
            print(f"Step: {step} | Reward: {reward:.2f} | CBF Correction (Mod): {mod:.4f} | Dist: {dist:.2f}")
            
            agent.store((obs, action, log_prob, reward, done, val))
            
            obs = next_obs
            episode_reward += reward
            
            if done:
                break
                
        # Update PPO
        agent.update()
        
        # Logging
        print(f"Episode {ep}: Reward = {episode_reward:.2f}, Steps = {step}")
        
        if ep % args.save_interval == 0:
            torch.save(agent.policy.state_dict(), f"{args.out_dir}/ppo_model_{ep}.pth")
            
        if episode_reward > best_reward:
            best_reward = episode_reward
            torch.save(agent.policy.state_dict(), f"{args.out_dir}/ppo_model_best.pth")
            
    print("Training finished.")

if __name__ == "__main__":
    main()
