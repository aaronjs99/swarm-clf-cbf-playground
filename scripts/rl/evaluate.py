import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

# Path hack to make imports work
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.config import load_config
from rl.environment import SwarmRLWrapper
from rl.agent import PPOAgent
from utils.animator import SwarmAnimator

def main():
    # 1. Load Config & Environment (Force 'TkAgg' to see the window)
    cfg = load_config("config/experiment/default.yaml")
    cfg["viz"]["backend"] = "TkAgg"  
    cfg["viz"]["record"] = False
    
    # 2. Init Environment
    env = SwarmRLWrapper(cfg)
    obs = env.reset()

    # 3. Load Trained Agent
    agent = PPOAgent(env.observation_dim, env.action_dim)
    model_path = "data/rl_models/ppo_model_best.pth"
    
    if os.path.exists(model_path):
        agent.policy.load_state_dict(torch.load(model_path, weights_only=True))
        print("Loaded trained model!")
    else:
        print("No model found, running with random weights (expect jitter).")

    # 4. Setup Animator
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d" if env.dim == 3 else None)
    animator = SwarmAnimator(
        ax, env.agents, env.goals_init, env.obs, env.dim,
        agent_radius=float(cfg["controller"]["cbf"]["agent_radius"]),
        buffer_agents=float(cfg["controller"]["cbf"]["buffer_agents"]),
        buffer_obs=float(cfg["controller"]["cbf"]["buffer_obstacles"]),
    )

    # 5. Simulation Loop
    for step in range(500):
        # Get Action from RL
        action, _, _ = agent.select_action(obs)
        scaled_action = action * env.a_max
        
        # Step Environment
        obs, reward, done, info = env.step(scaled_action)
        
        # Debug Print
        print(f"Step {step}: CBF Correction = {info['mod']:.4f}, Reward = {reward:.2f}")

        # Draw
        plt.cla() # Clear axis
        
        # Re-setup axis limits (simple hack for 2D visualization stability)
        if env.dim == 2:
            ax.set_xlim(-6, 6); ax.set_ylim(-6, 6)
            ax.set_aspect('equal')
        
        # Update Animator
        # Calculate crude centroid for camera
        positions = np.array([a["pos"] for a in env.agents])
        centroid = np.mean(positions, axis=0)
        
        animator.update(env.agents, env.obs, centroid, step)
        plt.pause(0.01)

        if done:
            print("Goal Reached!")
            break

if __name__ == "__main__":
    main()