import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


class ActorCritic(nn.Module):
    """
    Actor-Critic Network for PPO.
    Actor: Outputs mean of Gaussian distribution for actions.
    Critic: Estimates Value function V(s).
    """

    def __init__(self, obs_dim, action_dim, hidden_size=64, std=0.2):
        super(ActorCritic, self).__init__()

        # Critic
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

        # Actor
        self.actor_mean = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_dim),
            nn.Tanh(),  # Outputs in [-1, 1], scaled later
        )

        self.log_std = nn.Parameter(torch.ones(action_dim) * np.log(std))

    def forward(self):
        raise NotImplementedError

    def act(self, obs):
        mean = self.actor_mean(obs)
        std = self.log_std.exp()
        dist = Normal(mean, std)

        action = dist.sample()
        action_log_prob = dist.log_prob(action).sum(dim=-1)

        return action.detach(), action_log_prob.detach(), self.critic(obs)

    def evaluate(self, obs, action):
        mean = self.actor_mean(obs)
        std = self.log_std.exp()
        dist = Normal(mean, std)

        action_log_probs = dist.log_prob(action).sum(dim=-1)
        dist_entropy = dist.entropy().sum(dim=-1)
        values = self.critic(obs)

        return action_log_probs, values, dist_entropy


class PPOAgent:
    """
    Proximal Policy Optimization (PPO) Agent.
    """

    def __init__(
        self, obs_dim, action_dim, lr=3e-4, gamma=0.99, eps_clip=0.2, k_epochs=4
    ):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy = ActorCritic(obs_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self.buffer = []

    def select_action(self, obs):
        state = torch.FloatTensor(obs).to(self.device)
        action, action_log_prob, value = self.policy.act(state)
        return action.cpu().numpy(), action_log_prob.cpu().numpy(), value.detach()

    def store(self, transition):
        # transition: (obs, action, log_prob, reward, done, value)
        self.buffer.append(transition)

    def update(self):
        if not self.buffer:
            return

        # Unpack buffer
        obs_list = []
        act_list = []
        log_prob_list = []
        rew_list = []
        val_list = []
        done_list = []

        for t in self.buffer:
            obs_list.append(t[0])
            act_list.append(t[1])
            log_prob_list.append(t[2])
            rew_list.append(t[3])
            done_list.append(t[4])
            # t[5] is value (optional usage for GAE)

        # Convert to tensors
        old_states = torch.FloatTensor(np.array(obs_list)).to(self.device)
        old_actions = torch.FloatTensor(np.array(act_list)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(log_prob_list)).to(self.device)
        rewards = torch.FloatTensor(np.array(rew_list)).to(self.device)

        # Monte Carlo Estimate of Returns
        returns = []
        discounted_sum = 0
        for reward, is_done in zip(reversed(rewards), reversed(done_list)):
            if is_done:
                discounted_sum = 0
            discounted_sum = reward + (self.gamma * discounted_sum)
            returns.insert(0, discounted_sum)

        returns = torch.FloatTensor(returns).to(self.device)
        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        # Optimize Policy for K epochs
        for _ in range(self.k_epochs):
            # Evaluating old actions and values
            log_probs, state_values, dist_entropy = self.policy.evaluate(
                old_states, old_actions
            )

            # Match tensor shapes
            state_values = torch.squeeze(state_values)

            # Ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(log_probs - old_log_probs)

            # Surrogate Loss
            advantages = returns - state_values.detach()
            surr1 = ratios * advantages
            surr2 = (
                torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            )

            loss = (
                -torch.min(surr1, surr2)
                + 0.5 * nn.MSELoss()(state_values, returns)
                - 0.01 * dist_entropy
            )

            # Backprop
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.buffer = []
