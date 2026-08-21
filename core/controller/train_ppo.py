import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from core.device_utils import get_optimal_device
from core.controller.ppo_agent import ActorCritic
from core.controller.vectorized_env import VectorizedBuildingEnv


class RolloutBuffer:
    def __init__(self, num_steps: int, num_envs: int, state_dim: int, action_dim: int, device: str):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = device

        self.states = torch.zeros((num_steps, num_envs, state_dim), device=device)
        self.actions = torch.zeros((num_steps, num_envs, action_dim), device=device)
        self.log_probs = torch.zeros((num_steps, num_envs), device=device)
        self.rewards = torch.zeros((num_steps, num_envs), device=device)
        self.dones = torch.zeros((num_steps, num_envs), device=device)
        self.values = torch.zeros((num_steps, num_envs), device=device)
        self.advantages = torch.zeros((num_steps, num_envs), device=device)
        self.returns = torch.zeros((num_steps, num_envs), device=device)
        self.ptr = 0

    def store(self, state, action, log_prob, reward, done, value):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.values[self.ptr] = value
        self.ptr += 1

    def compute_gae(self, next_value: torch.Tensor, gamma: float = 0.99, gae_lambda: float = 0.95):
        last_gae = 0.0
        for step in reversed(range(self.num_steps)):
            if step == self.num_steps - 1:
                next_val = next_value
            else:
                next_val = self.values[step + 1]

            non_terminal = 1.0 - self.dones[step]
            delta = self.rewards[step] + gamma * next_val * non_terminal - self.values[step]
            last_gae = delta + gamma * gae_lambda * non_terminal * last_gae
            self.advantages[step] = last_gae

        self.returns = self.advantages + self.values
        self.ptr = 0


def train_safe_rl_ppo(
    total_timesteps: int = 50000,
    num_envs: int = 32,
    num_steps: int = 64,
    batch_size: int = 128,
    lr: float = 3e-4,
    epochs_per_update: int = 4,
    clip_ratio: float = 0.2,
    save_checkpoint_path: str = "models/checkpoints/ppo_safe_rl_best.pt",
    force_cpu: bool = False,
) -> Tuple[ActorCritic, Dict]:
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    device = get_optimal_device(force_cpu)
    print(f"[Safe-RL PPO] Starting training on device: {device} | Parallel Envs: {num_envs}")

    env = VectorizedBuildingEnv(num_envs=num_envs, num_zones=5, device=device)
    state_dim = 32
    action_dim = 11

    actor_critic = ActorCritic(state_dim=state_dim, action_dim=action_dim, hidden_dim=128).to(device)
    optimizer = torch.optim.Adam(actor_critic.parameters(), lr=lr)

    buffer = RolloutBuffer(num_steps=num_steps, num_envs=num_envs, state_dim=state_dim, action_dim=action_dim, device=device)

    num_updates = total_timesteps // (num_envs * num_steps)
    num_updates = max(1, num_updates)

    state = env.get_observations()
    best_reward = -float("inf")
    training_metrics = {}

    start_time = time.time()
    for update in range(1, num_updates + 1):
        for _ in range(num_steps):
            with torch.no_grad():
                dist, value = actor_critic(state)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1)

            next_state, reward, done, _ = env.step(action)
            buffer.store(state, action, log_prob, reward, done.float(), value.squeeze(-1))
            state = next_state

        with torch.no_grad():
            _, next_value = actor_critic(state)
            buffer.compute_gae(next_value.squeeze(-1))

        # Flatten rollout buffer for PPO update
        b_states = buffer.states.view(-1, state_dim)
        b_actions = buffer.actions.view(-1, action_dim)
        b_log_probs = buffer.log_probs.view(-1)
        b_advantages = buffer.advantages.view(-1)
        b_returns = buffer.returns.view(-1)

        # Normalize advantages
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        total_samples = b_states.shape[0]
        for _ in range(epochs_per_update):
            perm = torch.randperm(total_samples, device=device)
            for start_idx in range(0, total_samples, batch_size):
                end_idx = min(start_idx + batch_size, total_samples)
                idx = perm[start_idx:end_idx]

                mb_states = b_states[idx]
                mb_actions = b_actions[idx]
                mb_old_log_probs = b_log_probs[idx]
                mb_advantages = b_advantages[idx]
                mb_returns = b_returns[idx]

                dist, value = actor_critic(mb_states)
                new_log_probs = dist.log_prob(mb_actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()

                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = 0.5 * ((value.squeeze(-1) - mb_returns) ** 2).mean()

                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(actor_critic.parameters(), max_norm=0.5)
                optimizer.step()

        mean_ep_reward = float(buffer.rewards.mean().item() * 288)  # Approximate daily reward
        if mean_ep_reward > best_reward or update == num_updates:
            best_reward = mean_ep_reward
            save_path = Path(save_checkpoint_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "state_dict": actor_critic.state_dict(),
                "state_dim": state_dim,
                "action_dim": action_dim,
                "mean_reward": mean_ep_reward,
                "update": update
            }, str(save_path))

        if update % max(1, num_updates // 5) == 0 or update == num_updates:
            elapsed = time.time() - start_time
            sps = int((update * num_envs * num_steps) / elapsed)
            print(
                f"Update {update:3d}/{num_updates:3d} | Steps: {update*num_envs*num_steps:6d} | "
                f"Reward/Day: {mean_ep_reward:.2f} | Speed: {sps} steps/s"
            )

    print(f"[Safe-RL PPO] Training complete! Best Checkpoint saved to: {save_checkpoint_path}")
    return actor_critic, training_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Aetheris Safe-RL PPO Agent")
    parser.add_argument("--steps", type=int, default=30000, help="Total environment steps")
    parser.add_argument("--num-envs", type=int, default=32, help="Number of parallel environments")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--cpu", action="store_true", help="Force CPU training")
    args = parser.parse_args()

    train_safe_rl_ppo(
        total_timesteps=args.steps,
        num_envs=args.num_envs,
        batch_size=args.batch_size,
        force_cpu=args.cpu
    )
