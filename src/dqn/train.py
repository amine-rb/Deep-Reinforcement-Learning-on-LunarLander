"""
Deep Q-Network (DQN) — LunarLander-v3
======================================

Off-policy value-based RL with experience replay and a target network.

Algorithm overview
------------------
DQN differs fundamentally from policy-gradient methods (REINFORCE, A2C):
instead of directly optimising the policy, it learns a Q-function Q(s,a)
that estimates the expected discounted return from state s after taking action a.
The policy is then implicit: π(s) = argmax_a Q(s,a).

Two key stabilisation tricks (Mnih et al., 2015):
  1. Experience replay: store transitions (s,a,r,s',done) in a circular
     buffer and sample random mini-batches. This breaks temporal correlations
     that cause divergence when training on consecutive on-policy data.
  2. Target network: maintain a separate Q-network θ⁻ whose weights are
     only periodically copied from the online network θ. The Bellman target
     r + γ·max_{a'} Q_{θ⁻}(s',a') is then a stable regression target.

Bellman update rule
-------------------
For a sampled transition (s, a, r, s', done):
    y = r + γ·(1−done)·max_{a'} Q_{θ⁻}(s',a')   ← TD target
    L = MSE(Q_θ(s,a), y)                           ← regression loss

Compared to REINFORCE/A2C
--------------------------
- Sample efficiency: off-policy replay reuses past experience.
  Each environment step is used O(replay_ratio) times for training.
- Exploration: ε-greedy (random actions) rather than entropy regularisation.
  Simpler but less informed about the current policy's uncertainty.
- No policy gradient: DQN cannot directly optimise stochastic policies,
  making it unsuitable for continuous action spaces without extensions (DDPG, SAC).
"""

from __future__ import annotations

import csv
import os
import random
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

from src.utils import (
    set_seed,
    resolve_device,
    log_config,
    plot_training_curves,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """All hyperparameters and I/O paths for a DQN run."""

    # Environment
    env_id: str = "LunarLander-v3"
    seed: int = 42

    # Training schedule
    n_episodes: int = 2_000
    max_steps: int = 1_000      # Max steps per episode (safety cap)

    # Core hyperparameters
    gamma: float = 0.99          # Discount factor
    learning_rate: float = 1e-3  # Adam learning rate for Q-network

    # ε-greedy exploration schedule
    epsilon_start: float = 1.0   # Start fully random
    epsilon_min: float = 0.01    # Never less than 1% random
    epsilon_decay: float = 0.995 # Multiplicative decay per replay step

    # Experience replay
    batch_size: int = 64
    memory_size: int = 10_000   # Circular replay buffer capacity

    # Target network
    target_update_freq: int = 10  # Copy online → target every N episodes

    # Network architecture
    hidden_size: int = 128       # Hidden layer width

    # Logging
    log_every: int = 50          # Print progress every N episodes

    # Output paths
    save_dir: str = "models"
    save_name: str = "dqn_best.pth"
    log_dir: str = "experiments/metrics"
    plot_dir: str = "assets/plots"

    # Rendering / recording
    render_eval_human: bool = False


# ---------------------------------------------------------------------------
# Q-Network
# ---------------------------------------------------------------------------

class DQNetwork(nn.Module):
    """
    Q-function approximator: maps state → Q-values for all actions.

    The output dimension equals the number of discrete actions. The implicit
    greedy policy selects the action with the highest Q-value.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # shape: [batch, action_dim]


# ---------------------------------------------------------------------------
# DQN Agent
# ---------------------------------------------------------------------------

class DQNAgent:
    """
    Encapsulates the DQN agent: Q-networks, replay buffer, and learning logic.

    Attributes:
        q_network:      Online network updated every training step.
        target_network: Lagged copy of q_network used for Bellman targets.
        memory:         Circular replay buffer storing (s, a, r, s', done).
        epsilon:        Current exploration probability.
    """

    def __init__(self, state_dim: int, action_dim: int, cfg: Config, device: torch.device):
        self.action_dim = action_dim
        self.cfg        = cfg
        self.device     = device

        self.q_network      = DQNetwork(state_dim, action_dim, cfg.hidden_size).to(device)
        self.target_network = DQNetwork(state_dim, action_dim, cfg.hidden_size).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=cfg.learning_rate)
        self.loss_fn   = nn.MSELoss()
        self.memory    = deque(maxlen=cfg.memory_size)
        self.epsilon   = cfg.epsilon_start

    def remember(self, state: np.ndarray, action: int, reward: float,
                 next_state: np.ndarray, done: bool) -> None:
        """Store a transition in the replay buffer."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state: np.ndarray) -> int:
        """
        ε-greedy action selection.

        With probability ε, take a uniformly random action (explore).
        Otherwise, take the greedy action argmax_a Q(s, a) (exploit).
        """
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return int(self.q_network(state_t).argmax(dim=1).item())

    def replay(self) -> Optional[float]:
        """
        Sample a mini-batch from memory and perform one gradient step.

        The Bellman target uses the frozen target_network to compute
        max_{a'} Q(s', a'), which prevents the moving-target instability.

        Returns:
            Mean batch loss, or None if the buffer is not yet full enough.
        """
        if len(self.memory) < self.cfg.batch_size:
            return None

        batch = random.sample(self.memory, self.cfg.batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))

        states_t      = torch.as_tensor(states,      dtype=torch.float32, device=self.device)
        actions_t     = torch.as_tensor(actions,     dtype=torch.int64,   device=self.device)
        rewards_t     = torch.as_tensor(rewards,     dtype=torch.float32, device=self.device)
        next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t       = torch.as_tensor(dones,       dtype=torch.float32, device=self.device)

        # Current Q-values for the taken actions
        current_q = self.q_network(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Bellman target: r + γ·(1−done)·max_{a'} Q_target(s',a')
        with torch.no_grad():
            max_next_q = self.target_network(next_states_t).max(dim=1)[0]
            target_q   = rewards_t + (1.0 - dones_t) * self.cfg.gamma * max_next_q

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon after each learning step
        self.epsilon = max(self.cfg.epsilon_min, self.epsilon * self.cfg.epsilon_decay)

        return float(loss.item())

    def update_target_network(self) -> None:
        """Hard copy online weights → target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())


# ---------------------------------------------------------------------------
# Metrics writer
# ---------------------------------------------------------------------------

def _open_csv_writer(path: Path) -> Tuple:
    f = open(path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(
        f,
        fieldnames=["episode", "score", "avg_score_100", "epsilon", "steps", "mean_loss", "duration_s"],
    )
    writer.writeheader()
    return f, writer


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(cfg: Config) -> Dict:
    """
    Main DQN training loop.

    At each episode:
      1. Roll out the environment with ε-greedy actions.
      2. Store every transition in the replay buffer.
      3. After each step, sample a mini-batch and perform a gradient update.
      4. Every `target_update_freq` episodes, sync the target network.

    Args:
        cfg: Training configuration dataclass.

    Returns:
        Dict with scores, checkpoint path, and CSV log path.
    """
    set_seed(cfg.seed)
    device = resolve_device()
    log_config(cfg)
    print(f"[INFO] Device: {device}")

    os.makedirs(cfg.save_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)

    env = gym.make(cfg.env_id)
    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim, cfg, device)

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = Path(cfg.log_dir) / f"dqn_{ts}.csv"
    ckpt_path = Path(cfg.save_dir) / cfg.save_name
    f_csv = None

    # Persist configuration alongside the CSV
    import json
    with open(Path(cfg.log_dir) / f"dqn_{ts}_config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    scores: List[float] = []
    epsilon_history: List[float] = []

    try:
        f_csv, writer = _open_csv_writer(csv_path)

        for ep in range(cfg.n_episodes):
            t0 = time.time()
            state, _ = env.reset(seed=cfg.seed + ep)
            done, total_reward, steps = False, 0.0, 0
            losses: List[float] = []

            while not done and steps < cfg.max_steps:
                action = agent.act(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = bool(terminated or truncated)

                agent.remember(state, action, float(reward), next_state, done)
                loss = agent.replay()
                if loss is not None:
                    losses.append(loss)

                state         = next_state
                total_reward += float(reward)
                steps        += 1

            # Hard update target network periodically
            if ep % cfg.target_update_freq == 0:
                agent.update_target_network()

            scores.append(total_reward)
            epsilon_history.append(agent.epsilon)
            avg_100   = float(np.mean(scores[-100:]))
            mean_loss = float(np.mean(losses)) if losses else float("nan")
            dt        = time.time() - t0

            writer.writerow({
                "episode":      ep + 1,
                "score":        total_reward,
                "avg_score_100": avg_100,
                "epsilon":      agent.epsilon,
                "steps":        steps,
                "mean_loss":    mean_loss,
                "duration_s":   dt,
            })

            if (ep + 1) % cfg.log_every == 0:
                print(
                    f"Ep {ep+1:5d}/{cfg.n_episodes} | "
                    f"score={total_reward:8.2f} | "
                    f"avg100={avg_100:7.2f} | "
                    f"eps={agent.epsilon:5.3f} | "
                    f"loss={mean_loss:7.4f}"
                )

        torch.save(agent.q_network.state_dict(), ckpt_path)
        print(f"\n[SAVE] Checkpoint → {ckpt_path}")

    except KeyboardInterrupt:
        partial = ckpt_path.with_stem(ckpt_path.stem + "_partial")
        torch.save(agent.q_network.state_dict(), partial)
        print(f"\n[INTERRUPTED] Partial checkpoint → {partial}")
        raise

    finally:
        if f_csv is not None:
            f_csv.close()
        env.close()

    os.makedirs(cfg.plot_dir, exist_ok=True)
    plot_training_curves(
        scores,
        epsilon_history,
        algorithm_name="DQN",
        save_path=os.path.join(cfg.plot_dir, "dqn_training.png"),
    )

    return {
        "scores":       scores,
        "avg_last_100": float(np.mean(scores[-100:])),
        "checkpoint":   str(ckpt_path),
        "csv_log":      str(csv_path),
    }


# ---------------------------------------------------------------------------
# Checkpoint loading & evaluation
# ---------------------------------------------------------------------------

def load_agent(cfg: Config, checkpoint_path: str, render_human: bool = False) -> Tuple[gym.Env, DQNAgent]:
    """Load a trained DQNAgent from a checkpoint file."""
    device = resolve_device()
    env    = gym.make(cfg.env_id, render_mode="human" if render_human else None)

    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n, cfg, device)
    agent.q_network.load_state_dict(torch.load(checkpoint_path, map_location=device))
    agent.q_network.eval()
    agent.epsilon = 0.0  # fully greedy at evaluation time

    return env, agent


def evaluate(cfg: Config, checkpoint_path: str, n_episodes: int = 100, render_human: bool = False) -> Dict:
    """
    Evaluate a trained DQN agent deterministically.

    Args:
        cfg:             Configuration used during training.
        checkpoint_path: Path to the saved Q-network weights.
        n_episodes:      Number of evaluation episodes.
        render_human:    Show live rendering window.

    Returns:
        Dict with mean, std, min, max, success_rate.
    """
    env, agent = load_agent(cfg, checkpoint_path, render_human=render_human)
    scores = []

    with torch.no_grad():
        for ep in range(n_episodes):
            state, _ = env.reset(seed=cfg.seed + 10_000 + ep)
            done, total_reward, steps = False, 0.0, 0

            while not done and steps < cfg.max_steps:
                action = agent.act(state)
                state, reward, terminated, truncated, _ = env.step(action)
                done = bool(terminated or truncated)
                total_reward += float(reward)
                steps += 1

            scores.append(total_reward)
            print(f"Eval ep {ep+1:3d}/{n_episodes} | score={total_reward:8.2f}")

    env.close()
    scores_arr = np.array(scores)
    summary = {
        "mean":         float(scores_arr.mean()),
        "std":          float(scores_arr.std()),
        "min":          float(scores_arr.min()),
        "max":          float(scores_arr.max()),
        "success_rate": float((scores_arr >= 200).mean()),
    }
    print(f"\n[EVAL] mean={summary['mean']:.1f} ± {summary['std']:.1f} | "
          f"success={summary['success_rate']*100:.1f}% | "
          f"min={summary['min']:.1f} | max={summary['max']:.1f}")
    return summary


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from src.utils import setup_logging

    cfg = Config()
    log_path, tee = setup_logging("experiments/logs", "dqn")
    sys.stdout = tee
    try:
        train(cfg)
    finally:
        sys.stdout = tee.terminal
        tee.close()
