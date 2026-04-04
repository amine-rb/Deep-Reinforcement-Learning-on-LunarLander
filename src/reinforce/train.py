"""
REINFORCE with Baseline — LunarLander-v3
=========================================

Monte-Carlo policy gradient with a learned value baseline.

Algorithm overview
------------------
At each episode we collect a full trajectory τ = (s₀, a₀, r₀, …, sT).
The policy is updated by ascending the gradient:

    ∇J(θ) = E[ ∇ log π_θ(a|s) · Â_t ]

where the advantage Â_t = G_t − V(s_t) subtracts the value baseline to
reduce variance without introducing bias.

Key design choices
------------------
- Gradient accumulation over `batch_episodes` episodes before each update,
  which smooths the noisy Monte-Carlo gradient signal.
- Decaying entropy bonus: encourages exploration early, allows exploitation
  to dominate as training progresses.
- Advantage normalisation (per-batch): prevents large return magnitudes from
  dominating updates early in training.
- Separate optimisers for actor and critic — common in practice to allow
  independent learning-rate tuning.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym

from src.utils import (
    set_seed,
    resolve_device,
    make_env,
    make_eval_env,
    clip_gradients,
    plot_training_curves,
    log_config,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """All hyperparameters and I/O paths for a REINFORCE run."""

    # Environment
    env_id: str = "LunarLander-v3"
    seed: int = 42

    # Core hyperparameters
    gamma: float = 0.99          # Discount factor for future rewards
    lr_policy: float = 1e-4      # Actor learning rate
    lr_value: float = 5e-4       # Critic (baseline) learning rate
    hidden_size: int = 256        # Hidden layer width for both networks

    # Entropy regularisation (exploration schedule)
    entropy_coef: float = 0.01       # Initial entropy bonus coefficient
    entropy_coef_decay: float = 0.995 # Multiplicative decay per episode
    entropy_coef_min: float = 0.001  # Floor — prevents entropy collapsing to zero

    # Critic weighting
    value_coef: float = 0.5      # Weight of critic loss in the combined update

    # Training schedule
    max_episodes: int = 10_000
    batch_episodes: int = 4      # Accumulate gradients over N episodes per update
    grad_clip: float = 0.5       # Max gradient norm (joint actor + critic)

    # Evaluation
    eval_every: int = 50         # Evaluate every N training episodes
    eval_episodes: int = 10      # Episodes per evaluation call

    # Early stopping
    solved_mean_reward: float = 200.0
    solved_window: int = 100

    # Rendering / recording
    render_eval_human: bool = False
    record_video: bool = False
    video_dir: str = "assets/video/reinforce"

    # Output paths
    save_dir: str = "models"
    save_name: str = "reinforce_best.pt"
    plot_dir: str = "assets/plots"


# ---------------------------------------------------------------------------
# Neural networks
# ---------------------------------------------------------------------------

class PolicyNet(nn.Module):
    """
    Stochastic actor network.

    Maps an observation to a categorical action distribution (via logits).
    LayerNorm after each hidden layer improves training stability.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # returns logits, not probabilities


class ValueNet(nn.Module):
    """
    Scalar value baseline / critic network.

    Estimates V(s), the expected discounted return from state s under the
    current policy. Used to compute advantages Â_t = G_t − V(s_t).
    """

    def __init__(self, obs_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # shape: [batch]


# ---------------------------------------------------------------------------
# Episode collection
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_episode(
    env: gym.Env,
    policy: PolicyNet,
    device: torch.device,
) -> Tuple[List[np.ndarray], List[int], List[float], float, float]:
    """
    Collect one complete episode by rolling out the stochastic policy.

    Returns:
        states      — list of observations
        actions     — list of actions taken
        rewards     — list of per-step rewards
        ep_return   — undiscounted episode return
        avg_entropy — mean policy entropy over the episode
    """
    states, actions, rewards, entropies = [], [], [], []
    obs, _ = env.reset()
    done = False

    while not done:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        dist = Categorical(logits=policy(obs_t))
        action = dist.sample()

        next_obs, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated

        states.append(obs)
        actions.append(action.item())
        rewards.append(float(reward))
        entropies.append(dist.entropy().item())

        obs = next_obs

    ep_return = sum(rewards)
    avg_entropy = float(np.mean(entropies))
    return states, actions, rewards, ep_return, avg_entropy


def compute_returns(rewards: List[float], gamma: float) -> torch.Tensor:
    """
    Compute discounted returns G_t = Σ_{k≥0} γ^k · r_{t+k} via backward pass.

    Backward accumulation is equivalent to the recursive formula:
        G_{T} = r_T
        G_t   = r_t + γ · G_{t+1}
    """
    G = 0.0
    returns = []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.append(G)
    returns.reverse()
    return torch.tensor(returns, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Deterministic evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(cfg: Config, policy: PolicyNet, device: torch.device) -> float:
    """
    Evaluate the policy deterministically (argmax action) over several episodes.

    Returns the mean episode return.
    """
    env = make_eval_env(cfg.env_id, render_human=cfg.render_eval_human,
                        record_video=cfg.record_video, video_dir=cfg.video_dir)
    returns = []

    for _ in range(cfg.eval_episodes):
        obs, _ = env.reset()
        done, ep_return = False, 0.0

        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            action = Categorical(logits=policy(obs_t)).probs.argmax(dim=-1).item()
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_return += float(reward)

        returns.append(ep_return)

    env.close()
    return float(np.mean(returns))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(cfg: Config) -> Dict:
    """
    Main REINFORCE training loop.

    At every `batch_episodes` episodes:
      1. Compute discounted returns G_t for each collected trajectory.
      2. Compute advantages Â_t = G_t − V(s_t) and normalise them.
      3. Policy loss:  L_π = −E[log π(a|s) · Â_t] − α·H[π]
      4. Value loss:   L_V = MSE(G_t, V(s_t))
      5. Backpropagate joint loss, clip gradients, step both optimisers.

    Args:
        cfg: Training configuration dataclass.

    Returns:
        History dict with episode rewards and entropy values.
    """
    device = resolve_device()
    print(f"[INFO] Device: {device}")
    log_config(cfg)
    set_seed(cfg.seed)

    env = make_env(cfg.env_id, cfg.seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    policy = PolicyNet(obs_dim, act_dim, cfg.hidden_size).to(device)
    value  = ValueNet(obs_dim, cfg.hidden_size).to(device)

    opt_policy = optim.Adam(policy.parameters(), lr=cfg.lr_policy)
    opt_value  = optim.Adam(value.parameters(),  lr=cfg.lr_value)

    os.makedirs(cfg.save_dir, exist_ok=True)
    save_path = os.path.join(cfg.save_dir, cfg.save_name)

    reward_history: List[float] = []
    entropy_history: List[float] = []
    best_eval = -float("inf")
    entropy_coef = cfg.entropy_coef

    # Batch gradient buffers
    batch_policy_losses: List[torch.Tensor] = []
    batch_value_losses: List[torch.Tensor] = []

    t0 = time.time()

    for ep in range(1, cfg.max_episodes + 1):
        # --- Collect one episode ---
        states, actions, rewards, ep_return, avg_entropy = run_episode(env, policy, device)
        reward_history.append(ep_return)
        entropy_history.append(avg_entropy)

        # --- Compute targets ---
        states_t  = torch.tensor(np.array(states),  dtype=torch.float32, device=device)
        actions_t = torch.tensor(actions, dtype=torch.int64,   device=device)
        returns_t = compute_returns(rewards, cfg.gamma).to(device)

        # --- Advantage estimation: Â_t = G_t − V(s_t) ---
        # The baseline V(s_t) is the learned critic. Subtracting it reduces
        # gradient variance without introducing bias (since V does not depend on π).
        values_t   = value(states_t)
        advantages = returns_t - values_t.detach()
        # Normalise advantages per batch for stable updates
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        # --- Compute losses ---
        logits   = policy(states_t)
        dist     = Categorical(logits=logits)
        logp_t   = dist.log_prob(actions_t)
        entropy_t = dist.entropy().mean()

        # Policy gradient loss with entropy regularisation
        policy_loss = -(logp_t * advantages).mean() - entropy_coef * entropy_t
        # Critic loss: fit V(s_t) to the Monte-Carlo return G_t
        value_loss  = 0.5 * (returns_t - values_t).pow(2).mean()

        batch_policy_losses.append(policy_loss)
        batch_value_losses.append(value_loss)

        # --- Gradient update every batch_episodes episodes ---
        if ep % cfg.batch_episodes == 0 or ep == cfg.max_episodes:
            total_loss = (
                torch.stack(batch_policy_losses).mean()
                + cfg.value_coef * torch.stack(batch_value_losses).mean()
            )
            opt_policy.zero_grad()
            opt_value.zero_grad()
            total_loss.backward()
            clip_gradients([policy, value], cfg.grad_clip)
            opt_policy.step()
            opt_value.step()
            batch_policy_losses.clear()
            batch_value_losses.clear()

        # Decay entropy coefficient: shift from exploration to exploitation
        entropy_coef = max(cfg.entropy_coef_min, entropy_coef * cfg.entropy_coef_decay)

        rolling_mean = float(np.mean(reward_history[-cfg.solved_window:]))

        if ep % 10 == 0:
            print(
                f"Ep {ep:5d}/{cfg.max_episodes} | "
                f"score={ep_return:8.1f} | "
                f"avg{cfg.solved_window}={rolling_mean:7.1f} | "
                f"ent_coef={entropy_coef:.4f}"
            )

        # --- Periodic deterministic evaluation ---
        if ep % cfg.eval_every == 0:
            policy.eval()
            avg_eval = evaluate(cfg, policy, device)
            policy.train()
            print(f"[EVAL] ep={ep} | mean_return={avg_eval:.1f} (over {cfg.eval_episodes} eps)")

            if avg_eval > best_eval:
                best_eval = avg_eval
                torch.save(
                    {
                        "policy_state_dict": policy.state_dict(),
                        "value_state_dict":  value.state_dict(),
                        "cfg": cfg.__dict__,
                        "best_eval": best_eval,
                        "episode": ep,
                    },
                    save_path,
                )
                print(f"[SAVE] New best ({best_eval:.1f}) → {save_path}")

        # --- Early stopping ---
        if rolling_mean >= cfg.solved_mean_reward and ep >= cfg.solved_window:
            print(f"[DONE] Solved! rolling_mean={rolling_mean:.1f}")
            break

    env.close()
    elapsed = time.time() - t0
    print(f"\n[SUMMARY] Episodes: {len(reward_history)} | Best eval: {best_eval:.1f} | Time: {elapsed/60:.1f} min")

    os.makedirs(cfg.plot_dir, exist_ok=True)
    plot_training_curves(
        reward_history,
        entropy_history,
        algorithm_name="REINFORCE + Baseline",
        save_path=os.path.join(cfg.plot_dir, "reinforce_training.png"),
    )

    return {"episode_rewards": reward_history, "episode_entropies": entropy_history}


# ---------------------------------------------------------------------------
# Checkpoint loading & evaluation entry point
# ---------------------------------------------------------------------------

def load_policy(cfg: Config) -> PolicyNet:
    """Load a trained PolicyNet from the checkpoint specified in cfg."""
    ckpt_path = os.path.join(cfg.save_dir, cfg.save_name)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    env = gym.make(cfg.env_id)
    policy = PolicyNet(env.observation_space.shape[0], env.action_space.n, cfg.hidden_size)
    env.close()

    ckpt = torch.load(ckpt_path, map_location="cpu")
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.eval()
    return policy


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from src.utils import setup_logging

    cfg = Config()
    log_path, tee = setup_logging("experiments/logs", "reinforce")
    sys.stdout = tee
    try:
        train(cfg)
    finally:
        sys.stdout = tee.terminal
        tee.close()
