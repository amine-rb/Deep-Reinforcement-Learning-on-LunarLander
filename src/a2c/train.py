"""
A2C with Generalized Advantage Estimation (GAE) — LunarLander-v3
=================================================================

Synchronous Advantage Actor-Critic (A2C) with fixed-length rollouts.

Algorithm overview
------------------
Instead of waiting for full episodes (REINFORCE), A2C collects a fixed
number of environment steps (the rollout), then updates immediately:

    1. Collect K steps with the current policy, storing (s, a, r, done).
    2. Compute advantages using GAE(γ, λ):

           δ_t = r_t + γ·V(s_{t+1})·(1−done_t) − V(s_t)
           Â_t = Σ_{l≥0} (γλ)^l · δ_{t+l}

    3. Policy loss:  L_π = −E[log π(a|s) · Â_t] − α·H[π]
    4. Value loss:   L_V = SmoothL1(V(s_t), G_t)
    5. Joint backward pass, gradient clip, step optimisers.

Why GAE improves over plain REINFORCE
--------------------------------------
- REINFORCE uses full-episode Monte-Carlo returns → high variance.
- TD(0) bootstrapping → low variance but high bias.
- GAE(λ) interpolates between the two with a single λ parameter:
    λ=0 → TD(0) (low variance, higher bias)
    λ=1 → Monte-Carlo (high variance, zero bias)
  In practice λ≈0.95 gives the best empirical trade-off.

Why A2C is faster than REINFORCE
----------------------------------
- Online updates from sub-episode rollouts → more frequent gradient steps
  per wall-clock second, especially in long-horizon environments.
- The critic provides a tighter advantage estimate per step, reducing the
  number of samples needed to improve the policy significantly.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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
    """All hyperparameters and I/O paths for an A2C run."""

    # Environment
    env_id: str = "LunarLander-v3"
    seed: int = 42

    # Core hyperparameters
    gamma: float = 0.99       # Discount factor
    gae_lambda: float = 0.95  # GAE λ — controls bias/variance trade-off
    lr_policy: float = 5e-4   # Actor learning rate
    lr_value: float = 1e-3    # Critic learning rate
    hidden_size: int = 256    # Hidden layer width for both networks
    weight_decay: float = 0.0 # L2 regularisation (AdamW)

    # Entropy schedule (linear decay)
    entropy_coef_start: float = 0.05   # Exploration weight at update 0
    entropy_coef_final: float = 0.005  # Exploration weight at final update

    # Critic weighting
    value_coef: float = 0.5   # Weight of critic loss relative to policy loss

    # Training schedule
    rollout_steps: int = 2048  # Steps collected per rollout (one A2C "batch")
    max_updates: int = 10_000  # Total number of rollout-update cycles
    grad_clip: float = 0.5     # Max gradient norm

    # Observation normalisation (online Welford statistics)
    normalize_obs: bool = False
    obs_clip: float = 10.0     # Clip normalised obs to [-obs_clip, obs_clip]
    reward_clip: Optional[float] = None  # Clip rewards (None = disabled)

    # Evaluation
    eval_every: int = 50       # Evaluate every N updates
    eval_episodes: int = 30    # Episodes per evaluation call

    # Early stopping
    solved_mean_reward: float = 200.0
    solved_window: int = 100

    # Rendering / recording
    render_eval_human: bool = False
    record_video: bool = False
    video_dir: str = "assets/video/a2c"

    # Output paths
    save_dir: str = "models"
    save_name: str = "a2c_best.pt"
    plot_dir: str = "assets/plots"


# ---------------------------------------------------------------------------
# Online observation normaliser (Welford algorithm)
# ---------------------------------------------------------------------------

class RunningMeanStd:
    """
    Incremental mean and variance using Welford's online algorithm.

    Normalising observations (zero mean, unit variance) greatly stabilises
    actor-critic training in environments with heterogeneous observation scales.
    """

    def __init__(self, shape: tuple, epsilon: float = 1e-4):
        self.mean  = np.zeros(shape, dtype=np.float64)
        self.var   = np.ones(shape,  dtype=np.float64)
        self.count = epsilon  # small offset avoids division by zero at init

    def update(self, x: np.ndarray) -> None:
        """Update statistics with a new batch of observations (shape: [N, *shape])."""
        batch_mean  = np.mean(x, axis=0)
        batch_var   = np.var(x,  axis=0)
        batch_count = x.shape[0]

        delta     = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean  = self.mean + delta * batch_count / tot_count
        M2        = (self.var * self.count
                     + batch_var * batch_count
                     + delta**2 * self.count * batch_count / tot_count)

        self.mean  = new_mean
        self.var   = M2 / tot_count
        self.count = tot_count

    def normalize(self, x: np.ndarray, clip: float = 10.0) -> np.ndarray:
        """Normalise and clip a single observation."""
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -clip, clip)

    def state_dict(self) -> dict:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, d: dict) -> None:
        self.mean, self.var, self.count = d["mean"], d["var"], d["count"]


# ---------------------------------------------------------------------------
# Neural networks
# ---------------------------------------------------------------------------

class PolicyNet(nn.Module):
    """
    Actor network: maps observation → action logits.

    Tanh activations (rather than ReLU) work well for bounded continuous
    feature spaces; they saturate smoothly and avoid dead neurons in the
    early training phase.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),  nn.Tanh(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # logits


class ValueNet(nn.Module):
    """
    Critic network: estimates the state value V(s) used for advantage computation."""

    def __init__(self, obs_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),  nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # shape: [batch]


# ---------------------------------------------------------------------------
# GAE advantage computation
# ---------------------------------------------------------------------------

def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    terminateds: torch.Tensor,
    next_value: float,
    gamma: float,
    gae_lambda: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Generalized Advantage Estimation (GAE).

    GAE accumulates multi-step TD errors exponentially weighted by (γλ):

        δ_t   = r_t + γ·V(s_{t+1})·(1−done_t) − V(s_t)   ← TD error
        Â_t   = δ_t + γλ·Â_{t+1}·(1−done_t)               ← GAE recursion

    Crucially, `terminateds` distinguishes truly terminal states (crashed /
    landed) from truncated ones (time limit). Only true terminals set
    V(s_{t+1}) = 0; truncations still bootstrap from the critic.

    Args:
        rewards:     [T] float tensor of per-step rewards.
        values:      [T] float tensor of V(s_t) predictions.
        terminateds: [T] float tensor, 1.0 if episode ended naturally.
        next_value:  V(s_{T+1}) for bootstrapping (0 if terminal).
        gamma:       Discount factor.
        gae_lambda:  GAE λ parameter.

    Returns:
        advantages: [T] normalisation-ready advantage estimates.
        returns:    [T] bootstrapped target values for the critic.
    """
    T = len(rewards)
    advantages = torch.zeros(T, dtype=torch.float32)
    values_ext = torch.cat([values, torch.tensor([next_value])])
    gae = 0.0

    for t in reversed(range(T)):
        not_terminal = 1.0 - terminateds[t].item()
        delta = rewards[t] + gamma * not_terminal * values_ext[t + 1] - values[t]
        gae   = float(delta) + gamma * gae_lambda * not_terminal * gae
        advantages[t] = gae

    returns = advantages + values
    return advantages, returns


# ---------------------------------------------------------------------------
# Rollout collection
# ---------------------------------------------------------------------------

def collect_rollout(
    env: gym.Env,
    policy: PolicyNet,
    value: ValueNet,
    cfg: Config,
    device: torch.device,
    current_obs: np.ndarray,
    current_done: bool,
    obs_normalizer: Optional[RunningMeanStd],
) -> Tuple[Dict, np.ndarray, bool, List[float], List[np.ndarray]]:
    """
    Collect a fixed-length rollout across episode boundaries.

    Unlike REINFORCE, we do not wait for episode completion. When an episode
    ends mid-rollout, we reset the environment and continue collecting.

    Args:
        env:             Training environment.
        policy:          Current actor network.
        value:           Current critic network.
        cfg:             Training configuration.
        device:          Compute device.
        current_obs:     Observation at the start of this rollout.
        current_done:    Whether the last step was terminal.
        obs_normalizer:  Optional online observation normaliser.

    Returns:
        rollout_data:     Dict with keys: states, actions, rewards, terminateds, values.
        next_obs:         First observation of the next rollout.
        next_done:        Whether the last step was terminal.
        episode_returns:  Returns for any episodes that completed this rollout.
        raw_obs:          Unnormalised observations (used to update the normaliser).
    """
    states, actions, rewards, terminateds, values_list = [], [], [], [], []
    episode_returns: List[float] = []
    raw_obs_list: List[np.ndarray] = []
    ep_return = 0.0

    if current_done:
        current_obs, _ = env.reset()
        current_done = False

    for _ in range(cfg.rollout_steps):
        raw_obs_list.append(current_obs.copy())

        obs_in = (obs_normalizer.normalize(current_obs, clip=cfg.obs_clip)
                  if obs_normalizer else current_obs)
        obs_t = torch.tensor(obs_in, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            dist        = Categorical(logits=policy(obs_t))
            action      = dist.sample().item()
            value_pred  = value(obs_t).item()

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        if cfg.reward_clip is not None:
            reward = float(np.clip(reward, -cfg.reward_clip, cfg.reward_clip))

        states.append(obs_in)
        actions.append(action)
        rewards.append(float(reward))
        terminateds.append(1.0 if terminated else 0.0)
        values_list.append(value_pred)
        ep_return += float(reward)

        if done:
            episode_returns.append(ep_return)
            ep_return = 0.0
            next_obs, _ = env.reset()

        current_obs  = next_obs
        current_done = done

    rollout_data = {
        "states":      np.array(states,      dtype=np.float32),
        "actions":     np.array(actions,     dtype=np.int64),
        "rewards":     np.array(rewards,     dtype=np.float32),
        "terminateds": np.array(terminateds, dtype=np.float32),
        "values":      np.array(values_list, dtype=np.float32),
    }
    return rollout_data, current_obs, current_done, episode_returns, raw_obs_list


# ---------------------------------------------------------------------------
# Deterministic evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    cfg: Config,
    policy: PolicyNet,
    device: torch.device,
    obs_normalizer: Optional[RunningMeanStd] = None,
) -> float:
    """Evaluate policy deterministically and return mean episode return."""
    env = make_eval_env(cfg.env_id, render_human=cfg.render_eval_human,
                        record_video=cfg.record_video, video_dir=cfg.video_dir)
    returns = []

    for _ in range(cfg.eval_episodes):
        obs, _ = env.reset()
        done, ep_return = False, 0.0
        while not done:
            obs_in = obs_normalizer.normalize(obs, clip=cfg.obs_clip) if obs_normalizer else obs
            obs_t  = torch.tensor(obs_in, dtype=torch.float32, device=device).unsqueeze(0)
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
    Main A2C training loop.

    Alternates between:
      - collect_rollout() : gather cfg.rollout_steps transitions
      - compute_gae()     : estimate advantages
      - joint update      : single backward pass for actor + critic

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

    opt_policy = optim.AdamW(policy.parameters(), lr=cfg.lr_policy, eps=1e-5, weight_decay=cfg.weight_decay)
    opt_value  = optim.AdamW(value.parameters(),  lr=cfg.lr_value,  eps=1e-5, weight_decay=cfg.weight_decay)

    obs_normalizer = RunningMeanStd(shape=(obs_dim,)) if cfg.normalize_obs else None

    os.makedirs(cfg.save_dir, exist_ok=True)
    save_path = os.path.join(cfg.save_dir, cfg.save_name)

    all_returns: List[float] = []
    entropy_history: List[float] = []
    best_eval = -float("inf")

    current_obs, _ = env.reset(seed=cfg.seed)
    current_done = False
    t0 = time.time()

    for update in range(1, cfg.max_updates + 1):
        # --- 1. Collect rollout ---
        rollout, current_obs, current_done, ep_returns, raw_obs = collect_rollout(
            env, policy, value, cfg, device, current_obs, current_done, obs_normalizer
        )

        if obs_normalizer and raw_obs:
            obs_normalizer.update(np.array(raw_obs))

        all_returns.extend(ep_returns)

        # --- 2. Compute GAE advantages ---
        states_t      = torch.tensor(rollout["states"],      dtype=torch.float32, device=device)
        actions_t     = torch.tensor(rollout["actions"],     dtype=torch.int64,   device=device)
        rewards_t     = torch.tensor(rollout["rewards"],     dtype=torch.float32)
        terminateds_t = torch.tensor(rollout["terminateds"], dtype=torch.float32)
        old_values_t  = torch.tensor(rollout["values"],      dtype=torch.float32)

        # Bootstrap next-state value (0 if the rollout ended on a terminal state)
        with torch.no_grad():
            if rollout["terminateds"][-1] == 1.0:
                next_value = 0.0
            else:
                obs_in     = obs_normalizer.normalize(current_obs, clip=cfg.obs_clip) if obs_normalizer else current_obs
                next_obs_t = torch.tensor(obs_in, dtype=torch.float32, device=device).unsqueeze(0)
                next_value = value(next_obs_t).item()

        advantages_t, returns_t = compute_gae(
            rewards_t, old_values_t, terminateds_t, next_value, cfg.gamma, cfg.gae_lambda
        )
        advantages_t = advantages_t.to(device)
        returns_t    = returns_t.to(device)

        # Normalise advantages: zero-mean, unit-variance across the rollout
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std(unbiased=False) + 1e-8)

        # --- 3. Compute losses ---
        logits    = policy(states_t)
        dist      = Categorical(logits=logits)
        log_probs = dist.log_prob(actions_t)
        entropy   = dist.entropy().mean()

        # Linear entropy decay: warm exploration → focused exploitation
        progress     = update / cfg.max_updates
        entropy_coef = max(cfg.entropy_coef_final,
                           cfg.entropy_coef_start * (1.0 - progress))

        # Actor loss: maximise advantage-weighted log-prob + entropy bonus
        policy_loss = -(log_probs * advantages_t.detach()).mean() - entropy_coef * entropy

        # Critic loss: fit value predictions to bootstrapped returns
        new_values  = value(states_t)
        value_loss  = nn.SmoothL1Loss()(new_values, returns_t.detach())

        total_loss = policy_loss + cfg.value_coef * value_loss

        # --- 4. Optimise ---
        opt_policy.zero_grad()
        opt_value.zero_grad()
        total_loss.backward()
        clip_gradients([policy, value], cfg.grad_clip)
        opt_policy.step()
        opt_value.step()

        entropy_history.append(entropy.item())

        # Logging
        if ep_returns and update % 10 == 0:
            rolling_mean = float(np.mean(all_returns[-cfg.solved_window:]))
            print(
                f"Update {update:5d}/{cfg.max_updates} | "
                f"ep_ret={np.mean(ep_returns):8.1f} | "
                f"avg{cfg.solved_window}={rolling_mean:7.1f} | "
                f"ent_coef={entropy_coef:.4f}"
            )

        # --- 5. Periodic evaluation ---
        if update % cfg.eval_every == 0:
            policy.eval()
            avg_eval = evaluate(cfg, policy, device, obs_normalizer)
            policy.train()
            print(f"[EVAL] update={update} | mean_return={avg_eval:.1f} (over {cfg.eval_episodes} eps)")

            if avg_eval > best_eval:
                best_eval = avg_eval
                ckpt = {
                    "policy_state_dict": policy.state_dict(),
                    "value_state_dict":  value.state_dict(),
                    "cfg": cfg.__dict__,
                    "best_eval": best_eval,
                    "update": update,
                }
                if obs_normalizer:
                    ckpt["obs_normalizer"] = obs_normalizer.state_dict()
                torch.save(ckpt, save_path)
                print(f"[SAVE] New best ({best_eval:.1f}) → {save_path}")

        # --- 6. Early stopping ---
        if len(all_returns) >= cfg.solved_window:
            rolling_mean = float(np.mean(all_returns[-cfg.solved_window:]))
            if rolling_mean >= cfg.solved_mean_reward:
                print(f"[DONE] Solved! rolling_mean={rolling_mean:.1f}")
                break

    env.close()
    elapsed = time.time() - t0
    print(f"\n[SUMMARY] Updates: {update} | Best eval: {best_eval:.1f} | Time: {elapsed/60:.1f} min")

    os.makedirs(cfg.plot_dir, exist_ok=True)
    plot_training_curves(
        all_returns,
        entropy_history,
        algorithm_name="A2C + GAE",
        save_path=os.path.join(cfg.plot_dir, "a2c_training.png"),
    )

    return {"episode_rewards": all_returns, "episode_entropies": entropy_history}


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_policy(cfg: Config) -> Tuple[PolicyNet, Optional[RunningMeanStd]]:
    """Load a trained PolicyNet (and obs normaliser if present) from checkpoint."""
    ckpt_path = os.path.join(cfg.save_dir, cfg.save_name)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    env = gym.make(cfg.env_id)
    policy = PolicyNet(env.observation_space.shape[0], env.action_space.n, cfg.hidden_size)
    env.close()

    ckpt = torch.load(ckpt_path, map_location="cpu")
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.eval()

    obs_normalizer = None
    if "obs_normalizer" in ckpt:
        obs_normalizer = RunningMeanStd(shape=(8,))  # LunarLander obs dim
        obs_normalizer.load_state_dict(ckpt["obs_normalizer"])

    return policy, obs_normalizer


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from src.utils import setup_logging

    cfg = Config()
    log_path, tee = setup_logging("experiments/logs", "a2c")
    sys.stdout = tee
    try:
        train(cfg)
    finally:
        sys.stdout = tee.terminal
        tee.close()
