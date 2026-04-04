"""
Shared utility functions: seeding, device resolution, environment helpers,
and a reusable 4-panel training curve plot used across all algorithms.
"""

import os
import random
from typing import List

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility (Python, NumPy, PyTorch)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device() -> torch.device:
    """Return the best available compute device: CUDA > MPS (Apple) > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def make_env(env_id: str, seed: int, render_mode: str = None) -> gym.Env:
    """Create and seed a Gymnasium environment."""
    env = gym.make(env_id, render_mode=render_mode)
    env.reset(seed=seed)
    return env


def make_eval_env(env_id: str, render_human: bool = False, record_video: bool = False, video_dir: str = "assets/video") -> gym.Env:
    """
    Create an evaluation environment, optionally with human rendering or
    video recording.

    Args:
        env_id: Gymnasium environment ID.
        render_human: Show a live window during evaluation.
        record_video: Save mp4 episodes to video_dir.
        video_dir: Output directory for recorded videos.

    Returns:
        Configured Gymnasium environment.
    """
    render_mode = "human" if render_human else "rgb_array"
    env = gym.make(env_id, render_mode=render_mode)

    if record_video:
        from gymnasium.wrappers import RecordVideo
        os.makedirs(video_dir, exist_ok=True)
        env = RecordVideo(env, video_folder=video_dir, episode_trigger=lambda ep: True)

    return env


# ---------------------------------------------------------------------------
# Gradient utilities
# ---------------------------------------------------------------------------

def clip_gradients(networks: List[nn.Module], max_norm: float) -> None:
    """Apply gradient norm clipping across multiple networks jointly."""
    params = [p for net in networks for p in net.parameters()]
    nn.utils.clip_grad_norm_(params, max_norm=max_norm)


# ---------------------------------------------------------------------------
# Shared 4-panel performance plot
# ---------------------------------------------------------------------------

def plot_training_curves(
    episode_rewards: List[float],
    episode_entropies: List[float] = None,
    algorithm_name: str = "Algorithm",
    save_path: str = "training_curves.png",
) -> None:
    """
    Generate a 4-panel figure summarising training performance:
      1. Episode rewards with 100-episode moving average
      2. Entropy or epsilon over training (exploration schedule)
      3. Score distribution histogram
      4. Rolling success rate (score >= 200)

    Args:
        episode_rewards: Reward per episode from the training loop.
        episode_entropies: Entropy (policy gradient) or epsilon (DQN) values.
        algorithm_name: Label shown in the figure title.
        save_path: Output path for the PNG file.
    """
    episodes = np.arange(1, len(episode_rewards) + 1)

    plt.style.use("seaborn-v0_8-darkgrid")
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.28)

    # --- Panel 1: Rewards ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(episodes, episode_rewards, color="steelblue", alpha=0.4, linewidth=0.8, label="Episode score")
    if len(episode_rewards) >= 100:
        ma = np.convolve(episode_rewards, np.ones(100) / 100, mode="valid")
        ax1.plot(np.arange(100, len(episode_rewards) + 1), ma, color="darkorange", linewidth=2.2, label="Moving avg (100 ep)")
    ax1.axhline(200, color="seagreen", linestyle="--", linewidth=1.8, alpha=0.85, label="Solved threshold (200)")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Score")
    ax1.set_title("Training Rewards", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=9)

    # --- Panel 2: Exploration schedule ---
    ax2 = fig.add_subplot(gs[0, 1])
    if episode_entropies:
        ax2.plot(episodes[: len(episode_entropies)], episode_entropies, color="mediumpurple", linewidth=1.8, alpha=0.85)
        ax2.set_ylabel("Entropy / Epsilon")
    else:
        ax2.text(0.5, 0.5, "No entropy data", ha="center", va="center", transform=ax2.transAxes)
    ax2.set_xlabel("Episode")
    ax2.set_title("Exploration Schedule", fontsize=13, fontweight="bold")

    # --- Panel 3: Score distribution ---
    ax3 = fig.add_subplot(gs[1, 0])
    rewards_arr = np.array(episode_rewards)
    n, bins, patches = ax3.hist(rewards_arr, bins=50, color="steelblue", edgecolor="white", alpha=0.75)
    for i, patch in enumerate(patches):
        if bins[i] >= 200:
            patch.set_facecolor("seagreen")
    ax3.axvline(np.mean(rewards_arr), color="darkorange", linewidth=2, linestyle="--", label=f"Mean: {np.mean(rewards_arr):.1f}")
    ax3.set_xlabel("Score")
    ax3.set_ylabel("Count")
    ax3.set_title("Score Distribution", fontsize=13, fontweight="bold")
    ax3.legend(fontsize=9)

    # --- Panel 4: Rolling success rate ---
    ax4 = fig.add_subplot(gs[1, 1])
    window = 50
    if len(episode_rewards) >= window:
        success = np.array([(np.array(episode_rewards[max(0, i - window): i]) >= 200).mean() * 100 for i in range(window, len(episode_rewards) + 1)])
        x_range = np.arange(window, len(episode_rewards) + 1)
        ax4.plot(x_range, success, color="seagreen", linewidth=2)
        ax4.fill_between(x_range, success, alpha=0.2, color="seagreen")
        ax4.axhline(100, color="gold", linestyle="--", linewidth=1.5, alpha=0.7)
        ax4.set_ylim(-2, 105)
    ax4.set_xlabel("Episode")
    ax4.set_ylabel("Success rate (%)")
    ax4.set_title(f"Rolling Success Rate (window={window})", fontsize=13, fontweight="bold")

    fig.suptitle(f"{algorithm_name} — LunarLander-v3", fontsize=16, fontweight="bold")
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved: {save_path}")
