"""
Unified entry point for training and evaluating all RL agents.

Usage
-----
# Train
python run.py train reinforce
python run.py train a2c
python run.py train dqn

# Evaluate a saved checkpoint
python run.py eval reinforce --checkpoint models/reinforce_best.pt
python run.py eval a2c       --checkpoint models/a2c_best.pt
python run.py eval dqn       --checkpoint models/dqn_best.pth --episodes 100

# Watch a trained agent play
python run.py play reinforce --checkpoint models/reinforce_best.pt
python run.py play a2c       --checkpoint models/a2c_best.pt
python run.py play dqn       --checkpoint models/dqn_best.pth
"""

from __future__ import annotations

import argparse
import sys

from src.utils import setup_logging


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _train(algo: str, args: argparse.Namespace) -> None:
    log_path, tee = setup_logging("experiments/logs", algo)
    sys.stdout = tee
    try:
        if algo == "reinforce":
            from src.reinforce.train import Config, train
            cfg = Config()
            train(cfg)

        elif algo == "a2c":
            from src.a2c.train import Config, train
            cfg = Config()
            train(cfg)

        elif algo == "dqn":
            from src.dqn.train import Config, train
            cfg = Config()
            if args.episodes:
                cfg.n_episodes = args.episodes
            train(cfg)

    finally:
        sys.stdout = tee.terminal
        tee.close()
        print(f"Log saved → {log_path}")


def _eval(algo: str, args: argparse.Namespace) -> None:
    n_ep = args.episodes or 100

    if algo == "reinforce":
        from src.reinforce.train import Config, evaluate, load_policy
        import torch
        cfg    = Config()
        device = torch.device("cpu")
        policy = load_policy(cfg)
        result = evaluate(cfg, policy, device)
        print(f"\nResults: mean={result:.1f}")

    elif algo == "a2c":
        from src.a2c.train import Config, evaluate, load_policy
        import torch
        cfg    = Config()
        device = torch.device("cpu")
        policy, obs_norm = load_policy(cfg)
        result = evaluate(cfg, policy, device, obs_norm)
        print(f"\nResults: mean={result:.1f}")

    elif algo == "dqn":
        from src.dqn.train import Config, evaluate
        cfg  = Config()
        ckpt = args.checkpoint or f"models/{cfg.save_name}"
        evaluate(cfg, ckpt, n_episodes=n_ep, render_human=args.render)


def _play(algo: str, args: argparse.Namespace) -> None:
    """Launch a live rendering window with the trained policy."""
    if algo == "reinforce":
        from src.reinforce.train import Config, load_policy
        from src.utils import resolve_device
        import torch
        from torch.distributions import Categorical
        import gymnasium as gym
        cfg    = Config()
        device = resolve_device()
        policy = load_policy(cfg).to(device)
        env    = gym.make(cfg.env_id, render_mode="human")
        obs, _ = env.reset()
        while True:
            import numpy as np
            obs_t  = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            action = Categorical(logits=policy(obs_t)).probs.argmax(dim=-1).item()
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()

    elif algo == "a2c":
        from src.a2c.train import Config, load_policy
        from src.utils import resolve_device
        import torch
        from torch.distributions import Categorical
        import gymnasium as gym
        cfg    = Config()
        device = resolve_device()
        policy, obs_norm = load_policy(cfg)
        policy = policy.to(device)
        env    = gym.make(cfg.env_id, render_mode="human")
        obs, _ = env.reset()
        while True:
            import numpy as np
            obs_in = obs_norm.normalize(obs, clip=cfg.obs_clip) if obs_norm else obs
            obs_t  = torch.tensor(obs_in, dtype=torch.float32, device=device).unsqueeze(0)
            action = Categorical(logits=policy(obs_t)).probs.argmax(dim=-1).item()
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()

    elif algo == "dqn":
        from src.dqn.train import Config, load_agent
        cfg = Config()
        ckpt = args.checkpoint or f"models/{cfg.save_name}"
        env, agent = load_agent(cfg, ckpt, render_human=True)
        obs, _ = env.reset()
        while True:
            action = agent.act(obs)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

ALGORITHMS = ["reinforce", "a2c", "dqn"]
COMMANDS   = ["train", "eval", "play"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RL on LunarLander-v3 — REINFORCE / A2C / DQN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("command",   choices=COMMANDS,   help="Action to perform")
    parser.add_argument("algorithm", choices=ALGORITHMS, help="RL algorithm")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint file")
    parser.add_argument("--episodes",   type=int, default=None, help="Override number of episodes")
    parser.add_argument("--render",     action="store_true",    help="Enable human rendering during eval")

    args = parser.parse_args()

    dispatch = {"train": _train, "eval": _eval, "play": _play}
    dispatch[args.command](args.algorithm, args)


if __name__ == "__main__":
    main()
