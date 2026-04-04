# Deep Reinforcement Learning on LunarLander-v3

**Implementing and comparing REINFORCE, A2C, and DQN on a continuous-state control task.**

*Amine Rouibi · Thomas Sinapi — Master IASD, Paris-Dauphine / PSL*

---

## 🎬 Demo

![Preview](assets/demo.gif)

▶️ Full video:
https://raw.githubusercontent.com/tomasnp/rl-LunarLander/main/assets/video/visual_evaluation.mp4

*Average score: **247 ± 18** over 100 evaluation episodes. Success rate: **91%**.*

---

## Problem

[LunarLander-v3](https://gymnasium.farama.org/environments/box2d/lunar_lander/) is a discrete-action control task with an 8-dimensional continuous observation space (position, velocity, angle, angular velocity, leg contacts) and 4 actions (do nothing, left thruster, main engine, right thruster).

The reward function penalises fuel use, rewards proximity to the landing pad, and gives ±100 for a successful landing or crash. **An episode is considered solved when the agent scores ≥ 200.**

The environment is a meaningful benchmark for policy-gradient methods:
- Episodes have variable length and dense rewards — ideal for Monte-Carlo estimates.
- The observation space is continuous — eliminates tabular Q-learning.
- Safe landing requires coordinated multi-step decisions — tests credit assignment.

---

## Methods

### REINFORCE + Baseline

Classic Monte-Carlo policy gradient. The policy is updated at the end of each episode using the discounted return G_t as the learning signal. A learned value baseline V(s) is subtracted to form the advantage Â_t = G_t − V(s_t), which reduces gradient variance without introducing bias.

**Key choices:**
- Gradient accumulation over 4 episodes per update — smooths the noisy MC signal.
- Entropy regularisation with exponential decay — transitions from exploration to exploitation.
- Advantage normalisation per batch — prevents large early returns from destabilising training.

### A2C + GAE

Synchronous Advantage Actor-Critic with Generalized Advantage Estimation. Instead of full episodes, A2C collects fixed-length rollouts (2 048 steps) and updates immediately, decoupling update frequency from episode length.

**GAE interpolates between TD(0) and Monte-Carlo** via the λ parameter:

```
δ_t   = r_t + γ · V(s_{t+1}) · (1 − done_t) − V(s_t)   ← TD error
Â_t   = Σ_{l≥0} (γλ)^l · δ_{t+l}                       ← GAE (λ=0.95)
```

With λ=0.95, advantages are stable and low-bias — the key reason A2C converges faster and more reliably than REINFORCE on this task.

**Key choices:**
- Linear entropy decay (0.05 → 0.005) — coarser schedule than REINFORCE's multiplicative one.
- SmoothL1 critic loss — less sensitive to outlier returns than MSE.
- Correctly distinguishes *terminated* from *truncated* episodes when bootstrapping V(s_{T+1}).

### DQN

Off-policy value-based method. Instead of optimising the policy directly, DQN learns a Q-function Q(s,a) via the Bellman equation and derives an implicit greedy policy.

**Two stabilisation mechanisms:**
1. **Experience replay** — random mini-batch sampling from a 10 000-transition circular buffer breaks temporal correlations.
2. **Target network** — a periodically-synced copy of Q is used to compute Bellman targets, preventing the moving-target instability.

**Key choices:**
- ε-greedy with multiplicative decay (1.0 → 0.01) — simpler exploration than entropy regularisation.
- MSE loss on Q-values — standard Bellman regression objective.
- Hard target update every 10 episodes.

---

## Experimental Setup

| Setting | Value |
|---|---|
| Environment | `LunarLander-v3` (Gymnasium 0.29) |
| Random seed | 42 |
| Hardware | Apple M-series / NVIDIA GPU |
| Framework | PyTorch 2.1 |

**Architecture** — all networks use two hidden layers:

| Algorithm | Actor hidden | Critic hidden | Activation |
|---|---|---|---|
| REINFORCE | 256 | 256 | LayerNorm + ReLU |
| A2C | 256 | 256 | Tanh |
| DQN | — | 128 (Q-net) | ReLU |

---

## Results

### Training curves

| REINFORCE | A2C | DQN |
|---|---|---|
| ![reinforce](assets/plots/reinforce_training.png) | ![a2c](assets/plots/a2c_training.png) | ![dqn](assets/plots/dqn_training.png) |

### Final evaluation (100 deterministic episodes)

| Algorithm | Mean Score | Std | Success Rate | Episodes to Solve |
|---|---|---|---|---|
| REINFORCE + Baseline | 198 ± 52 | 52 | 68% | ~6 000 |
| A2C + GAE | **247 ± 18** | 18 | **91%** | ~1 200 updates (≈2.5M steps) |
| DQN | 231 ± 31 | 31 | 83% | ~1 500 episodes |

*"Solved" = rolling mean ≥ 200 over 100 episodes.*

---

## Key Insights

**Why A2C outperforms REINFORCE**

REINFORCE suffers from high variance because the advantage estimate G_t is a single Monte-Carlo sample over an entire episode. GAE in A2C replaces this with a multi-step TD estimate that trades a small amount of bias for a large reduction in variance. The result is a much smoother loss landscape and faster convergence.

Additionally, rollout-based updates (every 2 048 steps) are more frequent than episode-based updates, giving the critic more gradient steps early in training — which in turn produces better advantage estimates.

**Why DQN is more sample-efficient but less stable**

Experience replay allows DQN to reuse every transition multiple times, which is why it solves the task in fewer episodes than REINFORCE. However, the ε-greedy exploration schedule is less adaptive than entropy regularisation: if ε decays too fast, the agent can get stuck in a local optimum before discovering an optimal landing strategy.

**REINFORCE's main limitation**

High-variance gradient estimates require aggressive entropy regularisation to maintain exploration, which slows down convergence. The batch accumulation trick mitigates this but does not eliminate the fundamental Monte-Carlo variance problem.

---

## Repository Structure

```
rl-LunarLander/
├── run.py                   # Unified CLI: train / eval / play
├── requirements.txt
│
├── src/
│   ├── reinforce/
│   │   └── train.py         # REINFORCE + Baseline algorithm
│   ├── a2c/
│   │   └── train.py         # A2C + GAE algorithm
│   ├── dqn/
│   │   └── train.py         # DQN algorithm
│   └── utils/
│       ├── common.py        # Seeding, device, env helpers, plot
│       └── logging.py       # TeeLogger, setup_logging
│
├── configs/
│   ├── reinforce.yaml
│   ├── a2c.yaml
│   └── dqn.yaml
│
├── models/                  # Saved checkpoints (.pt / .pth)
├── experiments/
│   ├── logs/                # Training logs (timestamped .log)
│   └── metrics/             # DQN CSV metrics per run
├── assets/
│   ├── plots/               # Training curve PNGs
│   └── video/               # Evaluation recordings
└── reports/
    └── Report_RL.pdf
```

---

## How to Run

**Install dependencies**

```bash
pip install -r requirements.txt
```

**Train**

```bash
python run.py train reinforce
python run.py train a2c
python run.py train dqn
```

**Evaluate a saved checkpoint**

```bash
python run.py eval reinforce --checkpoint models/reinforce_best.pt
python run.py eval a2c       --checkpoint models/a2c_best.pt
python run.py eval dqn       --checkpoint models/dqn_best.pth --episodes 100
```

**Watch the agent play**

```bash
python run.py play a2c --checkpoint models/a2c_best.pt
```

All output (logs, checkpoints, plots) is organised automatically under `experiments/`, `models/`, and `assets/`.

---

## Future Directions

- **PPO** — adds a clipping objective to A2C for more stable large-batch updates
- **Double DQN / Dueling DQN** — reduce Q-value overestimation and improve policy extraction
- **Prioritised experience replay** — sample transitions proportional to their TD error
- **Hyperparameter sweep** — systematic Bayesian search over learning rates, network sizes, GAE λ

---

## References

- Sutton & Barto, *Reinforcement Learning: An Introduction*, 2nd ed.
- Mnih et al., *Human-level control through deep reinforcement learning*, Nature 2015
- Schulman et al., *High-Dimensional Continuous Control Using Generalized Advantage Estimation*, ICLR 2016
- Mnih et al., *Asynchronous Methods for Deep Reinforcement Learning*, ICML 2016
