"""Train and evaluate the Q-learning agent across every Blackjack variant.

Runs seven configurations (a fixed-threshold baseline plus six learned agents),
writes the learning curves to a CSV and renders a comparison plot.

Usage::

    uv run python train.py                 # full run
    uv run python train.py --games 300000  # quick smoke run
"""

import argparse
import time

import numpy as np
import pandas as pd

from blackjack_rl.agent import BlackjackAgent, epsilon_greedy, epsilon_greedy_masked
from blackjack_rl.envs import (
    COUNT_WEIGHTS_HI_LO,
    BlackjackCountEnv,
    BlackjackDoubleEnv,
    BlackjackSplitEnv,
)
import gymnasium as gym

HIT, STICK = 1, 0  # Gymnasium Blackjack-v1 convention: 1 = hit, 0 = stick.


def threshold_strategy_reward(env, threshold: int = 17) -> float:
    """Dealer-mimicking baseline: hit until the hand reaches ``threshold``."""
    obs, _ = env.reset()
    player = obs[0]
    terminated = truncated = False
    while player < threshold and not (terminated or truncated):
        obs, reward, terminated, truncated, _ = env.step(HIT)
        player = obs[0]
    if not (terminated or truncated):
        obs, reward, terminated, truncated, _ = env.step(STICK)
    return reward


def play_greedy(env, agent) -> float:
    """One greedy rollout (epsilon = 0) returning the episode reward."""
    state, _ = env.reset()
    terminated = truncated = False
    reward = 0.0
    while not (terminated or truncated):
        action = agent.policy_fn(agent.q, state, 0.0, env)
        state, reward, terminated, truncated, _ = env.step(action)
    return reward


def fit_agent(agent, env, n_games, step, eval_games, alpha, gamma, eps_start, eps_end):
    """Train in chunks of ``step`` episodes, evaluating greedily after each."""
    rewards = []
    n_iters = max(1, n_games // step)
    for it in range(n_iters):
        frac = it / max(1, n_iters - 1)
        epsilon = eps_start + (eps_end - eps_start) * frac  # linear decay
        agent.train(step, epsilon, alpha, gamma)
        rewards.append(np.mean([play_greedy(env, agent) for _ in range(eval_games)]))
    return rewards


def main():
    import os
    os.makedirs("results/data", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=3_500_000, help="training games per config")
    parser.add_argument("--step", type=int, default=35_000, help="games between evaluations")
    parser.add_argument("--eval-games", type=int, default=50_000, help="games per evaluation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-csv", default="results/data/rewards.csv")
    parser.add_argument("--out-plot", default="results/figures/rewards_plot.png")
    args = parser.parse_args()

    rng_seed = args.seed
    n_iters = max(1, args.games // args.step)

    def make_agent(env, policy):
        env.reset(seed=rng_seed)
        return BlackjackAgent(env, env.action_space.n, policy, seed=rng_seed)

    results = {}

    # 1. Fixed-threshold baseline (no learning) -> flat reference line.
    t0 = time.time()
    base_env = gym.make("Blackjack-v1", natural=True)
    base_env.reset(seed=rng_seed)
    baseline = np.mean([threshold_strategy_reward(base_env) for _ in range(args.eval_games)])
    results["threshold_baseline"] = [baseline] * n_iters
    print(f"[baseline] mean reward = {baseline:+.4f}  ({time.time() - t0:.1f}s)")

    # 2..7. Learned agents.
    configs = [
        ("default", gym.make("Blackjack-v1", natural=True), epsilon_greedy),
        ("double", BlackjackDoubleEnv(natural=True), epsilon_greedy),
        ("count_fractional", BlackjackCountEnv(natural=True), epsilon_greedy),
        ("count_hi_lo", BlackjackCountEnv(natural=True, weights=COUNT_WEIGHTS_HI_LO), epsilon_greedy),
        ("split_fractional", BlackjackSplitEnv(natural=True), epsilon_greedy_masked),
        ("split_hi_lo", BlackjackSplitEnv(natural=True, weights=COUNT_WEIGHTS_HI_LO), epsilon_greedy_masked),
    ]

    for name, env, policy in configs:
        t0 = time.time()
        agent = make_agent(env, policy)
        curve = fit_agent(
            agent, env,
            n_games=args.games, step=args.step, eval_games=args.eval_games,
            alpha=0.001, gamma=0.95, eps_start=0.9, eps_end=0.05,
        )
        results[name] = curve
        dt = time.time() - t0
        print(f"[{name}] final reward = {curve[-1]:+.4f}  states={agent.states_seen():,}  ({dt:.1f}s)")

    df = pd.DataFrame(results)
    df.to_csv(args.out_csv, index=False)
    print(f"saved {args.out_csv}")

    # Plot (rolling average for readability)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style("whitegrid")
    sns.set_palette("colorblind")

    labels = {
        "threshold_baseline": "baseline (mimic dealer)",
        "default": "hit/stick",
        "double": "+ double",
        "count_hi_lo": "+ count (Hi-Lo)",
        "count_fractional": "+ count (fractional)",
        "split_hi_lo": "+ count + split (Hi-Lo)",
        "split_fractional": "+ count + split (fractional)",
    }
    order = [c for c in labels if c in results]
    window = max(1, n_iters // 14)  # rolling window for readability
    x = np.arange(n_iters) * args.step
    plt.figure(figsize=(11, 6))
    for name in order:
        y = pd.Series(results[name]).rolling(window, min_periods=1, center=True).mean()
        plt.plot(x, y, label=labels[name], lw=2)
    plt.axhline(0, color="black", lw=1, ls="--", alpha=0.6)
    plt.xlabel("training games")
    plt.ylabel("mean reward per hand (rolling avg)")
    plt.title("Q-learning on Blackjack variants")
    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig(args.out_plot, dpi=150)
    print(f"saved {args.out_plot}")


if __name__ == "__main__":
    main()
