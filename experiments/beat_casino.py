"""Bankroll simulation: the case where you actually beat the casino.

Trains a count-agnostic near-optimal play policy, then simulates the bankroll
over many rounds for three players on the same deeply-penetrated 6-deck shoe:

  * house / flat player  — plays well but bets a flat unit  -> bankroll drifts down;
  * naive counter        — bets by count but on a poor (count-in-state) policy;
  * card counter         — basic strategy + count-based bet spread -> bankroll climbs.

Produces `bankroll_plot.png`.
"""

import argparse
import time

import numpy as np

from blackjack_rl.agent import BlackjackAgent, epsilon_greedy
from blackjack_rl.envs import COUNT_WEIGHTS_HI_LO, BlackjackCountEnv


def bet_spread(tc, threshold=2, cap=20, unit=2):
    if tc < threshold:
        return 1
    return min(cap, 1 + int(np.floor((tc - threshold + 1) * unit)))


def train_play(env, n_games, step, seed):
    agent = BlackjackAgent(env, env.action_space.n, epsilon_greedy, seed=seed)
    n_iters = max(1, n_games // step)
    for it in range(n_iters):
        frac = it / max(1, n_iters - 1)
        agent.train(step, 0.9 + (0.05 - 0.9) * frac, alpha=0.001, gamma=0.95)
    return agent


def simulate(env, agent, n_rounds, use_spread):
    """Return the cumulative bankroll trajectory over `n_rounds`."""
    bankroll = np.empty(n_rounds)
    total = 0.0
    for i in range(n_rounds):
        state, _ = env.reset()
        tc = env.true_count_bet
        bet = bet_spread(tc) if use_spread else 1
        terminated = truncated = False
        reward = 0.0
        while not (terminated or truncated):
            action = agent.policy_fn(agent.q, state, 0.0, env)
            state, reward, terminated, truncated, _ = env.step(action)
        total += reward * bet
        bankroll[i] = total
    return bankroll


def main():
    import os
    os.makedirs("results/data", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=3_000_000)
    p.add_argument("--step", type=int, default=50_000)
    p.add_argument("--rounds", type=int, default=200_000)
    p.add_argument("--trajectories", type=int, default=6)
    p.add_argument("--penetration", type=int, default=13)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    t0 = time.time()
    env = BlackjackCountEnv(natural=True, weights=COUNT_WEIGHTS_HI_LO, expose_count=False)
    env.reshuffle_threshold = args.penetration
    env.reset(seed=args.seed)
    agent = train_play(env, args.games, args.step, args.seed)
    print(f"trained play policy: {agent.states_seen()} states, {time.time()-t0:.0f}s")

    counter_runs, flat_runs = [], []
    for k in range(args.trajectories):
        env.reset(seed=1000 + k)
        counter_runs.append(simulate(env, agent, args.rounds, use_spread=True))
        env.reset(seed=1000 + k)
        flat_runs.append(simulate(env, agent, args.rounds, use_spread=False))
        print(f"  trajectory {k}: counter {counter_runs[-1][-1]:+.0f} u, flat {flat_runs[-1][-1]:+.0f} u")

    np.savez("results/data/bankroll_runs.npz", counter=np.array(counter_runs), flat=np.array(flat_runs))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style("whitegrid")

    x = np.arange(args.rounds)
    plt.figure(figsize=(11, 6))
    for run in counter_runs:
        plt.plot(x, run, color="tab:green", alpha=0.35, lw=1)
    for run in flat_runs:
        plt.plot(x, run, color="tab:red", alpha=0.35, lw=1)
    plt.plot(x, np.mean(counter_runs, axis=0), color="tab:green", lw=2.5,
             label="card counter (basic strategy + bet by count)")
    plt.plot(x, np.mean(flat_runs, axis=0), color="tab:red", lw=2.5,
             label="flat bet (same play, no bet spread)")
    plt.axhline(0, color="black", lw=1, ls="--")
    plt.xlabel("rounds played")
    plt.ylabel("cumulative profit (betting units)")
    plt.title("Beating the casino: bankroll with vs without betting by the count")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig("results/figures/bankroll_plot.png", dpi=150)
    print("saved bankroll_plot.png")


if __name__ == "__main__":
    main()
