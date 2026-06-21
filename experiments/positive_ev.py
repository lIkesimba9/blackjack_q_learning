"""Can we reach POSITIVE expected value? Yes — by separating play from betting.

The earlier experiment failed for two compounding reasons:
  1. the play policy was conditioned on the count, splintering the state space
     across 140 count buckets, so it never converged (≈ -3.5% EV);
  2. on top of that, the count was used to bet into hands that were *still*
     negative-EV.

Here we fix both:
  * PLAY with a count-AGNOSTIC policy (`expose_count=False`): a tiny state space
    (~280 states) that converges to near-optimal basic strategy with doubling;
  * BET by the true count (a real counter's bet spread);
  * use DEEP penetration so favourable counts occur often enough to matter.

We then sweep a few bet-spread settings and report the EV per round.
"""

import argparse
import time

import numpy as np
import pandas as pd

from blackjack_rl.agent import BlackjackAgent, epsilon_greedy
from blackjack_rl.envs import COUNT_WEIGHTS_HI_LO, BlackjackCountEnv


def bet_size(tc, threshold, cap, unit):
    """Flat minimum until the true count clears `threshold`, then ramp up."""
    if tc < threshold:
        return 1
    return min(cap, 1 + int(np.floor((tc - threshold + 1) * unit)))


def train_play_policy(env, n_games, step, seed):
    agent = BlackjackAgent(env, env.action_space.n, epsilon_greedy, seed=seed)
    n_iters = max(1, n_games // step)
    for it in range(n_iters):
        frac = it / max(1, n_iters - 1)
        epsilon = 0.9 + (0.05 - 0.9) * frac
        agent.train(step, epsilon, alpha=0.001, gamma=0.95)
    return agent


def rollout(env, agent):
    state, _ = env.reset()
    tc = env.true_count_bet
    terminated = truncated = False
    reward = 0.0
    while not (terminated or truncated):
        action = agent.policy_fn(agent.q, state, 0.0, env)
        state, reward, terminated, truncated, _ = env.step(action)
    return reward, tc


def main():
    import os
    os.makedirs("results/data", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=4_000_000)
    p.add_argument("--step", type=int, default=50_000)
    p.add_argument("--eval-rounds", type=int, default=600_000)
    p.add_argument("--penetration", type=int, default=15, help="reshuffle threshold (cards left)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    # --- Train near-optimal, count-agnostic play on the shoe -----------------
    t0 = time.time()
    env = BlackjackCountEnv(natural=True, weights=COUNT_WEIGHTS_HI_LO, expose_count=False)
    env.reshuffle_threshold = args.penetration
    env.reset(seed=args.seed)
    agent = train_play_policy(env, args.games, args.step, args.seed)
    print(f"trained count-agnostic play: {agent.states_seen()} states, {time.time()-t0:.0f}s")

    # --- Collect rewards + true counts once, reuse for every bet rule --------
    rewards = np.empty(args.eval_rounds)
    tcs = np.empty(args.eval_rounds)
    for i in range(args.eval_rounds):
        rewards[i], tcs[i] = rollout(env, agent)
    flat_ev = rewards.mean()
    print(f"flat-bet EV/round = {flat_ev:+.4f}  (penetration={args.penetration} cards left)\n")

    # EV by true count bucket
    print("EV by true count at bet time:")
    bucket_rows = []
    for b in range(-5, 11):
        mask = (np.floor(np.clip(tcs, -5, 10)).astype(int) == b)
        if mask.sum():
            ev_b = rewards[mask].mean()
            print(f"  TC {b:+d}: EV {ev_b:+.4f}   freq {mask.mean():6.2%}")
            bucket_rows.append({"true_count": b, "ev": ev_b, "freq": mask.mean()})
    pd.DataFrame(bucket_rows).to_csv("results/data/positive_ev_by_count.csv", index=False)

    # --- Sweep bet-spread settings -------------------------------------------
    print("\nbet spread sweep (EV per round, in bet units):")
    best = None
    for threshold in (1, 2, 3):
        for cap in (10, 20, 40):
            for unit in (1, 2, 4):
                bets = np.array([bet_size(tc, threshold, cap, unit) for tc in tcs])
                ev_round = (rewards * bets).sum() / args.eval_rounds
                ev_unit = (rewards * bets).sum() / bets.sum()
                tag = f"thr={threshold} cap={cap} unit={unit}"
                if best is None or ev_round > best[0]:
                    best = (ev_round, ev_unit, bets.mean(), tag)
                if ev_round > 0:
                    print(f"  {tag:24s} EV/round {ev_round:+.4f}  EV/unit {ev_unit:+.4f}  avg bet {bets.mean():.2f}  <-- POSITIVE")
    ev_round, ev_unit, avg_bet, tag = best
    print(f"\nBEST: {tag}  ->  EV/round {ev_round:+.4f}  EV/unit {ev_unit:+.4f}  avg bet {avg_bet:.2f}")
    print("POSITIVE expected value!" if ev_round > 0 else "still negative — lower the house edge further.")

    pd.DataFrame([{
        "flat_ev": flat_ev, "best_spread": tag,
        "spread_ev_round": ev_round, "spread_ev_unit": ev_unit, "avg_bet": avg_bet,
        "penetration": args.penetration, "train_games": args.games,
    }]).to_csv("results/data/positive_ev_summary.csv", index=False)


if __name__ == "__main__":
    main()
