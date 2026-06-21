"""Does betting by the count flip the expected value positive?

The agent still *plays* on a flat unit bet, but at the start of every hand the
stake is scaled by the true count (running count / decks remaining) — a classic
card-counter "bet spread". We then compare:

* flat bet              -> mean reward per round at 1 unit;
* count-based spread    -> mean reward per round (in units), where you bet more
                           only when the deck is favourable.

We also print EV bucketed by true count, which is *why* the spread can win:
favourable counts carry a positive edge, so wagering more there can outrun the
house edge paid on the (minimum-bet) unfavourable hands.
"""

import argparse
import time

import numpy as np
import pandas as pd

from blackjack_rl.agent import BlackjackAgent, epsilon_greedy, epsilon_greedy_masked
from blackjack_rl.envs import (
    COUNT_WEIGHTS_HI_LO,
    BlackjackCountEnv,
    BlackjackSplitEnv,
)


def bet_size(true_count: float, cap: int) -> int:
    """1-to-`cap` linear bet spread: bet ~ true count when it's favourable."""
    if true_count < 1:
        return 1
    return int(min(cap, np.floor(true_count)))


def train(agent, env, n_games, step, alpha, gamma, eps_start, eps_end):
    n_iters = max(1, n_games // step)
    for it in range(n_iters):
        frac = it / max(1, n_iters - 1)
        epsilon = eps_start + (eps_end - eps_start) * frac
        agent.train(step, epsilon, alpha, gamma)


def play_one(env, agent):
    """Play a single hand greedily; return (reward, true_count_at_bet)."""
    state, _ = env.reset()
    tc = env.true_count_bet
    terminated = truncated = False
    reward = 0.0
    while not (terminated or truncated):
        action = agent.policy_fn(agent.q, state, 0.0, env)
        state, reward, terminated, truncated, _ = env.step(action)
    return reward, tc


def evaluate(env, agent, n_rounds, cap):
    flat_sum = 0.0
    spread_sum = 0.0
    spread_wager = 0.0
    # EV bucketed by integer true count at bet time.
    buckets = {}
    for _ in range(n_rounds):
        reward, tc = play_one(env, agent)
        bet = bet_size(tc, cap)
        flat_sum += reward
        spread_sum += reward * bet
        spread_wager += bet
        b = int(np.clip(np.floor(tc), -5, 10))
        s, n = buckets.get(b, (0.0, 0))
        buckets[b] = (s + reward, n + 1)
    return {
        "flat_ev": flat_sum / n_rounds,                # per round, 1 unit
        "spread_ev_round": spread_sum / n_rounds,      # per round, in units
        "spread_ev_unit": spread_sum / spread_wager,   # per unit wagered
        "avg_bet": spread_wager / n_rounds,
        "buckets": buckets,
    }


def main():
    import os
    os.makedirs("results/data", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=6_000_000)
    p.add_argument("--step", type=int, default=50_000)
    p.add_argument("--eval-rounds", type=int, default=400_000)
    p.add_argument("--cap", type=int, default=10, help="max bet spread multiple")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    configs = [
        ("count_hi_lo", BlackjackCountEnv(natural=True, weights=COUNT_WEIGHTS_HI_LO), epsilon_greedy),
        ("split_hi_lo", BlackjackSplitEnv(natural=True, weights=COUNT_WEIGHTS_HI_LO), epsilon_greedy_masked),
    ]

    rows = []
    for name, env, policy in configs:
        t0 = time.time()
        env.reset(seed=args.seed)
        agent = BlackjackAgent(env, env.action_space.n, policy, seed=args.seed)
        train(agent, env, args.games, args.step, alpha=0.001, gamma=0.95,
              eps_start=0.9, eps_end=0.05)
        res = evaluate(env, agent, args.eval_rounds, args.cap)
        dt = time.time() - t0
        print(f"\n=== {name} (trained {args.games:,} games, {agent.states_seen():,} states, {dt:.0f}s) ===")
        print(f"  flat bet           EV/round = {res['flat_ev']:+.4f}")
        print(f"  count bet spread   EV/round = {res['spread_ev_round']:+.4f}  "
              f"(avg bet {res['avg_bet']:.2f}, EV/unit {res['spread_ev_unit']:+.4f})")
        print("  EV by true count at bet time:")
        for b in sorted(res["buckets"]):
            s, n = res["buckets"][b]
            print(f"    TC {b:+d}: EV {s/n:+.4f}   freq {n/args.eval_rounds:6.2%}")
        rows.append({
            "variant": name,
            "flat_ev_round": res["flat_ev"],
            "spread_ev_round": res["spread_ev_round"],
            "spread_ev_unit": res["spread_ev_unit"],
            "avg_bet": res["avg_bet"],
        })

    pd.DataFrame(rows).to_csv("results/data/bet_sizing_results.csv", index=False)
    print("\nsaved bet_sizing_results.csv")


if __name__ == "__main__":
    main()
