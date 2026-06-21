"""Tabular Q-learning agent for the Blackjack environments.

The agent is environment-agnostic: it only relies on the Gymnasium
``reset``/``step`` API and on a pluggable action-selection policy, so the very
same class trains on the stock ``Blackjack-v1`` and on every custom environment
defined in :mod:`blackjack_envs` (double / card-counting / split).
"""

from collections import defaultdict
from typing import Callable, Dict, Hashable, List

import numpy as np

# Type alias: a policy maps (Q-table, state, epsilon, env) -> action.
Policy = Callable[[dict, Hashable, float, object], int]


class BlackjackAgent:
    """Off-policy tabular Q-learning.

    Q-values are stored lazily in a ``defaultdict``: a state only consumes
    memory once it is actually visited. This matters a lot for the split
    environment, whose Cartesian state space is ~10^7 entries but whose
    *reachable* set is a tiny fraction of that.
    """

    def __init__(self, env, n_actions: int, policy_fn: Policy, seed: int | None = None):
        self.env = env
        self.n_actions = n_actions
        self.policy_fn = policy_fn
        self.q: Dict[Hashable, np.ndarray] = defaultdict(lambda: np.zeros(n_actions))
        # A dedicated RNG keeps exploration reproducible and independent from
        # the environment's own random stream.
        self.rng = np.random.default_rng(seed)

    def train(self, n_episodes: int, epsilon: float, alpha: float, gamma: float) -> None:
        """Run ``n_episodes`` of Q-learning with a fixed (epsilon, alpha, gamma)."""
        for _ in range(n_episodes):
            state, _ = self.env.reset()
            terminated = truncated = False
            while not (terminated or truncated):
                action = self.policy_fn(self.q, state, epsilon, self.env)
                next_state, reward, terminated, truncated, _ = self.env.step(action)

                # Bootstrapping must stop at terminal states: there is no
                # future return after the hand is over, so the TD target is
                # simply the reward. The original code always added
                # gamma * max(Q[next_state]), which leaked spurious value
                # through terminal observations.
                future = 0.0 if (terminated or truncated) else np.max(self.q[next_state])
                td_target = reward + gamma * future
                self.q[state][action] += alpha * (td_target - self.q[state][action])

                state = next_state

    def greedy_policy(self) -> Dict[Hashable, int]:
        """Return the learned deterministic policy (best action per state)."""
        return {state: int(np.argmax(values)) for state, values in self.q.items()}

    def states_seen(self) -> int:
        return len(self.q)


def epsilon_greedy(q: dict, state: Hashable, epsilon: float, env) -> int:
    """Standard epsilon-greedy over the full discrete action space.

    With probability ``epsilon`` pick a uniformly random action (explore),
    otherwise pick ``argmax_a Q(state, a)`` (exploit).
    """
    if env.np_random.random() < epsilon:
        return int(env.action_space.sample())
    return int(np.argmax(q[state]))


def epsilon_greedy_masked(q: dict, state: Hashable, epsilon: float, env) -> int:
    """Epsilon-greedy restricted to the actions that are legal in ``state``.

    Used by the split environment, where the set of legal actions depends on
    whether a hand can still be acted upon / split.
    """
    available: List[int] = env.get_available_actions()
    if env.np_random.random() < epsilon:
        return int(env.np_random.choice(available))
    values = np.asarray(q[state])[available]
    return int(available[int(np.argmax(values))])
