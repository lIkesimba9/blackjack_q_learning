# Beating the Casino: Blackjack and Q-learning

Can you train an agent to beat the casino at Blackjack? In this post I train a
tabular **Q-learning** agent and progressively loosen the rules — adding doubling
down, card counting and splitting — to see which tricks actually help. Spoiler:
the result was counter-intuitive. Giving the agent *more power* (card counting,
splitting) at first didn't move it closer to winning — it made play **worse**. But
by the end I rework the environment so the expected value *does* turn positive —
and we'll see exactly why that works.

All the code lives in the repo, split into three pieces:

- `blackjack_agent.py` — an environment-agnostic Q-learning agent;
- `blackjack_envs.py` — custom Gymnasium environments (double, card counting, split);
- `train.py` — trains and compares every variant.

> This is a reworked version of my first article. Along the way I found a few
> nasty bugs in the original code (bootstrapping at terminal states, an
> upside-down epsilon schedule, swapped action constants). There's a dedicated
> section on them below, because they're the kind of mistake that's easy to make
> in *any* RL project.

## The rules, briefly

Number cards are worth their face value (2–10), face cards 10, and an ace is 1
or 11 — whichever is better without busting. You play against the dealer; the
goal is a hand closer to 21 than the dealer's without going over. The two basic
actions are:

- **hit** — take another card;
- **stick** — stop.

The dealer follows a fixed rule: keep drawing while the hand is below 17. Going
over 21 (a bust) is an instant loss. A "natural" Blackjack (an ace + a ten on
the first two cards) pays out 1.5.

Later we'll add two more actions — **double** (double the bet, take exactly one
card, then stop) and **split** (split a pair into two independent hands) — and
let the agent *see the card count*.

## Q-learning in two minutes

Q-learning maintains a quality estimate for every (state, action) pair — the
function `Q(s, a)`, the expected total reward of taking action `a` in state `s`
and playing optimally afterwards. For each transition `(s, a, r, s')` the
estimate is nudged toward the temporal-difference target:

$$Q(s,a) \leftarrow Q(s,a) + \alpha\,[\,r + \gamma \max_{a'} Q(s',a') - Q(s,a)\,]$$

where `α` is the learning rate, `γ` the discount factor and `r` the reward.

A **state** in basic Blackjack is a triple: the player's sum, the dealer's
up-card, and whether the player holds a usable ace. The **actions** are
hit / stick. Reward arrives at the end of the hand: +1 for a win, −1 for a loss,
+1.5 for a natural Blackjack.

### The agent

The agent deliberately knows nothing about Blackjack — it works with any
environment that follows the Gymnasium `reset`/`step` API. The Q-table is a
`defaultdict`, so a state only costs memory once it's actually visited. That's
essential for the split environment, whose Cartesian state space is ~10⁷ entries
but whose *reachable* set is a tiny fraction of that.

```python
from collections import defaultdict
import numpy as np


class BlackjackAgent:
    def __init__(self, env, n_actions, policy_fn, seed=None):
        self.env = env
        self.n_actions = n_actions
        self.policy_fn = policy_fn
        self.q = defaultdict(lambda: np.zeros(n_actions))
        self.rng = np.random.default_rng(seed)

    def train(self, n_episodes, epsilon, alpha, gamma):
        for _ in range(n_episodes):
            state, _ = self.env.reset()
            terminated = truncated = False
            while not (terminated or truncated):
                action = self.policy_fn(self.q, state, epsilon, self.env)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                # No future return at a terminal state — don't bootstrap.
                future = 0.0 if (terminated or truncated) else np.max(self.q[next_state])
                td_target = reward + gamma * future
                self.q[state][action] += alpha * (td_target - self.q[state][action])
                state = next_state
```

## Three bugs that are easy to miss

Before the experiments, let's look at what was broken in the first version. All
three are *silent*: the code runs, something even learns, but the results are
biased.

### 1. Bootstrapping at terminal states

The classic one. The Q-learning target contains the term `γ · max Q(s', a')`.
But if `s'` is terminal (the hand is over), there is no future return, and the
target should be just `r`. The original code added `max Q(s')` **always**, even
after the game ended. Since the same triple (a sum of 20, say) can appear both
as a mid-hand state and as a terminal one, spurious value leaked into the
estimates through terminal observations.

The fix is one line: zero out `future` when the episode has ended (see the agent
code above).

### 2. The upside-down epsilon schedule

In an epsilon-greedy policy, `epsilon` is the probability of *exploring* (a
random move). You normally **decay** it over training: explore early, exploit
what you've learned later. The first version had two bugs at once:

- epsilon was **increased** (`epsilon + 0.05`), so by the end the agent moved
  almost randomly;
- the update condition `if i % 100_000 == 0` rarely fired, because the counter
  `i` advanced in steps of 35,000 and only occasionally landed on a multiple of
  100,000.

The reworked version decays epsilon linearly from 0.9 to 0.05:

```python
frac = it / max(1, n_iters - 1)
epsilon = eps_start + (eps_end - eps_start) * frac  # 0.9 -> 0.05
```

### 3. Swapped action constants

Gymnasium's stock `Blackjack-v1` uses `1 = hit`, `0 = stick`. The code, however,
declared `HIT = 0` and `STAND = 1` — exactly backwards. And in the custom
environments action `0` meant hit again. So a single project carried two
incompatible conventions, and the "basic strategy" didn't do what it read like.
The reworked version states the Gymnasium convention explicitly:
`HIT, STICK = 1, 0`.

Plus the small stuff: seeding was added for reproducibility, and the typos were
cleaned up (`mathing`, `deler`, `CoumputeReward`, `avilable`…).

## A baseline

To have something to compare the trained agent against, take a simple
non-strategy: keep hitting while the hand is below 17, then stop — essentially
copying the dealer.

```python
def threshold_strategy_reward(env, threshold=17):
    obs, _ = env.reset()
    player = obs[0]
    terminated = truncated = False
    while player < threshold and not (terminated or truncated):
        obs, reward, terminated, truncated, _ = env.step(HIT)
        player = obs[0]
    if not (terminated or truncated):
        obs, reward, terminated, truncated, _ = env.step(STICK)
    return reward
```

## The experiments

For each variant we train the agent on 3.5M hands, periodically measuring the
average reward of a fully greedy policy. The metric is the **average reward per
hand**: anything below zero is a losing game (the house wins), anything above is
a player edge.

### 1. Stock environment (hit / stick)

```python
env = gym.make('Blackjack-v1', natural=True)
agent = BlackjackAgent(env, env.action_space.n, epsilon_greedy, seed=SEED)
rewards_default = fit_agent(agent, env)
```

In a fair game with no extra tricks, Blackjack has negative expected value. The
best the agent can do is approach the baseline — it can't turn a profit.

### 2. Doubling down

We add a third action. `BlackjackDoubleEnv` widens the action space to three:
hit / stick / double. Double takes one card, doubles the stake (and the reward),
and ends the hand.

```python
env = BlackjackDoubleEnv(natural=True)
agent = BlackjackAgent(env, env.action_space.n, epsilon_greedy, seed=SEED)
rewards_double = fit_agent(agent, env)
```

### 3. Card counting

In `BlackjackCountEnv` the deck is a 6-deck *shoe* dealt **without replacement**,
and the running count is added to the state. Now the agent can change its
decisions based on how rich the remaining deck is in tens.

> An important caveat: these environments don't let you vary the bet size. Yet a
> variable bet (bet big when the deck is rich) is the *main* profit mechanism for
> a real card counter. Here the count can only influence the hit / stick /
> double / split choice. Spoiler: this is one reason counting doesn't buy the
> edge you'd expect.

I compare two counting systems:

- **fractional** — the author's weights with fractional values;
- **Hi-Lo** ("plus-minus") — the classic: low cards +1, neutral 0, tens and aces −1.

```python
from blackjack_rl.envs import COUNT_WEIGHTS_HI_LO

env = BlackjackCountEnv(natural=True, weights=COUNT_WEIGHTS_HI_LO)
agent = BlackjackAgent(env, env.action_space.n, epsilon_greedy, seed=SEED)
rewards_count_hi_lo = fit_agent(agent, env)
```

### 4. Splitting

The final variant — `BlackjackSplitEnv` — adds splitting a pair into two
independent hands. The action space grows to seven (hit / stick / double for each
hand, plus split), and not all of them are legal in every state. So we use a
**masked** epsilon-greedy policy that only picks from the available actions:

```python
def epsilon_greedy_masked(q, state, epsilon, env):
    available = env.get_available_actions()
    if env.np_random.random() < epsilon:
        return int(env.np_random.choice(available))
    values = np.asarray(q[state])[available]
    return int(available[int(np.argmax(values))])
```

## Results

Average reward per hand after 3.5M games (mean of the last 10 evaluations;
epsilon decayed 0.9 → 0.05; seed 42):

| Variant | Q-table states | Final | Best eval |
|---|---:|---:|---:|
| Baseline (mimic dealer) | — | −0.086 | −0.086 |
| hit / stick | 280 | −0.046 | −0.026 |
| **+ double** | 280 | **−0.011** | **+0.008** |
| + count (Hi-Lo) | 15,293 | −0.053 | −0.025 |
| + count (fractional) | 30,310 | −0.072 | −0.037 |
| + count + split (Hi-Lo) | 92,310 | −0.077 | −0.048 |
| + count + split (fractional) | 131,224 | −0.082 | −0.059 |

![Learning curves](../results/figures/rewards_plot.png)

The outcome was different from my first article — and far more instructive.

**1. Nobody beat the house.** Every final average is negative. That's the correct
answer: negative expectation is built into the rules. The closest to zero is the
doubling agent (−0.011), which occasionally peeks into positive territory
(+0.008) — roughly break-even.

**2. The *simplest* learned agent wins.** Adding **doubling** gave the biggest
improvement: one new action that barely grows the state space (still 280 states)
and noticeably more flexible play.

**3. Card counting and splitting made things worse.** This is the interesting
bit. The more "power" we handed the agent, the further it slid back toward the
baseline. The cause is the **curse of dimensionality**. Counting inflates the
state space to tens of thousands of states; splitting to hundreds of thousands.
With a fixed budget of 3.5M hands, each state gets far too few visits (about 30
per state for the split variant), and tabular Q-learning simply never converges.
You can see it on the plot: the complex variants start to **degrade** late in
training — as epsilon decays, the agent commits to a greedy policy over a badly
under-estimated table.

**4. Hi-Lo consistently beats the fractional scheme.** The integer count produces
fewer distinct count-states, so it converges better: −0.053 vs −0.072 for
counting, and −0.077 vs −0.082 with splitting.

And the big caveat again: in all of the variants above the **bet size is fixed**,
yet a variable bet keyed on the count is the primary way a real counter makes
money. The obvious question: what happens if we add it? That's the next section —
and the answer is surprising.

## Bonus experiment: betting by the count

Since the main lever — a variable bet — was missing, let's add it. The mechanics
mirror a real counter: the agent still *plays* on a unit bet, but at the start of
every hand (before any card is dealt) the stake is scaled by the **true count**
(running count / decks remaining). The spread is linear, 1 to 12: the richer the
deck, the larger the bet.

```python
def bet_size(true_count, cap=12):
    if true_count < 1:
        return 1                       # unfavourable deck -> minimum bet
    return int(min(cap, np.floor(true_count)))
```

I trained the Hi-Lo counting variants for 8M hands and measured the mean reward
per round with a flat bet and with the count-based spread:

| Variant | EV/round, flat | EV/round, spread | avg bet |
|---|---:|---:|---:|
| count (Hi-Lo) | −0.035 | −0.053 | 1.57 |
| count + split (Hi-Lo) | −0.060 | −0.089 | 1.56 |

Betting by the count made things **worse**. The reason is clear from the EV as a
function of the true count at bet time:

![EV by true count](../results/figures/ev_by_count.png)

EV really does climb with the count: at a poor deck (count −5) you lose ~8% a
hand; at a neutral deck (count ≈ 0) about −2%. But the curve **plateaus below zero
and never crosses it at any count**. There is no favourable count to bet into — and
since every situation is negative-EV, raising the stake on the "less bad" hands
just multiplies the loss.

The root cause: this simplified game has a **house edge of ~2–4%**, whereas real
Blackjack is ~0.5%. In a real casino a count of +1…+2 already flips EV positive —
that's where the big bet goes. Here even a perfect count can't drag a hand to
break-even. On top of that, the most profitable (high) counts are the rarest
(count +8 occurs in under 1% of hands), so they're also the least-trained.

The conclusion is paradoxical but honest: **betting by the count is the right
tool, but it only works when there's a positive edge to bet into. Here there
isn't — so the spread only amplifies the loss.**

## How to actually reach positive EV

The failure of the bet spread was the clue: the idea was fine, the implementation
wasn't. I reworked the environment and the approach on three points — and the
expected value turned positive.

**1. Separate the play policy from the bet policy.** The key mistake last time:
the count lived in the *play* state, splintering the Q-table across 140 count
buckets and killing convergence. But the count barely changes *how* to play (a
few index plays) — it changes *how much to bet*. So I added an `expose_count=False`
flag: the count is still tracked for betting, but kept out of the observation. Now
the play policy lives in a tiny state space (~280 states) and converges to
near-optimal basic strategy. That's exactly a real counter: play basic strategy,
vary only the bet.

**2. Deeper penetration.** Reshuffle at 13 cards left instead of 30 — favourable
high counts occur far more often.

**3. Bet by the true count.** Minimum on a neutral/poor deck, a multiple of that
on a rich one.

The result: the EV curve lifts ~2–3% and **crosses zero**:

![EV before and after](../results/figures/positive_ev_plot.png)

On a flat bet the game is now near break-even (−0.011), and crucially EV is
*positive* from a true count of +3 up. There's finally an edge to bet into. With
a count-based bet, total expectation goes positive:

| Bet spread | EV/round | EV per unit wagered | avg bet |
|---|---:|---:|---:|
| flat bet | −0.011 | −0.011 | 1.0 |
| moderate (1–20, from count +2) | +0.011 | **+0.0045** | 2.4 |
| aggressive (1–40) | +0.034 | +0.0066 | 5.2 |

What matters: the edge is positive not just *per round* (which you could dismiss
as variance from big bets) but **per unit wagered** (+0.4…0.7%). That's a genuine
player edge — the same order of magnitude a real card counter achieves.

### What it looks like over the long run

Simulating the bankroll over 200,000 rounds — 6 independent runs, thick line is
the mean:

![Bankroll: counter vs flat bet](../results/figures/bankroll_plot.png)

The flat bettor steadily sinks (≈ −2000 units), while the counter climbs on
average (≈ +1000). There it is — the casino, beaten. But look at the spread of the
green trajectories: the edge is real, yet the **variance is enormous** — over a
short stretch you can still lose (here 4 of 6 runs finish positive). This is
exactly why a real counter needs a large bankroll and thousands of hands: the edge
exists, but it only materialises over the long run.

## Takeaways

The story came in two acts. First: you can't beat the casino head-on — the base
game is negative-EV, and a naive bet-by-count makes it *worse* (the play policy
won't converge and there's no favourable count to bet into). But once you rework
the approach — separate play from betting, let play converge, deepen penetration —
the expected value turns positive, just like a real counter's.

The lessons:

- **More state ≠ better.** Counting and splitting in the *play* state inflate a
  Q-table you can't fill — they only pay off with orders of magnitude more games
  or Q-function approximation.
- **Put information where it belongs.** The very same count is useless in the play
  table but decisive in the bet decision. You beat the house with information
  applied at the right point, not with brute "power" (more actions).
- **In RL, details decide everything.** One forgotten terminal-state term or an
  upside-down epsilon schedule will quietly spoil your results without ever
  crashing the code — most likely the very bugs that created the illusion of a
  "profitable" strategy in the first version.

The full code is in the repo. *If you found this useful, follow for more.*
