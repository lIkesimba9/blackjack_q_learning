# Blackjack & Q-learning

A tabular **Q-learning** agent that learns to play Blackjack from scratch across
rule variants of growing realism — the stock game, *double down*, *card counting*
(6-deck shoe) and pair *splitting* — and a final setup that reaches a **positive
expected value** by separating play from betting.

![Beating the casino](results/figures/bankroll_plot.png)

## Project layout

```
blackjack_q_learning/
├── pyproject.toml          # project metadata + pinned dependencies (uv)
├── uv.lock                 # fully resolved lockfile for reproducibility
├── requirements.txt        # pip fallback (pinned)
├── src/blackjack_rl/       # the library (installed as a package)
│   ├── agent.py            # environment-agnostic tabular Q-learning agent
│   └── envs.py             # custom Gymnasium envs + counting schemes
├── experiments/            # runnable scripts (write to results/)
│   ├── train.py            # all six variants -> rewards.csv + comparison plot
│   ├── bet_sizing.py       # naive count-based bet spread (stays negative)
│   ├── positive_ev.py      # count-agnostic play + spread -> positive EV
│   └── beat_casino.py      # bankroll simulation -> bankroll_plot.png
├── notebooks/
│   └── blackjack.ipynb     # narrative walkthrough
├── articles/
│   ├── article_ru.md       # write-up (Russian)
│   └── article_en.md       # write-up (English / Medium)
└── results/
    ├── data/               # csv / npz outputs
    └── figures/            # png plots
```

## Setup

Reproducible install with [uv](https://docs.astral.sh/uv/) (recommended):

```bash
uv sync                  # creates .venv from uv.lock
uv sync --extra notebook # also install Jupyter for the notebook
```

Or with pip:

```bash
pip install -r requirements.txt && pip install -e .
```

## Run

Run from the repository root (outputs land in `results/`):

```bash
uv run python experiments/train.py            # full run (3.5M games per config)
uv run python experiments/train.py --games 300000   # quick smoke run

uv run python experiments/bet_sizing.py       # flat bet vs count-based spread
uv run python experiments/positive_ev.py      # separate play from betting -> positive EV
uv run python experiments/beat_casino.py      # bankroll trajectories
```

## Key finding

A fair game is negative-EV, and naively folding the count into the *play* state
hurts — the Q-table explodes and never converges. But separating **play** (a
small, count-agnostic, near-optimal basic strategy) from **betting** (scale the
stake by the true count) on a deeply-penetrated shoe yields a genuine positive
edge — positive even per unit wagered. See `articles/` for the full story.

[More in the Telegram channel](https://t.me/+9fl51jd750A3MTIy).
