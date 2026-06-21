"""Custom Blackjack environments built on top of Gymnasium's ``Blackjack-v1``.

Three extensions over the stock environment, each adding one degree of realism:

* :class:`BlackjackDoubleEnv` ........ adds the *double down* action.
* :class:`BlackjackCountEnv` ......... adds *double* + a running card count
  (a 6-deck shoe is dealt without replacement and the count is exposed to the
  agent as part of the observation).
* :class:`BlackjackSplitEnv` ......... adds *double*, the card count and the
  *split* action (a pair can be split into two independently played hands).

All the card mechanics (``draw_card``, ``score``, ``is_bust`` …) are reused
from Gymnasium so the rules stay identical to the original game.
"""

from typing import List, Optional

import numpy as np
from gymnasium.envs.toy_text.blackjack import (
    BlackjackEnv,
    cmp,
    is_bust,
    is_natural,
    score,
    spaces,
    sum_hand,
    usable_ace,
)
from gymnasium.utils import seeding

# --------------------------------------------------------------------------- #
# Card-counting weight presets
# --------------------------------------------------------------------------- #
# A fine-grained, fractional counting scheme (the "half-deck" weights from the
# original experiments).
COUNT_WEIGHTS_FRACTIONAL = {
    1: -1.0, 2: 0.5, 3: 1.0, 4: 1.0, 5: 1.5,
    6: 1.0, 7: 0.5, 8: 0.0, 9: -0.5, 10: -1.0,
}

# The classic Hi-Lo ("plus-minus") system: low cards +1, neutral 0, tens/aces -1.
COUNT_WEIGHTS_HI_LO = {
    1: -1, 2: 1, 3: 1, 4: 1, 5: 1,
    6: 1, 7: 0, 8: 0, 9: 0, 10: -1,
}

# One standard 52-card deck encoded by value (four 10-valued cards per suit).
STANDARD_DECK = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10] * 4
N_DECKS = 6  # a 6-deck shoe, as used in most casinos
RESHUFFLE_THRESHOLD = 30  # reshuffle once fewer than this many cards remain

# Discretisation grid for the running count exposed in the observation.
_COUNT_GRID = np.arange(-35, 35, 0.5)


class _CountingMixin:
    """Shared shoe / card-counting machinery for the custom environments."""

    def _init_shoe(self, weights: Optional[dict]) -> None:
        self.weights = dict(weights) if weights is not None else dict(COUNT_WEIGHTS_FRACTIONAL)
        self.deck = STANDARD_DECK * N_DECKS
        self.cards_count = 0.0
        self._count_to_state = {value: i for i, value in enumerate(_COUNT_GRID)}
        # Reshuffle once fewer than this many cards remain. Lower = deeper
        # penetration = stronger counting signal (extreme counts occur more often).
        self.reshuffle_threshold = RESHUFFLE_THRESHOLD
        # Count snapshot taken at "bet time" (before the hand is dealt). This is
        # what a real card counter sees when deciding how much to wager.
        self.true_count_bet = 0.0

    def _capture_bet_context(self) -> None:
        """Record the true count just before dealing — i.e. when the bet is placed."""
        decks_left = max(0.5, len(self.deck) / 52.0)
        self.true_count_bet = self.cards_count / decks_left

    def draw_card(self, np_random) -> int:
        """Draw without replacement from the shoe (unlike the stock game)."""
        card = int(np_random.choice(self.deck))
        self.deck.remove(card)
        return card

    def draw_hand(self, np_random) -> List[int]:
        return [self.draw_card(np_random), self.draw_card(np_random)]

    def count(self, card: int) -> None:
        self.cards_count += self.weights[card]

    def _count_state(self) -> int:
        if self.cards_count in self._count_to_state:
            return self._count_to_state[self.cards_count]
        return 0 if self.cards_count < 0 else len(self._count_to_state) - 1

    def _maybe_reshuffle(self) -> None:
        if len(self.deck) < self.reshuffle_threshold:
            self.deck = STANDARD_DECK * N_DECKS
            self.cards_count = 0.0

    def _seed(self, seed) -> None:
        if seed is not None:
            self._np_random, _ = seeding.np_random(seed)


class BlackjackDoubleEnv(BlackjackEnv):
    """Stock Blackjack plus a *double down* action (0=hit, 1=stick, 2=double)."""

    def __init__(self, render_mode: Optional[str] = None, natural=False, sab=False):
        super().__init__(render_mode, natural, sab)
        self.action_space = spaces.Discrete(3)

    def _settle_against_dealer(self, multiplier: float) -> float:
        while sum_hand(self.dealer) < 17:
            self.dealer.append(self._deal())
        reward = cmp(score(self.player), score(self.dealer))
        if self.sab and is_natural(self.player) and not is_natural(self.dealer):
            return 1.0
        if not self.sab and self.natural and is_natural(self.player) and reward == 1.0:
            return 1.5
        return reward * multiplier

    def _deal(self) -> int:
        from gymnasium.envs.toy_text.blackjack import draw_card
        return draw_card(self.np_random)

    def step(self, action):
        assert self.action_space.contains(action)
        if action == 0:  # hit
            self.player.append(self._deal())
            if is_bust(self.player):
                return self._get_obs(), -1.0, True, False, {}
            return self._get_obs(), 0.0, False, False, {}
        if action == 1:  # stick
            return self._get_obs(), self._settle_against_dealer(1.0), True, False, {}
        # action == 2: double — one card, double stake, hand ends
        self.player.append(self._deal())
        if is_bust(self.player):
            return self._get_obs(), -2.0, True, False, {}
        return self._get_obs(), self._settle_against_dealer(2.0), True, False, {}


class BlackjackCountEnv(_CountingMixin, BlackjackEnv):
    """Double-down + a 6-deck shoe with the running count in the observation."""

    def __init__(self, render_mode: Optional[str] = None, natural=False, sab=False,
                 weights=None, expose_count=True):
        super().__init__(render_mode, natural, sab)
        self.action_space = spaces.Discrete(3)
        self._init_shoe(weights)
        # When False, the count is tracked (for bet sizing) but hidden from the
        # observation, so the *play* policy lives in a tiny, well-trainable state
        # space — a counter who plays basic strategy and only varies the bet.
        self.expose_count = expose_count
        obs = [spaces.Discrete(32), spaces.Discrete(11), spaces.Discrete(2)]
        if expose_count:
            obs.append(spaces.Discrete(len(_COUNT_GRID)))
        self.observation_space = spaces.Tuple(tuple(obs))

    def _settle_against_dealer(self, multiplier: float) -> float:
        while sum_hand(self.dealer) < 17:
            self.dealer.append(self.draw_card(self.np_random))
        # Reveal (and count) every dealer card except the up-card, which was
        # already counted when the hand was dealt.
        for card in self.dealer[1:]:
            self.count(card)
        reward = cmp(score(self.player), score(self.dealer))
        if not self.sab and self.natural and is_natural(self.player) and reward == 1.0:
            reward = 1.5
        return reward * multiplier

    def step(self, action):
        assert self.action_space.contains(action)
        if action == 0:  # hit
            card = self.draw_card(self.np_random)
            self.count(card)
            self.player.append(card)
            if is_bust(self.player):
                return self._get_obs(), -1.0, True, False, {}
            return self._get_obs(), 0.0, False, False, {}
        if action == 1:  # stick
            return self._get_obs(), self._settle_against_dealer(1.0), True, False, {}
        # action == 2: double
        card = self.draw_card(self.np_random)
        self.count(card)
        self.player.append(card)
        if is_bust(self.player):
            return self._get_obs(), -2.0, True, False, {}
        return self._get_obs(), self._settle_against_dealer(2.0), True, False, {}

    def _get_obs(self):
        base = (sum_hand(self.player), self.dealer[0], usable_ace(self.player))
        return base + (self._count_state(),) if self.expose_count else base

    def reset(self, seed=None, options=None):
        self._seed(seed)
        self._maybe_reshuffle()
        self._capture_bet_context()  # bet is placed before any card is dealt
        self.dealer = self.draw_hand(self.np_random)
        self.count(self.dealer[0])  # only the up-card is visible to the counter
        self.player = self.draw_hand(self.np_random)
        for card in self.player:
            self.count(card)
        return self._get_obs(), {}


class Hand:
    """A single player hand: tracks its cards, whether it is finished and the
    stake multiplier (2 after a double)."""

    def __init__(self, cards: Optional[List[int]] = None):
        self.cards: List[int] = cards if cards is not None else []
        self.multiplier = 1
        self.reward = 0.0
        # An empty placeholder hand counts as already "done" (nothing to play).
        self.done = not cards

    def add_card(self, card: int) -> None:
        if self.done:
            return
        self.cards.append(card)
        if is_bust(self.cards):
            self.reward = -1.0
            self.done = True

    def stick(self) -> None:
        self.done = True

    def double(self, card: int) -> None:
        if self.done:
            return
        self.cards.append(card)
        self.multiplier = 2
        self.done = True

    def split(self, first_card: int, second_card: int) -> "tuple[Hand, Hand]":
        return Hand([self.cards[0], first_card]), Hand([self.cards[1], second_card])

    def compute_reward(self, dealer_cards: List[int]) -> float:
        if not self.cards:
            return 0.0
        if self.reward == -1.0:  # already busted
            return self.reward
        self.reward = cmp(score(self.cards), score(dealer_cards)) * self.multiplier
        return self.reward


def can_split(first_hand: List[int], second_hand: List[int]) -> bool:
    return len(first_hand) == 2 and len(second_hand) == 0 and first_hand[0] == first_hand[1]


class BlackjackSplitEnv(_CountingMixin, BlackjackEnv):
    """Full game: double, card counting and pair *splitting*.

    Action layout (7 discrete actions)::

        0 stick first hand   1 hit first hand   2 double first hand   3 split
        4 stick second hand  5 hit second hand  6 double second hand

    Only the actions returned by :meth:`get_available_actions` are ever legal in
    a given state; the masked policy enforces this.
    """

    def __init__(self, render_mode: Optional[str] = None, natural=False, sab=False, weights=None):
        super().__init__(render_mode, natural, sab)
        self.action_space = spaces.Discrete(7)
        self._init_shoe(weights)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(32),                # first hand sum
            spaces.Discrete(11),                # dealer up-card
            spaces.Discrete(2),                 # first hand usable ace
            spaces.Discrete(len(_COUNT_GRID)),  # running count
            spaces.Discrete(32),                # second hand sum
            spaces.Discrete(2),                 # second hand usable ace
        ))

    def step(self, action):
        assert self.action_space.contains(action)
        if action == 0:
            self.first_hand.stick()
        elif action == 1:
            card = self.draw_card(self.np_random); self.count(card)
            self.first_hand.add_card(card)
        elif action == 2:
            card = self.draw_card(self.np_random); self.count(card)
            self.first_hand.double(card)
        elif action == 3:
            first_card = self.draw_card(self.np_random); self.count(first_card)
            second_card = self.draw_card(self.np_random); self.count(second_card)
            self.first_hand, self.second_hand = self.first_hand.split(first_card, second_card)
        elif action == 4:
            self.second_hand.stick()
        elif action == 5:
            card = self.draw_card(self.np_random); self.count(card)
            self.second_hand.add_card(card)
        elif action == 6:
            card = self.draw_card(self.np_random); self.count(card)
            self.second_hand.double(card)

        if self.first_hand.done and self.second_hand.done:
            while sum_hand(self.dealer) < 17:
                self.dealer.append(self.draw_card(self.np_random))
            for card in self.dealer[1:]:
                self.count(card)
            reward = (self.first_hand.compute_reward(self.dealer)
                      + self.second_hand.compute_reward(self.dealer))
            return self._get_obs(), reward, True, False, {}
        return self._get_obs(), 0.0, False, False, {}

    def get_available_actions(self) -> List[int]:
        if can_split(self.first_hand.cards, self.second_hand.cards):
            return [0, 1, 2, 3]
        actions: List[int] = []
        if not self.first_hand.done:
            actions += [0, 1, 2]
        if self.second_hand.cards and not self.second_hand.done:
            actions += [4, 5, 6]
        return actions

    def _get_obs(self):
        return (
            sum_hand(self.first_hand.cards),
            self.dealer[0],
            usable_ace(self.first_hand.cards),
            self._count_state(),
            sum_hand(self.second_hand.cards) if self.second_hand.cards else 0,
            usable_ace(self.second_hand.cards) if self.second_hand.cards else 0,
        )

    def reset(self, seed=None, options=None):
        self._seed(seed)
        self._maybe_reshuffle()
        self._capture_bet_context()  # bet is placed before any card is dealt
        self.dealer = self.draw_hand(self.np_random)
        self.count(self.dealer[0])
        cards = self.draw_hand(self.np_random)
        self.first_hand = Hand(cards=cards)
        self.second_hand = Hand()
        for card in cards:
            self.count(card)
        return self._get_obs(), {}
