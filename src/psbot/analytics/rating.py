"""A team-adjusted rating for each opponent, and a tier list.

Everyone plays the same bot, so the bot is the yardstick. For each battle we
know (from ``team_strength``) the opponent's calibrated chance of winning with
the teams they were dealt, if both sides played like the bot. A player's skill
``s`` shifts that chance on the log-odds scale:

    P(player wins battle i) = sigmoid(s + logit(team chance_i))

``s = 0`` means "plays like the bot". Winning with a weak team says more than
winning with a strong one. ``s`` gets a normal prior centred on 0 (standard
deviation ``PRIOR_SD``), which keeps a player with two lucky games from being
crowned: few games give a rating near the bot's and a wide interval.

Reported three ways:

* **rating**: Elo-style, the bot fixed at 1500 (400 points = 10x the odds)
* **power level**: the chance to beat the bot when teams are even, 0-100
* **tier**: from the rating, S to D; *provisional* below 5 games

A battle without a team-strength measurement counts as an even draw.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

BOT_RATING = 1500
ELO_PER_LOGIT = 400 / math.log(10)  # about 174 rating points per unit of log-odds
PRIOR_SD = 1.0
PROVISIONAL_BELOW = 5

# (tier, lowest rating difference from the bot, plain description)
TIERS = (
    ("S", 200, "much stronger than the bot"),
    ("A", 75, "stronger than the bot"),
    ("B", -75, "about level with the bot"),
    ("C", -200, "weaker than the bot"),
    ("D", -10_000, "much weaker than the bot"),
)


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def _logit(p: float) -> float:
    p = min(max(p, 0.02), 0.98)
    return math.log(p / (1 - p))


@dataclass(frozen=True)
class Rating:
    battles: int
    wins: int
    skill: float  # log-odds versus the bot, team luck removed
    skill_sd: float
    measured: int  # battles with a team-strength measurement

    @property
    def rating(self) -> int:
        return round(BOT_RATING + self.skill * ELO_PER_LOGIT)

    @property
    def interval(self) -> tuple[int, int]:
        half = 1.96 * self.skill_sd * ELO_PER_LOGIT
        return round(self.rating - half), round(self.rating + half)

    @property
    def power_level(self) -> int:
        """Chance (0-100) of beating the bot with teams of equal strength."""
        return round(100 * _sigmoid(self.skill))

    @property
    def tier(self) -> str:
        return tier_of(self.rating)

    @property
    def provisional(self) -> bool:
        return self.battles < PROVISIONAL_BELOW


def tier_of(rating: float) -> str:
    difference = rating - BOT_RATING
    return next(name for name, low, _ in TIERS if difference >= low)


def rate(results: Sequence[tuple[bool, float | None]], prior_sd: float = PRIOR_SD) -> Rating:
    """``results``: (player won, player's calibrated team chance or None)."""
    offsets = [_logit(chance) if chance is not None else 0.0 for _, chance in results]
    wins = [1.0 if won else 0.0 for won, _ in results]
    s = 0.0
    for _ in range(100):  # Newton's method on the log-posterior
        probs = [_sigmoid(s + o) for o in offsets]
        gradient = sum(w - p for w, p in zip(wins, probs)) - s / prior_sd**2
        curvature = sum(p * (1 - p) for p in probs) + 1 / prior_sd**2
        step = gradient / curvature
        s += step
        if abs(step) < 1e-10:
            break
    probs = [_sigmoid(s + o) for o in offsets]
    curvature = sum(p * (1 - p) for p in probs) + 1 / prior_sd**2
    return Rating(
        battles=len(results),
        wins=int(sum(wins)),
        skill=s,
        skill_sd=1 / math.sqrt(curvature),
        measured=sum(chance is not None for _, chance in results),
    )
