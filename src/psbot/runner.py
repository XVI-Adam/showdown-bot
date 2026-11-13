"""Run a batch of battles between two players and summarise the outcome."""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass
from typing import Iterable

from poke_env.battle import AbstractBattle
from poke_env.player import Player

logger = logging.getLogger(__name__)


class BattleRunError(RuntimeError):
    """Raised when a batch of battles does not complete."""


@dataclass(frozen=True)
class BattleSummary:
    """Results of a batch of battles, from the first player's point of view."""

    player: str
    opponent: str
    n_battles: int
    wins: int
    losses: int
    ties: int
    avg_turns: float
    avg_remaining_pokemon: float

    @property
    def win_rate(self) -> float:
        return self.wins / self.n_battles if self.n_battles else 0.0

    def __str__(self) -> str:
        return (
            f"{self.player} vs {self.opponent}: {self.wins}W-{self.losses}L-{self.ties}T "
            f"over {self.n_battles} battles ({self.win_rate:.0%} win rate), "
            f"avg {self.avg_turns:.1f} turns, "
            f"avg {self.avg_remaining_pokemon:.2f} Pokémon remaining"
        )


def wilson_interval(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval for a win rate (Wilson score interval).

    Unlike the naive ``p +/- 1.96*sqrt(p(1-p)/n)``, it behaves at 0% and 100%
    and with small samples, which is exactly where win rates get over-read.
    """
    if n == 0:
        return 0.0, 1.0
    p = wins / n
    denominator = 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (centre - half) / denominator), min(1.0, (centre + half) / denominator)


def remaining_pokemon(battle: AbstractBattle) -> int:
    """Number of the player's own Pokémon that have not fainted."""
    return sum(1 for mon in battle.team.values() if not mon.fainted)


def summarise(player: Player, opponent: Player, battles: Iterable[AbstractBattle]) -> BattleSummary:
    """Aggregate finished battles into a :class:`BattleSummary`."""
    finished = [b for b in battles if b.finished]
    n = len(finished)
    wins = sum(1 for b in finished if b.won is True)
    losses = sum(1 for b in finished if b.won is False)
    return BattleSummary(
        player=player.username,
        opponent=opponent.username,
        n_battles=n,
        wins=wins,
        losses=losses,
        ties=n - wins - losses,
        avg_turns=sum(b.turn for b in finished) / n if n else 0.0,
        avg_remaining_pokemon=sum(remaining_pokemon(b) for b in finished) / n if n else 0.0,
    )


async def run_battles(
    player: Player,
    opponent: Player,
    n_battles: int = 1,
    timeout_per_battle: float = 120.0,
) -> BattleSummary:
    """Play ``n_battles`` between ``player`` and ``opponent`` and summarise them.

    Both players are disconnected from the server afterwards, so construct
    fresh players for each call.

    :raises BattleRunError: if the batch times out or finishes short.
    """
    if n_battles < 1:
        raise ValueError("n_battles must be >= 1")
    try:
        await asyncio.wait_for(
            player.battle_against(opponent, n_battles=n_battles),
            timeout=timeout_per_battle * n_battles,
        )
    except asyncio.TimeoutError as exc:
        raise BattleRunError(
            f"battles did not finish within {timeout_per_battle * n_battles:.0f}s "
            f"({player.n_finished_battles}/{n_battles} completed)"
        ) from exc
    finally:
        for p in (player, opponent):
            try:
                await p.ps_client.stop_listening()
            except Exception:  # teardown must not mask the real error
                logger.warning("failed to disconnect %s", p.username, exc_info=True)

    summary = summarise(player, opponent, player.battles.values())
    if summary.n_battles != n_battles:
        raise BattleRunError(f"expected {n_battles} finished battles, got {summary.n_battles}")
    return summary
