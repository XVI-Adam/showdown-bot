"""Unit tests for result aggregation, using stand-ins for poke-env objects."""

from types import SimpleNamespace

from psbot.runner import remaining_pokemon, summarise


def _battle(won: bool | None, turn: int, fainted: list[bool], finished: bool = True):
    team = {f"p1: mon{i}": SimpleNamespace(fainted=f) for i, f in enumerate(fainted)}
    return SimpleNamespace(won=won, turn=turn, team=team, finished=finished)


PLAYER = SimpleNamespace(username="A")
OPPONENT = SimpleNamespace(username="B")


def test_remaining_pokemon_counts_non_fainted() -> None:
    assert remaining_pokemon(_battle(True, 10, [False, True, False, True, True, True])) == 2


def test_summarise_counts_wins_losses_and_ties() -> None:
    battles = [
        _battle(True, 20, [False] * 3 + [True] * 3),
        _battle(False, 40, [True] * 6),
        _battle(None, 30, [False] + [True] * 5),  # tie: poke-env leaves `won` as None
    ]
    summary = summarise(PLAYER, OPPONENT, battles)
    assert (summary.wins, summary.losses, summary.ties) == (1, 1, 1)
    assert summary.n_battles == 3
    assert summary.avg_turns == 30
    assert summary.avg_remaining_pokemon == (3 + 0 + 1) / 3
    assert summary.win_rate == 1 / 3


def test_summarise_ignores_unfinished_battles() -> None:
    battles = [_battle(True, 10, [False] * 6), _battle(None, 3, [False] * 6, finished=False)]
    assert summarise(PLAYER, OPPONENT, battles).n_battles == 1


def test_summarise_handles_no_battles() -> None:
    summary = summarise(PLAYER, OPPONENT, [])
    assert summary.n_battles == 0
    assert summary.win_rate == 0.0
    assert summary.avg_turns == 0.0
