"""Team-adjusted ratings, play-style measures and the player cards."""

from __future__ import annotations

import pytest

from psbot.analytics.cards import Game, Player, play_style, player_card, something_to_try, tier_list_page
from psbot.analytics.rating import BOT_RATING, rate, tier_of
from psbot.analytics.style import StyleTally, effectiveness, matchup, stab_threat


# ---------------------------------------------------------------- rating


def test_no_games_and_even_games_rate_like_the_bot() -> None:
    assert rate([]).rating == BOT_RATING and rate([]).power_level == 50
    even = rate([(True, 0.5), (False, 0.5)] * 5)
    assert even.rating == pytest.approx(BOT_RATING, abs=1)


def test_a_win_with_a_weak_team_counts_for_more() -> None:
    lucky = rate([(True, 0.85)] * 3)
    skilled = rate([(True, 0.15)] * 3)
    neutral = rate([(True, None)] * 3)
    assert skilled.rating > neutral.rating > lucky.rating > BOT_RATING


def test_few_games_stay_near_the_bot_with_wide_intervals() -> None:
    two, twenty = rate([(True, 0.5)] * 2), rate([(True, 0.5)] * 20)
    assert BOT_RATING < two.rating < twenty.rating
    width = lambda r: r.interval[1] - r.interval[0]
    assert width(two) > width(twenty)
    assert two.provisional and not twenty.provisional
    assert rate([(False, 0.5)] * 4).rating < BOT_RATING


def test_tiers_are_relative_to_the_bot() -> None:
    assert tier_of(BOT_RATING) == "B"
    assert tier_of(BOT_RATING + 250) == "S" and tier_of(BOT_RATING + 100) == "A"
    assert tier_of(BOT_RATING - 100) == "C" and tier_of(BOT_RATING - 300) == "D"


# ----------------------------------------------------------------- style


def test_matchups_from_types() -> None:
    assert effectiveness("earthquake", "skarmory") == 0
    assert effectiveness("thunderbolt", "gyarados") == 4
    assert effectiveness("swordsdance", "gyarados") is None
    assert stab_threat("raichu", "gyarados") == 4
    # Our Raichu threatens their Gyarados; Gyarados's types do not threaten Raichu.
    assert matchup("raichu", ["thunderbolt"], "gyarados") == "bad"
    assert matchup("gyarados", ["waterfall"], "raichu") == "good"
    assert matchup("snorlax", ["bodyslam"], "snorlax") == "even"


def record() -> dict:
    """Two decisions: their Gyarados is in a bad matchup and switches into our
    Thunderbolt with Garchomp (a read); then it sets up while unthreatened."""
    ours = {"ident": "p1: Raichu", "species": "raichu", "moves": ["thunderbolt", "nastyplot"], "fainted": False, "hp": 1.0}
    gyarados = {"ident": "p2: Gyarados", "species": "gyarados", "hp": 1.0, "fainted": False}
    chomp = {"ident": "p2: Garchomp", "species": "garchomp", "hp": 1.0, "fainted": False}
    normal = {"ident": "p1: Snorlax", "species": "snorlax", "moves": ["bodyslam"], "fainted": False, "hp": 1.0}
    return {
        "battle_id": "b1", "role": "p1", "outcome": {"won": False},
        "decisions": [
            {"active": ours, "opponent_active": gyarados, "selected": {"action": "move thunderbolt"},
             "opponent_actions": [{"type": "switch", "id": "garchomp"}]},
            {"active": normal, "opponent_active": chomp, "selected": {"action": "move bodyslam"},
             "opponent_actions": [{"type": "move", "id": "swordsdance", "called": False}]},
        ],
        "events": [["switch", "p1a: Raichu", "Raichu, L80", "100/100"], ["switch", "p2a: Gyarados", "Gyarados, L80", "100/100"],
                   ["turn", "1"], ["switch", "p2a: Garchomp", "Garchomp, L77", "100/100"], ["turn", "2"],
                   ["-terastallize", "p2a: Garchomp", "Steel"], ["move", "p2a: Garchomp", "Swords Dance", "p2a: Garchomp"]],
    }  # fmt: skip


def test_style_counts_choices_in_context() -> None:
    from psbot.analytics.style import tally_battle

    t = tally_battle(record())
    assert t.choices["bad"]["switch"] == 1 and t.total() == 2
    assert (t.reads, t.switches_into_attack) == (1, 1)  # Ground resists... is immune to Electric
    assert (t.setup, t.setup_unthreatened) == (1, 1)
    assert t.tera_battles == 1 and t.tera_turns == [2]
    assert t.wins == 1


# ----------------------------------------------------------------- cards


def player(name: str, results: list[tuple[bool, float | None]], style: StyleTally | None = None) -> Player:
    games = [Game(f"2026-11-{10 + i:02d}", won, chance) for i, (won, chance) in enumerate(results)]
    return Player(name, rate(results), games, style, bot_name="TestBot")


def test_card_states_counts_luck_and_early_read() -> None:
    from psbot.analytics.style import tally_battle

    alice = player("Alice", [(True, 0.3), (False, 0.6)], tally_battle(record()))
    bob = player("Bob <script>", [(True, 0.5)] * 6)
    html = player_card(alice, [alice, bob])
    assert "Early read" in html and "Alice" in html and "TestBot" in html
    assert "stronger one in 1 of 2 games" in html
    assert "weaker team on paper" in html  # a win with a 30% team
    assert "Bob" not in html and "Another player" in html  # others anonymous by default
    named = player_card(alice, [alice, bob], name_others=True)
    assert "Bob &lt;script&gt;" in named and "<script>" not in named  # escaped


def test_statements_need_three_examples() -> None:
    t = StyleTally(battles=1)
    t.choices["bad"]["switch"] = 2
    assert not any("matchup was against you" in s for s in play_style(t))
    t.choices["bad"]["attack"] = 1
    assert any("switched out 2 of 3 times" in s for s in play_style(t))


def test_something_to_try_prefers_a_clear_habit() -> None:
    t = StyleTally(battles=3, into_immunity=3)
    assert "immune" in something_to_try(player("A", [(True, 0.5)], t))
    assert "Nothing stood out" in something_to_try(player("B", [(True, 0.5)], None))


def test_tier_list_places_the_bot_and_everyone() -> None:
    page = tier_list_page([player("Strong", [(True, 0.3)] * 8), player("Weak", [(False, 0.7)] * 8)])
    assert "TestBot (the bot)" in page and "Strong" in page and "Weak" in page
    assert page.index("Strong") < page.index("TestBot (the bot)") < page.index("Weak")
