"""Rebuilding teams for replay: keep what was revealed, sample only the rest."""

from __future__ import annotations

import random

import pytest
from poke_env.data import to_id_str
from poke_env.teambuilder import Teambuilder

from psbot.analytics.team_strength import (
    _tera_options,
    bootstrap_interval,
    chance_of_at_most,
    pack,
    rebuild_ours,
    sample_theirs,
    team_species,
)
from psbot.heuristics.randbats import _load_sets

pytestmark = pytest.mark.skipif(not _load_sets(), reason="random-battle sets not installed")


def ours(ident: str, species: str, moves: list[str], ability: str, item: str | None, tera: str | None = None) -> dict:
    return {"ident": ident, "species": species, "level": 80, "moves": moves, "ability": ability, "item": item, "tera_type": tera}


def theirs(ident: str, species: str, moves: list[str], ability=None, item=None, tera=None) -> dict:
    def label(value):
        return {"value": value, "source": "revealed"} if value else {"value": None, "source": "unknown"}

    return {"ident": ident, "species": species, "level": 82, "revealed_moves": moves,
            "ability": label(ability), "item": label(item), "tera_type": label(tera)}  # fmt: skip


RECORD = {
    "battle_id": "battle-test-1",
    "role": "p1",
    "events": [
        ["switch", "p1a: Garchomp", "Garchomp, L77", "250/250"],
        ["switch", "p2a: Skarmory", "Skarmory, L80", "100/100"],
        ["turn", "1"],
        ["-enditem", "p1a: Azumarill", "Sitrus Berry", "[eat]"],
        ["-terastallize", "p1a: Garchomp", "Steel"],
    ],
    "team": [
        ours("p1: Azumarill", "azumarill", ["aquajet", "playrough", "liquidation", "knockoff"], "hugepower", None),
        ours("p1: Garchomp", "garchomp", ["earthquake", "scaleshot", "swordsdance", "ironhead"], "roughskin", "loadeddice"),
        ours("p1: Palafin", "palafinhero", ["jetpunch", "wavecrash", "closecombat", "bulkup"], "zerotohero", "choiceband"),
    ],
    "opponent_team": [
        theirs("p2: Garchomp", "garchomp", ["earthquake"]),
        theirs("p2: Skarmory", "skarmory", ["bodypress", "roost"], ability="sturdy", item="rockyhelmet"),
    ],
}


def test_battle_only_formes_start_as_their_base() -> None:
    assert team_species("palafinhero") == "palafin"
    assert team_species("mimikyubusted") == "mimikyu"
    assert team_species("ogerponcornerstonetera") == "ogerponcornerstone"
    assert team_species("greninja", "battlebond") == "greninjabond"
    assert team_species("raichualola") == "raichualola"


def test_our_team_is_exact_and_gaps_are_filled_from_the_log() -> None:
    team = rebuild_ours(RECORD, random.Random(0))
    by_species = {to_id_str(m.species): m for m in team.members}
    assert to_id_str(team.members[0].species) == "garchomp"  # the lead goes first
    assert by_species["garchomp"].tera_type == "Steel"  # shown in the battle
    assert to_id_str(by_species["azumarill"].item) == "sitrusberry"  # eaten, recovered from the log
    assert by_species["palafin"].moves == ["jetpunch", "wavecrash", "closecombat", "bulkup"]
    assert all(m.evs == [85] * 6 and m.level == 80 for m in team.members)
    assert team.sampled["our item recovered from the log"] == 1
    assert by_species["azumarill"].tera_type in _tera_options("azumarill", by_species["azumarill"].moves)


def test_their_team_keeps_every_revealed_fact_and_samples_the_rest() -> None:
    for seed in range(20):
        team = sample_theirs(RECORD, random.Random(seed))
        assert len(team.members) == 6
        lead, chomp = team.members[0], team.members[1]
        assert to_id_str(lead.species) == "skarmory"  # their lead first
        assert lead.ability == "sturdy" and lead.item == "rockyhelmet"
        assert {"bodypress", "roost"} <= set(lead.moves) and len(lead.moves) == 4
        assert chomp.moves[0] == "earthquake" and len(set(chomp.moves)) == 4
        sets = _load_sets()["garchomp"]["sets"]
        assert any(set(chomp.moves) <= {to_id_str(x) for x in s["movepool"]} for s in sets)
        assert len({to_id_str(m.species) for m in team.members}) == 6  # no repeated species
        assert team.sampled["their unseen Pokémon sampled"] == 4


def test_abilities_are_valid_ids() -> None:
    from psbot.analytics.team_strength import team_ability

    assert team_ability("calyrexshadow", "As One") == "asonespectrier"
    assert team_ability("calyrexice", "asone") == "asoneglastrier"
    assert team_ability("garchomp", "Rough Skin") == "roughskin"


@pytest.mark.skipif(__import__("shutil").which("node") is None, reason="node not installed")
def test_rebuilt_teams_pass_the_simulators_validator() -> None:
    from psbot.analytics.team_strength import validate_teams

    teams = [pack(rebuild_ours(RECORD, random.Random(0)).members)]
    teams += [pack(sample_theirs(RECORD, random.Random(seed)).members) for seed in range(10)]
    assert validate_teams(teams) == [None] * len(teams)
    bad = pack(sample_theirs(RECORD, random.Random(0)).members).replace("sturdy", "notanability")
    assert validate_teams([bad])[0]


def test_packed_teams_round_trip() -> None:
    team = sample_theirs(RECORD, random.Random(1)).members
    parsed = Teambuilder.parse_packed_team(pack(team))
    assert [to_id_str(p.species) for p in parsed] == [to_id_str(m.species) for m in team]
    assert [int(p.level) for p in parsed] == [m.level for m in team]
    assert [p.moves for p in parsed] == [m.moves for m in team]


def test_chance_of_a_result_under_team_strength_alone() -> None:
    assert chance_of_at_most(0, [0.5] * 5) == pytest.approx(1 / 32)
    assert chance_of_at_most(5, [0.5] * 5) == pytest.approx(1.0)
    assert chance_of_at_most(0, [0.9, 0.1]) == pytest.approx(0.1 * 0.9)


def test_bootstrap_interval_widens_with_disagreeing_samples() -> None:
    steady = bootstrap_interval([(15, 30)] * 6)
    spread = bootstrap_interval([(3, 30), (27, 30)] * 3)
    assert steady[0] < 0.5 < steady[1]
    assert (spread[1] - spread[0]) > (steady[1] - steady[0])


def test_recorder_takes_our_tera_type_from_the_request() -> None:
    """Phase 4 records left our Tera type empty unless we Terastallized."""
    from factories import make_battle, make_mon

    from psbot.dataset.human import own_pokemon

    me = make_mon("garchomp", ["earthquake"])
    battle = make_battle(me, make_mon("blissey", own=False))
    battle._last_request = {"side": {"pokemon": [{"ident": me.identifier("p1"), "teraType": "Steel"}]}}
    assert own_pokemon("p1: Garchomp", me)["tera_type"] is None  # the old behaviour
    assert own_pokemon("p1: Garchomp", me, battle)["tera_type"] == "steel"


def test_calibration_shrinks_overconfident_strengths() -> None:
    from psbot.analytics.team_strength import calibrated, fit_calibration

    rng = random.Random(0)
    strengths = [rng.random() for _ in range(2000)]
    # The truth is half as extreme as the raw strength claims.
    outcomes = [rng.random() < 0.5 + (s - 0.5) * 0.5 for s in strengths]
    fit = fit_calibration(outcomes, strengths)
    assert fit["b"] < 1
    assert 0.2 < calibrated(0.05, fit) < 0.45 and 0.55 < calibrated(0.95, fit) < 0.8
    assert calibrated(0.5, fit) == pytest.approx(0.5, abs=0.05)
    assert calibrated(0.3, None) == 0.3
