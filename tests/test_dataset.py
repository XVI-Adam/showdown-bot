"""The human-battle dataset: what counts as revealed, what is recorded, and
what the validator refuses."""

from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path
from typing import Any

import pytest
from poke_env import AccountConfiguration
from poke_env.player import RandomPlayer, SimpleHeuristicsPlayer

from psbot.analysis import battle_factors, build_timeline
from psbot.config import BATTLE_FORMAT, local_server_configuration
from psbot.dataset import HumanBattleRecorder, agent_info, to_legacy_record, validate_record
from psbot.dataset.revealed import Revealed, RevealTracker, normalise_ident, revealed_at
from psbot.dataset.schema import SCHEMA, load_dataset
from psbot.opponents import make_opponent
from psbot.runner import run_battles
from psbot.server import is_server_reachable

HYBRID = "checkpoints/hybrid-v2-sparse/final.zip"
PPO = "checkpoints/ppo-v1/final.zip"


class _Facts(dict):
    """A Pokémon the log never mentioned has no facts: same as an empty entry."""

    def __missing__(self, ident: str) -> Revealed:
        return Revealed()


def reveal(*lines: str) -> RevealTracker:
    tracker = RevealTracker().update(line.split("|") for line in lines)
    tracker.mons = _Facts(tracker.mons)
    return tracker


# ------------------------------------------------------- what is "revealed"


def test_ident_drops_the_position_letter() -> None:
    assert normalise_ident("p1a: Raichu") == "p1: Raichu"
    assert normalise_ident("p2: Iron Treads") == "p2: Iron Treads"
    assert normalise_ident("Sandstorm") is None


def test_used_moves_are_revealed_once_and_in_order() -> None:
    mons = reveal(
        "switch|p1a: Raichu|Raichu-Alola, L88, F|100/100",
        "move|p1a: Raichu|Thunderbolt|p2a: Komala",
        "move|p1a: Raichu|Psychic|p2a: Komala",
        "move|p1a: Raichu|Thunderbolt|p2a: Komala",
    ).mons
    raichu = mons["p1: Raichu"]
    assert (raichu.species, raichu.level) == ("raichualola", 88)
    assert raichu.moves == ["thunderbolt", "psychic"]
    assert raichu.ability is None and raichu.item is None and raichu.tera_type is None


def test_called_moves_and_struggle_reveal_nothing() -> None:
    mons = reveal(
        "move|p1a: Snorlax|Sleep Talk|p1a: Snorlax",
        "move|p1a: Snorlax|Body Slam|p2a: Komala|[from] move: Sleep Talk",
        "move|p1a: Oricorio|Quiver Dance|p1a: Oricorio|[from] ability: Dancer",
        "move|p1a: Snorlax|Struggle|p2a: Komala",
        "move|p1a: Dragonite|Outrage|p2a: Komala|[from]lockedmove",
    ).mons
    assert mons["p1: Snorlax"].moves == ["sleeptalk"]
    assert mons["p1: Oricorio"].moves == []
    assert mons["p1: Oricorio"].ability == "dancer"
    assert mons["p1: Dragonite"].moves == ["outrage"]  # a locked-in move is its own


def test_abilities_named_by_the_log() -> None:
    mons = reveal(
        "-ability|p1a: Gyarados|Intimidate|boost",
        "-weather|Sandstorm|[from] ability: Sand Stream|[of] p1a: Tyranitar",
        "-immune|p1a: Rotom|[from] ability: Levitate",
        "-damage|p2a: Komala|88/100|[from] ability: Rough Skin|[of] p1a: Garchomp",
        "-activate|p1a: Umbreon|ability: Synchronize",
    ).mons
    assert mons["p1: Gyarados"].ability == "intimidate"
    assert mons["p1: Tyranitar"].ability == "sandstream"
    assert mons["p1: Rotom"].ability == "levitate"
    assert mons["p1: Garchomp"].ability == "roughskin"
    assert mons["p1: Umbreon"].ability == "synchronize"
    assert mons["p2: Komala"].ability is None  # it only took the damage


def test_ambiguous_ownership_is_left_unrevealed() -> None:
    """A Water Absorb heal names the *attacker* in [of]. Reading [of] as the
    owner credited Floatzel with Water Absorb in a real battle; it had Water Veil."""
    mons = reveal(
        "-heal|p2a: Vaporeon|100/100|[from] ability: Water Absorb|[of] p1a: Floatzel",
        "-ability|p1a: Toxapex|Mummy|[from] ability: Mummy|[of] p2a: Cofagrigus",
        "-unboost|p1a: Lucario|spe|1|[from] ability: Cotton Down|[of] p2a: Eldegoss",
    ).mons
    assert mons["p2: Vaporeon"].ability == "waterabsorb"
    assert mons["p1: Floatzel"].ability is None
    assert mons["p1: Toxapex"].ability is None  # overwritten, not its own
    assert mons["p1: Lucario"].ability is None
    assert mons["p2: Eldegoss"].ability is None  # true, but not certain from the log's shape


def test_trace_reveals_both_abilities() -> None:
    mons = reveal("-ability|p1a: Gardevoir|Intimidate|[from] ability: Trace|[of] p2a: Gyarados").mons
    assert mons["p1: Gardevoir"].ability == "trace"
    assert mons["p2: Gyarados"].ability == "intimidate"


def test_items_named_by_the_log() -> None:
    mons = reveal(
        "-heal|p1a: Snorlax|60/100|[from] item: Leftovers",
        "-damage|p2a: Komala|88/100|[from] item: Rocky Helmet|[of] p1a: Skarmory",
        "-enditem|p1a: Azumarill|Sitrus Berry|[eat]",
        "-item|p1a: Blissey|Choice Scarf|[from] ability: Frisk|[of] p2a: Dusknoir",
        "-damage|p1a: Gengar|90/100|[from] item: Life Orb",
    ).mons
    assert mons["p1: Snorlax"].item == "leftovers"
    assert mons["p1: Skarmory"].item == "rockyhelmet"
    assert (mons["p1: Azumarill"].item, mons["p1: Azumarill"].item_consumed) == ("sitrusberry", True)
    assert mons["p1: Blissey"].item == "choicescarf"
    assert mons["p2: Dusknoir"].ability == "frisk" and mons["p2: Dusknoir"].item is None
    assert mons["p1: Gengar"].item == "lifeorb"
    assert mons["p2: Komala"].item is None


def test_tera_type_and_faint() -> None:
    mons = reveal("-terastallize|p1a: Raichu|Fairy", "faint|p1a: Raichu").mons
    assert mons["p1: Raichu"].tera_type == "fairy" and mons["p1: Raichu"].fainted


def test_transformed_pokemon_shows_nothing_of_its_own() -> None:
    mons = reveal(
        "switch|p1a: Ditto|Ditto, L87|100/100",
        "-transform|p1a: Ditto|p2a: Gengar|[from] ability: Imposter",
        "move|p1a: Ditto|Shadow Ball|p2a: Gengar",
        "switch|p1a: Raichu|Raichu, L88|100/100",
        "move|p1a: Raichu|Thunderbolt|p2a: Gengar",
    ).mons
    assert mons["p1: Ditto"].moves == [] and mons["p1: Ditto"].ability == "imposter"
    assert mons["p1: Raichu"].moves == ["thunderbolt"]


def test_broken_illusion_moves_facts_to_the_real_pokemon() -> None:
    mons = reveal(
        "switch|p1a: Diancie|Diancie, L80|100/100",
        "move|p1a: Diancie|Moonblast|p2a: Komala",
        "switch|p1a: Raichu|Raichu, L88|100/100",
        "switch|p1a: Diancie|Diancie, L80|100/100",  # really Zoroark this time
        "move|p1a: Diancie|Dark Pulse|p2a: Komala",
        "-damage|p1a: Diancie|90/100|[from] item: Life Orb",
        "replace|p1a: Zoroark|Zoroark, L84, M",
        "-end|p1a: Zoroark|Illusion",
    ).mons
    assert mons["p1: Diancie"].moves == ["moonblast"] and mons["p1: Diancie"].item is None
    assert mons["p1: Zoroark"].moves == ["darkpulse"] and mons["p1: Zoroark"].item == "lifeorb"
    assert mons["p1: Zoroark"].species == "zoroark"


def test_revealed_at_respects_the_cutoff() -> None:
    events = [line.split("|") for line in ("move|p1a: A|Tackle|p2a: B", "move|p1a: A|Growl|p2a: B")]
    assert revealed_at(events, 1)["p1: A"].moves == ["tackle"]
    assert revealed_at(events)["p1: A"].moves == ["tackle", "growl"]


# ----------------------------------------------------------- real battles

needs_simulator = pytest.mark.skipif(
    not is_server_reachable(), reason="local Showdown server not running"
)


def play(bot: str, opponent_cls: type, n: int, path: Path) -> tuple[Any, Any, HumanBattleRecorder]:
    async def run():
        kwargs = dict(
            battle_format=BATTLE_FORMAT,
            server_configuration=local_server_configuration(),
            max_concurrent_battles=4,
        )
        player = make_opponent(bot, account_configuration=AccountConfiguration.generate("Rec", rand=True), **kwargs)
        opponent = opponent_cls(account_configuration=AccountConfiguration.generate("Opp", rand=True), **kwargs)
        recorder = HumanBattleRecorder(agent_info(bot, player), source="local", path=path)
        player.recorder = recorder
        await run_battles(player, opponent, n_battles=n)
        return player, opponent, recorder

    return asyncio.run(run())


@pytest.fixture(scope="module")
def heuristic_run(tmp_path_factory: pytest.TempPathFactory):
    if not is_server_reachable():
        pytest.skip("local Showdown server not running")
    return play("heuristic", SimpleHeuristicsPlayer, 8, tmp_path_factory.mktemp("ds") / "h.jsonl")


@pytest.mark.integration
def test_recorded_battles_are_complete_and_valid(heuristic_run) -> None:
    player, _, recorder = heuristic_run
    assert recorder.n_written == 8
    loaded = list(load_dataset(recorder.path))
    assert len(loaded) == 8
    for _, record in loaded:
        assert validate_record(record) == []
        assert record["schema"] == SCHEMA and record["format"] == "gen9randombattle"
        assert record["source"] == "local" and record["started_at"] and record["finished_at"]
        assert record["agent"]["type"] == "heuristic" and record["agent"]["checkpoint"] is None
        assert record["opponent"]["username"].startswith("Opp")
        assert record["outcome"]["won"] in (True, False)
        assert len(record["team"]) == 6 and all(mon["moves"] and mon["ability"] for mon in record["team"])
        for decision in record["decisions"]:
            assert sum(decision["action_mask"]) == len(decision["legal_actions"]) >= 1
            assert decision["action_mask"][decision["selected"]["index"]] == 1
            assert decision["observation"]["size"] == 112
            scored = decision["heuristic"]["actions"]
            assert scored and all(action["components"] for action in scored)
            # every legal action was scored (Tera variants share their move's score)
            base = {entry["action"].removesuffix(" +tera") for entry in decision["legal_actions"]}
            assert {action["action"].removesuffix(" +tera") for action in scored} == base
        assert any(decision["opponent_actions"] for decision in record["decisions"])
    assert sum(len(r["decisions"]) for _, r in loaded) == player.n_decisions


@pytest.mark.integration
def test_recorded_knowledge_matches_the_opponents_real_team(heuristic_run) -> None:
    """The opponent here is our own process, so its true team is known.
    Nothing marked revealed may contradict it."""
    _, opponent, recorder = heuristic_run
    checked = 0
    for record in recorder.records:
        truth = opponent.battles[record["battle_id"]].team
        for mon in record["opponent_team"]:
            real = truth[mon["ident"]]
            if "zoroark" in {m.species[:7] for m in truth.values()}:
                continue  # an unbroken Illusion fools the log itself
            assert set(mon["revealed_moves"]) <= set(real.moves) or real.species == "ditto"
            if mon["ability"]["source"] == "revealed":
                assert (real._ability or "").startswith(mon["ability"]["value"]) or real.species.startswith("ogerpon")
            if mon["tera_type"]["source"] == "revealed":
                assert mon["tera_type"]["value"] == real.tera_type.name.lower()
            assert not set(mon["possible_moves"]["value"]) & set(mon["revealed_moves"])
            checked += 1
    assert checked > 20


@pytest.mark.integration
def test_early_decisions_do_not_know_later_reveals(heuristic_run) -> None:
    """No hindsight: a decision records only what had been shown by then."""
    _, _, recorder = heuristic_run
    grew = False
    for record in recorder.records:
        final = {mon["ident"]: mon for mon in record["opponent_team"]}
        first = record["decisions"][0]
        seen_at_start = [m for m in [first["opponent_active"], *first["opponent_bench"]] if m]
        assert len(seen_at_start) <= 1 and first["opponent_unseen"] >= 5
        for mon in seen_at_start:
            assert mon["revealed_moves"] == []
            grew = grew or bool(final[mon["ident"]]["revealed_moves"])
    assert grew


@needs_simulator
@pytest.mark.integration
@pytest.mark.parametrize("bot, kind, size", [(HYBRID, "hybrid", 165), (PPO, "ppo", 112)])
def test_learned_agents_record_probabilities(tmp_path: Path, bot: str, kind: str, size: int) -> None:
    if not Path(bot).is_file():
        pytest.skip("trained checkpoint not available")
    _, _, recorder = play(bot, RandomPlayer, 3, tmp_path / "p.jsonl")
    assert recorder.n_written == 3
    for record in recorder.records:
        assert validate_record(record) == []
        agent = record["agent"]
        assert agent["type"] == kind and agent["checkpoint"] == bot and len(agent["checkpoint_sha256"]) == 64
        for decision in record["decisions"]:
            policy = decision["policy"]
            assert decision["heuristic"] is None and decision["observation"]["size"] == size
            assert abs(sum(policy["probabilities"]) - 1) < 1e-3 and policy["deterministic"] is True
            best = max(range(26), key=policy["probabilities"].__getitem__)
            assert decision["selected"]["index"] == best  # deterministic play = most likely action
            assert ("heuristic_scores" in policy) == (kind == "hybrid")


# -------------------------------------------------- validator and converter


@pytest.fixture
def record(heuristic_run) -> dict[str, Any]:
    return copy.deepcopy(heuristic_run[2].records[0])


def errors(record: dict[str, Any]) -> str:
    return "\n".join(str(issue) for issue in validate_record(record) if issue.level == "error")


def some_opponent(record: dict[str, Any], with_moves: bool = False) -> dict[str, Any]:
    for decision in reversed(record["decisions"]):
        mon = decision["opponent_active"]
        if mon and (mon["revealed_moves"] or not with_moves):
            return mon
    raise AssertionError("no opposing Pokémon recorded")


@pytest.mark.integration
def test_validator_rejects_a_fabricated_move(record) -> None:
    some_opponent(record)["revealed_moves"].append("spore")
    assert "revealed_moves" in errors(record)


@pytest.mark.integration
def test_validator_rejects_an_inference_passed_off_as_revealed(record) -> None:
    for decision in record["decisions"]:
        mon = decision["opponent_active"]
        if mon and mon["ability"]["source"] == "inferred":
            mon["ability"]["source"] = "revealed"
            assert "marked revealed but the log shows None" in errors(record)
            return
    pytest.skip("no inferred ability in this battle")


@pytest.mark.integration
def test_validator_rejects_bad_labels_and_illegal_choices(record) -> None:
    broken = copy.deepcopy(record)
    some_opponent(broken)["item"] = {"value": "leftovers", "source": "unknown"}
    assert "'unknown' but has a value" in errors(broken)

    broken = copy.deepcopy(record)
    decision = broken["decisions"][0]
    illegal = decision["action_mask"].index(0)
    decision["selected"]["index"] = illegal
    assert "is not legal" in errors(broken)

    broken = copy.deepcopy(record)
    broken["decisions"][0]["heuristic"]["actions"][0]["total"] += 0.5
    assert "do not add up" in errors(broken)

    broken = copy.deepcopy(record)
    broken["outcome"]["won"] = not broken["outcome"]["won"]
    assert "disagrees with the log" in errors(broken)

    broken = copy.deepcopy(record)
    del broken["opponent"]
    assert "missing fields: opponent" in errors(broken)

    assert "schema" in errors({**record, "schema": "something/9"})


@pytest.mark.integration
def test_record_survives_json_and_feeds_the_existing_analysis(record) -> None:
    legacy = to_legacy_record(json.loads(json.dumps(record)))
    assert legacy["battle_tag"] == record["battle_id"] and legacy["won"] == record["outcome"]["won"]
    assert len(legacy["decisions"]) == len(record["decisions"])
    assert legacy["decisions"][0]["reasons"] and "predicted" in legacy["decisions"][0]
    assert build_timeline(legacy).moves
    assert battle_factors(legacy)
