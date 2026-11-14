"""End-to-end check against the real local simulator."""

import asyncio

import pytest
from poke_env import AccountConfiguration
from poke_env.player import RandomPlayer

from psbot.config import BATTLE_FORMAT, local_server_configuration
from psbot.players import InspectingRandomPlayer
from psbot.runner import run_battles
from psbot.server import is_server_reachable

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not is_server_reachable(), reason="local Showdown server not running"),
]


def test_two_random_players_complete_battles() -> None:
    async def play():
        server = local_server_configuration()
        player = InspectingRandomPlayer(
            account_configuration=AccountConfiguration.generate("TestA", rand=True),
            battle_format=BATTLE_FORMAT,
            server_configuration=server,
        )
        opponent = RandomPlayer(
            account_configuration=AccountConfiguration.generate("TestB", rand=True),
            battle_format=BATTLE_FORMAT,
            server_configuration=server,
        )
        return await run_battles(player, opponent, n_battles=3)

    summary = asyncio.run(play())

    assert summary.n_battles == 3
    assert summary.wins + summary.losses + summary.ties == 3
    assert summary.avg_turns > 0
    assert 0 <= summary.avg_remaining_pokemon <= 6


def test_heuristic_player_completes_battles_and_logs_them(tmp_path) -> None:
    from psbot.players import HeuristicPlayer
    from psbot.recording import BattleRecorder

    recorder = BattleRecorder(tmp_path / "battles.jsonl")

    async def play():
        server = local_server_configuration()
        player = HeuristicPlayer(
            account_configuration=AccountConfiguration.generate("TestH", rand=True),
            battle_format=BATTLE_FORMAT,
            server_configuration=server,
            recorder=recorder,
            opponent_kind="random",
        )
        opponent = RandomPlayer(
            account_configuration=AccountConfiguration.generate("TestR", rand=True),
            battle_format=BATTLE_FORMAT,
            server_configuration=server,
        )
        summary = await run_battles(player, opponent, n_battles=3)
        return player, summary

    player, summary = asyncio.run(play())

    assert summary.n_battles == 3
    assert player.n_decisions > 0
    assert player.n_fallback_decisions == 0  # the engine never raised
    assert len(recorder.path.read_text().splitlines()) == 3
    assert all(r["decisions"] and r["opponent_actions"] for r in recorder.records)


def test_environment_trains_saves_and_plays(tmp_path) -> None:
    """The whole RL loop on a tiny budget: env -> PPO update -> save -> load -> battle."""
    import numpy as np
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    from psbot.players.ppo import PPOPlayer
    from psbot.rl.env import make_env
    from psbot.rl.observation import OBSERVATION_SIZE
    from psbot.rl.policy import MaskedActorCriticPolicy

    env = DummyVecEnv([lambda: make_env("max_base_power")])
    try:
        assert env.observation_space["observation"].shape == (OBSERVATION_SIZE,)
        assert env.action_space.n == 26
        model = PPO(MaskedActorCriticPolicy, env, n_steps=128, batch_size=64, n_epochs=2, device="cpu", seed=0)
        model.learn(total_timesteps=256)
        assert model.num_timesteps == 256

        obs = env.reset()
        for _ in range(50):  # every action the policy picks must be legal
            action, _ = model.predict(obs, deterministic=False)
            assert obs["action_mask"][0, action[0]] == 1
            obs, _, _, _ = env.step(action)
            assert np.isfinite(obs["observation"]).all()
        path = tmp_path / "model.zip"
        model.save(path)
    finally:
        env.close()

    async def play():
        server = local_server_configuration()
        agent = PPOPlayer.from_checkpoint(
            path,
            account_configuration=AccountConfiguration.generate("TestP", rand=True),
            battle_format=BATTLE_FORMAT,
            server_configuration=server,
        )
        opponent = RandomPlayer(
            account_configuration=AccountConfiguration.generate("TestQ", rand=True),
            battle_format=BATTLE_FORMAT,
            server_configuration=server,
        )
        return agent, await run_battles(agent, opponent, n_battles=2)

    agent, summary = asyncio.run(play())
    assert summary.n_battles == 2
    assert agent.n_decisions > 0
    assert agent.n_illegal_actions == 0


def test_hybrid_trains_saves_and_plays(tmp_path) -> None:
    """Hybrid observation + prior policy through the same loop, with critic warm-up."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    from psbot.players.ppo import PPOPlayer
    from psbot.rl.callbacks import CriticWarmupCallback
    from psbot.rl.env import REWARD_PRESETS, make_env
    from psbot.rl.hybrid import HYBRID_OBSERVATION_SIZE
    from psbot.rl.policy import HybridActorCriticPolicy

    env = DummyVecEnv([lambda: make_env("heuristic", REWARD_PRESETS["sparse"], observation="hybrid")])
    try:
        assert env.observation_space["observation"].shape == (HYBRID_OBSERVATION_SIZE,)
        model = PPO(
            HybridActorCriticPolicy,
            env,
            policy_kwargs={"prior_scale": 15.0},
            n_steps=128,
            batch_size=64,
            n_epochs=2,
            device="cpu",
            seed=0,
        )
        model.learn(total_timesteps=256, callback=CriticWarmupCallback(128))
        assert model.policy.prior_scale.requires_grad  # unfrozen after warm-up
        path = tmp_path / "hybrid.zip"
        model.save(path)
    finally:
        env.close()

    async def play():
        server = local_server_configuration()
        agent = PPOPlayer.from_checkpoint(
            path,
            account_configuration=AccountConfiguration.generate("TestHy", rand=True),
            battle_format=BATTLE_FORMAT,
            server_configuration=server,
        )
        opponent = RandomPlayer(
            account_configuration=AccountConfiguration.generate("TestRn", rand=True),
            battle_format=BATTLE_FORMAT,
            server_configuration=server,
        )
        return agent, await run_battles(agent, opponent, n_battles=2)

    agent, summary = asyncio.run(play())
    assert summary.n_battles == 2
    assert agent.n_illegal_actions == 0
    assert agent.n_choices > 0
    assert agent.policy.prior_scale.item() == pytest.approx(15.0, abs=0.5)  # restored from the file


@pytest.mark.parametrize("bot", ["heuristic", "checkpoints/ppo-v1/final.zip"])
def test_hosted_bot_accepts_a_challenge_and_logs_the_battle(bot, tmp_path) -> None:
    """What a person does in the browser, done by a stand-in challenger."""
    from pathlib import Path

    from psbot.hosting import create_hosted_bot

    if bot.endswith(".zip") and not Path(bot).is_file():
        pytest.skip("trained checkpoint not available")

    async def play():
        hosted, recorder = create_hosted_bot(bot, name="HostedTestBot", log_path=tmp_path / "human.jsonl")
        challenger = RandomPlayer(
            account_configuration=AccountConfiguration.generate("Visitor", rand=True),
            battle_format=BATTLE_FORMAT,
            server_configuration=local_server_configuration(),
        )
        await asyncio.wait_for(
            asyncio.gather(
                hosted.accept_challenges(challenger.username, 1),
                challenger.send_challenges(hosted.username, 1),
            ),
            timeout=60,
        )
        await hosted.ps_client.stop_listening()
        await challenger.ps_client.stop_listening()
        return hosted, recorder

    hosted, recorder = asyncio.run(play())
    assert hosted.n_finished_battles == 1
    (record,) = recorder.records
    assert record["opponent_kind"] == "human"
    assert record["decisions"] and record["events"]
    assert len((tmp_path / "human.jsonl").read_text().splitlines()) == 1


def test_hosted_bot_rejects_a_bad_name() -> None:
    from psbot.hosting import create_hosted_bot

    for name in ("", "x" * 19):
        with pytest.raises(ValueError, match="name must be"):
            create_hosted_bot("heuristic", name=name)
