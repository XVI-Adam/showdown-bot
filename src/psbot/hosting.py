"""Host a bot on the local server so a person can challenge it from a browser."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from poke_env import AccountConfiguration
from poke_env.player import Player

from psbot.config import BATTLE_FORMAT, LOG_DIR, local_server_configuration
from psbot.opponents import make_opponent
from psbot.recording import BattleRecorder

DEFAULT_BOT_NAME = "ShowdownBot"
MAX_USERNAME_LENGTH = 18  # Showdown's limit


def create_hosted_bot(
    bot: str, name: str = DEFAULT_BOT_NAME, log_path: Path | None = None
) -> tuple[Player, BattleRecorder | None]:
    """Create a bot that accepts challenges under the username ``name``.

    :param bot: ``heuristic``, another scripted player, or a path to a PPO or
        hybrid checkpoint ``.zip`` (see ``psbot.opponents.make_opponent``).
    :param log_path: where to save each finished battle, with the bot's
        reasoning and the full event log. Only the heuristic and learned agents
        record; for other bots the second value is None.
    """
    if not name or len(name) > MAX_USERNAME_LENGTH:
        raise ValueError(f"name must be 1-{MAX_USERNAME_LENGTH} characters, got {name!r}")
    player = make_opponent(
        bot,
        account_configuration=AccountConfiguration(name, None),
        battle_format=BATTLE_FORMAT,
        server_configuration=local_server_configuration(),
        max_concurrent_battles=1,
    )
    recorder = None
    if log_path is not None and hasattr(player, "recorder"):
        kind = Path(bot).parent.name if bot.endswith(".zip") else bot
        recorder = BattleRecorder(log_path, player_kind=kind, save_events=True)
        player.recorder = recorder
        player.opponent_kind = "human"
    return player, recorder


def default_log_path(name: str = DEFAULT_BOT_NAME) -> Path:
    return LOG_DIR / "human" / f"{name}-{datetime.now():%Y%m%d-%H%M%S}.jsonl"
