"""Connections to Showdown servers.

* local simulator: ``psbot.hosting`` and ``psbot.config`` (password-less,
  loopback only);
* official server: ``psbot.battle_adapters.official`` (password login, rate
  limited, nothing secret in logs).
"""

from psbot.battle_adapters.credentials import BotCredentials, CredentialsError, load_credentials
from psbot.battle_adapters.official import (
    LoginError,
    OfficialPSClient,
    create_official_player,
)

from psbot.battle_adapters.session import BattleOutcome, ChoiceGuard, OfficialSession

__all__ = [
    "BattleOutcome",
    "BotCredentials",
    "ChoiceGuard",
    "CredentialsError",
    "LoginError",
    "OfficialPSClient",
    "OfficialSession",
    "create_official_player",
    "load_credentials",
]
