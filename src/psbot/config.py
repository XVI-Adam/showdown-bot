"""Project-wide constants: where the simulator lives and what we play."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

from poke_env import ServerConfiguration

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
LOG_DIR: Final[Path] = PROJECT_ROOT / "logs"

BATTLE_FORMAT: Final[str] = "gen9randombattle"

# Loopback only. We use 127.0.0.1 rather than "localhost" so the client never
# tries IPv6 (::1) first; the server binds to the IPv4 loopback address.
SERVER_HOST: Final[str] = "127.0.0.1"
SERVER_PORT: Final[int] = int(os.environ.get("PS_PORT", "8000"))

_LOOPBACK_HOSTS: Final[frozenset[str]] = frozenset({"127.0.0.1", "localhost", "::1"})


def local_server_configuration(
    host: str = SERVER_HOST, port: int = SERVER_PORT
) -> ServerConfiguration:
    """Build the poke-env server configuration for the local simulator.

    The second field is the login-server URL poke-env requires. It is never
    contacted for password-less accounts on a ``--no-security`` server.

    :raises ValueError: if ``host`` is not a loopback address. Training against
        a remote server is out of scope and would be a mistake.
    """
    if host not in _LOOPBACK_HOSTS:
        raise ValueError(f"refusing non-loopback Showdown host: {host!r}")
    if not 0 < port < 65536:
        raise ValueError(f"invalid port: {port}")
    return ServerConfiguration(
        f"ws://{host}:{port}/showdown/websocket",
        "https://play.pokemonshowdown.com/action.php?",
    )
