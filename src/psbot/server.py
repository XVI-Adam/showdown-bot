"""Reachability checks for the local Pokémon Showdown server.

poke-env opens its websocket when a ``Player`` is constructed and, if the
server is down, fails with a low-level connection error from a background
thread. Checking first lets every script fail fast with an actionable message.
"""

from __future__ import annotations

import logging
import socket

from psbot.config import SERVER_HOST, SERVER_PORT

logger = logging.getLogger(__name__)


class ServerUnavailableError(RuntimeError):
    """Raised when the local Showdown server cannot be reached."""


def is_server_reachable(
    host: str = SERVER_HOST, port: int = SERVER_PORT, timeout: float = 2.0
) -> bool:
    """Return True if something is accepting TCP connections on ``host:port``."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError as exc:
        logger.debug("server check failed for %s:%s: %s", host, port, exc)
        return False


def require_server(host: str = SERVER_HOST, port: int = SERVER_PORT) -> None:
    """Raise :class:`ServerUnavailableError` unless the server is reachable."""
    if not is_server_reachable(host, port):
        raise ServerUnavailableError(
            f"No Pokémon Showdown server at {host}:{port}. "
            "Start one in another terminal with: scripts/start_server.sh"
        )
    logger.info("Showdown server reachable at %s:%s", host, port)
