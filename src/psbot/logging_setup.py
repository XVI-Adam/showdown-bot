"""Logging configuration shared by all scripts."""

from __future__ import annotations

import logging


def setup_logging(verbose: bool = True) -> None:
    """Configure console logging.

    Only the ``psbot`` logger is raised to INFO. poke-env names each player's
    logger after its username and logs every websocket message at INFO, so
    leaving the root logger at WARNING keeps the raw protocol out of the way.
    Pass ``log_level=logging.INFO`` to a ``Player`` to see that protocol.
    """
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("psbot").setLevel(logging.INFO if verbose else logging.WARNING)
