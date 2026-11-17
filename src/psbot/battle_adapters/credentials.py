"""The bot account's credentials, read from the environment or a local file.

Nothing here is ever written to disk or to a log. The password lives in a
dataclass field excluded from ``repr`` so it cannot leak through an error
message or a debug print of the object.

Sources, first match wins per variable:

1. environment variables;
2. a ``.env`` file in the project root (git-ignored), ``KEY=value`` per line.

Variables:

``PSBOT_USERNAME``        the dedicated bot account's username
``PSBOT_PASSWORD``        its password
``PSBOT_BOT_REGISTERED``  set to ``1`` once Showdown staff have registered the
                          account as a bot (mandatory since the policy of
                          2026-10-09: https://www.smogon.com/forums/threads/3789528/)
"""

from __future__ import annotations

import logging
import os
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from poke_env.data import to_id_str

from psbot.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

ENV_FILE: Path = PROJECT_ROOT / ".env"
USERNAME_VAR = "PSBOT_USERNAME"
PASSWORD_VAR = "PSBOT_PASSWORD"
REGISTERED_VAR = "PSBOT_BOT_REGISTERED"
MAX_USERNAME_LENGTH = 18  # Showdown's limit
_TRUE = frozenset({"1", "true", "yes"})


class CredentialsError(RuntimeError):
    """Raised when the bot account is not configured. Never contains a secret."""


@dataclass(frozen=True)
class BotCredentials:
    username: str
    password: str = field(repr=False)
    registered: bool = False

    @property
    def user_id(self) -> str:
        """Showdown's id for the username: lower-case letters and digits only."""
        return to_id_str(self.username)


def read_env_file(path: Path) -> dict[str, str]:
    """Parse ``KEY=value`` lines. Blank lines and ``#`` comments are skipped;
    one pair of surrounding quotes is removed. A missing file is not an error."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError as exc:
        raise CredentialsError(f"cannot read {path}: {exc.strerror}") from None
    try:
        if stat.S_IMODE(path.stat().st_mode) & 0o077:
            logger.warning("%s is readable by other users; run: chmod 600 %s", path, path)
    except OSError:
        pass
    values: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        values[key.strip().removeprefix("export ").strip()] = value
    return values


def load_credentials(
    env: Mapping[str, str] | None = None, env_file: Path | None = ENV_FILE
) -> BotCredentials:
    """Load the bot account, or raise :class:`CredentialsError` saying what is missing."""
    env = os.environ if env is None else env
    from_file = read_env_file(env_file) if env_file is not None else {}

    def get(name: str) -> str:
        return env.get(name) or from_file.get(name) or ""

    username, password = get(USERNAME_VAR).strip(), get(PASSWORD_VAR)
    missing = [n for n, v in ((USERNAME_VAR, username), (PASSWORD_VAR, password)) if not v]
    if missing:
        raise CredentialsError(
            f"missing {' and '.join(missing)}. Set them in your shell or in {ENV_FILE} "
            "(git-ignored). Use a dedicated bot account, never your personal one."
        )
    if len(username) > MAX_USERNAME_LENGTH or not to_id_str(username):
        raise CredentialsError(
            f"{USERNAME_VAR} must be 1-{MAX_USERNAME_LENGTH} characters with a letter or digit"
        )
    return BotCredentials(username, password, get(REGISTERED_VAR).strip().lower() in _TRUE)
