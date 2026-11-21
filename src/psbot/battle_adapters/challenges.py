"""Challenging people who agreed to play the bot: series and rotations.

Only players on the consent list are ever challenged (``data/challenges/
consenting.txt``, one username per line, managed with
``scripts/challenge_list.py``). Everything else follows from being polite to
them and to the server:

* one challenge pending at a time, at least ``min_gap`` seconds apart (the
  server itself refuses a second challenge within 10 seconds);
* a challenge nobody answers within two minutes is withdrawn;
* after a decline (or no answer), that player is left alone for
  ``decline_cooldown`` seconds, and after ``max_declines`` of them they are
  not challenged again that day;
* at most ``daily_cap`` challenges per day in total;
* a player who is offline or blocking challenges is skipped for this run and
  not held against them.

The only state kept between runs is what those limits need: today's date,
how many challenges went out, and each player's declines today
(``data/challenges/state.json``). It resets every day.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Callable

from poke_env.data import to_id_str

from psbot.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

CHALLENGE_DIR: Path = PROJECT_ROOT / "data" / "challenges"
CONSENT_FILE: Path = CHALLENGE_DIR / "consenting.txt"
STATE_FILE: Path = CHALLENGE_DIR / "state.json"


# ------------------------------------------------------------ consent list


def read_consent(path: Path = CONSENT_FILE) -> list[str]:
    """Usernames, in file order, without duplicates. ``#`` starts a comment."""
    if not path.exists():
        return []
    names, seen = [], set()
    for line in path.read_text(encoding="utf-8").splitlines():
        name = line.split("#", 1)[0].strip()
        if name and to_id_str(name) and to_id_str(name) not in seen:
            seen.add(to_id_str(name))
            names.append(name)
    return names


def add_consent(name: str, note: str = "", path: Path = CONSENT_FILE) -> bool:
    """Add a player. Returns False if they were already on the list."""
    if to_id_str(name) in {to_id_str(n) for n in read_consent(path)}:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = f"  # agreed {date.today():%Y-%m-%d}" + (f": {note}" if note else "")
    with path.open("a", encoding="utf-8") as fh:
        fh.write(f"{name}{stamp}\n")
    return True


def remove_consent(name: str, path: Path = CONSENT_FILE) -> bool:
    if not path.exists():
        return False
    target = to_id_str(name)
    lines = path.read_text(encoding="utf-8").splitlines()
    kept = [l for l in lines if to_id_str(l.split("#", 1)[0].strip()) != target]
    path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    return len(kept) != len(lines)


# --------------------------------------------------------- limits and state


@dataclass(frozen=True)
class Limits:
    daily_cap: int = 20  # challenges sent per day, all players together
    max_declines: int = 2  # per player per day; an unanswered challenge counts
    decline_cooldown: float = 600.0  # seconds before asking the same player again
    min_gap: float = 15.0  # seconds between challenges (the server requires 10)


@dataclass
class DailyState:
    day: str
    sent: int = 0
    declines: dict[str, list[float]] = field(default_factory=dict)  # player id -> times

    @classmethod
    def load(cls, path: Path = STATE_FILE, today: str | None = None) -> "DailyState":
        today = today or date.today().isoformat()
        try:
            raw = json.loads(path.read_text())
        except (OSError, ValueError):
            return cls(today)
        if raw.get("day") != today:
            return cls(today)  # a new day: everything resets
        return cls(today, int(raw.get("sent", 0)), {k: list(v) for k, v in raw.get("declines", {}).items()})

    def save(self, path: Path = STATE_FILE) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"day": self.day, "sent": self.sent, "declines": self.declines}, indent=1))

    def record_decline(self, player: str, now: float) -> None:
        self.declines.setdefault(to_id_str(player), []).append(now)


def refusal(state: DailyState, limits: Limits, player: str, now: float) -> tuple[str, float] | None:
    """Why ``player`` may not be challenged right now, and from when they may
    be again (``inf`` for not today), or None if they may."""
    if state.sent >= limits.daily_cap:
        return f"daily cap of {limits.daily_cap} challenges reached", float("inf")
    declines = state.declines.get(to_id_str(player), [])
    if len(declines) >= limits.max_declines:
        return f"declined or didn't answer {len(declines)} times today", float("inf")
    if declines and now - declines[-1] < limits.decline_cooldown:
        return "declined recently", declines[-1] + limits.decline_cooldown
    return None


# ----------------------------------------------------------------- runner


@dataclass
class Attempt:
    player: str
    status: str  # accepted / declined / expired / unavailable / refused
    detail: str = ""


class ChallengeRunner:
    """Sends challenges for an ``OfficialSession`` and waits for each answer.

    ``targets`` are tried in order, one game each per pass (rotation); with a
    single target this is a series. Stops after ``games`` battles, when
    nobody left can be challenged, or after ``idle_timeout`` seconds without
    anything to do.
    """

    def __init__(
        self,
        session: Any,
        targets: list[str],
        games: int,
        limits: Limits = Limits(),
        state: DailyState | None = None,
        save: Callable[[DailyState], None] | None = None,
        idle_timeout: float = 1800.0,
        clock: Callable[[], float] = time.time,
        say: Callable[[str], None] = print,
    ) -> None:
        self.session = session
        self.targets = targets
        self.games = games
        self.limits = limits
        self.state = state or DailyState.load()
        self._save = save if save is not None else (lambda s: s.save())
        self.idle_timeout = idle_timeout
        self.clock = clock
        self.say = say
        self.attempts: list[Attempt] = []
        self._last_sent = -1e18

    @property
    def played(self) -> int:
        return len(self.session.outcomes)

    def _closed(self) -> bool:
        return bool(getattr(self.session.client, "closed", False))

    async def _sleep_until(self, deadline: float) -> None:
        while self.clock() < deadline and not self._closed():
            await asyncio.sleep(min(1.0, max(0.05, deadline - self.clock())))

    async def _wait_idle(self) -> None:
        """Wait while a battle (perhaps one somebody else started) is running."""
        while (self.session.active is not None or self.session._expected is not None) and not self._closed():
            await asyncio.sleep(0.5)

    async def _challenge_once(self, player: str) -> str:
        await self._sleep_until(self._last_sent + self.limits.min_gap)
        await self._wait_idle()
        if self._closed() or self.played >= self.games:
            return "stopped"
        before = self.played
        try:
            await self.session.challenge(player)
        except RuntimeError:  # a battle started meanwhile
            return "busy"
        self._last_sent = self.clock()
        self.state.sent += 1
        self._save(self.state)
        self.say(f"challenged {player} ({self.state.sent} today)")
        while self.session.challenge_status == "pending" and not self._closed():
            await asyncio.sleep(0.25)
        status, detail = self.session.challenge_status or "stopped", self.session.challenge_detail
        if status == "accepted":
            self.say(f"{player} accepted; battle in progress")
            while self.played == before and not self._closed():
                await asyncio.sleep(0.5)
            if self.played > before:
                self.say(f"  {self.session.outcomes[-1]}")
        elif status in ("declined", "expired"):
            self.state.record_decline(player, self.clock())
            self._save(self.state)
            self.say(f"{player} {'declined' if status == 'declined' else 'did not answer'}")
        elif status == "too_soon":
            self.state.sent -= 1  # refused by the server, not delivered
            self._save(self.state)
        else:
            self.say(f"{player}: {status}{f' ({detail})' if detail else ''}")
        self.attempts.append(Attempt(player, status, detail))
        return status

    async def run(self) -> list[Attempt]:
        unavailable: set[str] = set()
        idle_since = self.clock()
        index = 0
        while self.played < self.games and not self._closed():
            await self._wait_idle()
            if self.played >= self.games:
                break
            now = self.clock()
            open_players, soonest = [], float("inf")
            for player in self.targets:
                if to_id_str(player) in unavailable:
                    continue
                why = refusal(self.state, self.limits, player, now)
                if why is None:
                    open_players.append(player)
                else:
                    soonest = min(soonest, why[1])
            if not open_players:
                if soonest == float("inf") or soonest - now > self.idle_timeout:
                    self.say("nobody left to challenge: " + self._why_nobody(unavailable, now))
                    break
                self.say(f"waiting {soonest - now:.0f}s for a cooldown to pass")
                await self._sleep_until(soonest)
                continue
            # rotation: the next open player after the last one asked
            ordered = [p for p in self.targets if p in open_players]
            player = ordered[index % len(ordered)]
            index += 1
            status = await self._challenge_once(player)
            if status == "unavailable":
                unavailable.add(to_id_str(player))
            elif status == "too_soon":
                self._last_sent = self.clock()  # wait out the server's 10 s rule, then retry
                index -= 1
            if status == "accepted":
                idle_since = self.clock()
            elif self.clock() - idle_since > self.idle_timeout:
                self.say("stopping: nothing has happened for a while")
                break
        return self.attempts

    def _why_nobody(self, unavailable: set[str], now: float) -> str:
        reasons = []
        for player in self.targets:
            if to_id_str(player) in unavailable:
                reasons.append(f"{player} offline or not taking challenges")
            else:
                why = refusal(self.state, self.limits, player, now)
                reasons.append(f"{player}: {why[0] if why else 'available'}")
        return "; ".join(reasons) or "the list is empty"
