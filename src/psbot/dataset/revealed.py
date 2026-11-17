"""What the battle log has revealed about each Pokémon, and nothing more.

A Pokémon object in poke-env cannot answer "has this ability been revealed?":
poke-env fills in the ability when a species has only one, and our own damage
code fills in abilities and items that are certain for a species. Those are
inferences. This module reads only the protocol events both players can see
and records what they show:

* a move, when the Pokémon uses it (a move *called* by another move or
  ability, e.g. through Sleep Talk or Dancer, proves nothing about the
  moveset and is ignored);
* an ability, when the log names it (``-ability`` or an ``ability: X`` tag);
* an item, when the log names it (``-item``, ``-enditem``, ``item: X`` tag);
* a Tera type, when the Pokémon Terastallizes.

When the log is ambiguous the fact is left unrevealed. An effect tagged
``[from] ability: X|[of] POKEMON`` names two Pokémon, and which of them owns X
depends on the effect (Rough Skin damage names the owner in ``[of]``; a Water
Absorb heal names the attacker there). Only the cases whose meaning is fixed
are read; see ``_OWNER_IS_OF`` and ``_OWNER_IS_SUBJECT``. Under-claiming costs
a little information; a wrong "revealed" would poison everything built on it.

A transformed Pokémon (Ditto) uses its target's moves and ability, so nothing
it shows while transformed is recorded as its own.

Under Illusion (Zoroark) the log attributes everything to the disguise. When
the disguise breaks (``replace``), what the "disguise" showed since it came in
is moved to the real Pokémon. If the disguise never breaks, the log never
tells the truth and neither can we: the facts stay with the disguise.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from poke_env.data import to_id_str

Event = Sequence[str]

_IDENT = re.compile(r"^(p\d)[a-z]?: (.+)$")

# For an effect line ``KIND|SUBJECT|...|[from] ability-or-item: X|[of] OTHER``
# with OTHER different from SUBJECT: who owns X?
_OWNER_IS_OF = frozenset({"-damage", "-status", "-weather", "-fieldstart", "-sidestart"})
_OWNER_IS_SUBJECT = frozenset({"-heal"})


def normalise_ident(ident: str) -> str | None:
    """``"p1a: Raichu"`` -> ``"p1: Raichu"`` (the key poke-env uses for a team
    member), or None if the text does not name a Pokémon."""
    match = _IDENT.match(ident)
    return f"{match.group(1)}: {match.group(2)}" if match else None


@dataclass
class Revealed:
    """Facts the log has shown about one Pokémon."""

    species: str | None = None
    level: int | None = None
    moves: list[str] = field(default_factory=list)
    ability: str | None = None  # the first ability shown, i.e. its own
    item: str | None = None  # the last item shown
    item_consumed: bool = False  # ... and it has since been used up or removed
    tera_type: str | None = None
    fainted: bool = False


def _tags(args: Event) -> tuple[dict[str, str], str | None]:
    """Read ``[from] ability: X`` / ``[from] item: X`` and ``[of] POKEMON``."""
    source: dict[str, str] = {}
    of = None
    for arg in args:
        if arg.startswith("[from]"):
            kind, _, name = arg[6:].strip().partition(": ")
            if kind in ("ability", "item") and name:
                source[kind] = to_id_str(name)
        elif arg.startswith("[of] "):
            of = normalise_ident(arg[5:])
    return source, of


class RevealTracker:
    """Reads events in order; ``mons`` maps a Pokémon's ident to its facts.

    Events are lists with the event name first, e.g.
    ``["move", "p1a: Raichu", "Thunderbolt", "p2a: Komala"]``.
    """

    def __init__(self) -> None:
        self.mons: dict[str, Revealed] = {}
        self.n_events = 0
        self._transformed: set[str] = set()
        # side -> (ident on the field, its facts when it came in), for Illusion
        self._entered: dict[str, tuple[str, Revealed]] = {}

    def get(self, ident: str) -> Revealed:
        return self.mons.setdefault(ident, Revealed())

    def update(self, events: Iterable[Event]) -> "RevealTracker":
        for event in events:
            self.n_events += 1
            if event:
                self._read(event[0], list(event[1:]))
        return self

    def _read(self, kind: str, args: list[str]) -> None:
        subject = normalise_ident(args[0]) if args else None
        source, of = _tags(args)

        if kind in ("switch", "drag") and subject:
            # Whoever was transformed on this side has left the field.
            self._transformed = {i for i in self._transformed if i[:2] != subject[:2]}
        if kind == "-transform" and subject:
            if "ability" in source:  # Imposter is Ditto's own
                self._ability(subject, source["ability"])
            self._transformed.add(subject)
            return
        if subject in self._transformed and kind not in ("faint", "-enditem", "-item"):
            return  # borrowed moves and ability

        if kind == "replace" and subject:
            self._unmask(subject)
        if kind in ("switch", "drag", "replace", "detailschange") and subject and len(args) > 1:
            mon = self.get(subject)
            details = args[1].split(", ")
            mon.species = to_id_str(details[0])
            for part in details[1:]:
                if part.startswith("L") and part[1:].isdigit():
                    mon.level = int(part[1:])
            if mon.level is None and kind != "detailschange":
                mon.level = 100
            if kind != "detailschange":
                self._entered[subject[:2]] = (subject, copy.deepcopy(mon))
        elif kind == "move" and subject and len(args) > 1:
            called = any(
                a.startswith("[from]") and "lockedmove" not in a for a in args[2:]
            )  # fmt: skip
            move = to_id_str(args[1])
            if not called and move != "struggle" and move not in self.get(subject).moves:
                self.get(subject).moves.append(move)
        elif kind == "-ability" and subject and len(args) > 1:
            shown = to_id_str(args[1])
            if source.get("ability") == "trace":
                # The subject owns Trace and copied the shown ability from [of].
                self._ability(subject, "trace")
                if of:
                    self._ability(of, shown)
            elif "ability" not in source:
                self._ability(subject, shown)
            # Otherwise (Mummy, Wandering Spirit, ...) an ability is being
            # overwritten or swapped: not a reliable view of anyone's own.
            return
        elif kind == "-item" and subject and len(args) > 1:
            mon = self.get(subject)
            mon.item, mon.item_consumed = to_id_str(args[1]), False
        elif kind == "-enditem" and subject and len(args) > 1:
            mon = self.get(subject)
            mon.item, mon.item_consumed = to_id_str(args[1]), True
        elif kind == "-terastallize" and subject and len(args) > 1:
            self.get(subject).tera_type = to_id_str(args[1])
        elif kind == "faint" and subject:
            self.get(subject).fainted = True

        # Effects credited to an ability or item reveal it for their owner.
        if of is None or of == subject:
            owner = subject
        elif kind in _OWNER_IS_OF:
            owner = of
        elif kind in _OWNER_IS_SUBJECT:
            owner = subject
        elif kind == "-item" and source.get("ability") == "frisk":
            owner = of  # the [of] Pokémon looked at the subject's item
            source.pop("item", None)
        else:
            owner = None  # ambiguous: leave unrevealed
        if owner:
            if "ability" in source:
                self._ability(owner, source["ability"])
            if "item" in source and self.get(owner).item is None:
                self.get(owner).item = source["item"]
        if subject and (of is None or of == subject):
            for arg in args[1:]:
                if arg.startswith("ability: "):
                    self._ability(subject, to_id_str(arg[9:]))
                elif arg.startswith("item: ") and self.get(subject).item is None:
                    self.get(subject).item = to_id_str(arg[6:])

    def _unmask(self, real: str) -> None:
        """An Illusion broke: ``real`` was on the field all along. Give it what
        the disguise appeared to show since coming in, and take that back."""
        disguise, before = self._entered.get(real[:2], (real, Revealed()))
        if disguise == real:
            return
        shown, mon = self.get(disguise), self.get(real)
        mon.moves += [m for m in shown.moves if m not in before.moves and m not in mon.moves]
        if before.ability is None and shown.ability:
            mon.ability = mon.ability or shown.ability
        if (shown.item, shown.item_consumed) != (before.item, before.item_consumed):
            mon.item, mon.item_consumed = shown.item, shown.item_consumed
        if before.tera_type is None and shown.tera_type:
            mon.tera_type = shown.tera_type
        self.mons[disguise] = before

    def _ability(self, ident: str, ability: str) -> None:
        mon = self.get(ident)
        if mon.ability is None and ability:
            mon.ability = ability


def revealed_at(events: Sequence[Event], upto: int | None = None) -> dict[str, Revealed]:
    """Everything revealed by the first ``upto`` events (all if None)."""
    return RevealTracker().update(events[:upto]).mons
