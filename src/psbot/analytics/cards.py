"""Shareable player cards and the tier list, as standalone HTML pages.

A card speaks to the player in plain language: their record against the bot,
their tier and power level, how lucky their teams were, and what their games
show about how they play. Every statement carries its count ("4 of 5"),
nothing is claimed from fewer than three examples, and below five games the
card says plainly that it is an early read.

Other players on a card's tier list are anonymous unless asked otherwise, so
a card can be passed on without revealing anyone else's results.
"""

from __future__ import annotations

import html
from dataclasses import dataclass, field
from datetime import date

from psbot.analytics.rating import BOT_RATING, PROVISIONAL_BELOW, TIERS, Rating
from psbot.analytics.style import StyleTally

MIN_EXAMPLES = 3


@dataclass
class Game:
    day: str
    won: bool | None  # the player's result
    team_chance: float | None  # the player's calibrated chance from the teams alone


@dataclass
class Player:
    name: str
    rating: Rating
    games: list[Game]
    style: StyleTally | None  # None when only older records exist
    bot_name: str = "the bot"
    notes: list[str] = field(default_factory=list)


def _n(count: int, noun: str) -> str:
    return f"{count} {noun}" + ("" if count == 1 else "s")


def play_style(style: StyleTally | None) -> list[str]:
    """Plain statements about how they played, each with its count."""
    if style is None or not style.total():
        return []
    out = []
    bad, good = style.total("bad"), style.total("good")
    if bad >= MIN_EXAMPLES:
        out.append(f"When the matchup was against you, you switched out {style.kind('switch', 'bad')} of {bad} times.")
    if good >= MIN_EXAMPLES:
        out.append(f"When the matchup was in your favour, you stayed in and attacked {style.kind('attack', 'good')} of {good} times.")
    if style.switches_into_attack >= MIN_EXAMPLES:
        out.append(
            f"{style.reads} of your {style.switches_into_attack} switches into the bot's attacks brought in a "
            "Pokémon that resisted the hit or took nothing from it."
        )
    if style.setup:
        safe = f", {style.setup_unthreatened} of them when the bot's Pokémon couldn't hit yours super-effectively" if style.setup_unthreatened else ""
        out.append(f"You used {_n(style.setup, 'setup move')}{safe}.")
    if style.tera_turns:
        ordered = [str(t) for t in sorted(style.tera_turns)]
        turns = ordered[0] if len(ordered) == 1 else ", ".join(ordered[:-1]) + " and " + ordered[-1]
        label = "turn" if len(ordered) == 1 else "turns"
        out.append(f"You Terastallized in {style.tera_battles} of {style.battles} games (on {label} {turns}).")
    elif style.battles >= MIN_EXAMPLES:
        out.append(f"You didn't Terastallize in any of your {style.battles} games.")
    if style.led_by_2:
        lead = "every time" if style.led_by_2_won == style.led_by_2 else f"{style.led_by_2_won} of {style.led_by_2} times"
        out.append(f"After getting two Pokémon ahead, you went on to win {lead}.")
    if style.trailed_by_2:
        out.append(f"You came back from two Pokémon down {style.trailed_by_2_won} of {style.trailed_by_2} times.")
    total = style.total()
    if total >= MIN_EXAMPLES * 3:
        share = {k: round(100 * style.kind(k) / total) for k in ("attack", "switch", "support")}
        out.append(f"Overall: {share['attack']}% of your choices were attacks, {share['switch']}% switches and "
                   f"{share['support']}% setup, status or recovery.")  # fmt: skip
    return out


def how_they_won(player: Player) -> list[str]:
    style, wins = player.style, sum(g.won is True for g in player.games)
    if not wins:
        return []
    out = []
    if style is not None and style.wins:
        if style.wins_by_sweep:
            out.append(f"In {style.wins_by_sweep} of your {style.wins} recorded wins, one of your Pokémon knocked out three or more of the bot's.")
        if style.wins_by_setup_sweep:
            games = "win" if style.wins_by_setup_sweep == 1 else "wins"
            out.append(f"In {style.wins_by_setup_sweep} {games}, a Pokémon of yours set up and then knocked out "
                       "the bot's Pokémon before it could be stopped.")  # fmt: skip
    lucky = sum(1 for g in player.games if g.won and g.team_chance is not None and g.team_chance < 0.5)
    if lucky:
        out.append(f"You won {lucky} with the weaker team on paper: that's play, not the draw.")
    return out


def something_to_try(player: Player) -> str:
    style = player.style
    if style is not None:
        if style.into_immunity >= 2:
            return (f"{style.into_immunity} of your attacks hit a Pokémon immune to them. A quick check of the "
                    "type chart before clicking would have saved those turns.")  # fmt: skip
        bad = style.total("bad")
        if bad >= 5 and style.kind("attack", "bad") / bad >= 0.6:
            return "In matchups against you, you mostly stayed in and attacked. The bot punishes that; switching out more often could pay."
        if not style.tera_turns and style.battles >= MIN_EXAMPLES:
            return "You never used Tera. It can turn a losing matchup around, and the bot doesn't plan for it."
    return "Nothing stood out as a habit to change. Keep doing what you're doing."


# ------------------------------------------------------------------- HTML

_CSS = """
:root{--bg:#f6f5f2;--card:#ffffff;--ink:#1d1c1a;--muted:#6b6760;--line:#e4e1da;--accent:#4b5bd6;
--S:#d4572a;--A:#d9932b;--B:#3f9a6b;--C:#3d7fc2;--D:#7a6fb0;--bot:#1d1c1a}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#151514;--card:#1f1e1c;--ink:#f1efe9;
--muted:#a39f96;--line:#34322e;--accent:#8e9bff;--bot:#f1efe9}}
:root[data-theme="dark"]{--bg:#151514;--card:#1f1e1c;--ink:#f1efe9;--muted:#a39f96;--line:#34322e;--accent:#8e9bff;--bot:#f1efe9}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:16px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
main{max-width:760px;margin:0 auto;padding:28px 16px 48px}
.card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:24px;margin-bottom:16px}
h1{font-size:30px;line-height:1.15;margin:0 0 4px;overflow-wrap:anywhere}h2{font-size:15px;letter-spacing:.04em;
text-transform:uppercase;color:var(--muted);margin:0 0 12px}.sub{color:var(--muted);margin:0}
.stats{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin-top:20px}
.stat{border:1px solid var(--line);border-radius:12px;padding:14px}.stat .k{font-size:13px;color:var(--muted)}
.stat .v{font-size:30px;font-weight:700;line-height:1.1;margin-top:4px}.stat .x{font-size:13px;color:var(--muted);margin-top:4px}
.badge{display:inline-flex;align-items:center;justify-content:center;width:44px;height:44px;border-radius:10px;
color:#fff;font-weight:800;font-size:24px}.bar{height:8px;background:var(--line);border-radius:4px;margin-top:8px;overflow:hidden}
.bar>span{display:block;height:100%;background:var(--accent)}
.note{background:color-mix(in srgb,var(--accent) 10%,transparent);border-radius:10px;padding:10px 14px;margin-top:16px;font-size:14px}
ul{margin:0;padding-left:20px}li{margin:6px 0}
.games{display:grid;gap:8px}.game{display:grid;grid-template-columns:96px 44px 1fr 44px;gap:10px;align-items:center;font-size:14px}
.res{font-weight:700}.res.w{color:var(--B)}.res.l{color:var(--S)}.mini{height:10px;background:var(--line);border-radius:5px;position:relative}
.mini>span{position:absolute;left:0;top:0;bottom:0;border-radius:5px;background:var(--accent)}
.mini>i{position:absolute;left:50%;top:-3px;bottom:-3px;width:1px;background:var(--muted)}
.tiers{display:grid;gap:8px}.tier{display:grid;grid-template-columns:44px 1fr;gap:12px;align-items:center;
border:1px solid var(--line);border-radius:12px;padding:8px}.chips{display:flex;flex-wrap:wrap;gap:6px}
.chip{border:1px solid var(--line);border-radius:999px;padding:3px 10px;font-size:14px;white-space:nowrap}
.chip.me{border-color:var(--accent);background:color-mix(in srgb,var(--accent) 14%,transparent);font-weight:700}
.chip.bot{border-style:solid;border-color:var(--bot);font-style:italic}.chip.prov{border:1.5px dashed var(--muted)}
.desc{font-size:13px;color:var(--muted);margin:2px 0 6px}footer{color:var(--muted);font-size:13px;margin-top:8px}
@media (max-width:560px){.stats{grid-template-columns:1fr}.game{grid-template-columns:80px 36px 1fr 40px}h1{font-size:26px}}
"""


def _e(text: object) -> str:
    return html.escape(str(text))


def _page(title: str, body: str) -> str:
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1"><title>{_e(title)}</title>'
            f"<style>{_CSS}</style></head><body><main>{body}</main></body></html>")  # fmt: skip


def _tier_badge(tier: str, size: int = 44) -> str:
    return f'<span class="badge" style="background:var(--{tier});width:{size}px;height:{size}px">{tier}</span>'


def tier_list_html(players: list[Player], highlight: str | None = None, name_others: bool = True) -> str:
    bot_name = players[0].bot_name if players else "the bot"
    rows = []
    for tier, _, description in TIERS:
        chips = []
        entries = [(p.rating.rating, p) for p in players if p.rating.tier == tier]
        if tier == "B":
            entries.append((BOT_RATING, None))
        for rating, p in sorted(entries, key=lambda e: -e[0]):
            if p is None:
                chips.append(f'<span class="chip bot">{_e(bot_name)} (the bot) · {BOT_RATING}</span>')
                continue
            me = p.name == highlight
            label = p.name if (me or name_others) else "Another player"
            classes = "chip" + (" me" if me else "") + (" prov" if p.rating.provisional else "")
            chips.append(f'<span class="{classes}">{_e(label)} · {p.rating.rating}</span>')
        rows.append(
            f'<div class="tier">{_tier_badge(tier)}<div><div class="desc">{_e(description)}</div>'
            f'<div class="chips">{"".join(chips) or "<span class=desc>nobody yet</span>"}</div></div></div>'
        )
    return (f'<div class="tiers">{"".join(rows)}</div><p class="desc" style="margin-top:10px">'
            f"Dashed outline: fewer than {PROVISIONAL_BELOW} games, still provisional.</p>")  # fmt: skip


def player_card(player: Player, everyone: list[Player], name_others: bool = False) -> str:
    r = player.rating
    wins = sum(g.won is True for g in player.games)
    losses = sum(g.won is False for g in player.games)
    low, high = r.interval
    early = (f'<div class="note"><b>Early read.</b> Based on {_n(r.battles, "game")}. Ratings settle after '
             f"{PROVISIONAL_BELOW} or more; until then treat the tier as a first guess.</div>") if r.provisional else ""  # fmt: skip
    header = f"""
<section class="card"><h1>{_e(player.name)}</h1>
<p class="sub">Your games against {_e(player.bot_name)}, a Pokémon Showdown bot · Gen 9 Random Battles</p>
<div class="stats">
 <div class="stat"><div class="k">Tier</div><div class="v">{_tier_badge(r.tier, 40)}</div>
  <div class="x">{_e(next(d for t, _, d in TIERS if t == r.tier))}</div></div>
 <div class="stat"><div class="k">Power level</div><div class="v">{r.power_level}</div>
  <div class="bar"><span style="width:{r.power_level}%"></span></div>
  <div class="x">Your chance to beat the bot with even teams</div></div>
 <div class="stat"><div class="k">Rating</div><div class="v">{r.rating}</div>
  <div class="x">Likely {low}-{high}. Bot = {BOT_RATING}</div></div>
</div>
<div class="note"><b>Record:</b> {wins} won, {losses} lost.</div>{early}</section>"""

    measured = [g for g in player.games if g.team_chance is not None]
    luck = ""
    if measured:
        stronger = sum(g.team_chance >= 0.5 for g in measured)
        rows = "".join(
            f'<div class="game"><span>{_e(g.day)}</span><span class="res {"w" if g.won else "l"}">{"Won" if g.won else "Lost"}</span>'
            f'<div class="mini"><i></i><span style="width:{round(100 * g.team_chance)}%"></span></div><span>{round(100 * g.team_chance)}%</span></div>'
            for g in sorted(measured, key=lambda g: g.day)
        )
        unmeasured = len(player.games) - len(measured)
        extra = f" {_n(unmeasured, 'older game')} couldn't be measured." if unmeasured else ""
        luck = f"""<section class="card"><h2>Team luck</h2>
<p>Random teams aren't equally strong. Each matchup was replayed hundreds of times with the bot playing both sides;
the bar is your team's chance of winning on the teams alone (50% = an even draw).</p>
<p><b>Your team was the stronger one in {stronger} of {len(measured)} games.</b>{extra} Your rating already accounts for this.</p>
<div class="games">{rows}</div></section>"""

    style = play_style(player.style)
    style_html = (f'<section class="card"><h2>How you play</h2><ul>{"".join(f"<li>{_e(s)}</li>" for s in style)}</ul>'
                  '<p class="desc" style="margin-top:10px">Matchups judged by types alone: &quot;against you&quot; means the bot&#39;s '
                  "Pokémon could hit yours super-effectively and yours couldn't do the same back.</p></section>") if style else ""  # fmt: skip
    won = how_they_won(player)
    won_html = f'<section class="card"><h2>How you beat the bot</h2><ul>{"".join(f"<li>{_e(s)}</li>" for s in won)}</ul></section>' if won else ""
    tip = f'<section class="card"><h2>Something to try</h2><p style="margin:0">{_e(something_to_try(player))}</p></section>'
    tiers = f'<section class="card"><h2>Tier list</h2>{tier_list_html(everyone, player.name, name_others)}</section>'
    footer = (f"<footer>Ratings compare everyone through their games against the same bot, adjusted for how strong each "
              f"side's team was: a win with a weaker team counts for more. With few games, ratings stay close to the bot's "
              f"{BOT_RATING}. Made {date.today():%d %B %Y}.</footer>")  # fmt: skip
    return _page(f"{player.name} vs {player.bot_name}", header + luck + style_html + won_html + tip + tiers + footer)


def tier_list_page(players: list[Player]) -> str:
    rows = "".join(
        f"<tr><td>{_tier_badge(p.rating.tier, 28)}</td><td>{_e(p.name)}{' <span class=desc>(provisional)</span>' if p.rating.provisional else ''}</td>"
        f"<td>{p.rating.rating}</td><td>{p.rating.interval[0]}-{p.rating.interval[1]}</td><td>{p.rating.power_level}</td>"
        f"<td>{p.rating.wins}-{p.rating.battles - p.rating.wins}</td></tr>"
        for p in sorted(players, key=lambda p: -p.rating.rating)
    )
    table = ('<table style="width:100%;border-collapse:collapse;font-size:14px"><thead><tr style="text-align:left;color:var(--muted)">'
             "<th></th><th>Player</th><th>Rating</th><th>Likely range</th><th>Power</th><th>W-L vs bot</th></tr></thead>"
             f"<tbody>{rows}</tbody></table>")  # fmt: skip
    body = (f'<section class="card"><h1>Tier list</h1><p class="sub">Everyone who has played {_e(players[0].bot_name if players else "the bot")}, '
            "rated through those games and adjusted for team luck.</p></section>"
            f'<section class="card">{tier_list_html(players)}</section><section class="card" style="overflow-x:auto">{table}</section>'
            f"<footer>Made {date.today():%d %B %Y}. Fewer than {PROVISIONAL_BELOW} games: provisional.</footer>")  # fmt: skip
    return _page("Tier list", body)
