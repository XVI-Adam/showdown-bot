"""Opponent analytics: what the people we have played tend to do.

Built only from what battle logs show. See ``psbot.analytics.opponents``.
"""

from psbot.analytics.opponents import (
    MIN_BATTLES_FOR_PROFILE,
    BattleView,
    build_profile,
    build_profiles,
    load_battles,
    render_profile,
)

__all__ = [
    "MIN_BATTLES_FOR_PROFILE",
    "BattleView",
    "build_profile",
    "build_profiles",
    "load_battles",
    "render_profile",
]
