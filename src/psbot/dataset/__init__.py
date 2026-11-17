"""The human-battle dataset: what is recorded, how, and how it is checked.

* ``revealed``  what the battle log has actually shown about each Pokémon
* ``human``     the recorder that writes one record per finished battle
* ``schema``    the record layout, its validator, and a converter to the
                older log format that ``psbot.analysis`` reads

Human battles live in ``data/human/``, apart from the synthetic battles the
bot plays against other bots (``logs/``). Both directories are git-ignored.
"""

from psbot.dataset.human import AgentInfo, HumanBattleRecorder, agent_info
from psbot.dataset.revealed import RevealTracker
from psbot.dataset.schema import SCHEMA, load_dataset, to_legacy_record, validate_record

__all__ = [
    "SCHEMA",
    "AgentInfo",
    "HumanBattleRecorder",
    "RevealTracker",
    "agent_info",
    "load_dataset",
    "to_legacy_record",
    "validate_record",
]
