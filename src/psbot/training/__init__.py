"""Offline learning: datasets of decisions, behaviour cloning, opponent adaptation.

Nothing here touches the production agents. A trained model is saved as a new
checkpoint and has to beat the existing agents in fresh battles before anyone
should play it (``scripts/evaluate.py``).

* ``dataset``              (observation, legal-action mask, action) arrays,
                           split by battle so no battle feeds two splits
* ``bc``                   behaviour cloning into the same masked policy the
                           PPO agents use, so the result loads like any checkpoint
* ``opponent_adaptation``  does knowing who we play predict what they do?
"""
