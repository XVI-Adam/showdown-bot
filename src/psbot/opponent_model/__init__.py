"""What the opponent's Pokémon probably carry, as probabilities.

``sets``     candidate sets per species and a belief over them
"""

from psbot.opponent_model.sets import SetBelief, belief_for, set_belief

__all__ = ["SetBelief", "belief_for", "set_belief"]
