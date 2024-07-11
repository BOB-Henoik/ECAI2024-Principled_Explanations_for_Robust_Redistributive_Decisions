"""Function :
    - checking Restricted Lorenz dominance
"""
from numpy import all as npall
from package.generalized_lorenz import lorenz_vector


def restricted_lorenz_dominance(looser, winner):
    """Returns true if the second argument restricted Lorenz dominates first argument.

    Args:
        looser (ArrayLike): First candidate.
        winner (ArrayLike): Second candidate.
    """
    a, b = lorenz_vector(winner), lorenz_vector(looser)
    return npall(a >= b) and a[-1] == b[-1]
