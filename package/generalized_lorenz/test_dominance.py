"""Functions :
    - building Lorenz vector
    - checking Generalized Lorenz dominance
"""
from numpy import cumsum as npcumsum
from numpy import all as npall


def lorenz_vector(a):
    """Returns the Lorenz vector of the given candidate

    Args:
        a (ArrayLike): Vector of values of the candidate.
    """
    return npcumsum(a)


def generalized_lorenz_dominance(looser, winner):
    """Returns true if the second argument Lorenz dominates first argument.

    Args:
        looser (ArrayLike): First candidate.
        winner (ArrayLike): Second candidate.
    """
    a, b = lorenz_vector(winner), lorenz_vector(looser)
    return npall(a >= b) and a[-1] > b[-1]
