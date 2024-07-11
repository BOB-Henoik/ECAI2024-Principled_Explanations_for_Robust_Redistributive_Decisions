"""Functions :
    - building the receiving and giving indexes
    - computing the largest redistributive transfer possible between two indexes
    - applying a redistributive transfer on a candidate
"""
from numpy import argwhere, ravel
from numpy import min as npmin
from package.restricted_lorenz.test_dominance import lorenz_vector


def positive_negative_modification_indexes(looser, winner, ndigits):
    """Returns the set of indexes where the looser has to receive (pos)
    and where the looser has to give (neg) to become the winner.

    Args:
        looser (ArrayLike): First candidate.
        winner (ArrayLike): Second candidate.
    """
    neg = ravel(argwhere(looser - winner > 10 ** (-ndigits - 2)))
    pos = ravel(argwhere(looser - winner < -(10 ** (-ndigits - 2))))
    return neg, pos


def largest_redistributive_transfer(a, b, i: int, j: int):
    """Returns the largest amount of a redistributive transfer from index j to i.
    Must be the minimum between :
        - The total amount to be received
        - The total amount to be given
        - The maximum amount which can be received at this step
        - The maximum amount which can be given at this step
        - The maximum amount which can be taken from j and given to i while keeping Lorenz dominance
    Note : the only the two  first are relevent in the HLP algorithm

    Args:
        a (ArrayLike): First candidate (looser).
        b (ArrayLike): Second candidate (winner).
        i (ArrayLike): Chosen where the looser receives.
        j (ArrayLike): Chosen index where the looser gives.
    """
    lor_a, lor_b = lorenz_vector(a), lorenz_vector(b)
    lor_diff = lor_b - lor_a
    return min(
        b[i] - a[i],
        a[j] - b[j],
        a[i + 1] - a[i],
        a[j] - a[j - 1],
        npmin(lor_diff[i:j]),
    )


def redistributive_transfer(a, i: int, j: int, epsilon):
    """Return the candidate after applying the given Progressive Transfer.

    Args:
        a (ArrayLike): Initial candidate.
        i (int): Index of the criteria to receive the transfer.
        j (int): Index of the criteria to give the transfer.
        epsilon (int | float): Magnitude of the transfer
    """
    b = a.copy()
    b[i] += epsilon
    b[j] -= epsilon
    return b
