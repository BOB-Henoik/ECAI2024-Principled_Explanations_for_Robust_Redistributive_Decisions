"""Function providing the PT-ATX for restricted Lorenz dominance using 
the [Hardy, Littlewood,Poly;1934] algorithm"""
from numpy import where as npwhere
from numpy import argwhere as npargwhere
from numpy import max as npmax
from numpy import delete as npdelete
from numpy import equal as npequal
from numpy import all as npall
from package.plot import REDISTRIBUTIVE_TRANSFER
from .commons import (
    positive_negative_modification_indexes,
    redistributive_transfer,
    largest_redistributive_transfer,
)
from package import timeout_decorator

FILE_NAME = "Restricted\\hlp.csv"


@timeout_decorator
def hardy_littlewood_polya(looser, winner, ndigits: int = 0):
    """Builds the explanation for restricted Lorenz dominance between two
    candidates using the [Hardy, Littlewood,Poly;1934] algorithm.
    Returns the length, the explanation and the symbols for display.

    Args:
        looser (ArrayLike): First candidate.
        winner (ArrayLike): Second candidate.
        ndigits (int, optional): Precision (number of digit after the coma).
        The default value is 0.
    """
    threshold = 0.5 if ndigits == 0 else 2 * 10 ** (-ndigits - 1)

    def recurcive_hlp(a, b, n, p):
        """Recurcive process of the explanation generation.
        Takes the first giving index and match it with the biggest smaller
        receiving index with the largest trade possible.

        Args:
            a (ArrayLike): First candidate.
            b (ArrayLike): Second candidate.
            n (ArrayLike): Indexes where the looser has to give.
            p (ArrayLike): Indexes where the looser has to receive.
        """
        if npall(npequal(a, b)) or len(n) == 0:
            return []

        j = n[0]
        i = npmax(p[npwhere(p < j)])
        epsilon = largest_redistributive_transfer(a, b, i, j)
        updated_cand = redistributive_transfer(a, i, j, epsilon)

        if abs(b[i] - a[i] - epsilon) < threshold:
            p = npdelete(p, npargwhere(p == i))
        if abs(a[j] - b[j] - epsilon) < threshold:
            n = npdelete(n, npargwhere(n == j))
        explanation_end = recurcive_hlp(updated_cand, b, n, p)
        explanation_end.append(updated_cand)
        return explanation_end

    neg, pos = positive_negative_modification_indexes(looser, winner, ndigits)
    explanation = recurcive_hlp(looser, winner, neg, pos)
    expl_len = len(explanation)
    explanation.append(looser)
    explanation.reverse()
    expl_symbols = [REDISTRIBUTIVE_TRANSFER] * expl_len

    return expl_len, explanation, expl_symbols
