"""Function providing the PT-ATX for restricted Lorenz dominance using 
our contribution algorithm"""
from functools import reduce
from numpy import where as npwhere
from numpy import argwhere as npargwhere
from numpy import argmax as npargmax
from numpy import delete as npdelete
from numpy import equal as npequal
from numpy import all as npall
from numpy import round as npround
from numpy import array as nparray
from package import timeout_decorator
from package.plot import REDISTRIBUTIVE_TRANSFER
from .commons import (
    positive_negative_modification_indexes,
    redistributive_transfer,
    largest_redistributive_transfer,
)

FILE_NAME = "Restricted\\contribution_algo.csv"


@timeout_decorator
def contribution_heuristics(looser, winner, ndigits: int = 0, low=None, high=None):
    """Builds the explanation for restricted Lorenz dominance between two
    candidates using our cautious contribution algorithm.
    Returns the length, the explanation and the symbols for display.

    Args:
        looser (ArrayLike): First candidate.
        winner (ArrayLike): Second candidate.
        ndigits (int, optional): Precision (number of digit after the coma).
        The default value is 0.
    """
    threshold = 0.5 if ndigits == 0 else 2 * 10 ** (-ndigits - 1)
    nb_var = len(looser)

    def recurcive_contribution(a, b, n, p):
        """Recurcive process of the explanation generation.
        Performs the biggest trade available at each step.

        Args:
            a (ArrayLike): First candidate.
            b (ArrayLike): Second candidate.
            n (ArrayLike): Indexes where the looser has to give.
            p (ArrayLike): Indexes where the looser has to receive.
        """
        if npall(npequal(a, b)) or len(n) == 0:
            return []

        resolvable_n = nparray(
            [
                j
                for j in n
                if (ndigits != 0 and a[j] - a[j - 1] - (a[j] - b[j]) >= threshold)
                or (a[j] - a[j - 1] >= a[j] - b[j])
            ]
        )
        resolvable_p = nparray(
            [
                i
                for i in p
                if i == nb_var - 1
                or (ndigits != 0 and a[i + 1] - a[i] - (b[i] - a[i]) >= threshold)
                or (a[i + 1] - a[i] >= b[i] - a[i])
            ]
        )

        def best_transfer(j):
            i = npargmax(
                [
                    largest_redistributive_transfer(a, b, i, j)
                    for i in resolvable_p[npwhere(resolvable_p < j)]
                ]
            )
            return (
                resolvable_p[i],
                j,
                largest_redistributive_transfer(a, b, resolvable_p[i], j),
            )

        i, j, epsilon = reduce(
            lambda t_1, t_2: (t_1[0], t_1[1], t_1[2])
            if t_1[2] - t_2[2] >= 0
            else (t_2[0], t_2[1], t_2[2]),
            (best_transfer(j) for j in resolvable_n),
        )
        if ndigits != 0:
            epsilon = round(epsilon, ndigits)

        updated_cand = (
            npround(redistributive_transfer(a, i, j, epsilon), ndigits)
            if ndigits != 0
            else redistributive_transfer(a, i, j, epsilon)
        )

        if abs(b[i] - a[i] - epsilon) < threshold:
            p = npdelete(p, npargwhere(p == i))
        if abs(a[j] - b[j] - epsilon) < threshold:
            n = npdelete(n, npargwhere(n == j))
        explanation_end = recurcive_contribution(updated_cand, b, n, p)
        explanation_end.append(updated_cand)
        return explanation_end

    neg, pos = positive_negative_modification_indexes(looser, winner, ndigits)
    explanation = recurcive_contribution(looser, winner, neg, pos)
    expl_len = len(explanation)
    explanation.append(looser)
    explanation.reverse()
    expl_symbols = [REDISTRIBUTIVE_TRANSFER] * expl_len

    return expl_len, explanation, expl_symbols
