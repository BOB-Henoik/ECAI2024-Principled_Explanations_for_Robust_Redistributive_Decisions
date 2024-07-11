"""Function providing the (G u PT)-ATX for generalized Lorenz dominance using 
our contribution algorithm first and a unique gift afterwards"""
from package.restricted_lorenz import contribution_heuristics
from package.plot import GIFT
from package import timeout_decorator

FILE_NAME = "Generalized\\after_contrib.csv"


@timeout_decorator
def gift_after_contribution_heuristics(looser, winner, ndigits: int = 0):
    """Builds the explanation for generalized Lorenz dominance between two
    candidates using our contribution algorithm first and
    applying a unique gift afterwards.
    Returns the length, the explanation and the symbols for display.

    Args:
        looser (ArrayLike): First candidate.
        winner (ArrayLike): Second candidate.
        ndigits (int, optional): Precision (number of digit after the coma).
        The default value is 0.
    """
    expl_len, explanation, symbols = contribution_heuristics(looser, winner, ndigits)
    expl_len += 1
    explanation.append(winner)
    symbols.append(GIFT)
    return expl_len, explanation, symbols
