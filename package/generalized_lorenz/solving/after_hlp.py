"""Function providing the (G u PT)-ATX for generalized Lorenz dominance using 
the [Hardy, Littlewood,Poly;1934] algorithm first and a unique gift afterwards"""
from package.plot import GIFT
from package.restricted_lorenz import hardy_littlewood_polya
from package import timeout_decorator

FILE_NAME = "Generalized\\after_hlp.csv"


@timeout_decorator
def gift_after_hardy_littlewood_polya(looser, winner, ndigits: int = 0):
    """Builds the explanation for generalized Lorenz dominance between two
    candidates using the [Hardy, Littlewood,Poly;1934] algorithm first and
    applying a unique gift afterwards.
    Returns the length, the explanation and the symbols for display.

    Args:
        looser (ArrayLike): First candidate.
        winner (ArrayLike): Second candidate.
        ndigits (int, optional): Precision (number of digit after the coma).
        The default value is 0.
    """
    expl_len, explanation, symbols = hardy_littlewood_polya(looser, winner, ndigits)
    expl_len += 1
    explanation.append(winner)
    symbols.append(GIFT)
    return expl_len, explanation, symbols
