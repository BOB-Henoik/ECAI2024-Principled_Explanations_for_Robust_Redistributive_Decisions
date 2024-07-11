"""Functions :
    - giving the file name from the farkas computation method used
    - providing a (G u PT u I)-CTX for robust redistributive OWA dominance using 
MILP solving by moving toward the origin."""
from decimal import Decimal, ROUND_DOWN
from sys import float_info
from numpy import zeros as npzeros
from numpy import float64
from package.restricted_lorenz.solving import restricted_optimum
from package.robust_owa.solving.ctx.commons import ctx_from_farkas_factory


def file_name(farkas_name: str):
    """Returns the file name of the method from the farkas computation name.

    Args:
        farkas_name (str): Farkas certificate function name.
    """
    return f"RobustOWA\\{farkas_name}_ctx_displaced.csv"


def alternatives_displaced(looser, winner, c, delta_plus, delta_minus, low):
    """Computes the new alternatives x' and y' by using translation and homothety.

    Args:
        looser (ArrayLike): Values of the first candidate on criteria.
        winner (ArrayLike): Values of the second candidate on criteria.
        c (int | float): Reduction factor to apply for the new alternatives to remain in the
        definition domain.
        delta_plus (ArrayLike): Negative evolution of the criteria given by the Farkas certificate.
        delta_minus (ArrayLike): Positive evolution of the criteria given by the Farkas certificate.
        low (int | float): Lower boundary of the definition domain.
    """
    nb_var = len(winner)
    y = npzeros(nb_var, dtype=float64)
    x = npzeros(nb_var, dtype=float64)
    y[0] = low + c * delta_plus[0]
    x[0] = low - c * delta_minus[0]
    cumsum_plus = 0
    for i in range(1, nb_var):
        cumsum_plus += delta_plus[i - 1]
        x[i] = x[i - 1] + c * (cumsum_plus - delta_minus[i])
        y[i] = x[i - 1] + c * (cumsum_plus + delta_plus[i])
    return x, y


def compute_reduction_factor_displaced(
    looser, winner, delta_plus, delta_minus, low, high
) -> float:
    """Compute the reduction factor c enabling the intermediate candidate of the CTX to
    belong to the definition domain.

    Args:
        looser (ArrayLike): Values of the first candidate on criteria.
        winner (ArrayLike): Values of the second candidate on criteria.
        delta_plus (ArrayLike): Negative evolution of the criteria given by the Farkas certificate.
        delta_minus (ArrayLike): Positive evolution of the criteria given by the Farkas certificate.
        low (int | float): Lower boundary of the definition domain.
        high (int | float): Upper boundary of the definition domain.
    """
    c = 1
    x, y = alternatives_displaced(looser, winner, c, delta_plus, delta_minus, low)
    if max(x[-1], y[-1]) > high:
        c = (high - low) / max(x[-1], y[-1])

    if c != 1:
        precision = 1
        valid = False
        while not valid:
            new_c = Decimal(c).quantize(
                Decimal("1." + "".zfill(precision)), rounding=ROUND_DOWN
            )
            if new_c <= 2 * float_info.epsilon:
                # print(f"new_c: {new_c}")
                precision += 1
                continue
            valid = True
            c = round(float(new_c), precision)
    # print(f"c: {c}")
    return c


def ctx_from_farkas_displaced(
    looser, winner, pi_statements, low, high, ndigits, nu_minus, nu_plus, mu, lmbd
):
    """Builds the (G u PT u I)-CTX for robust redistributive OWA dominance between
    two candidates by using translation and homothety so that the Farkas certificate
    could be applyed.
    Returns the length, the explanation and the symbols for display.

    Args:
        looser (ArrayLike): Values of the first candidate on criteria.
        winner (ArrayLike): Values of the second candidate on criteria.
        pi_statements (NDArray): Matrix containing the preferential information
        statements. Each row contains one statement.
        low (int | float): Lower boundary of the definition domain.
        high (int | float): Upper boundary of the definition domain.
        ndigits (int): Precision (number of digit after the coma).
        nu_minus (ArrayLike): Giving part of the redistributive transfer computed by
        the farkas certificate.
        nu_plus (ArrayLike): Receiving part of the redistributive transfer computed by
        the farkas certificate.
        mu (ArrayLike): Gift computed by the farkas certificate.
        lmbd (ArrayLike): Magnitude of the PI statements used in the farkas certificate.
    """
    ctx_from_farkas = ctx_from_farkas_factory(
        alternatives_displaced, compute_reduction_factor_displaced, restricted_optimum
    )

    return ctx_from_farkas(
        looser, winner, pi_statements, low, high, ndigits, nu_minus, nu_plus, mu, lmbd
    )
