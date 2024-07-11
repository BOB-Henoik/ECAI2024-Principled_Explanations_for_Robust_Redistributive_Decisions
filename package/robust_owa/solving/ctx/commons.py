"""Functions :
    - building the CTX computation function from a given alternative, reduction factor and 
    redistributive transfers computation functions.
    - computing the delta+ and delta- variation of criteria from a given Farkas certificate."""
from numpy import sum as npsum
from numpy import zeros as npzeros
from package.plot.types import GIFT, PREFERENTIAL_INFORMATION
from package import timeout_decorator


def compute_deltas(nb_var, pi_statements, nu_minus, nu_plus, mu, lmbd):
    """Computes the positive and negative evolution of criteria by the given Farkas certificate.

    Args:
        nb_var (int): Number of criteria.
        pi_statements (NDArray): Matrix containing the preferential information
        statements. Each row contains one statement.
        nu_minus (ArrayLike): Giving part of the redistributive transfer computed by
        the farkas certificate.
        nu_plus (ArrayLike): Receiving part of the redistributive transfer computed by
        the farkas certificate.
        mu (ArrayLike): Gift computed by the farkas certificate.
        lmbd (ArrayLike): Magnitude of the PI statements used in the farkas certificate.
    """
    delta_plus = npzeros(nb_var) + mu + nu_plus
    delta_minus = npzeros(nb_var) - nu_minus
    for k, l in enumerate(lmbd):
        if l > 0.0:
            for i in range(nb_var):
                if pi_statements[k][i] > 0.0:
                    delta_plus[i] += l * pi_statements[k][i]
                elif pi_statements[k][i] < 0.0:
                    delta_minus[i] += l * pi_statements[k][i]
    return delta_plus, delta_minus


def ctx_from_farkas_factory(
    congruence_computation,
    reduction_factor_computation,
    redistributive_transfer_computation,
):
    @timeout_decorator
    def ctx_from_farkas(
        looser, winner, pi_statements, low, high, ndigits, nu_minus, nu_plus, mu, lmbd
    ):
        """Builds the (G u PT u I)-CTX for robust redistributive OWA dominance between
        two candidates from the values given by the Farkas certificate.
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
        nb_var = len(looser)
        delta_plus, delta_minus = compute_deltas(
            nb_var, pi_statements, nu_minus, nu_plus, mu, lmbd
        )
        c = reduction_factor_computation(
            looser, winner, delta_plus, delta_minus, low, high
        )
        x, y = congruence_computation(looser, winner, c, delta_plus, delta_minus, low)

        explanation = [x.copy()]
        explanation_len = 0
        explanation_symbols = []
        if npsum(mu) > 10 ** (-ndigits - 2):
            x += c * mu
            explanation.append(x.copy())
            explanation_len += 1
            explanation_symbols.append(GIFT)

        for k, l in enumerate(lmbd):
            if l > 10 ** (-ndigits - 2):
                x += c * l * pi_statements[k]
                explanation.append(x.copy())
                explanation_len += 1
                explanation_symbols.append(PREFERENTIAL_INFORMATION)

        if npsum(nu_minus) > 10 ** (-ndigits - 2):
            # print(x)
            # print(y)
            # print(y - x)
            length, expl, symbols = redistributive_transfer_computation(
                looser=x,
                winner=y,
                low=low,
                high=high,
                ndigits=ndigits + 3,
            )
            explanation_len += length
            explanation += expl[1:]
            explanation_symbols += symbols

        return explanation_len, explanation, explanation_symbols

    return ctx_from_farkas
