"""Functions generating :
    - integer candidate with and without fixed total sum 
    - float candidate rounded with precision
    - redistributive OWA weights
Function for pareto dominance checking
"""
from numpy import sort as npsort
from numpy import all as npall
from numpy import round as npround
from numpy import zeros


def sorted_fixed_sum_int_candidate_factory(rng, total_sum: int):
    """Builds the gen_fixed_sum_int_candidate function
    from the given numpy random Generator and total sum.

    Args:
        rng (Generator): Numpy random Generator.
        total_sum (int): Value of the sum of candidate criteria.
    """

    def fixed_sum_int_candidate(
        nb_var: int, low: int = 0, high: int = 100, epsilon: int = 1
    ):
        """Generates an ordered candidate with integer values of fixed sum over criteria.

        Args:
            nb_var (int): Number of criteria.
            low (int, optional): Lower boundary of the output interval.
            All values generated will be greater than or equal to low. The default value is 0.
            high (int, optional): Upper boundary of the output interval.
            All values generated will be less than or equal to high.
            The high limit may be included. The default value is 100.
            epsilon (float, optional): Minimum separation from interval bounds.
            The default value is 1.
        """
        cand = zeros(nb_var, dtype=int)
        total_remaining = total_sum
        effective_low = low + epsilon
        effective_high = high - epsilon
        for i in range(nb_var):
            low_value = max(
                effective_low, total_remaining - (nb_var - i - 1) * effective_high
            )
            high_value = min(
                effective_high, total_remaining - (nb_var - i - 1) * effective_low
            )
            cand[i] = rng.integers(low_value, high_value, endpoint=True)
            total_remaining -= cand[i]
        return npsort(cand)

    return fixed_sum_int_candidate


def sorted_uniform_candidate_factory(rng):
    """Builds the gen_sorted_uniform_candidate function from the given numpy random Generator.

    Args:
        rng (Generator): Numpy random Generator.
    """

    def sorted_uniform_candidate(
        nb_var: int,
        low: float = 0.0,
        high: float = 1.0,
        epsilon: float = 1e-3,
        ndigits: int = 3,
    ):
        """Generates an ordered candidate from uniform distribution.

        Args:
            nb_var (int): Number of criteria.
            low (float, optional): Lower boundary of the output interval.
            All values generated will be greater than or equal to low. The default value is 0.
            high (float, optional): Upper boundary of the output interval.
            All values generated will be less than or equal to high.
            The high limit may be included in the returned array of floats due to floating-point
            rounding. The default value is 1.0.
            epsilon (float, optional): Minimum separation from interval bounds.
            The default value is 1e-3.
            ndigits (int, optional): Precision (number of digit after the coma).
            The default value is 3.
        """
        return npround(
            npsort(rng.uniform(low + epsilon, high - epsilon, nb_var)),
            ndigits,
        )

    return sorted_uniform_candidate


def sorted_int_candidate_factory(rng):
    """Builds the gen_sorted_int_candidate function from the given numpy random Generator

    Args:
        rng (Generator): numpy random Generator
    """

    def sorted_int_candidate(
        nb_var: int, low: int = 0, high: int = 100, epsilon: int = 1
    ):
        """Generates an ordered candidate with integer values on criteria

        Args:
            nb_var (int): Number of criteria.
            low (int, optional): Lower boundary of the output interval.
            All values generated will be greater than or equal to low. The default value is 0.
            high (int, optional): Upper boundary of the output interval.
            All values generated will be less than or equal to high.
            The high limit may be included. The default value is 100.
            epsilon (int, optional): Minimum separation from interval bounds.
            The default value is 1.
        """
        return npsort(
            rng.integers(low + epsilon, high - epsilon, nb_var, endpoint=True)
        )

    return sorted_int_candidate


def pareto_dominance(x1, x2) -> bool:
    """Returns True if x1 (weekly) Pareto dominates x2"""
    if npall(x1 >= x2):
        return True
    return False


def true_rowa(nb_var: int, rng):
    """Returns from the Diriclet distribution a redistributive Ordered Weighted Average"""
    return npsort(rng.dirichlet([nb_var] * nb_var))[::-1]
