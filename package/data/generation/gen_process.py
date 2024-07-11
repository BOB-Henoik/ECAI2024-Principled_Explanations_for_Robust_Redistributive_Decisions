"""Functions generating :
    - array of integer candidates with and without fixed total sum 
    - array of float candidates rounded with precision
"""
from os import makedirs
from os.path import exists as pathexists
from itertools import combinations
from math import ceil
from typing import List, Tuple
from numpy.random import default_rng
from numpy import zeros, int_, array_equiv, float64
from numpy import sum as npsum
from numpy import array as nparray
from package.data.generation import (
    sorted_int_candidate_factory,
    sorted_fixed_sum_int_candidate_factory,
    sorted_uniform_candidate_factory,
    pareto_dominance,
    true_rowa,
)
from package.generalized_lorenz import generalized_lorenz_dominance
from package.restricted_lorenz import restricted_lorenz_dominance
from package.robust_owa import compute_redistributive_owa_dominance
from package.data.save import save_data, save_meta_data


def generation_type(
    nb_var: int,
    nb_cand: int,
    low,
    high,
    epsilon,
    max_sum=0.0,
    fixed_sum: int = 0,
    ndigits: int = 0,
    seed: int = 404,
):
    """Wrapper for calling right generation process given the value of the parameters.
    If the precision does not have its default value, the countinuous generation is called.
    Else, if the fixed sum has its default value, the integer generation with fixed total
    sum is called.
    Else, the integer generation is called.

    Args:
        path (str): Path to the root of the experiment folder.
        nb_fold (int): Number of repeat of the generation process.
        nb_var (int): Number of criteria.
        nb_cand (int): Number of candidates to generate.
        nb_pi (int): Number of preferential information to generate from precise
        redistributive OWA.
        low (int | float): Lower boundary of the output interval.
        All values generated will be greater than or equal to low.
        high (int | float): Upper boundary of the output interval.
        All values generated will be less than or equal to high.
        The high limit may be included.
        epsilon (int | float): Minimum separation from interval bounds.
        max_sum (int | float): Maximum value for the sum of criteria of a candidate.
        Avoids infinite loop when a Pareto dominating candidate is generated in the set.
        Defaults to 2/3 * nb_var * high.
        fixed_sum (int, optional): Value for the total sum of criteria of a candidate.
        ndigits (int, optional): Precision (number of digit after the coma).
        The default value is 3.
        seed (int, optional): Seed value for fixing randomness. The default value is 404.
    """
    if ndigits == 0:
        if fixed_sum == 0:
            return gen_int_candidates(
                nb_var, nb_cand, low, high, epsilon, int(max_sum), seed
            )
        return gen_int_candidates_fixed_sum(
            nb_var, nb_cand, low, high, epsilon, fixed_sum, seed
        )
    return gen_uniform_candidates(
        nb_var, nb_cand, low, high, epsilon, ndigits, max_sum, seed
    )


def generation_process(
    path: str,
    nb_fold: int,
    nb_var: int,
    nb_cand: int,
    nb_pi: int,
    low,
    high,
    epsilon,
    max_sum=0.0,
    fixed_sum: int = 0,
    ndigits: int = 0,
    seed: int = 404,
):
    """Wrapper for the generation process. Generates for each repetition the
    candidates, the set of Lorenz dominance pairs, the meta data file,
    the redistributive OWA and preferetial information statements and saves them in csv files.

    Args:
        path (str): Path to the root of the experiment folder.
        nb_fold (int): Number of repeat of the generation process.
        nb_var (int): Number of criteria.
        nb_cand (int): Number of candidates to generate.
        nb_pi (int): Number of preferential information to generate from precise
        redistributive OWA.
        low (int | float): Lower boundary of the output interval.
        All values generated will be greater than or equal to low.
        high (int | float): Upper boundary of the output interval.
        All values generated will be less than or equal to high.
        The high limit may be included.
        epsilon (int | float): Minimum separation from interval bounds.
        max_sum (int | float): Maximum value for the sum of criteria of a candidate.
        Avoids infinite loop when a Pareto dominating candidate is generated in the set.
        Defaults to 2/3 * nb_var * high.
        fixed_sum (int, optional): Value for the total sum of criteria of a candidate.
        ndigits (int, optional): Precision (number of digit after the coma).
        The default value is 3.
        seed (int, optional): Seed value for fixing randomness. The default value is 404.
    """
    if not pathexists(path):
        makedirs(path)

    save_meta_data(
        path,
        nb_fold,
        nb_cand,
        nb_pi,
        nb_var,
        low,
        high,
        epsilon,
        max_sum,
        fixed_sum,
        ndigits,
    )

    for i in range(nb_fold):
        fold_path = f"{path}\\{i}"
        if not pathexists(f"{fold_path}\\Restricted"):
            makedirs(f"{fold_path}\\Restricted")
        if not pathexists(f"{fold_path}\\Generalized"):
            makedirs(f"{fold_path}\\Generalized")
        if not pathexists(f"{fold_path}\\RobustOWA"):
            makedirs(f"{fold_path}\\RobustOWA")

        data = generation_type(
            nb_var, nb_cand, low, high, epsilon, max_sum, fixed_sum, ndigits, seed + i
        )
        save_data(fold_path, "candidates", data)
        redistributive_owa = gen_rowa(nb_var, seed + i)
        save_data(fold_path, "true_owa", redistributive_owa)
        restricted_lorenz_dom, generalized_lorenz_dom = lorenz_dominances(data)
        save_data(
            f"{fold_path}\\Restricted", "restricted_lorenz_dom", restricted_lorenz_dom
        )
        save_data(
            f"{fold_path}\\Generalized",
            "generalized_lorenz_dom",
            generalized_lorenz_dom,
        )
        owa_dom = owa_dominances(data, redistributive_owa)
        save_data(fold_path, "owa_dominances", owa_dom)
        pi, pi_statements = gen_pi_statements(
            nb_pi,
            data,
            redistributive_owa,
            restricted_lorenz_dom,
            generalized_lorenz_dom,
            seed + i,
        )
        save_data(fold_path, "pi_indexes", pi)
        save_data(fold_path, "pi_statements", pi_statements)
        rowa_dom = compute_redistributive_owa_dominance(
            data, pi_statements, ndigits, restricted_lorenz_dom, generalized_lorenz_dom
        )
        save_data(f"{fold_path}\\RobustOWA", "rowa_dominances", rowa_dom)


def gen_int_candidates(
    nb_var: int,
    nb_cand: int,
    low: int = 0,
    high: int = 100,
    epsilon: int = 1,
    max_sum: int = 0,
    seed: int = 404,
):
    """Generates a ndarray of non Pareto-dominated candidates with integer values on criteria

    Args:
        nb_var (int): Number of criteria.
        nb_cand (int): Number of candidates to generate.
        low (int, optional): Lower boundary of the output interval.
        All values generated will be greater than or equal to low. The default value is 0.
        high (int, optional): Upper boundary of the output interval.
        All values generated will be less than or equal to high.
        The high limit may be included. The default value is 100.
        epsilon (int, optional): Minimum separation from interval bounds.
        The default value is 1.
        max_sum (int, optional): Maximum value for the sum of criteria of a candidate.
        Avoids infinite loop when a Pareto dominating candidate is generated in the set.
        Defaults to 2/3 * nb_var * high.
        seed (int, optional): Seed value for fixing randomness. The default value is 404.
    """
    rng = default_rng(seed)
    max_sum = ceil(2 / 3 * nb_var * high) if max_sum == 0 else max_sum
    sorted_int_candidate = sorted_int_candidate_factory(rng)
    candidates = zeros((nb_cand, nb_var), dtype=int_)
    for i in range(nb_cand):
        valid = False
        while not valid:
            new_cand = sorted_int_candidate(nb_var, low, high, epsilon)
            if npsum(new_cand) <= max_sum and not any(
                pareto_dominance(x, new_cand) or pareto_dominance(new_cand, x)
                for x in candidates[0:i, :]
            ):
                candidates[i] = new_cand
                valid = True

    return candidates


def gen_int_candidates_fixed_sum(
    nb_var: int,
    nb_cand: int,
    low: int = 0,
    high: int = 100,
    epsilon: int = 1,
    fixed_sum: int = 50,
    seed: int = 404,
):
    """Generates a ndarray of non Pareto-dominated candidates with integer values on criteria

    Args:
        nb_var (int): Number of criteria.
        nb_cand (int): Number of candidates to generate.
        low (int, optional): Lower boundary of the output interval.
        All values generated will be greater than or equal to low. The default value is 0.
        high (int, optional): Upper boundary of the output interval.
        All values generated will be less than or equal to high.
        The high limit may be included. The default value is 100.
        epsilon (int, optional): Minimum separation from interval bounds.
        The default value is 1.
        fixed_sum (int, optional): Value for the total sum of criteria of a candidate.
        seed (int, optional): Seed value for fixing randomness. The default value is 404.
    """
    rng = default_rng(seed)
    sorted_fixed_sum_int_candidate = sorted_fixed_sum_int_candidate_factory(
        rng, fixed_sum
    )
    candidates = zeros((nb_cand, nb_var), dtype=int_)
    for i in range(nb_cand):
        valid = False
        while not valid:
            new_cand = sorted_fixed_sum_int_candidate(nb_var, low, high, epsilon)
            if not any(array_equiv(x, new_cand) for x in candidates[0:i, :]):
                candidates[i] = new_cand
                valid = True

    return candidates


def gen_uniform_candidates(
    nb_var: int,
    nb_cand: int,
    low: float = 0.0,
    high: float = 1.0,
    epsilon: float = 1e-3,
    ndigits: int = 3,
    max_sum: float = 0.0,
    seed: int = 404,
):
    """Generates a ndarray of non Pareto-dominated candidates with integer values on criteria

    Args:
        nb_var (int): Number of criteria.
        nb_cand (int): Number of candidates to generate.
        low (float, optional): Lower boundary of the output interval.
        All values generated will be greater than or equal to low. The default value is 0.
        high (float, optional): Upper boundary of the output interval.
        All values generated will be less than or equal to high.
        The high limit may be included. The default value is 1.
        epsilon (int, optional): Minimum separation from interval bounds.
        The default value is 1e-3.
        ndigit (int, optional): Precision (number of digit after the coma).
        The default value is 3.
        max_sum (float, optional): Maximum value for the sum of criteria of a candidate.
        Avoids infinite loop when a Pareto dominating candidate is generated in the set.
        Defaults to 2/3 * nb_var * high.
        seed (int, optional): Seed value for fixing randomness. The default value is 404.
    """
    rng = default_rng(seed)
    max_sum = 2 / 3 * high * nb_var if max_sum == 0.0 else max_sum
    sorted_uniform_candidate = sorted_uniform_candidate_factory(rng)
    candidates = zeros((nb_cand, nb_var), dtype=float64)
    for i in range(nb_cand):
        valid = False
        while not valid:
            new_cand = sorted_uniform_candidate(nb_var, low, high, epsilon, ndigits)
            if npsum(new_cand) <= max_sum and not any(
                pareto_dominance(x, new_cand) or pareto_dominance(new_cand, x)
                for x in candidates[0:i, :]
            ):
                candidates[i] = new_cand
                valid = True

    return candidates


def lorenz_dominances(data):
    """Returns the list of pairs (i,j) such that j Lorenz dominates i.

    Args:
        data (NDArray): Dataset of candidates.
    """
    nb_cand = data.shape[0]
    generalized_dom = []
    restricted_dom = []
    for i in range(nb_cand):
        for j in range(i + 1, nb_cand):
            if restricted_lorenz_dominance(data[i], data[j]):
                restricted_dom.append((i, j))
            if restricted_lorenz_dominance(data[j], data[i]):
                restricted_dom.append((j, i))
            if generalized_lorenz_dominance(data[i], data[j]):
                generalized_dom.append((i, j))
            if generalized_lorenz_dominance(data[j], data[i]):
                generalized_dom.append((j, i))

    return restricted_dom, generalized_dom


def gen_rowa(nb_var: int, seed: int = 404):
    """Returns from the Diriclet distribution a redistributive Ordered Weighted Average.

    Args:
        nb_var (int): Number of criteria.
        seed (int, optional): Seed value for fixing randomness. The default value is 404.
    """
    rng = default_rng(seed)
    return true_rowa(nb_var, rng)


def gen_pi_statements(
    nb_pi: int,
    data,
    redistributive_owa,
    restricted_dom: List[Tuple[int, int]],
    generalized_dom: List[Tuple[int, int]],
    seed: int = 404,
):
    """Generates Preferential Information compatible with the given redistributive OWA.
    Returns the indexes of concerned candidates and the values of the associated statements.

    Args:
        nb_pi (int): Number of Preferential Information statements to generate.
        data (NDArray): Dataset of candidates.
        redistributive_owa (ArrayLike): Value of the redistributive OWA operator.
        restricted_dom (List[Tuple[int,int]]): List of Restricted Lorenz statements in data.
        generalized_dom (List[Tuple[int,int]]): List of Generalized Lorenz statements in data.
        seed (int, optional): Seed value for fixing randomness. The default value is 404.
    """
    rng = default_rng(seed)
    nb_cand, nb_var = data.shape
    pi = zeros((nb_pi, 2), dtype=int_)
    pi_statement = zeros((nb_pi, nb_var))
    for i in range(nb_pi):
        valid = False
        while not valid:
            (u, v) = rng.integers(0, nb_cand, 2)
            if (
                u != v
                and (u, v) not in pi
                and (v, u) not in pi
                and (u, v) not in restricted_dom
                and (v, u) not in restricted_dom
                and (u, v) not in generalized_dom
                and (v, u) not in generalized_dom
            ):
                score = (data[v] - data[u]) @ redistributive_owa
                if score > 0.0:
                    pi_statement[i] = data[v] - data[u]
                    pi[i] = (u, v)
                    valid = True
                elif score < 0.0:
                    pi_statement[i] = data[u] - data[v]
                    pi[i] = (v, u)
                    valid = True

    return pi, pi_statement


def owa_dominances(data, redistributive_owa):
    """Returns the list of pairs (i,j) such that j owa dominates i.
    (The relation is complete)

    Args:
        data (NDArray): Dataset of candidates.
        redistributive_owa (ArrayLike): Weights of the precise redistributive OWA.
    """
    nb_cand = data.shape[0]
    owa_dom = []
    for i, j in combinations(range(nb_cand), 2):
        score_diff = (data[j] - data[i]) @ redistributive_owa
        if score_diff == 0.0:
            owa_dom.append((i, j))
            owa_dom.append((j, i))
        elif score_diff > 0.0:
            owa_dom.append((i, j))
        else:
            owa_dom.append((j, i))

    return owa_dom


def generation_example(path: str):
    """Wrapper for the generation process. Generates for each repetition the
    candidates, the set of Lorenz dominance pairs, the meta data file,
    the redistributive OWA and preferetial information statements and saves them in csv files.

    Args:
        path (str): Path to the root of the experiment folder.
        nb_fold (int): Number of repeat of the generation process.
        nb_var (int): Number of criteria.
        nb_cand (int): Number of candidates to generate.
        nb_pi (int): Number of preferential information to generate from precise
        redistributive OWA.
        low (int | float): Lower boundary of the output interval.
        All values generated will be greater than or equal to low.
        high (int | float): Upper boundary of the output interval.
        All values generated will be less than or equal to high.
        The high limit may be included.
        epsilon (int | float): Minimum separation from interval bounds.
        max_sum (int | float): Maximum value for the sum of criteria of a candidate.
        Avoids infinite loop when a Pareto dominating candidate is generated in the set.
        Defaults to 2/3 * nb_var * high.
        fixed_sum (int, optional): Value for the total sum of criteria of a candidate.
        ndigits (int, optional): Precision (number of digit after the coma).
        The default value is 3.
        seed (int, optional): Seed value for fixing randomness. The default value is 404.
    """
    if not pathexists(path):
        makedirs(path)

    save_meta_data(
        path,
        1,
        4,
        1,
        5,
        0,
        100,
        0,
        0,
        0,
        0,
    )

    fold_path = f"{path}\\0"
    if not pathexists(f"{fold_path}\\Restricted"):
        makedirs(f"{fold_path}\\Restricted")
    if not pathexists(f"{fold_path}\\Generalized"):
        makedirs(f"{fold_path}\\Generalized")
    if not pathexists(f"{fold_path}\\RobustOWA"):
        makedirs(f"{fold_path}\\RobustOWA")

    data = nparray(
        [
            [16, 31, 51, 70, 83],
            [2, 25, 28, 84, 98],
            [22, 23, 34, 76, 82],
            [6, 17, 18, 88, 96],
        ],
        dtype=int_,
    )
    save_data(fold_path, "candidates", data)
    redistributive_owa = nparray([0.615385, 0.142308, 0.142308, 0.05, 0.05])
    save_data(fold_path, "true_owa", redistributive_owa)
    restricted_lorenz_dom, generalized_lorenz_dom = lorenz_dominances(data)
    save_data(
        f"{fold_path}\\Restricted", "restricted_lorenz_dom", restricted_lorenz_dom
    )
    save_data(
        f"{fold_path}\\Generalized",
        "generalized_lorenz_dom",
        generalized_lorenz_dom,
    )
    owa_dom = owa_dominances(data, redistributive_owa)
    save_data(fold_path, "owa_dominances", owa_dom)
    pi = nparray([[1, 3]], dtype=int_)
    pi_statements = nparray([data[3] - data[1]], dtype=float64)
    save_data(fold_path, "pi_indexes", pi)
    save_data(fold_path, "pi_statements", pi_statements)
    rowa_dom = compute_redistributive_owa_dominance(
        data, pi_statements, 0, restricted_lorenz_dom, generalized_lorenz_dom
    )
    save_data(f"{fold_path}\\RobustOWA", "rowa_dominances", rowa_dom)
