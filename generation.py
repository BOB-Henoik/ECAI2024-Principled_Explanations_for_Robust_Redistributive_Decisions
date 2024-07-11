"""Main example aiming at generating data for 3 scenarios:
    - candidates with integer values (i.e. oriented toward generalized Lorenz dominance)
    - candidates with integer values with fixed sum (i.e. oriented toward restricted 
    Lorenz dominance)
    - candidates with continuous values (i.e. oriented toward redistributive OWA)
"""
from sys import stderr
from package.data.generation import generation_process


def process_int(
    path: str,
    nb_fold: int,
    nb_var: int,
    nb_cand: int,
    low: int = 0,
    high: int = 100,
    epsilon: int = 1,
    nb_pi: int = 0,
    max_sum: int = 0,
    seed: int = 404,
):
    """Generates and save an experiment with integer valued candidates.

    Args:
        path (str): Path to the root of the experiment folder.
        nb_fold (int): Number of repeat of the generation process.
        nb_var (int): Number of criteria.
        nb_cand (int): Number of candidates to generate.
        low (int, optional): Lower boundary of the output interval.
        All values generated will be greater than or equal to low. The default value is 0.
        high (int, optional): Upper boundary of the output interval.
        All values generated will be less than or equal to high.
        The high limit may be included. The default value is 100.
        epsilon (int, optional): Minimum separation from interval bounds.
        The default value is 1.
        nb_pi (int, optional): Number of preferential information to generate from precise
        redistributive OWA. The default value is 0.
        max_sum (int, optional): Maximum value for the sum of criteria of a candidate.
        Avoids infinite loop when a Pareto dominating candidate is generated in the set.
        Defaults to 2/3 * nb_var * high.
        seed (int, optional): Seed value for fixing randomness. The default value is 404.
    """
    return generation_process(
        path=path,
        nb_fold=nb_fold,
        nb_pi=nb_pi,
        nb_var=nb_var,
        nb_cand=nb_cand,
        low=low,
        high=high,
        epsilon=epsilon,
        max_sum=max_sum,
        seed=seed,
    )


def process_int_fixed(
    path: str,
    nb_fold: int,
    nb_var: int,
    nb_cand: int,
    fixed_sum: int,
    low: int = 0,
    high: int = 100,
    epsilon: int = 1,
    nb_pi: int = 0,
    seed: int = 404,
):
    """Generates and save an experiment with integer valued candidates with the same
    total sum on criteria.

    Args:
        path (str): Path to the root of the experiment folder.
        nb_fold (int): Number of repeat of the generation process.
        nb_var (int): Number of criteria.
        nb_cand (int): Number of candidates to generate.
        fixed_sum (int): Value for the total sum of criteria of a candidate.
        low (int, optional): Lower boundary of the output interval.
        All values generated will be greater than or equal to low. The default value is 0.
        high (int, optional): Upper boundary of the output interval.
        All values generated will be less than or equal to high.
        The high limit may be included. The default value is 100.
        epsilon (int, optional): Minimum separation from interval bounds.
        The default value is 1.
        nb_pi (int, optional): Number of preferential information to generate from precise
        redistributive OWA. The default value is 0.
        seed (int, optional): Seed value for fixing randomness. The default value is 404.
    """
    return generation_process(
        path=path,
        nb_fold=nb_fold,
        nb_pi=nb_pi,
        nb_var=nb_var,
        nb_cand=nb_cand,
        low=low,
        high=high,
        epsilon=epsilon,
        max_sum=0,
        fixed_sum=fixed_sum,
        seed=seed,
    )


def process_float(
    path: str,
    nb_fold: int,
    nb_var: int,
    nb_cand: int,
    low: float = 0.0,
    high: float = 1.0,
    epsilon: float = 1e-3,
    ndigits: int = 3,
    nb_pi: int = 0,
    max_sum: float = 0.0,
    seed: int = 404,
):
    """Generates and save an experiment with continuous valued candidates.

    Args:
        path (str): Path to the root of the experiment folder.
        nb_fold (int): Number of repeat of the generation process.
        nb_var (int): Number of criteria.
        nb_cand (int): Number of candidates to generate.
        fixed_sum (int): Value for the total sum of criteria of a candidate.
        low (float, optional): Lower boundary of the output interval.
        All values generated will be greater than or equal to low. The default value is 0.
        high (float, optional): Upper boundary of the output interval.
        All values generated will be less than or equal to high.
        The high limit may be included. The default value is 1.
        epsilon (float, optional): Minimum separation from interval bounds.
        The default value is 1e-3.
        ndigits (int, optional): Precision (number of digit after the coma).
        The default value is 3.
        nb_pi (int, optional): Number of preferential information to generate from precise
        redistributive OWA. The default value is 0.
        max_sum (int, optional): Maximum value for the sum of criteria of a candidate.
        Avoids infinite loop when a Pareto dominating candidate is generated in the set.
        Defaults to 2/3 * nb_var * high.
        seed (int, optional): Seed value for fixing randomness. The default value is 404.
    """
    return generation_process(
        path=path,
        nb_fold=nb_fold,
        nb_pi=nb_pi,
        nb_var=nb_var,
        nb_cand=nb_cand,
        low=low,
        high=high,
        epsilon=epsilon,
        ndigits=ndigits,
        max_sum=max_sum,
        seed=seed,
    )


if __name__ == "__main__":
    print("Experimetation generation", file=stderr)
    NB_VAR = 5
    NB_FOLD = 1
    NB_CAND = 10
    NB_PI = 2

    EXP_PATH = f".\\Datasets\\Int_{NB_VAR}cri_{NB_CAND}cand"
    process_int(
        path=EXP_PATH, nb_fold=NB_FOLD, nb_var=NB_VAR, nb_cand=NB_CAND, nb_pi=NB_PI
    )

    FIXED_SUM = NB_VAR * 20
    EXP_PATH = f".\\Datasets\\Int_{NB_VAR}cri_{FIXED_SUM}sum_{NB_CAND}cand"
    process_int_fixed(
        path=EXP_PATH,
        nb_fold=NB_FOLD,
        nb_var=NB_VAR,
        nb_cand=NB_CAND,
        nb_pi=NB_PI,
        fixed_sum=FIXED_SUM,
    )

    EXP_PATH = f".\\Datasets\\Float_{NB_VAR}cri_{NB_CAND}cand"
    process_float(
        path=EXP_PATH,
        nb_fold=NB_FOLD,
        nb_var=NB_VAR,
        nb_cand=NB_CAND,
        nb_pi=NB_PI,
    )
