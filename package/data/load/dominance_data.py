"""Functions loading :
    - restricted Lorenz dominance pairs
    - generalized Lorenz dominance pairs
    - precise redistributive owa dominance pairs
    - robust redistributive owa dominance pairs
"""
from csv import reader


def load_indexes_pairs(file: str):
    """Loads pairs of dominance candidates indexes from csv file.

    Args:
        path (str): Path to the experiment's csv folder.
    """
    with open(file, "r", newline="", encoding="utf8") as f:
        for x in reader(f, delimiter=";"):
            yield (int(x[0]), int(x[1]))


def load_restricted_lorenz_dominances(exp_path: str):
    """Loads pairs of restricted Lorenz dominance candidates indexes from csv file.

    Args:
        path (str): Path to the experiment's csv folder.
    """
    return load_indexes_pairs(f"{exp_path}\\Restricted\\restricted_lorenz_dom.csv")


def load_generalized_lorenz_dominances(exp_path: str):
    """Loads pairs of generalized Lorenz dominance candidates indexes from csv file.

    Args:
        path (str): Path to the experiment's csv folder.
    """
    return load_indexes_pairs(f"{exp_path}\\Generalized\\generalized_lorenz_dom.csv")


def load_robust_redistributive_owa_dominances(exp_path: str):
    """Loads pairs of robust redistributive OWA dominance candidates indexes from csv file.

    Args:
        path (str): Path to the experiment's csv folder.
    """
    return load_indexes_pairs(f"{exp_path}\\RobustOWA\\rowa_dominances.csv")


def load_redistributive_owa_dominances(exp_path: str):
    """Loads pairs of robust redistributive OWA dominance candidates indexes from csv file.

    Args:
        path (str): Path to the experiment's csv folder.
    """
    return load_indexes_pairs(f"{exp_path}\\owa_dominances.csv")
