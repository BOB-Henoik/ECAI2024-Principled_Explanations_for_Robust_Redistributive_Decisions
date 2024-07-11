"""Functions loading :
    - candidates from csv file in ndarray
    - problem meta parameters
"""
from csv import reader
from warnings import catch_warnings, simplefilter
from numpy import loadtxt, int_, float64


def load_dataset(exp_path: str, ndigits: int = 0):
    """Load candidates and Preferential Information statements from csv file.

    Args:
        exp_path (str): Path to the experiment's csv folder.
        ndigits (int, optional): Precision (number of digit after the coma).
        The default value is 0 (loads integer values).
    """
    with catch_warnings():
        simplefilter("ignore")
        dtype = int_ if ndigits == 0 else float64
        cand = loadtxt(f"{exp_path}\\candidates.csv", delimiter=";", dtype=dtype)
        pi = loadtxt(f"{exp_path}\\pi_statements.csv", delimiter=";", dtype=dtype)
        if len(pi.shape) == 1:
            pi = pi.reshape((1, pi.shape[0]))
    return cand, pi


def load_meta_data(path: str):
    """Load experiment meta data.

    Args:
        path (str): Path to the experiment's root folder.
    """
    with open(f"{path}\\meta.csv", "r", newline="", encoding="utf8") as f:
        [
            nb_exp,
            nb_cand,
            nb_pi,
            nb_var,
            low,
            high,
            epsilon,
            max_sum,
            fixed_sum,
            precision,
        ] = [
            float(x[0]) if "." in x[0] else int(x[0]) for x in reader(f, delimiter=";")
        ]
        return (
            nb_exp,
            nb_cand,
            nb_pi,
            nb_var,
            low,
            high,
            epsilon,
            max_sum,
            fixed_sum,
            precision,
        )
