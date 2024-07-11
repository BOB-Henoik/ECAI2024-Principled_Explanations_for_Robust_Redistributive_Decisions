"""Function providing the smallest (in terms of non zeros variables) Farkas lemma 
for robust redistributive OWA dominance"""
from typing import Tuple
from math import ceil
from multiprocessing.context import TimeoutError as TimedOut
from numpy import sum as npsum
from numpy import array as nparray
from numpy import ones as npones
from numpy import float64, transpose, zeros
from gurobipy import GRB, Model, MVar, Var
from package import timeout_decorator

FARKAS_NAME = "min_farkas"


@timeout_decorator
def minimum_length_farkas(
    looser, winner, ndigits: int, preferential_information
) -> Tuple:
    """Builds the smallest (in terms of non zeros variables) Farkas lemma
    of the robust OWA dominance between first and second candidate.

    Args:
        looser (ArrayLike): First candidate.
        winner (ArrayLike): Second candidate.
        preferential_information (NDArray): Matrix containing the preferential information
        statements. Each row contains one statement.
        ndigits (int, optional): Precision (number of digit after the coma).
    """
    m = Model("MinLengthFarkas")
    m.Params.LogToConsole = 0
    m.Params.TimeLimit = 150

    # if ndigits != 0:
    m.Params.FeasibilityTol = 10 ** (-ndigits - 3)
    m.Params.IntFeasTol = 10 ** (-ndigits - 3)
    m.Params.IntegralityFocus = 1
    nb_var: int = winner.shape[0]
    nb_pi: int = preferential_information.shape[0]
    nb_redistributive_transfers: int = (nb_var * nb_var - nb_var) // 2
    big_m_rt = ceil(npsum(winner) + npsum(looser))
    big_m_pi: int = 1000

    mu: MVar = m.addMVar(shape=nb_var, vtype=GRB.CONTINUOUS, lb=0.0, name="mu")
    mu_norm: Var = m.addVar(vtype=GRB.BINARY, name="mu_norm")
    lmbd_norm: MVar = m.addMVar(shape=nb_pi, vtype=GRB.BINARY, name="lambda_norm")
    lmbd: MVar = m.addMVar(shape=nb_pi, vtype=GRB.CONTINUOUS, lb=0.0, name="lambda")
    transfer = m.addMVar(
        shape=(nb_redistributive_transfers), vtype=GRB.CONTINUOUS, name="transfers"
    )
    transfer_norm: Var = m.addVar(lb=0, ub=nb_var, name="transfer_norm")

    matrix = zeros((nb_var, nb_redistributive_transfers))
    for i in range(nb_var):
        positive_index = [
            i + ((k - 1) ** 2 - (k - 1)) // 2 for k in range(i + 2, nb_var + 1)
        ]
        negative_index = [k - 1 + ((i) ** 2 - (i)) // 2 for k in range(1, i + 1)]
        matrix[i][positive_index] = 1
        matrix[i][negative_index] = -1

    m.addConstr(
        transpose(preferential_information) @ lmbd
        + mu
        + matrix @ transfer
        - (winner - looser)
        == 0,
        name="Farkas",
    )
    m.addConstr(
        big_m_rt * mu_norm - npones((1, nb_var)) @ mu >= 0,
        name="Mu norm",
    )
    # m.addConstr(npones((1, nb_var)) @ lmbd_norm == 2)
    # m.addGenConstrNorm(lmbd_norm, lmbd, 0, "Lambda norm")
    m.addConstr(
        big_m_pi * lmbd_norm - lmbd >= 0,
        name="Lambda norm",
    )

    m.addGenConstrNorm(transfer_norm, transfer, 0, "Transfer norm")

    m.setObjective(
        npones((1, nb_pi)) @ lmbd_norm + transfer_norm + mu_norm, GRB.MINIMIZE
    )
    m.optimize()
    if m.Status == GRB.TIME_LIMIT:
        raise TimedOut

    farkas_lmbd = nparray(lmbd.X, dtype=float64)
    farkas_mu = nparray(mu.X, dtype=float64)

    fuzed_nu = nparray(matrix @ transfer.X)
    farkas_nu_plus = nparray([x if x > 0 else 0 for x in fuzed_nu], dtype=float64)
    farkas_nu_minus = -nparray([x if x < 0 else 0 for x in fuzed_nu], dtype=float64)
    # print("Farkas :")
    # print(farkas_lmbd)
    # print(farkas_mu)
    # print(-farkas_nu_minus)
    # print(farkas_nu_plus)
    # print(f"F obj {m.getObjective().getValue()}")
    # print(lmbd_norm.X)
    # print(transfer_norm.X)
    # print(mu_norm.X)
    return farkas_nu_minus, farkas_nu_plus, farkas_mu, farkas_lmbd
