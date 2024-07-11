"""Function providing the first Farkas lemma for robust redistributive OWA dominance"""
from multiprocessing.context import TimeoutError as TimedOut
from numpy import array as nparray
from numpy import ones as npones
from numpy import int_, float64, transpose
from scipy.sparse import spmatrix
from scipy.sparse import diags as spdiags
from gurobipy import GRB, Model, MVar
from package import timeout_decorator

FARKAS_NAME = "first_farkas"


@timeout_decorator
def first_farkas(looser, winner, ndigits: int, preferential_information):
    """Builds a Farkas lemma of the robust OWA dominance between first
    and second candidate.

    Args:
        looser (ArrayLike): First candidate.
        winner (ArrayLike): Second candidate.
        ndigits (int, optional): Precision (number of digit after the coma).
        preferential_information (NDArray): Matrix containing the preferential information
        statements. Each row contains one statement.
    """
    m = Model("FirstFarkas")
    m.Params.LogToConsole = 0
    m.Params.TimeLimit = 150
    m.Params.MIPFocus = 1
    nb_var: int = winner.shape[0]
    nb_pi: int = preferential_information.shape[0]

    # if ndigits == 0:
    #     mu = m.addMVar(shape=nb_var, vtype=GRB.INTEGER, lb=0.0, name="mu")
    #     nu = m.addMVar(shape=nb_var, vtype=GRB.INTEGER, lb=0.0, name="nu")
    # else:
    m.Params.FeasibilityTol = 10 ** (-ndigits - 3)
    m.Params.IntFeasTol = 10 ** (-ndigits - 3)
    m.Params.IntegralityFocus = 1
    mu = m.addMVar(shape=nb_var, vtype=GRB.CONTINUOUS, lb=0.0, name="mu")
    nu = m.addMVar(shape=nb_var, vtype=GRB.CONTINUOUS, lb=0.0, name="nu")

    lmbd: MVar = m.addMVar(shape=nb_pi, vtype=GRB.CONTINUOUS, lb=0.0, name="lambda")

    bidiag: spmatrix = spdiags([npones(nb_var), -npones(nb_var - 1)], [0, -1])
    m.addConstr(
        transpose(preferential_information) @ lmbd + mu + bidiag @ nu
        == winner - looser,
        name="Farkas",
    )

    # if ndigits == 0:
    #     lmbd_int = m.addMVar(
    #         shape=(nb_pi, nb_var), vtype=GRB.INTEGER, lb=0.0, name="IntLmbd"
    #     )
    #     m.addConstrs(
    #         (
    #             lmbd_int[j, :] == preferential_information[j, :] * lmbd[j]
    #             for j in range(nb_pi)
    #         )
    #     )

    m.optimize()
    if m.Status == GRB.TIME_LIMIT:
        raise TimedOut

    farkas_lmbd = nparray(lmbd.X)
    farkas_mu = nparray(mu.X, dtype=float64 if ndigits else int_)

    fuzed_nu = bidiag @ nparray(nu.X)
    farkas_nu_plus = nparray(
        [x if x > 0 else 0 for x in fuzed_nu], dtype=float64 if ndigits else int_
    )
    farkas_nu_minus = -nparray(
        [x if x < 0 else 0 for x in fuzed_nu], dtype=float64 if ndigits else int_
    )

    return farkas_nu_minus, farkas_nu_plus, farkas_mu, farkas_lmbd
