"""Functions building :
    - the function checking the robust redistributive OWA dominance
    - the Guroby Linear Model of the robust redistributive OWA
    - the set of robust redistributive OWA dominances
"""
from typing import List, Tuple
from itertools import permutations
from gurobipy import Model, GRB
from multiprocessing.context import TimeoutError as TimedOut
from numpy import ones as npones
from scipy.sparse import diags as spdiags


def robust_redistributive_owa_dominance_factory(lpmodel, w):
    """Builds the robust_redistributive_owa_dominance function from the LP Model.

    Args:
        lpmodel : Guroby robust redistributive OWA Model.
        w : Guroby variable for the OWA weights.
    """

    def robust_redistributive_owa_dominance(a, b):
        """Returns true if the second argument dominates first argument in the
        robust redistributive OWA.

        Args:
            a (ArrayLike): First candidate.
            b (ArrayLike): Second candidate.
        """
        lpmodel.setObjective((b - a) @ w, GRB.MINIMIZE)
        lpmodel.optimize()

        if lpmodel.status == GRB.TIME_LIMIT:
            raise TimedOut

        if lpmodel.status in (GRB.INF_OR_UNBD, GRB.INFEASIBLE, GRB.UNBOUNDED):
            return False
        return True

    return robust_redistributive_owa_dominance


def build_lpmodel(pi_statements, nb_var: int, ndigits: int):
    """Builds the LP Guroby model of the robust redistributive OWA and returns
    the dominance checking function associated

    Args:
        pi_statements (NDArray): Preferential information statements to apply as
        constraints in the robust redistributive OWA.
        nb_var (int): Number of criteria.
        ndigits (int): Precision (number of digit after the coma).
    """
    nb_pi = pi_statements.shape[0] if len(pi_statements) > 0 else 0
    lpmodel = Model("Dominance")
    lpmodel.Params.LogToConsole = 0
    lpmodel.Params.TimeLimit = 150
    if ndigits != 0:
        lpmodel.Params.FeasibilityTol = 10 ** (-ndigits - 2)

    w = lpmodel.addMVar(shape=nb_var, vtype=GRB.CONTINUOUS, lb=0.0, name="w")

    if nb_pi:
        lpmodel.addConstr(pi_statements @ w >= 0, name="PI")
    lpmodel.addConstr(
        spdiags([npones(nb_var), -npones(nb_var - 1)], [0, 1]) @ w >= 0,
        name="BalancedOWA",
    )

    return robust_redistributive_owa_dominance_factory(lpmodel, w)


def compute_redistributive_owa_dominance(
    data,
    pi_statements,
    ndigits: int,
    restricted_dom: List[Tuple[int, int]],
    generalized_dom: List[Tuple[int, int]],
):
    """Returns the list of pairs (i,j) such that j dominates i in the robust
    redistributive OWA obtained from the preferential information statements.

    Args:
        data (NDArray): Dataset of candidates.
        pi_statements (NDArray): Preferential information statements to apply as
        constraints in the robust redistributive OWA.
        ndigits (int): Precision (number of digit after the coma).
        restricted_dom (List[Tuple[int,int]]): List of Restricted Lorenz statements in data.
        generalized_dom (List[Tuple[int,int]]): List of Generalized Lorenz statements in data.
        Allows to return robust redistributive OWA only dominances.
    """
    nb_cand, nb_var = data.shape
    dom = []
    if len(pi_statements) > 0:
        robust_redistributive_owa_dominance = build_lpmodel(
            pi_statements, nb_var, ndigits
        )
        for a, b in permutations(range(nb_cand), 2):
            if (
                (a, b) not in restricted_dom
                and (a, b) not in generalized_dom
                and robust_redistributive_owa_dominance(data[a], data[b])
            ):
                dom.append((a, b))
    return dom
