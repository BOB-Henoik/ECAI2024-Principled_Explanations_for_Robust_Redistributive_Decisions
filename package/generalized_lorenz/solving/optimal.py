"""Functions :
    - providing the shortest (G u PT)-ATX for generalized Lorenz dominance using 
MILP solving
    - yielding the MILP formulation
    - adds constraints for gifts in the MILP"""
from math import ceil
from multiprocessing.context import TimeoutError as TimedOut
from numpy import sum as npsum
from numpy import array as nparray
from numpy import ones as npones
from numpy import int_, float64
from gurobipy import GRB, Model, MVar
from gurobipy import quicksum as gpquicksum
from package.plot import REDISTRIBUTIVE_TRANSFER, GIFT
from package.restricted_lorenz.solving import (
    positive_negative_modification_indexes,
    add_candidate_for_step_factory,
    add_ordered_candidate_constraint_factory,
    add_redistributive_constraints_factory,
)
from package import timeout_decorator

FILE_NAME = "Generalized\\optim.csv"


@timeout_decorator
def generalized_optimum(looser, winner, low, high, ndigits):
    """Builds the shortest explanation for restricted Lorenz dominance between
    two candidates.
    Returns the length, the explanation and the symbols for display.

    Args:
        looser (ArrayLike): First candidate.
        winner (ArrayLike): Second candidate.
    """
    minimum_k = (
        len(positive_negative_modification_indexes(looser, winner, ndigits)[0]) + 1
    )
    k = minimum_k
    model_generator = build_generalized_base_model(
        looser, winner, minimum_k, low, high, ndigits
    )
    m = next(model_generator)
    found = False

    while not found:
        m.update()
        m.optimize()
        m.display()
        if m.Status == GRB.TIME_LIMIT:
            raise TimedOut
        if m.status == GRB.OPTIMAL:
            nb_var = len(looser)
            explanation = []
            symbols = []
            explanation.append(
                nparray(
                    [
                        m.getVarByName(name).X
                        for name in (f"x0[{i}]" for i in range(nb_var))
                    ],
                    dtype=int_ if ndigits == 0 else float64,
                )
            )
            for step in range(1, k + 1):
                explanation.append(
                    nparray(
                        [
                            m.getVarByName(name).X
                            for name in (f"x{step}[{i}]" for i in range(nb_var))
                        ],
                        dtype=int_ if ndigits == 0 else float64,
                    )
                )
                symbols.append(
                    GIFT
                    if m.getVarByName(f"g{step}").X != 0.0
                    else REDISTRIBUTIVE_TRANSFER
                )
            found = True
        else:
            m = next(model_generator)
            k += 1

    m.dispose()
    return k, explanation, symbols


def build_generalized_base_model(
    looser, winner, minimum_k: int, low, high, ndigits: int = 0
):
    """Generator returning the MILP model.
    Updated at each call by adding a new step to the explanation search.
    Starts with a minimum number of steps.

    Args:
        looser (ArrayLike): First candidate.
        winner (ArrayLike): Second candidate.
        minimum_k (int): Number of steps to be initialised.
        low (int | float): Lower boundary of the definition domain.
        high (int | float): Upper boundary of the definition domain.
        ndigits (int, optional): Precision (number of digit after the coma).
        The default value is 0.
    """
    m = Model("ExactKExplanation")
    m.Params.LogToConsole = 0
    m.Params.TimeLimit = 150
    # m.Params.MIPFocus = 1

    if ndigits == 0:
        cand_type = GRB.INTEGER
    else:
        cand_type = GRB.CONTINUOUS
        m.Params.FeasibilityTol = 10 ** (-ndigits - 2)

    nb_var: int = winner.shape[0]
    nb_redistributive_transfers: int = (nb_var * nb_var - nb_var) // 2
    big_m_1: int = ceil(npsum(winner) + npsum(looser))

    add_redistributive_constraints = add_redistributive_constraints_factory(
        nb_var,
        cand_type,
        big_m_1,
        nb_redistributive_transfers,
    )

    add_candidate_for_step = add_candidate_for_step_factory(
        nb_var, cand_type, low, high
    )

    add_ordered_candidate_constraint = add_ordered_candidate_constraint_factory(nb_var)

    add_gift_for_step = add_gift_for_step_factory(nb_var, cand_type)
    add_gift_use_constraint = add_gift_use_constraint_factory(nb_var, big_m_1)

    # Initial state
    x: MVar = add_candidate_for_step(m, 0)
    m.addConstr(x == looser, name="X0")

    for step in range(1, minimum_k + 1):
        # Create redistributive transfers
        nu_plus, nu_minus, transfer = add_redistributive_constraints(m, step)

        # Create gifts
        mu_k = add_gift_for_step(m, step)
        g_k = add_gift_use_constraint(m, mu_k, step)

        # Create variable
        xk = add_candidate_for_step(m, step)
        if step < minimum_k:
            add_ordered_candidate_constraint(m, xk, step)

        # Explanation steps
        m.addConstr(
            x + mu_k + nu_plus - nu_minus == xk,
            name=f"StepsATX{step}",
        )

        # One action by step
        m.addConstr(
            npones((1, nb_redistributive_transfers)) @ transfer + g_k <= 1,
            name=f"AtMost1ArgumentStep{step}",
        )

        x = xk

    # Final states
    m.addConstr(x == winner, name="Xk")
    m.update()
    g = (m.getVarByName(f"g{step}") for step in range(1, minimum_k + 1))
    coef = (max(1, 2 * min(i, minimum_k - 1 - i)) for i in range(minimum_k))
    m.setObjective(gpquicksum([x * y for x, y in zip(g, coef)]), GRB.MINIMIZE)
    # print(m.getObjective())

    yield m

    for step in range(minimum_k + 1, nb_var + 1):
        add_ordered_candidate_constraint(m, x, step - 1)
        xk = add_candidate_for_step(m, step)
        nu_plus, nu_minus, transfer = add_redistributive_constraints(m, step)
        mu_k = add_gift_for_step(m, step)
        g_k = add_gift_use_constraint(m, mu_k, step)
        m.addConstr(
            x + mu_k + nu_plus - nu_minus == xk,
            name=f"StepsATX{step}",
        )
        m.addConstr(
            npones((1, nb_redistributive_transfers)) @ transfer + g_k <= 1,
            name=f"AtMost1ArgumentStep{step}",
        )

        m.remove(
            [m.getConstrByName(name) for name in (f"Xk[{i}]" for i in range(nb_var))]
        )
        x = xk
        m.addConstr(x == winner, name="Xk")
        m.update()

        g = (m.getVarByName(f"g{s}") for s in range(1, step + 1))
        coef = (max(1, 2 * min(i, step - 1 - i)) for i in range(step))
        m.setObjective(gpquicksum([x * y for x, y in zip(g, coef)]), GRB.MINIMIZE)
        # print(m.getObjective())

        yield m


def add_gift_for_step_factory(nb_var: int, cand_type):
    """Builds the add_gift_for_step function from parameters deduced
    from the candidates.

    Args:
        nb_var (int): Number of criteria.
        cand_type (int_ | float64): Data type of the candidate criteria.
    """

    def add_gift_for_step(m: Model, step: int):
        """Adds and returns the gift variable in the MILP for the given step.

        Args:
            m (Model): Gurobi MILP.
            step (int): Number of the step introducing the gift.
        """
        return m.addMVar(shape=nb_var, vtype=cand_type, lb=0.0, name=f"mu{step}")

    return add_gift_for_step


def add_gift_use_constraint_factory(nb_var: int, big_m_g: int):
    """Builds the add_gift_use_constraint function from the number of criteria
    and the "Big M" constant.

    Args:
        nb_var (int): Number of criteria.
        big_m_g (int): Value of the "Big M" for gifts.
    """

    def add_gift_use_constraint(m: Model, muk: MVar, step: int):
        """Add to the MILP the variable and the constraint for capturing the use of gifts.

        Args:
            m (Model): Gurobi MILP.
            muk (MVar): New candidate variables whose values will be constrained.
            step (int): Number of the step introducing the new candidate.
        """
        g_k = m.addVar(vtype=GRB.BINARY, name=f"g{step}")
        m.addConstr(
            big_m_g * g_k - npones((1, nb_var)) @ muk >= 0,
            name=f"Gift{step}",
        )
        return g_k

    return add_gift_use_constraint
