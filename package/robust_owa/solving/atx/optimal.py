"""Functions :
    - providing the shortest (G u PT u CC(I))-ATX for robust redistributive OWA 
    dominance using MILP solving
    - yielding the MILP formulation
    - adds constraints for gifts in the MILP"""
from math import ceil
from multiprocessing.context import TimeoutError as TimedOut
from numpy import sum as npsum
from numpy import array as nparray
from numpy import ones as npones
from numpy import float64, transpose, argwhere
from gurobipy import GRB, Model, MVar
from package.plot import REDISTRIBUTIVE_TRANSFER, GIFT, PREFERENTIAL_INFORMATION
from package.restricted_lorenz.solving import (
    add_candidate_for_step_factory,
    add_ordered_candidate_constraint_factory,
    add_redistributive_constraints_factory,
)
from package.generalized_lorenz.solving import (
    add_gift_for_step_factory,
    add_gift_use_constraint_factory,
)
from package import timeout_decorator

FILE_NAME = "RobustOWA\\atx_optim.csv"


@timeout_decorator
def robust_optimum(looser, winner, low, high, ndigits, preferential_information):
    """Builds the shortest explanation for robust redistributive OWA dominance between
    two candidates.
    Returns the length, the explanation and the symbols for display.

    Args:
        looser (ArrayLike): First candidate.
        winner (ArrayLike): Second candidate.
    """
    minimum_k = 1
    k = minimum_k
    model_generator = build_robust_base_model(
        looser, winner, minimum_k, low, high, ndigits, preferential_information
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
            nb_pi = preferential_information.shape[0]
            explanation = []
            symbols = []
            explanation.append(
                nparray(
                    [
                        m.getVarByName(name).X
                        for name in (f"x0[{i}]" for i in range(nb_var))
                    ],
                    dtype=float64,
                )
            )
            for step in range(1, k + 1):
                explanation.append(
                    nparray(
                        [
                            m.getVarByName(name).X
                            for name in (f"x{step}[{i}]" for i in range(nb_var))
                        ],
                        dtype=float64,
                    )
                )
                # print(f"Step : {step}")
                # print(
                #     list(m.getVarByName(f"lambda{step}[{i}]").X for i in range(nb_pi))
                # )
                # print(list(m.getVarByName(f"nu+{step}[{i}]").X for i in range(nb_var)))
                # print(list(-m.getVarByName(f"nu-{step}[{i}]").X for i in range(nb_var)))
                # print(list(m.getVarByName(f"mu{step}[{i}]").X for i in range(nb_var)))
                if m.getVarByName(f"g{step}").X != 0.0:
                    symbols.append(GIFT)
                else:
                    pi_k = argwhere(
                        nparray(
                            [
                                m.getVarByName(name).X
                                for name in (f"pi{step}[{i}]" for i in range(nb_pi))
                            ]
                        )
                        > 0.0
                    )
                    if len(pi_k) > 0:
                        symbols.append(PREFERENTIAL_INFORMATION)
                    else:
                        symbols.append(REDISTRIBUTIVE_TRANSFER)
            found = True
        else:
            m = next(model_generator)
            k += 1

    m.dispose()
    return k, explanation, symbols


def build_robust_base_model(
    looser, winner, minimum_k: int, low, high, ndigits: int, preferential_information
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
        preferential_information (NDArray): Matrix containing the preferential information
        statements. Each row contains one statement.
    """
    m = Model("ExactKExplanation")
    m.Params.LogToConsole = 0
    m.Params.MIPFocus = 1
    m.Params.TimeLimit = 150

    if ndigits == 0:
        cand_type = GRB.INTEGER
    else:
        cand_type = GRB.CONTINUOUS
        m.Params.FeasibilityTol = 10 ** (-ndigits - 3)
        m.Params.IntFeasTol = 10 ** (-ndigits - 3)
        m.Params.IntegralityFocus = 1

    nb_var: int = winner.shape[0]
    nb_pi: int = preferential_information.shape[0]
    preferential_information_transpose = transpose(preferential_information)
    nb_redistributive_transfers: int = (nb_var * nb_var - nb_var) // 2
    big_m_1: int = ceil(npsum(winner) + npsum(looser))
    big_m_pi: int = 1000

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
    add_pi_for_step = add_pi_for_step_factory(nb_pi)
    add_pi_use_constraint = add_pi_use_constraint_factory(nb_pi, big_m_pi)

    # Initial state
    x: MVar = add_candidate_for_step(m, 0)
    m.addConstr(x == looser, name="X0")

    for step in range(1, minimum_k + 1):
        # Create redistributive transfers
        nu_plus, nu_minus, transfer = add_redistributive_constraints(m, step)

        # Create gifts
        mu_k = add_gift_for_step(m, step)
        g_k = add_gift_use_constraint(m, mu_k, step)

        # Create PI
        lmbd_k = add_pi_for_step(m, step)
        pi_k = add_pi_use_constraint(m, lmbd_k, step)

        # Create variable
        xk = add_candidate_for_step(m, step)
        if step < minimum_k:
            add_ordered_candidate_constraint(m, xk, step)

        # Explanation steps
        m.addConstr(
            x + mu_k + nu_plus - nu_minus + preferential_information_transpose @ lmbd_k
            == xk,
            name=f"StepsATX{step}",
        )

        # One action by step
        m.addConstr(
            npones((1, nb_redistributive_transfers)) @ transfer
            + g_k
            + npones((1, nb_pi)) @ pi_k
            <= 1,
            name=f"AtMost1ArgumentStep{step}",
        )

        x = xk

    # Final states
    m.addConstr(x == winner, name="Xk")
    m.update()

    yield m

    for step in range(minimum_k + 1, nb_pi * nb_var + 2 * nb_var + 1):
        # if step % 2:
        #     print(f"Step{step} : {strftime('%H:%M:%S', localtime())}")
        add_ordered_candidate_constraint(m, x, step - 1)
        xk = add_candidate_for_step(m, step)
        nu_plus, nu_minus, transfer = add_redistributive_constraints(m, step)
        mu_k = add_gift_for_step(m, step)
        g_k = add_gift_use_constraint(m, mu_k, step)
        lmbd_k = add_pi_for_step(m, step)
        pi_k = add_pi_use_constraint(m, lmbd_k, step)

        m.addConstr(
            x + mu_k + nu_plus - nu_minus + preferential_information_transpose @ lmbd_k
            == xk,
            name=f"StepsATX{step}",
        )

        m.addConstr(
            npones((1, nb_redistributive_transfers)) @ transfer
            + g_k
            + npones((1, nb_pi)) @ pi_k
            <= 1,
            name=f"AtMost1ArgumentStep{step}",
        )

        m.remove(
            [m.getConstrByName(name) for name in (f"Xk[{i}]" for i in range(nb_var))]
        )
        x = xk
        m.addConstr(x == winner, name="Xk")
        m.update()

        yield m


def add_pi_for_step_factory(nb_pi: int):
    """Builds the add_gift_for_step function from parameters deduced
    from the candidates.

    Args:
        nb_pi (int): Number of preferential information statements.
    """

    def add_pi_for_step(m: Model, step: int):
        """Adds and returns the gift variable in the MILP for the given step.

        Args:
            m (Model): Gurobi MILP.
            step (int): Number of the step introducing the gift.
        """
        return m.addMVar(
            shape=nb_pi, vtype=GRB.CONTINUOUS, lb=0.0, name=f"lambda{step}"
        )

    return add_pi_for_step


def add_pi_use_constraint_factory(nb_pi: int, big_m_pi: int):
    """Builds the add_pi_use_constraint function from the number of criteria
    and the preferential information.

    Args:
        nb_pi (int): Number of preferential information statements.
        big_m_g (int): Value of the "Big M" for preferential information.
    """

    def add_pi_use_constraint(m: Model, lmbd_k: MVar, step: int):
        """Add to the MILP the variable and the constraint for capturing the use of preferential
        information.

        Args:
            m (Model): Gurobi MILP.
            lmbd (MVar): New candidate variables whose values will be constrained.
            step (int): Number of the step introducing the new candidate.
        """
        pi_k = m.addMVar(shape=nb_pi, vtype=GRB.BINARY, name=f"pi{step}")
        m.addConstr(
            big_m_pi * pi_k - lmbd_k >= 0,
            name=f"PI{step}",
        )
        return pi_k

    return add_pi_use_constraint
