"""Functions :
    - providing the shortest PT-ATX for restricted Lorenz dominance using 
MILP solving
    - yielding the MILP formulation
    - adds new candidate to the MILP
    - adds ordering constraints for candidate in the MILP
    - adds constraints for redistributive transfers in the MILP"""
from math import ceil
from multiprocessing.context import TimeoutError as TimedOut
from numpy import sum as npsum
from numpy import array as nparray
from numpy import ones as npones
from numpy import int_, float64
from scipy.sparse import spmatrix
from scipy.sparse import diags as spdiags
from gurobipy import GRB, Model, MVar
from package import timeout_decorator
from package.plot import REDISTRIBUTIVE_TRANSFER
from .commons import positive_negative_modification_indexes

FILE_NAME = "Restricted\\optimum.csv"


@timeout_decorator
def restricted_optimum(looser, winner, low, high, ndigits):
    """Builds the shortest explanation for restricted Lorenz dominance between
    two candidates.
    Returns the length, the explanation and the symbols for display.

    Args:
        looser (ArrayLike): First candidate.
        winner (ArrayLike): Second candidate.
    """
    minimum_k = max(
        (
            len(x)
            for x in positive_negative_modification_indexes(looser, winner, ndigits)
        )
    )
    k = minimum_k
    model_generator = build_restricted_base_model(
        looser, winner, minimum_k, low, high, ndigits
    )
    m = next(model_generator)
    found = False

    while not found:
        m.update()
        m.optimize()
        if m.Status == GRB.TIME_LIMIT:
            raise TimedOut
        m.display()
        if m.status == GRB.OPTIMAL:
            nb_var = len(looser)
            explanation = []
            for step in range(k + 1):
                explanation.append(
                    nparray(
                        [
                            m.getVarByName(name).X
                            for name in (f"x{step}[{i}]" for i in range(nb_var))
                        ],
                        dtype=int_ if ndigits == 0 else float64,
                    )
                )
            found = True
        else:
            m = next(model_generator)
            k += 1

    m.dispose()
    return k, explanation, [REDISTRIBUTIVE_TRANSFER] * k


def build_restricted_base_model(
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
    m.Params.MIPFocus = 1
    m.Params.TimeLimit = 150

    if ndigits == 0:
        cand_type = GRB.INTEGER
    else:
        cand_type = GRB.CONTINUOUS
        m.Params.FeasibilityTol = 10 ** (-ndigits - 2)

    nb_var: int = winner.shape[0]
    nb_redistributive_transfers: int = (nb_var * nb_var - nb_var) // 2

    add_redistributive_constraints = add_redistributive_constraints_factory(
        nb_var,
        cand_type,
        ceil(npsum(winner) + npsum(looser)),
        nb_redistributive_transfers,
    )

    add_candidate_for_step = add_candidate_for_step_factory(
        nb_var, cand_type, low, high
    )

    add_ordered_candidate_constraint = add_ordered_candidate_constraint_factory(nb_var)

    # Initial state
    x: MVar = add_candidate_for_step(m, 0)
    m.addConstr(x == looser, name="X0")

    for step in range(1, minimum_k + 1):
        nu_plus, nu_minus, transfer = add_redistributive_constraints(m, step)

        # Create variable
        xk = add_candidate_for_step(m, step)
        if step < minimum_k:
            add_ordered_candidate_constraint(m, xk, step)

        # Explanation steps
        m.addConstr(
            x + nu_plus - nu_minus == xk,
            name=f"StepsATX{step}",
        )

        # One action by step
        m.addConstr(
            npones((1, nb_redistributive_transfers)) @ transfer <= 1,
            name=f"AtMost1ArgumentStep{step}",
        )

        x = xk

    # Final states
    m.addConstr(x - winner == 0, name="Xk")

    yield m

    for step in range(minimum_k + 1, nb_var + 1):
        add_ordered_candidate_constraint(m, x, step - 1)
        xk = add_candidate_for_step(m, step)
        nu_plus, nu_minus, transfer = add_redistributive_constraints(m, step)
        m.addConstr(
            x + nu_plus - nu_minus == xk,
            name=f"StepsATX{step}",
        )
        m.addConstr(
            npones((1, nb_redistributive_transfers)) @ transfer <= 1,
            name=f"AtMost1ArgumentStep{step}",
        )
        m.remove(
            [m.getConstrByName(name) for name in (f"Xk[{i}]" for i in range(nb_var))]
        )
        x = xk
        m.addConstr(x == winner, name="Xk")

        yield m


def add_candidate_for_step_factory(nb_var: int, cand_type, low, high):
    """Builds the add_candidate_for_step function from parameters deduced
    from the candidates.

    Args:
        nb_var (int): Number of criteria.
        cand_type (int_ | float64): Data type of the candidate criteria.
        low (int | float): Lower boundary of the definition domain.
        high (int | float): Upper boundary of the definition domain.
    """

    def add_candidate_for_step(m: Model, step: int):
        """Adds and returns a new candidate in the MILP for the given step.

        Args:
            m (Model): Gurobi MILP.
            step (int): Number of the step introducing the new candidate.
        """
        return m.addMVar(
            shape=nb_var, vtype=cand_type, lb=low, ub=high, name=f"x{step}"
        )

    return add_candidate_for_step


def add_ordered_candidate_constraint_factory(nb_var: int):
    """Builds the add_ordered_candidate_constraint function from the number of criteria.

    Args:
        nb_var (int): Number of criteria.
    """
    bidiag: spmatrix = spdiags([npones(nb_var), -npones(nb_var - 1)], [0, -1])

    def add_ordered_candidate_constraint(m: Model, xk: MVar, step: int):
        """Add to the MILP the constraints for the freshly added new candidate to be ordered.

        Args:
            m (Model): Gurobi MILP.
            xk (MVar): New candidate variables whose values will be constrained.
            step (int): Number of the step introducing the new candidate.
        """
        m.addConstr(
            bidiag @ xk >= 0,
            name=f"OrderedCandidates{step}",
        )

    return add_ordered_candidate_constraint


def add_redistributive_constraints_factory(
    nb_var: int, cand_type, big_m_rt: int, nb_redistributive_transfers: int
):
    """Builds the add_ordered_candidate_constraint function from parameters deduced
    from the candidates.

    Args:
        nb_var (int): Number of criteria.
        cand_type (int_ | float64): Data type of the candidate criteria.
        big_m_rt (int): Value of the "Big M" for redistributive transfers.
        nb_redistributive_transfers (int): Total number of possible pair (i,j) for redistributive
        transfers.
    """

    def add_redistributive_constraints(m: Model, step: int):
        """Adds to the models the constraints for performing redistributive transfers at the given
        step.
        Returns the guroby variables containing the receiving and giving vector contributions
        and the vector encoding the pair (i,j) of indexes.

        Args:
            m (Model): Gurobi MILP.
            step (int): Number of the step needing the constraints.
        """
        nu_plus = m.addMVar(shape=nb_var, vtype=cand_type, lb=0.0, name=f"nu+{step}")
        nu_minus = m.addMVar(shape=nb_var, vtype=cand_type, lb=0.0, name=f"nu-{step}")
        gamma_plus = m.addMVar(shape=nb_var, vtype=GRB.BINARY, name=f"Gamma+{step}")
        gamma_minus = m.addMVar(shape=nb_var, vtype=GRB.BINARY, name=f"Gamma-{step}")
        transfer = m.addMVar(
            shape=(nb_redistributive_transfers), vtype=GRB.BINARY, name=f"Tji{step}"
        )

        m.addConstr(
            npones((1, nb_var)) @ nu_minus == npones((1, nb_var)) @ nu_plus,
            name=f"EqualizingRT{step}",
        )

        m.addConstr(
            big_m_rt * gamma_plus - nu_plus >= 0,
            name=f"SelectIndexGiverRT{step}",
        )

        m.addConstr(
            big_m_rt * gamma_minus - nu_minus >= 0,
            name=f"SelectIndexReceiverRT{step}",
        )

        m.addConstr(
            npones((1, nb_var)) @ gamma_minus == npones((1, nb_var)) @ gamma_plus,
            name=f"Same#ReceiverGiver{step}",
        )

        m.addConstr(
            npones((1, nb_var)) @ gamma_minus
            == npones((1, nb_redistributive_transfers)) @ transfer,
            name=f"FairRedistributiveReceiver{step}",
        )

        m.addConstrs(
            (
                -2 * transfer[l] + gamma_minus[j] + gamma_plus[i] <= 1
                for l, (j, i) in enumerate(
                    (j, i) for j in range(1, nb_var) for i in range(j)
                )
            ),
            name=f"Redistributive1RT{step}",
        )

        m.addConstrs(
            (
                -2 * transfer[l] + gamma_minus[j] + gamma_plus[i] >= 0
                for l, (j, i) in enumerate(
                    (j, i) for j in range(1, nb_var) for i in range(j)
                )
            ),
            name=f"Redistributive2RT{step}",
        )

        # m.addConstr(
        #     npones((1, nb_redistributive_transfers)) @ transfer <= 1,
        #     name=f"AtMost1RTStep{step}",
        # )

        return nu_plus, nu_minus, transfer

    return add_redistributive_constraints
