"""Main example aiming at generating explanations from previously generated data."""
from time import perf_counter, strftime, localtime, sleep
from multiprocessing.context import TimeoutError as TimedOut
from package.plot.types import PREFERENTIAL_INFORMATION
from package.data.save import save_experiment_data_factory
from package.data.load import (
    load_dataset,
    load_meta_data,
    load_restricted_lorenz_dominances,
    load_generalized_lorenz_dominances,
    load_robust_redistributive_owa_dominances,
)

from package.restricted_lorenz import (
    hardy_littlewood_polya,
    contribution_heuristics,
    restricted_optimum,
)
from package.restricted_lorenz.solving.hlp import FILE_NAME as hlp_file
from package.restricted_lorenz.solving.contribution_algo import (
    FILE_NAME as contrib_file,
)
from package.restricted_lorenz.solving.optimum import FILE_NAME as r_optim_file


from package.generalized_lorenz import (
    gift_after_hardy_littlewood_polya,
    gift_after_contribution_heuristics,
    generalized_optimum,
)
from package.generalized_lorenz.solving.after_hlp import FILE_NAME as after_hlp_file
from package.generalized_lorenz.solving.after_contribution_algo import (
    FILE_NAME as after_contrib_file,
)
from package.generalized_lorenz.solving.optimal import FILE_NAME as g_optim_file


from package.robust_owa.solving.farkas import (
    first_farkas,
    minimum_length_farkas,
)
from package.robust_owa.solving.farkas.first_farkas import (
    FARKAS_NAME as first_farkas_name,
)
from package.robust_owa.solving.farkas.min_nb_pi import FARKAS_NAME as min_farkas_name


from package.robust_owa.solving.atx import robust_optimum
from package.robust_owa.solving.atx.optimal import FILE_NAME as atx_optim_file

from package.robust_owa.solving.ctx import (
    ctx_from_farkas_displaced,
)
from package.robust_owa.solving.ctx.from_farkas_displaced import (
    file_name as ctx_displaced_file,
)


def save_with_perf(writer, computation_func, args, output_file=None):
    """Executes the given explanation computation and saves its length and compute time.

    Args:
        writer (Generator): Data saver.
        computation_func (Callable): Explanation function to compute.
        args (Tuple): Explanation function arguments.
    """
    start_compute = perf_counter()
    try:
        l, ex, sy = computation_func(*args)
        if output_file:
            print(
                ex,
                sy,
                file=open(output_file, "a", encoding="utf8"),
            )
    except TimedOut:
        l = -3
        end_compute = perf_counter()
        writer.send([l, end_compute - start_compute, l])
        return
    end_compute = perf_counter()
    writer.send([l, end_compute - start_compute, sy.count(PREFERENTIAL_INFORMATION)])


def save_with_perf_farkas(
    writer,
    computation_func,
    farkas_func,
    args_computation,
    args_farkas,
    output_file=None,
):
    """Executes the given explanation computation and the given farkas certificate
    computation it requires and saves its length and compute time.
    If the Farkas function did not find a certificate, saves -2 as legnth and
    if explanation method did not find an explanation, saves -1 as length instead.

    Args:
        writer (Generator): Data saver.
        computation_func (Callable): Explanation function to compute.
        farkas_func (Callable): Farkas certificate function to compute.
        args_computation (Tuple): Explanation function arguments.
        args_farkas (Tuple): Farkas certificate function arguments.
    """
    start_compute = perf_counter()
    try:
        nu_minus, nu_plus, mu, lmbd = farkas_func(*args_farkas)
    except StopIteration:
        l = -2
        end_compute = perf_counter()
        writer.send([l, end_compute - start_compute, l])
        return
    except TimedOut:
        l = -3
        end_compute = perf_counter()
        writer.send([l, end_compute - start_compute, l])
        return
    try:
        l, ex, sy = computation_func(*args_computation, nu_minus, nu_plus, mu, lmbd)
        end_compute = perf_counter()
        if output_file:
            print(
                ex,
                sy,
                file=open(output_file, "a", encoding="utf8"),
            )
        writer.send(
            [l, end_compute - start_compute, sy.count(PREFERENTIAL_INFORMATION)]
        )
        return
    except StopIteration:
        l = -1
    except TimedOut:
        l = -3
    end_compute = perf_counter()
    writer.send([l, end_compute - start_compute, l])


def restricted_explain(path: str, data, low, high, precision: int, output_file=None):
    """Saves explanation length and compute times for methods :
        - [Hardy, Littlewood, Poly, 1934]
        - contribution algorithm
        - shortest length using optimum MILP formulation
    for restricted Lorenz dominances.

    Args:
        path (str): Path to the root of the experiment folder.
        data (NDArray): Dataset of candidates.
        low (int | float): Lower boundary of the definition domain.
        high (int | float): Upper boundary of the definition domain.
        precision (int): Precision (number of digit after the coma).
    """
    restricted_lorenz_dom = load_restricted_lorenz_dominances(path)

    save_hlp = save_experiment_data_factory(path, hlp_file)
    save_contrib = save_experiment_data_factory(path, contrib_file)

    save_optim = save_experiment_data_factory(path, r_optim_file)

    for i, j in restricted_lorenz_dom:
        if output_file:
            print(
                "HLP :",
                file=open(output_file, "a", encoding="utf8"),
            )
        save_with_perf(
            save_hlp, hardy_littlewood_polya, (data[i], data[j], precision), output_file
        )
        if output_file:
            print(
                "Contrib :",
                file=open(output_file, "a", encoding="utf8"),
            )
        save_with_perf(
            save_contrib,
            contribution_heuristics,
            (data[i], data[j], precision),
            output_file,
        )
        if output_file:
            print(
                "Optim L*",
                file=open(output_file, "a", encoding="utf8"),
            )
        save_with_perf(
            save_optim,
            restricted_optimum,
            (data[i], data[j], low, high, precision),
            output_file,
        )


def generalized_explain(path: str, data, low, high, precision: int, output_file=None):
    """Saves explanation length and compute times for methods :
        - [Hardy, Littlewood, Poly, 1934] with Gift afterwards
        - contribution algorithm with Gift afterwards
        - shortest length using optimum MILP formulation
    for generalized Lorenz dominances.

    Args:
        path (str): Path to the root of the experiment folder.
        data (NDArray): Dataset of candidates.
        low (int | float): Lower boundary of the definition domain.
        high (int | float): Upper boundary of the definition domain.
        precision (int): Precision (number of digit after the coma).
    """
    generalized_lorenz_dom = load_generalized_lorenz_dominances(path)

    save_hlp = save_experiment_data_factory(path, after_hlp_file)
    save_contrib = save_experiment_data_factory(path, after_contrib_file)
    save_optim = save_experiment_data_factory(path, g_optim_file)

    for i, j in generalized_lorenz_dom:
        if output_file:
            print(
                "HLP after :",
                file=open(output_file, "a", encoding="utf8"),
            )
        save_with_perf(
            save_hlp,
            gift_after_hardy_littlewood_polya,
            (data[i], data[j], precision),
            output_file,
        )
        if output_file:
            print(
                "Contrib after :",
                file=open(output_file, "a", encoding="utf8"),
            )
        save_with_perf(
            save_contrib,
            gift_after_contribution_heuristics,
            (data[i], data[j], precision),
            output_file,
        )
        if output_file:
            print(
                "Optim L :",
                file=open(output_file, "a", encoding="utf8"),
            )
        save_with_perf(
            save_optim,
            generalized_optimum,
            (data[i], data[j], low, high, precision),
            output_file,
        )


def robust_explain(
    path: str, data, pi_statements, low, high, precision: int, output_file=None
):
    """_summary_

    Args:
        path (str): Path to the root of the experiment folder.
        data (NDArray): Dataset of candidates.
        low (int | float): Lower boundary of the definition domain.
        high (int | float): Upper boundary of the definition domain.
        precision (int): Precision (number of digit after the coma).
    """
    rowa_dom = load_robust_redistributive_owa_dominances(path)

    save_optim = save_experiment_data_factory(path, atx_optim_file)

    save_ctx_farkas_displaced_min = save_experiment_data_factory(
        path, ctx_displaced_file(min_farkas_name)
    )
    save_ctx_farkas_displaced_first = save_experiment_data_factory(
        path, ctx_displaced_file(first_farkas_name)
    )

    for i, j in rowa_dom:
        # print(i, j, strftime("%H:%M:%S", localtime()))
        if output_file:
            print(
                "Optim ATX :",
                file=open(output_file, "a", encoding="utf8"),
            )
        save_with_perf(
            save_optim,
            robust_optimum,
            (data[i], data[j], low, high, precision, pi_statements),
            output_file,
        )
        if output_file:
            print(
                "Optim CTX displaced :",
                file=open(output_file, "a", encoding="utf8"),
            )
        save_with_perf_farkas(
            save_ctx_farkas_displaced_min,
            ctx_from_farkas_displaced,
            minimum_length_farkas,
            (data[i], data[j], pi_statements, low, high, precision),
            (data[i], data[j], precision, pi_statements),
            output_file,
        )
        if output_file:
            print(
                "CTX displaced with fast Farkas :",
                file=open(output_file, "a", encoding="utf8"),
            )
        save_with_perf_farkas(
            save_ctx_farkas_displaced_first,
            ctx_from_farkas_displaced,
            first_farkas,
            (data[i], data[j], pi_statements, low, high, precision),
            (data[i], data[j], precision, pi_statements),
            output_file,
        )


def explain_int_fixed(exp_path: str, output_file=None):
    """Launch the explanation computation for Restricted dominances.

    Args:
        exp_path (str): Path to the root of the experiment folder.
    """
    (
        nb_exp,
        _,
        _,
        _,
        low,
        high,
        _,
        _,
        _,
        precision,
    ) = load_meta_data(exp_path)

    for f in range(nb_exp):
        print(f"Fold {f}: {strftime('%H:%M:%S', localtime())}")
        fold_path = f"{exp_path}\\{f}"
        data, _ = load_dataset(fold_path, precision)

        restricted_explain(fold_path, data, low, high, precision, output_file)
        sleep(0.01)


def explain_int(exp_path: str, output_file=None):
    """Launch the explanation computation for Generalized and Restricted dominances.

    Args:
        exp_path (str): Path to the root of the experiment folder.
    """
    (
        nb_exp,
        _,
        _,
        _,
        low,
        high,
        _,
        _,
        _,
        precision,
    ) = load_meta_data(exp_path)

    for f in range(nb_exp):
        print(f"Fold {f}: {strftime('%H:%M:%S', localtime())}")
        fold_path = f"{exp_path}\\{f}"
        data, _ = load_dataset(fold_path, precision)

        restricted_explain(fold_path, data, low, high, precision, output_file)
        generalized_explain(fold_path, data, low, high, precision, output_file)
        sleep(0.01)


def explain_float(exp_path: str, output_file=None):
    """Launch the explanation computation for ROWA, Generalized and Restricted dominances.

    Args:
        exp_path (str): Path to the root of the experiment folder.
    """
    (
        nb_exp,
        _,
        _,
        _,
        low,
        high,
        _,
        _,
        _,
        precision,
    ) = load_meta_data(exp_path)

    for f in range(nb_exp):
        print(f"Fold {f}: {strftime('%H:%M:%S', localtime())}")
        fold_path = f"{exp_path}\\{f}"
        data, pi_statements = load_dataset(fold_path, precision)

        restricted_explain(fold_path, data, low, high, precision, output_file)
        generalized_explain(fold_path, data, low, high, precision, output_file)
        robust_explain(
            fold_path, data, pi_statements, low, high, precision, output_file
        )
        sleep(0.01)


if __name__ == "__main__":
    NB_VAR = 5
    NB_FOLD = 1
    NB_CAND = 10
    NB_PI = 2
    EXP_PATH = f".\\Datasets\\Int_{NB_VAR}cri_{NB_CAND}cand"
    explain_int(EXP_PATH)

    FIXED_SUM = NB_VAR * 20
    EXP_PATH = f".\\Datasets\\Int_{NB_VAR}cri_{FIXED_SUM}sum_{NB_CAND}cand"
    explain_int_fixed(EXP_PATH)

    EXP_PATH = f".\\Datasets\\Float_{NB_VAR}cri_{NB_CAND}cand"
    explain_float(EXP_PATH)
