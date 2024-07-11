from package.data.generation import generation_example
from package.restricted_lorenz.solving.hlp import FILE_NAME as hlp_file
from package.restricted_lorenz.solving.contribution_algo import (
    FILE_NAME as contrib_file,
)
from package.restricted_lorenz.solving.optimum import FILE_NAME as r_optim_file
from package.plot.expermiment_comparisons import (
    experiment_comparison,
    experiment_pairwise_comparison,
    pairwise_generator_timedout,
)
from explanation import explain_float, explain_int_fixed
from generation import process_int_fixed


def restricted_lorenz_comparison(exp_path: str):
    """Compares the HLP, our algorithm and the optimum method for restricted Lorenz dominance.

    Args:
        exp_path (str): Root path containing the computations of the methods.
    """
    experiment_comparison(exp_path, hlp_file, True)
    experiment_comparison(exp_path, contrib_file, True)
    experiment_comparison(exp_path, r_optim_file, True)
    experiment_pairwise_comparison(
        exp_path, contrib_file, hlp_file, pairwise_generator_timedout
    )
    experiment_pairwise_comparison(
        exp_path, r_optim_file, contrib_file, pairwise_generator_timedout
    )
    experiment_pairwise_comparison(
        exp_path, r_optim_file, hlp_file, pairwise_generator_timedout
    )


if __name__ == "__main__":
    generation_example(".\\ECAI\\Paper_Example")
    explain_float(".\\ECAI\\Paper_Example", ".\\ECAI\\Paper_Example\\output.txt")

    # for n in [5, 10, 20, 50]:
    for n in [5]:
        print(f"Restricted Lorenz dominance for n={n} :")
        EXP_PATH = f".\\ECAI\\Int_{n}cri_{n*200}sum_10_cand"
        process_int_fixed(
            path=EXP_PATH,
            nb_fold=10,
            nb_var=n,
            nb_cand=10,
            nb_pi=0,
            fixed_sum=n * 200,
            high=1000,
        )
        explain_int_fixed(EXP_PATH)
        restricted_lorenz_comparison(EXP_PATH)
