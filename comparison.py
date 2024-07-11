from itertools import combinations
from package.plot.expermiment_comparisons import (
    experiment_comparison,
    experiment_pairwise_comparison,
    pairwise_generator_timedout,
)

from package.restricted_lorenz.solving.hlp import FILE_NAME as hlp_file
from package.restricted_lorenz.solving.contribution_algo import (
    FILE_NAME as contrib_file,
)
from package.restricted_lorenz.solving.optimum import FILE_NAME as r_optim_file


from package.generalized_lorenz.solving.optimal import FILE_NAME as gen_optim_file
from package.generalized_lorenz.solving.after_contribution_algo import (
    FILE_NAME as after_contrib_file,
)
from package.generalized_lorenz.solving.after_hlp import FILE_NAME as after_hlp_file


from package.robust_owa.solving.farkas.first_farkas import (
    FARKAS_NAME as first_farkas_name,
)
from package.robust_owa.solving.farkas.min_nb_pi import FARKAS_NAME as min_farkas_name


from package.robust_owa.solving.atx.optimal import FILE_NAME as atx_optim_file

from package.robust_owa.solving.ctx.from_farkas_displaced import (
    file_name as ctx_displaced_file_name,
)

if __name__ == "__main__":
    EXP_PATH = ".\\Datasets\\Float_5cri_10cand"

    # Restricted Lorenz comparison
    experiment_comparison(EXP_PATH, hlp_file, True)
    experiment_comparison(EXP_PATH, contrib_file, True)
    experiment_comparison(EXP_PATH, r_optim_file, True)

    experiment_pairwise_comparison(
        EXP_PATH, contrib_file, hlp_file, pairwise_generator_timedout
    )
    experiment_pairwise_comparison(
        EXP_PATH, r_optim_file, contrib_file, pairwise_generator_timedout
    )
    experiment_pairwise_comparison(
        EXP_PATH, r_optim_file, hlp_file, pairwise_generator_timedout
    )

    # Generalized Lorenz comparison
    experiment_comparison(EXP_PATH, gen_optim_file, True)
    experiment_comparison(EXP_PATH, after_contrib_file, True)
    experiment_comparison(EXP_PATH, after_hlp_file, True)
    experiment_pairwise_comparison(
        EXP_PATH,
        after_contrib_file,
        after_hlp_file,
        pairwise_generator_timedout,
    )
    experiment_pairwise_comparison(
        EXP_PATH,
        gen_optim_file,
        after_contrib_file,
        pairwise_generator_timedout,
    )
    experiment_pairwise_comparison(
        EXP_PATH, gen_optim_file, after_hlp_file, pairwise_generator_timedout
    )

    # Robust redistributive OWA comparison
    methods = [
        ctx_displaced_file_name(first_farkas_name),
        ctx_displaced_file_name(min_farkas_name),
        atx_optim_file,
    ]
    for m in methods:
        experiment_comparison(EXP_PATH, m, True)
    for m1, m2 in combinations(methods, 2):
        experiment_pairwise_comparison(
            EXP_PATH,
            m1,
            m2,
            pairwise_generator_timedout,
        )
