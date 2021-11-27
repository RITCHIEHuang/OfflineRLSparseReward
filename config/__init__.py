from loguru import logger
import warnings

warnings.filterwarnings("ignore")

from offlinerl.algo.modelfree import cql
from offlinerl.utils.config import parse_config

from offlinerl.config.algo import (
    # cql_config,
    plas_config,
    # mopo_config,
    moose_config,
    bcqd_config,
    bcq_config,
    bc_config,
    crr_config,
    combo_config,
    bremen_config,
)
from offlinerl.utils.config import parse_config
from offlinerl.algo.modelfree import plas, bcqd, bcq, bc, crr
from offlinerl.algo.modelbase import moose, combo, bremen

from algos import mopo, mopo_discrete, cql, iql
from algos import reward_shaper, reward_decoposer, reward_giver
from config import (
    cql_config,
    iql_config,
    shaping_config,
    mopo_config,
    decomposer_config,
    reward_giver_config,
)

algo_dict = {
    # custom
    "reward_giver": {"algo": reward_giver, "config": reward_giver_config},
    "reward_shaper": {"algo": reward_shaper, "config": shaping_config},
    "reward_decomposer": {
        "algo": reward_decoposer,
        "config": decomposer_config,
    },
    "cql": {"algo": cql, "config": cql_config},
    "iql": {"algo": iql, "config": iql_config},
    "mopo": {"algo": mopo, "config": mopo_config},
    "mopo_discrete": {"algo": mopo_discrete, "config": mopo_config},
    # default
    "bc": {"algo": bc, "config": bc_config},
    "bcq": {"algo": bcq, "config": bcq_config},
    "bcqd": {"algo": bcqd, "config": bcqd_config},
    "combo": {"algo": combo, "config": combo_config},
    # "cql" : {"algo" : cql, "config" : cql_config},
    "crr": {"algo": crr, "config": crr_config},
    "plas": {"algo": plas, "config": plas_config},
    "moose": {"algo": moose, "config": moose_config},
    "bremen": {"algo": bremen, "config": bremen_config},
}


def algo_select(command_args, algo_config_module=None):
    algo_name = command_args["algo_name"]
    logger.info("Use {} algorithm!", algo_name)
    assert algo_name in algo_dict.keys()
    algo = algo_dict[algo_name]["algo"]

    if algo_config_module is None:
        algo_config_module = algo_dict[algo_name]["config"]
    algo_config = parse_config(algo_config_module)
    algo_config.update(command_args)

    algo_init = algo.algo_init
    algo_trainer = algo.AlgoTrainer

    return algo_init, algo_trainer, algo_config
