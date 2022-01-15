from loguru import logger
import warnings

warnings.filterwarnings("ignore")

from offlinerl.algo.modelfree import cql
from offlinerl.utils.config import parse_config

from offlinerl.config.algo import (
    cql_config,
    iql_config,
    plas_config,
    mopo_config,
    mopod_config,
    moose_config,
    bcqd_config,
    bcq_config,
    bc_config,
    crr_config,
    combo_config,
    bremen_config,
    shaping_config,
    decomposer_config,
    reward_giver_config,
    cqld_config,
    iqld_config,
    online_sacd_config,
    online_dqn_config,
    online_qr_dqn_config,
    offline_dqn_config,
    offline_qr_dqn_config,
    bcd_config,
    online_rem_config,
    offline_rem_config,
)

from offlinerl.utils.config import parse_config
from offlinerl.algo.modelfree import (
    plas,
    bcqd,
    bcq,
    bc,
    crr,
    cql,
    iql,
    cql_discrete,
    iql_discrete,
    offline_dqn,
    offline_qr_dqn,
    bcd,
    offline_rem,
)
from offlinerl.algo.modelbase import moose, combo, bremen, mopo, mopo_discrete
from offlinerl.algo.custom import reward_shaper, reward_decoposer, reward_giver
from offlinerl.algo.online import qr_dqn, rem, sac_discrete, dqn

algo_dict = {
    # custom
    "reward_giver": {"algo": reward_giver, "config": reward_giver_config},
    "reward_shaper": {"algo": reward_shaper, "config": shaping_config},
    "reward_decomposer": {
        "algo": reward_decoposer,
        "config": decomposer_config,
    },
    "cql": {"algo": cql, "config": cql_config},
    "cqld": {"algo": cql_discrete, "config": cqld_config},
    "iql": {"algo": iql, "config": iql_config},
    "iqld": {"algo": iql_discrete, "config": iqld_config},
    "mopo": {"algo": mopo, "config": mopo_config},
    "mopod": {"algo": mopo_discrete, "config": mopod_config},
    "offline_dqn": {"algo": offline_dqn, "config": offline_dqn_config},
    "offline_qr_dqn": {
        "algo": offline_qr_dqn,
        "config": offline_qr_dqn_config,
    },
    "offline_rem": {"algo": offline_rem, "config": offline_rem_config},
    "bcd": {"algo": bcd, "config": bcd_config},
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
    # online
    "sacd": {"algo": sac_discrete, "config": online_sacd_config},
    "dqn": {"algo": dqn, "config": online_dqn_config},
    "qr_dqn": {"algo": qr_dqn, "config": online_qr_dqn_config},
    "rem": {"algo": rem, "config": online_rem_config},
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
