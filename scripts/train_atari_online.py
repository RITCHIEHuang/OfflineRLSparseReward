from offlinerl.algo import algo_select
from offlinerl.evaluation import CallBackFunctionList

from utils.exp_util import setup_exp_args
from evaluation.atari import atari_eval_fn


def run_algo(kwargs):
    algo_init_fn, algo_trainer_obj, algo_config = algo_select(kwargs)
    algo_init = algo_init_fn(algo_config)
    algo_trainer = algo_trainer_obj(algo_init, algo_config)

    callback_list = CallBackFunctionList(
        [atari_eval_fn(task=algo_config["task"])]
    )

    algo_trainer.train(callback_fn=callback_list)


if __name__ == "__main__":
    args = setup_exp_args()

    run_algo(args)
