from offlinerl.algo import algo_select
from offlinerl.evaluation import OnlineCallBackFunction, CallBackFunctionList

from datasets import delay_neorl_dataset
from utils.exp_util import setup_exp_args
from evaluation.d4rl_score import d4rl_eval_fn


def run_algo(kwargs):
    algo_init_fn, algo_trainer_obj, algo_config = algo_select(kwargs)

    if (
        algo_config["delay_mode"] == "none"
        or algo_config["strategy"] == "none"
    ):
        train_buffer = delay_neorl_dataset.load_neorl_buffer(algo_config)
    else:
        train_buffer = delay_neorl_dataset.load_neorl_traj_buffer(algo_config)

    algo_init = algo_init_fn(algo_config)
    algo_trainer = algo_trainer_obj(algo_init, algo_config)

    callback = OnlineCallBackFunction()
    callback.initialize(
        train_buffer=train_buffer, val_buffer=None, task=algo_config["task"]
    )
    callback_list = CallBackFunctionList(
        [callback, d4rl_eval_fn(task=algo_config["task"])]
    )

    algo_trainer.train(train_buffer, None, callback_fn=callback_list)


# python train_d4rl.py --algo_name=mopo --task=walker2d-medium-replay-v0 --delay_mode=constant --delay=20 --strategy=interval_average
if __name__ == "__main__":
    args = setup_exp_args()

    run_algo(args)
