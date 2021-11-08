from offlinerl.evaluation import OnlineCallBackFunction, CallBackFunctionList
from offlinerl.evaluation.d4rl import d4rl_eval_fn
from offlinerl.data import load_data_from_neorl

from datasets import delay_neorl_dataset
from config import algo_select
from utils.exp_util import setup_exp_args


def run_algo(kwargs):
    algo_init_fn, algo_trainer_obj, algo_config = algo_select(kwargs)

    if algo_config["delay_mode"] == "none":
        train_buffer = load_data_from_neorl(
            algo_config["task_name"],
            algo_config["task_data_type"],
            algo_config["task_train_num"],
        )
    elif algo_config["strategy"] == "none":
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
