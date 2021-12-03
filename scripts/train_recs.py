from offlinerl.algo import algo_select
from offlinerl.evaluation import OnlineCallBackFunction, CallBackFunctionList

from datasets import recs_dataset
from utils.exp_util import setup_exp_args
from evaluation.d4rl_score import d4rl_eval_fn


def run_algo(kwargs):
    algo_init_fn, algo_trainer_obj, algo_config = algo_select(kwargs)
    train_buffer = recs_dataset.load_rec_sim_buffer(algo_config)

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


if __name__ == "__main__":
    args = setup_exp_args()

    run_algo(args)
