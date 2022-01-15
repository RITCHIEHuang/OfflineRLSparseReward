from loguru import logger
from ray import tune

from offlinerl.algo import algo_select
from offlinerl.evaluation import CallBackFunctionList

from datasets import recs_dataset
from utils.exp_util import setup_exp_args
from utils.io_util import proj_path
from evaluation.recs import recs_eval_fn


def training_function(config):
    algo_init_fn, algo_trainer_obj, algo_config = algo_select(config["kwargs"])

    train_buffer = recs_dataset.load_recs_buffer(algo_config)

    algo_config.update(config)
    algo_config["device"] = "cuda"
    algo_init = algo_init_fn(algo_config)
    algo_trainer = algo_trainer_obj(algo_init, algo_config)

    callback_list = CallBackFunctionList(
        [recs_eval_fn(task=algo_config["task"])]
    )

    score = algo_trainer.train(train_buffer, None, callback_fn=callback_list)

    return score


def run_algo(kwargs):
    config = {}
    config["kwargs"] = kwargs
    logger.info(
        f"Task: {kwargs['task']}, algo: {kwargs['algo_name']}, exp_name: {kwargs['exp_name']}"
    )
    _, _, algo_config = algo_select(kwargs)
    # Prepare Dataset
    grid_tune = algo_config["grid_tune"]
    for k, v in grid_tune.items():
        config[k] = tune.grid_search(v)

    analysis = tune.run(
        training_function,
        config=config,
        local_dir=f"{proj_path}/tune_results",
        stop={"training_iteration": 1500},
        resources_per_trial={"gpu": 1},
        queue_trials=True,
    )


if __name__ == "__main__":
    args = setup_exp_args()
    run_algo(args)
