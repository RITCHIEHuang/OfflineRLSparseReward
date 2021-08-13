import fire

from offlinerl.algo import algo_select
from offlinerl.evaluation import OnlineCallBackFunction

from datasets.d4rl import load_d4rl_buffer

# python train_d4rl.py --algo_name=cql --exp_name=d4rl-halfcheetah-medium-cql --task=d4rl-halfcheetah-medium-v0 --delay=20


def run_algo(**kwargs):
    algo_init_fn, algo_trainer_obj, algo_config = algo_select(kwargs)
    train_buffer = load_d4rl_buffer(algo_config)
    algo_init = algo_init_fn(algo_config)
    algo_trainer = algo_trainer_obj(algo_init, algo_config)
    callback = OnlineCallBackFunction()
    callback.initialize(
        train_buffer=train_buffer, val_buffer=None, task=algo_config["task"]
    )

    algo_trainer.train(train_buffer, None, callback_fn=callback)


if __name__ == "__main__":
    fire.Fire(run_algo)
