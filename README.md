# Model Based Sparse Offline RL

The official implementation of "On Offline Reinforcement Learning for Sparse Reward Tasks"

## Installation

To reproduce reported results, please follow the steps below inside the project folder:

```shell
./install.sh
```


## Tasks and Datasets

- [D4RL](datasets/d4rl_dataset.py) with artificially delayed-reward tasks and sparse reward tasks.

- [NeoRL](datasets/neorl_dataset.py) with artifically delayed-reward tasks.

- [RecS](datasets/recs_dataset.py) with real-world simulated sparse reward tasks.

## Run Experiments 

All running scripts are placed under the [scripts](scripts/) folder, some examples are provided below:

To run d4rl delayde-reward task:
```shell
python train_d4rl.py --algo_name=mopo --strategy=average \
--task=halfcheetah-medium-expert-v0 --delay_mode=constant --seed=10
```

To run d4rl sparse-reward task:
```shell
python train_d4rl.py --algo_name=mopo --strategy=average \
--task=antmaze-medium-play-v2 --delay_mode=none --seed=10
```

To run neorl delayed-reward task:
```shell
python train_neorl.py --algo_name=mopo --strategy=average \
 --task=Halfcheetah-v3-low-100 --delay_mode=constant --seed=10
```

To run recs sparse-reward task:
```shell
python train_recs.py --algo_name=mopo --strategy=average \
--task=recs-random-v0 --seed=10
```


## Experiments Logging and Visualization

This project record the training log with `Tensorboard` in local directory `logs/` and [Wandb](#https://wandb.ai/site) on website.


## Reference

This project includes experiments on [d4rl](https://github.com/rail-berkeley/d4rl) benchmark and [neorl](https://github.com/polixir/NeoRL) benchmark, our implementation based on the [OfflineRL](https://github.com/polixir/OfflineRL) codebase for efficiency.

To cite this repository:
```
@xxxx{

}
```

