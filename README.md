# Model Based Sparse Offline RL

The official implementation of []().

## Paper Manuscripts

Overleaf link:

[RL_Offline_Delay_Rewards](https://www.overleaf.com/project/617812aa8bb2ee6d5adccf00)

Local Latex Folder:

[paper/RL_Offline_Delay_Rewards](paper/RL_Offline_Delay_Rewards)

## Installation

To reproduce reported results, please follow the steps below inside the project folder:

```shell
# install d4rl
cd d4rl/ && pip install -e .
# install neorl
cd ../neorl/ && pip install -e .
# install OfflineRL
cd ../OfflineRL/ && pip install -e .
# install project
cd ../ && pip install -e .
```
This project includes experiments on [d4rl](https://github.com/rail-berkeley/d4rl) benchmark and [neorl](https://github.com/polixir/NeoRL) benchmark, our implementation modified based on the [OfflineRL](https://github.com/polixir/OfflineRL) codebase for efficiency.


## Datasets

### 1. D4RL Offline Datasets in Gym domain and Antmaze domain

[d4rl_dataset.py](datasets/d4rl_dataset.py).

### 2. NeoRL Offline Datasets for Gym

[neorl_dataset.py](datasets/neorl_dataset.py).

### 3. Custom Industrial Offline Datasets based on RecSim

[recs_dataset.py](datasets/recs_dataset.py)

## Run Experiments 

All running scripts are placed under the [scripts](scripts/) folder, some examples are provided below:

To run d4rl delay tasks:
```shell
python train_d4rl.py --algo_name=mopo --strategy=average  --task=halfcheetah-medium-expert-v0 --delay_mode=constant --seed=10
```

To run d4rl sparse tasks:
```shell
python train_d4rl.py --algo_name=mopo --strategy=average  --task=antmaze-medium-play-v2 --delay_mode=none --seed=10
```

To run neorl tasks:
```shell
python train_neorl.py --algo_name=mopo --strategy=average  --task=Halfcheetah-v3-low-100 --delay_mode=constant --seed=10
```

To run recs tasks:
```shell
python train_recs.py --algo_name=mopo --strategy=average  --task=recs-random-v0 --seed=10
```


## Citation

To cite this repository:
```
@xxxx{

}
```

