# RecSim: A Configurable Recommender Systems Simulation Platform

RecSim is a configurable platform for authoring simulation environments for
recommender systems (RSs) that naturally supports **sequential interaction**
with users. RecSim allows the creation of new environments that reflect
particular aspects of user behavior and item structure at a level of abstraction
well-suited to pushing the limits of current reinforcement learning (RL) and RS
techniques in sequential interactive recommendation problems. Environments can
be easily configured that vary assumptions about: user preferences and item
familiarity; user latent state and its dynamics; and choice models and other
user response behavior. We outline how RecSim offers value to RL and RS
researchers and practitioners, and how it can serve as a vehicle for
academic-industrial collaboration. For a detailed description of the RecSim
architecture please read [Ie et al](https://arxiv.org/abs/1909.04847). Please
cite the paper if you use the code from this repository in your work.

### Bibtex

```
@article{ie2019recsim,
    title={RecSim: A Configurable Simulation Platform for Recommender Systems},
    author={Eugene Ie and Chih-wei Hsu and Martin Mladenov and Vihan Jain and Sanmit Narvekar and Jing Wang and Rui Wu and Craig Boutilier},
    year={2019},
    eprint={1909.04847},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

<a id='Disclaimer'></a>

## Disclaimer

This is not an officially supported Google product.

## What's new

*   **12/13/2019:** Added (abstract) classes for both multi-user environments
    and agents. Added bandit algorithms for generalized linear models.

## Installation and Sample Usage

It is recommended to install RecSim using (https://pypi.org/project/recsim/):

```shell
pip install recsim
```

However, the latest version of Dopamine is not in PyPI as of December, 2019. We
want to install the latest version from Dopamine's repository like the following
before we install RecSim. Note that Dopamine requires Tensorflow 1.15.0 which is
the final 1.x release including GPU support for Ubuntu and Windows.

```
pip install git+https://github.com/google/dopamine.git
```

Here are some sample commands you could use for testing the installation:

```
git clone https://github.com/google-research/recsim
cd recsim/recsim
python main.py --logtostderr \
  --base_dir="/tmp/recsim/interest_exploration_full_slate_q" \
  --agent_name=full_slate_q \
  --environment_name=interest_exploration \
  --episode_log_file='episode_logs.tfrecord' \
  --gin_bindings=simulator.runner_lib.Runner.max_steps_per_episode=100 \
  --gin_bindings=simulator.runner_lib.TrainRunner.num_iterations=10 \
  --gin_bindings=simulator.runner_lib.TrainRunner.max_training_steps=100 \
  --gin_bindings=simulator.runner_lib.EvalRunner.max_eval_episodes=5
```

You could then start a tensorboard and view the output

```
tensorboard --logdir=/tmp/recsim/interest_exploration_full_slate_q/ --port=2222
```

You could also find the simulated logs in /tmp/recsim/episode_logs.tfrecord

## Tutorials

To get started, please check out our Colab tutorials. In
[**RecSim: Overview**](recsim/colab/RecSim_Overview.ipynb),
we give a brief overview about RecSim. We then talk about each configurable
component:
[**environment**](recsim/colab/RecSim_Developing_an_Environment.ipynb)
and
[**recommender agent**](recsim/colab/RecSim_Developing_an_Agent.ipynb).

## Documentation


Please refer to the [white paper](http://arxiv.org/abs/1909.04847) for the
high-level design.
