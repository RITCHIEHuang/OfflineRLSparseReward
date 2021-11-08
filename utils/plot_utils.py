import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from utils.io_util import proj_path

sns.set_style("white")
sns.set_context("paper", font_scale=1)


def plot_ep_reward(data_list: list, names: list, config: dict, suffix=""):
    plt.figure()
    df = pd.DataFrame(data_list)
    df = df.T
    df.columns = names
    sns.lineplot(data=df, legend=False, ci=None)
    sns.despine(top=False, right=False, left=False, bottom=False)
    plt.xlabel("Time step")
    plt.ylabel("Reward")
    # plt.title(f"{config['task']}-{config['delay_tag']}")
    plt.legend(names)
    fig_dir = f"{proj_path}/assets/{config['task']}"
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    plt.savefig(f"{fig_dir}/{config['delay_tag']}_{suffix}.png")

    for name in names:
        plt.figure()
        sns.displot(data=df[name], kind="hist", legend=False)
        plt.xlabel("Value")
        sns.despine(top=False, right=False, left=False, bottom=False)
        plt.savefig(
            f"{fig_dir}/{config['delay_tag']}_distribution_{name}_{suffix}.png"
        )


def plot_reward_dist(data_list: list, names: list, config: dict, suffix=""):
    plt.figure()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()
    for idx, ax in enumerate(axes):
        if idx >= len(data_list):
            break
        ax.hist(data_list[idx], color="blue", edgecolor="black", bins=1000)
        ax.set_xlabel("val")
        ax.set_ylabel("proportion")
        ax.set_title(names[idx])

    fig_dir = f"{proj_path}/assets/{config['task']}"
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    plt.savefig(f"{fig_dir}/{config['delay_tag']}_distribution_{suffix}.png")
