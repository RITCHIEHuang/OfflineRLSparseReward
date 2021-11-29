import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from utils.io_util import proj_path

sns.set_style("white")
sns.set_context("paper", font_scale=1)


def plot_ep_reward(data_list: list, names: list, config: dict, suffix=""):
    df = pd.DataFrame(data_list)
    df = df.T
    df.columns = names
    n_row, n_col = 2, len(names)

    fig, axes = plt.subplots(n_row, n_col, figsize=(12, 8))
    cur_r, cur_c = 0, 0
    ax = axes[cur_r, cur_c]
    sns.lineplot(data=df, legend=False, ci=None, ax=ax)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Reward")

    for idx in range(n_col):
        name = names[idx]
        cur_c += 1
        if cur_c == n_col:
            cur_r += 1
            cur_c = 0
        ax = axes[cur_r, cur_c]
        sns.distplot(df[name], ax=ax)

    sns.despine(top=False, right=False, left=False, bottom=False)
    fig_dir = f"{proj_path}/assets/{config['task']}"
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    fig.tight_layout()
    fig.savefig(f"{fig_dir}/{config['delay_tag']}_{suffix}.png")


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


def plot_line(df, x_label, y_label, hue_label, style_label, fig_name):
    # Plot the responses for different events and regions
    plt.figure()
    sns.pointplot(
        x=x_label,
        y=y_label,
        hue=hue_label,
        style=style_label,
        data=df,
        linestyles="-.",
        markers="^",
    )
    plt.savefig(f"{proj_path}/assets/{fig_name}.png")


if __name__ == "__main__":
    df = pd.read_csv(f"{proj_path}/assets/Delay_results.csv")
    plot_line(
        df,
        x_label="Delay",
        y_label="D4rl_Score",
        hue_label="Strategy",
        style_label="Strategy",
        fig_name="performanc_vs_delay_size",
    )
