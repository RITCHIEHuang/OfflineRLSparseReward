import os

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from utils.io_util import proj_path

flatui = [
    "#9b59b6",
    "#3498db",
    "#95a5a6",
    "#e74c3c",
    "#34495e",
    "#2ecc71",
    "#e67e22",
    "#f1c40f",
]


material = [
    "#E91E63",
    "#FFC107",
    "#9C27B0",
    "#3F51B5",
    "#2196F3",
    "#009688",
    "#795548",
    "#607D8B",
]

sns.set(
    # style="white",
    style="darkgrid",
    # font_scale=1.8,
    font_scale=1.0,
    context="paper",
    # rc={"lines.linewidth": 1.5},
    rc={"lines.linewidth": 1.2},
    palette=sns.color_palette(material),
)

# sns.set_context("paper", font_scale=1)


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


def process_pointplot(dfs, x_label, y_label, hue_label, style_label, fig_name):
    # Plot the responses for different events and regions
    fig, axes = plt.subplots(1, len(dfs), figsize=(12, 5))

    for idx, ax in enumerate(axes):
        df = dfs[idx]
        dset_name = (
            df["Environment"][0].capitalize() + "-" + df["Dataset Type"][0]
        )
        print("dataset name", dset_name)

        # sns.barplot(
        #     x=x_label,
        #     y=y_label,
        #     hue=hue_label,
        #     # style=style_label,
        #     data=df,
        #     estimator=np.median,
        #     # errcolor="c",
        #     # linestyles="-.",
        #     # markers=True,
        #     ax=ax,
        #     hue_order=["IUPM", "IUS", "None"],
        # )
        g = sns.pointplot(
            x=x_label,
            y=y_label,
            hue=hue_label,
            # style=style_label,
            data=df,
            linestyles=["-", "-.", "--"],
            markers=["o", "s", "^"],
            ax=ax,
            hue_order=["IUPM", "IUS", "None"],
            err_kws={"alpha": 0.1},
            errwidth=0.8,
            ci=None,
        )
        ax.set_xlabel(r"$Delay\ interval\ size\ (K)$")
        ax.set_ylabel(r"$D4rl\ score$")
        ax.set_title(dset_name)

    # plt.suptitle(fig_name)

    for ax in axes:
        ax.legend([], [], frameon=False)
    plt.tight_layout()
    plt.savefig(f"{proj_path}/assets/{fig_name}.pdf")


def plot_delay_interval(
    file_paths=[
        f"{proj_path}/assets/Delay_results_walker2d.csv",
        f"{proj_path}/assets/Delay_results_halfcheetah.csv",
    ],
    fig_name="Performance vs Delay interval size",
):
    dfs = []
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        df = df.replace("interval_average", "IUS")
        df = df.replace("interval_ensemble", "IUPM")
        df = df.replace("none", "None")
        # print(df)
        dfs.append(df)
    process_pointplot(
        dfs,
        x_label="Delay",
        y_label="D4rl_Score",
        hue_label="Strategy",
        style_label="Strategy",
        fig_name=fig_name,
    )


def process_barplot(dfs, x_label, y_label, hue_label, style_label, fig_name):
    # Plot the responses for different events and regions
    fig, axes = plt.subplots(1, len(dfs), figsize=(12, 5))

    for idx, ax in enumerate(axes):
        df = dfs[idx]
        dset_name = (
            df["Environment"][0].capitalize() + "-" + df["Dataset Type"][0]
        )
        print("dataset name", dset_name)

        # sns.barplot(
        #     x=x_label,
        #     y=y_label,
        #     hue=hue_label,
        #     # style=style_label,
        #     data=df,
        #     estimator=np.median,
        #     # errcolor="c",
        #     # linestyles="-.",
        #     # markers=True,
        #     ax=ax,
        #     hue_order=["IUPM", "IUS", "None"],
        # )
        ax.set_xlabel(r"$Delay\ interval\ size\ (K)$")
        ax.set_ylabel(r"$D4rl\ score$")
        ax.set_title(dset_name)

    # plt.suptitle(fig_name)

    for ax in axes:
        ax.legend([], [], frameon=False)
    plt.tight_layout()
    plt.savefig(f"{proj_path}/assets/{fig_name}.pdf")


def plot_delay_vs_nodelay(
    file_paths=[
        f"{proj_path}/assets/Delay_results_walker2d.csv",
        f"{proj_path}/assets/Delay_results_halfcheetah.csv",
    ],
    fig_name="Delay vs None-Delay",
):
    dfs = []
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        df = df.replace("interval_average", "IUS")
        df = df.replace("interval_ensemble", "IUPM")
        df = df.replace("none", "None")
        # print(df)
        dfs.append(df)
    process_barplot(
        dfs,
        x_label="Delay",
        y_label="D4rl_Score",
        hue_label="Strategy",
        style_label="Strategy",
        fig_name=fig_name,
    )


def process_lineplot(
    dfs, x_label, y_label, hue_label=None, style_label=None, fig_name=""
):
    # Plot the responses for different events and regions
    fig, axes = plt.subplots(3, len(dfs) // 3, figsize=(12, 8))

    for idx, ax in enumerate(axes.flatten()):
        df = dfs[idx]
        dset_name = (
            df["Environment"][0].capitalize() + "-" + df["Dataset Type"][0]
        )
        print("dataset name", dset_name)

        sns.lineplot(
            x=x_label,
            y=y_label,
            hue=hue_label,
            style=style_label,
            data=df,
            ax=ax,
            ci="sd",
            hue_order=["IUPM", "IUS", "None"],
        )
        # ax.set_xlabel(r"$Delay\ interval\ size\ (K)$")
        ax.set_xlabel("")
        ax.set_ylabel(r"$D4rl\ score$")
        ax.set_title(dset_name)

    # plt.suptitle(fig_name)

    for ax in axes.flatten():
        ax.legend([], [], frameon=False)
    plt.tight_layout()
    plt.savefig(f"{proj_path}/assets/{fig_name}.pdf")


def plot_implementation(
    file_dir=f"{proj_path}/assets/iql/d4rl",
    fig_name="IQL_performance_mujoco",
):

    file_paths = sorted([p for p in os.listdir(file_dir) if p.endswith("v0-strategy.csv")])
    # file_paths = sorted([p for p in os.listdir(file_dir) if "strategy" in p])
    dfs = []
    for p in file_paths:
        df = pd.read_csv(os.path.join(file_dir, p))
        df = df.replace("interval_ensemble", "IUPM")
        df = df.replace("interval_average", "IUS")
        df = df.replace("none", "None")
        dfs.append(df)
    print("Datasets num", len(dfs))

    process_lineplot(
        dfs,
        x_label="Iteration",
        y_label="D4rl_Score",
        hue_label="Strategy",
        style_label="Strategy",
        fig_name=fig_name,
    )


if __name__ == "__main__":
    # plot_delay_interval()
    plot_implementation()
