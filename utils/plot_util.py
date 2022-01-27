import math
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
    "#2196F3",
    "#3F51B5",
    "#009688",
    "#795548",
    "#607D8B",
]

sns.set(
    # style="white",
    style="darkgrid",
    # font_scale=1.8,
    # font_scale=1.0,
    font_scale=1.5,
    context="paper",
    # rc={"lines.linewidth": 1.5},
    rc={"lines.linewidth": 1.5},
    # rc={"lines.linewidth": 1.4},
    palette=sns.color_palette(material),
)

# sns.set_context("paper", font_scale=1)


def plot_ep_reward(data_list: list, names: list, config: dict, suffix=""):
    df = pd.DataFrame(data_list)
    df = df.T
    names = [name.capitalize() for name in names]
    df.columns = names

    fig1, ax = plt.subplots(1, 1, figsize=(6, 4))
    sns.lineplot(data=df, legend=False, ci=None, ax=ax)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Reward")

    if len(names) >= 4:
        row = 2
        col = math.ceil(len(names) // 2)
        figsize = (12, 7)
    else:
        row = 1
        col = len(names)
        figsize = (14, 4)
    fig2, axes = plt.subplots(row, col, figsize=figsize)
    for idx, ax in enumerate(axes):
        sns.distplot(df[names[idx]], ax=ax, color=material[idx])

    fig_dir = f"{proj_path}/assets/{config['task']}"
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    fig1.tight_layout()
    fig1.savefig(f"{fig_dir}/{config['delay_tag']}_{suffix}.pdf")

    fig2.tight_layout()
    fig2.savefig(f"{fig_dir}/{config['delay_tag']}_{suffix}_distribution.pdf")


##############################################################################
def process_pointplot(
    dfs,
    r,
    c,
    x_key,
    y_key,
    hue_key,
    style_key,
    x_label="Delay interval size (K)",
    y_label="D4RL Score",
    fig_name="",
    title_func=lambda: "",
):
    fig, axes = plt.subplots(r, c, figsize=(12, 5))

    for idx, ax in enumerate(axes):
        df = dfs[idx]
        title = title_func(df)

        sns.pointplot(
            x=x_key,
            y=y_key,
            hue=hue_key,
            style=style_key,
            data=df,
            linestyles=["-", "-.", "--"],
            markers=["o", "s", "^"],
            ax=ax,
            hue_order=["IUPM", "IUS", "None"],
            errwidth=0.8,
            ci=None,
        )
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)

    for ax in axes:
        ax.legend([], [], frameon=False)
    plt.tight_layout()
    plt.savefig(f"{proj_path}/assets/{fig_name}.pdf")


def process_barplot(
    dfs, row, col, x_key, y_key, hue_key, fig_name="", hue_order=None
):
    fig, axes = plt.subplots(row, col, figsize=(12, 5))
    if row * col == 1:
        axes = [axes]

    for idx, ax in enumerate(axes):
        df = dfs[idx]
        dset_name = ()
        print("dataset name", dset_name)

        sns.barplot(
            x=x_key,
            y=y_key,
            hue=hue_key,
            data=df,
            estimator=np.median,
            ax=ax,
            hue_order=hue_order,
        )
        ax.set_xlabel("Delay interval size (K)")
        ax.set_ylabel("D4RL Score")
        ax.set_title(dset_name)

    for ax in axes:
        ax.legend([], [], frameon=False)
    plt.tight_layout()
    plt.savefig(f"{proj_path}/assets/{fig_name}.pdf")


def process_lineplot(
    dfs,
    row,
    col,
    x_key,
    y_key,
    hue_key=None,
    style_key=None,
    x_label="",
    y_label="D4RL Score",
    fig_name="",
    hue_order=None,
    ci=95,
    title_func=lambda x: "",
):
    if row == 1:
        fig_size = (12, 5)
    elif row == 2:
        fig_size = (12, 7)
    elif row == 3:
        fig_size = (15, 10)
    fig, axes = plt.subplots(row, col, figsize=fig_size)
    if row * col == 1:
        axes = [axes]

    for idx, ax in enumerate(axes.flatten()):
        df = dfs[idx]
        title = title_func(df)
        print("Processing plot", title)

        sns.lineplot(
            x=x_key,
            y=y_key,
            hue=hue_key,
            style=style_key,
            data=df,
            ax=ax,
            ci=ci,
            hue_order=hue_order,
        )
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)

    for ax in axes.flatten():
        ax.legend([], [], frameon=False)
    plt.tight_layout()
    plt.savefig(f"{proj_path}/assets/{fig_name}.pdf")


def plot_performance_under_shapings():
    file_dir = f"{proj_path}/assets/antmaze-medium-play-v2"
    fig_name = "Performance_vs_reward_shapings"
    file_paths = sorted(
        [p for p in os.listdir(file_dir) if p.endswith("v2.csv")]
    )
    dfs = []

    df = pd.read_csv(os.path.join(file_dir, file_paths[0]))
    df["class"] = (
        df["Reward_Scale"].astype("str")
        + "-"
        + df["Reward_Shift"].astype("str")
    )
    df = df[df["Iteration"] <= 1000]
    df = df[df["class"].isin(["10--0.5", "4--0.5", "1--1.0", "1-0.0"])]

    dfs.append(df[df["Algo"] == "iql"].reset_index())
    dfs.append(df[df["Algo"] == "cql"].reset_index())
    print("Datasets num", len(dfs))

    process_lineplot(
        dfs,
        1,
        2,
        x_key="Iteration",
        y_key="D4rl_Score",
        hue_key="class",
        style_key="class",
        fig_name=fig_name,
        hue_order=["10--0.5", "4--0.5", "1--1.0", "1-0.0"],
        title_func=lambda df: df["Environment"][0].capitalize()
        + "-"
        + df["Dataset Type"][0],
    )


def plot_iql_reimp_antmaze():
    file_dir = f"{proj_path}/assets/iql/d4rl"
    fig_name = "IQL_reimpl_antmaze"
    file_paths = sorted(
        [p for p in os.listdir(file_dir) if p.endswith("v2.csv")]
    )
    dfs = []

    for p in file_paths:
        df = pd.read_csv(os.path.join(file_dir, p))
        dfs.append(df)
    print("Datasets num", len(dfs))
    process_lineplot(
        dfs,
        2,
        3,
        x_key="Iteration",
        y_key="D4rl_Score",
        fig_name=fig_name,
        title_func=lambda df: df["Environment"][0].capitalize() + "-" + df["Dataset Type"][0]
    )


def plot_iql_reimp_mujoco():
    file_dir = f"{proj_path}/assets/iql/d4rl"
    fig_name = "IQL_reimpl_mujoco"
    file_paths = sorted(
        [p for p in os.listdir(file_dir) if p.endswith("v0.csv")]
    )
    dfs = []

    for p in file_paths:
        df = pd.read_csv(os.path.join(file_dir, p))
        dfs.append(df)
    print("Datasets num", len(dfs))
    process_lineplot(
        dfs,
        3,
        4,
        x_key="Iteration",
        y_key="D4rl_Score",
        fig_name=fig_name,
        title_func=lambda df: df["Environment"][0].capitalize() + "-" + df["Dataset Type"][0]
    )


def plot_mopo_reimp_mujoco():
    file_dir = f"{proj_path}/assets/mopo/d4rl"
    fig_name = "MOPO_reimpl_mujoco"
    file_paths = sorted(
        [p for p in os.listdir(file_dir) if p.endswith("v0.csv")]
    )
    dfs = []

    for p in file_paths:
        df = pd.read_csv(os.path.join(file_dir, p))
        dfs.append(df)
    print("Datasets num", len(dfs))
    process_lineplot(
        dfs,
        3,
        4,
        x_key="Iteration",
        y_key="D4rl_Score",
        fig_name=fig_name,
        title_func=lambda df: df["Environment"][0].capitalize() + "-" + df["Dataset Type"][0]
    )


def plot_iql_mujoco_strategy():
    file_dir = f"{proj_path}/assets/iql/d4rl"
    fig_name = "IQL_mujoco_strategy"
    file_paths = sorted(
        [p for p in os.listdir(file_dir) if p.endswith("v0-strategy.csv")]
    )
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
        3,
        4,
        x_key="Iteration",
        y_key="D4rl_Score",
        hue_key="Strategy",
        style_key="Strategy",
        fig_name=fig_name,
        hue_order=["IUPM", "IUS", "None"],
        title_func=lambda df: df["Environment"][0].capitalize() + "-" + df["Dataset Type"][0]
    )


def plot_mopo_mujoco_strategy():
    file_dir = f"{proj_path}/assets/mopo/d4rl"
    fig_name = "MOPO_mujoco_strategy"
    file_paths = sorted(
        [p for p in os.listdir(file_dir) if p.endswith("v0-strategy.csv")]
    )
    dfs = []

    for p in file_paths:
        df = pd.read_csv(os.path.join(file_dir, p), skiprows=lambda x: x > 0 and x % 50 != 0)
        df = df.replace("interval_ensemble", "IUPM")
        df = df.replace("interval_average", "IUS")
        df = df.replace("none", "None")
        dfs.append(df)
    print("Datasets num", len(dfs))
    process_lineplot(
        dfs,
        3,
        4,
        x_key="Iteration",
        y_key="D4rl_Score",
        hue_key="Strategy",
        style_key="Strategy",
        fig_name=fig_name,
        hue_order=["IUPM", "IUS", "None"],
        title_func=lambda df: df["Environment"][0].capitalize() + "-" + df["Dataset Type"][0]
    )


def plot_iql_antmaze_strategy():
    file_dir = f"{proj_path}/assets/iql/d4rl"
    fig_name = "IQL_antmaze_strategy"
    file_paths = sorted(
        [p for p in os.listdir(file_dir) if p.endswith("v2-strategy.csv")]
    )
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
        2,
        3,
        x_key="Iteration",
        y_key="D4rl_Score",
        hue_key="Strategy",
        style_key="Strategy",
        fig_name=fig_name,
        hue_order=["IUPM", "IUS", "None"],
        title_func=lambda df: df["Environment"][0].capitalize() + "-" + df["Dataset Type"][0]
    )


def plot_delay_vs_nodelay(
    file_paths=[
        f"{proj_path}/assets/results_mopo_walker2d_medium_v0.csv",
        f"{proj_path}/assets/results_mopo_halfcheetah_medium_replay_v0.csv",
    ],
    fig_name="MOPO_Delay_vs_None-Delay",
):
    dfs = []
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        df = df.replace("interval_average", "IUS")
        df = df.replace("interval_ensemble", "IUPM")
        df = df.replace("none", "None")
        dfs.append(df)
    process_barplot(
        dfs,
        1,
        2,
        x_key="Delay",
        y_key="D4rl_Score",
        hue_key="Strategy",
        row=1,
        col=2,
        fig_name=fig_name,
        hue_order=["IUPM", "IUS", "None"],
    )


def plot_delay_interval(
    file_paths=[
        f"{proj_path}/assets/results_mopo_delay_walker2d_medium_v0.csv",
        f"{proj_path}/assets/results_mopo_delay_halfcheetah_medium_replay_v0.csv",
    ],
    fig_name="MOPO_vs_Delay_interval_size",
):
    dfs = []
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        df = df.replace("interval_average", "IUS")
        df = df.replace("interval_ensemble", "IUPM")
        df = df.replace("none", "None")
        dfs.append(df)
    process_pointplot(
        dfs,
        1,
        2,
        x_key="Delay",
        y_key="D4rl_Score",
        hue_key="Strategy",
        style_key="Strategy",
        fig_name=fig_name,
        title_func=lambda df: df["Environment"][0].capitalize()
        + "-"
        + df["Dataset Type"][0],
    )


if __name__ == "__main__":
    # mopo with different delay intervals
    # plot_delay_interval()

    # # mopo mujoco v0 impl
    # plot_mopo_reimp_mujoco()

    # # mopo mujoco v0 strategy
    plot_mopo_mujoco_strategy()

    # # iql, cql under different reward shapings
    # plot_performance_under_shapings()

    # # iql antmaze reimpl
    # plot_iql_reimp_antmaze()

    # # iql mujoco v0 impl
    # plot_iql_reimp_mujoco()

    # # iql antmaze strategy
    # plot_iql_antmaze_strategy()

    # # iql mujoco strategy
    # plot_iql_mujoco_strategy()
