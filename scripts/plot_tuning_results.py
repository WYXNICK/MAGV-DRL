from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev


RUN_LABELS = {
    "tune_base": "Base\n(default)",
    "tune_lr_low": "1e-4",
    "tune_lr_high": "5e-4",
    "tune_entropy_low": "0.015",
    "tune_entropy_high": "0.060",
    "tune_reward_mild": "Mild",
    "tune_reward_strong": "Strong",
    "tune_steps_short": "150k",
    "tune_map_1layout": "1 layout",
    "tune_map_6layouts": "6 layouts",
}


CATEGORY_SPECS = [
    {
        "key": "learning_rate",
        "title": "Learning Rate Sensitivity",
        "subtitle": "300k training steps, 4 generated layouts, 5 evaluation seeds",
        "x_label": "Learning rate",
        "runs": ["tune_lr_low", "tune_base", "tune_lr_high"],
        "color": "#2F80ED",
        "accent": "#0B3D91",
        "filename": "comparison_learning_rate.png",
    },
    {
        "key": "entropy",
        "title": "Exploration Coefficient Sensitivity",
        "subtitle": "Higher entropy keeps the local follower stochastic in congested regions",
        "x_label": "Entropy coefficient",
        "runs": ["tune_entropy_low", "tune_base", "tune_entropy_high"],
        "color": "#8E63D9",
        "accent": "#4C2A85",
        "filename": "comparison_entropy.png",
    },
    {
        "key": "reward",
        "title": "Reward Shaping Sensitivity",
        "subtitle": "Waypoint reward and active-wait penalty jointly tune path following",
        "x_label": "Reward setting",
        "runs": ["tune_reward_mild", "tune_base", "tune_reward_strong"],
        "color": "#F28E2B",
        "accent": "#9A4D00",
        "filename": "comparison_reward.png",
    },
    {
        "key": "layout_count",
        "title": "Training Layout Count Sensitivity",
        "subtitle": "Generated layouts share the warehouse family but are not the target map",
        "x_label": "Number of generated training layouts",
        "runs": ["tune_map_1layout", "tune_base", "tune_map_6layouts"],
        "color": "#14A085",
        "accent": "#005B4F",
        "filename": "comparison_training_layouts.png",
    },
    {
        "key": "training_steps",
        "title": "Training Steps Sensitivity",
        "subtitle": "Shorter training is compared with the 300k-step default",
        "x_label": "Training steps",
        "runs": ["tune_steps_short", "tune_base"],
        "color": "#D45087",
        "accent": "#7A1745",
        "filename": "comparison_training_steps.png",
    },
]


OVERALL_ORDER = [
    "tune_base",
    "tune_lr_low",
    "tune_lr_high",
    "tune_entropy_low",
    "tune_entropy_high",
    "tune_reward_mild",
    "tune_reward_strong",
    "tune_steps_short",
    "tune_map_1layout",
    "tune_map_6layouts",
]


OVERALL_COLORS = {
    "tune_base": "#2F80ED",
    "tune_lr_low": "#56CC9D",
    "tune_lr_high": "#27AE60",
    "tune_entropy_low": "#B388EB",
    "tune_entropy_high": "#7B61FF",
    "tune_reward_mild": "#F7B267",
    "tune_reward_strong": "#F28E2B",
    "tune_steps_short": "#D45087",
    "tune_map_1layout": "#7AC7C4",
    "tune_map_6layouts": "#14A085",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot MAGV-DRL tuning results as publication-quality PNG figures."
    )
    parser.add_argument(
        "--batch-dir",
        type=Path,
        default=Path("artifacts/agv_runs/tuning_batch"),
        help="Tuning batch directory containing tuning_seed_results.csv.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to artifacts/figures.",
    )
    parser.add_argument(
        "--planner-summary",
        type=Path,
        default=Path("artifacts/agv_runs/eval_planner_5seeds/summary_planner.json"),
        help="Planner baseline 5-seed summary JSON used for the baseline-vs-LTF-PPO figure.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_csv = args.batch_dir / "tuning_seed_results.csv"
    if not seed_csv.exists():
        raise FileNotFoundError(f"Missing seed-level results: {seed_csv}")

    out_dir = args.out_dir or Path("artifacts/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_seed_rows(seed_csv)
    summaries = summarize(rows)

    import matplotlib.pyplot as plt

    configure_matplotlib(plt)
    for spec in CATEGORY_SPECS:
        plot_category(plt, spec, summaries, rows, out_dir)
    plot_overall(plt, summaries, rows, out_dir)
    if args.planner_summary.exists() and "tune_base" in summaries:
        plot_planner_vs_ltfp(plt, args.planner_summary, summaries["tune_base"], rows, out_dir)
    print(f"Wrote figures to {out_dir}")


def read_seed_rows(path: Path) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            parsed: dict[str, object] = dict(row)
            for key in [
                "learning_rate",
                "entropy_coef",
                "waypoint_reward",
                "active_wait_penalty",
                "total_steps",
                "training_layout_count",
                "seed",
                "completed_tasks",
                "delivered_to_goal",
                "generated_tasks",
                "mean_agent_reward",
                "blocked_moves",
                "forced_toggles",
            ]:
                parsed[key] = to_float(row.get(key, ""))
            result.append(parsed)
    if not result:
        raise ValueError(f"No rows in {path}")
    return result


def to_float(value: object) -> float:
    if value is None or value == "":
        return math.nan
    return float(value)


def summarize(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    by_run: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_run[str(row["run_name"])].append(row)

    result: dict[str, dict[str, object]] = {}
    for run_name, run_rows in by_run.items():
        completed = [float(row["completed_tasks"]) for row in run_rows]
        blocked = [float(row["blocked_moves"]) for row in run_rows]
        first = run_rows[0]
        result[run_name] = {
            "run_name": run_name,
            "label": RUN_LABELS.get(run_name, run_name),
            "mean_completed": mean(completed),
            "std_completed": pstdev(completed) if len(completed) > 1 else 0.0,
            "mean_blocked": mean(blocked),
            "std_blocked": pstdev(blocked) if len(blocked) > 1 else 0.0,
            "learning_rate": float(first["learning_rate"]),
            "entropy_coef": float(first["entropy_coef"]),
            "waypoint_reward": float(first["waypoint_reward"]),
            "active_wait_penalty": float(first["active_wait_penalty"]),
            "total_steps": float(first["total_steps"]),
            "training_layout_count": float(first["training_layout_count"]),
            "n_seeds": len(completed),
        }
    return result


def configure_matplotlib(plt) -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 360,
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.labelsize": 11.5,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.linewidth": 0.95,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def plot_category(plt, spec, summaries, rows, out_dir: Path) -> None:
    items = [summaries[name] for name in spec["runs"] if name in summaries]
    if not items:
        return
    seed_by_name = collect_seed_values(rows, metric="completed_tasks")

    fig, ax = plt.subplots(figsize=(7.2, 4.9))
    x = list(range(len(items)))
    means = [item["mean_completed"] for item in items]
    stds = [item["std_completed"] for item in items]
    labels = [item["label"] for item in items]
    colors = make_gradient(spec["color"], len(items))

    bars = ax.bar(
        x,
        means,
        yerr=stds,
        color=colors,
        edgecolor="#172033",
        linewidth=0.9,
        capsize=5,
        width=0.62,
        error_kw={"elinewidth": 1.35, "capthick": 1.35, "ecolor": "#172033"},
        zorder=2,
    )
    for idx, item in enumerate(items):
        values = seed_by_name[str(item["run_name"])]
        offsets = symmetric_offsets(len(values), width=0.34)
        ax.scatter(
            [idx + off for off in offsets],
            values,
            s=34,
            facecolor="white",
            edgecolor=spec["accent"],
            linewidth=1.1,
            zorder=4,
        )
        ax.text(
            idx,
            means[idx] + stds[idx] + 42,
            f"{means[idx]:.1f}\n±{stds[idx]:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#172033",
        )

    best_idx = max(range(len(means)), key=lambda i: means[i])
    bars[best_idx].set_edgecolor("#111827")
    bars[best_idx].set_linewidth(1.5)
    bars[best_idx].set_hatch("///")

    ax.set_title(spec["title"], pad=18)
    ax.text(
        0.5,
        1.01,
        spec["subtitle"],
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        color="#4A5568",
        fontsize=9.5,
    )
    ax.set_xlabel(spec["x_label"])
    ax.set_ylabel("Completed tasks in 5,000 steps")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(means[i] + stds[i] for i in range(len(means))) + 230)
    style_axis(ax)
    fig.tight_layout()
    save_figure(fig, out_dir, Path(spec["filename"]).stem)
    plt.close(fig)


def plot_overall(plt, summaries, rows, out_dir: Path) -> None:
    available = [name for name in OVERALL_ORDER if name in summaries]
    items = [summaries[name] for name in available]
    seed_by_name = collect_seed_values(rows, metric="completed_tasks")

    fig, ax = plt.subplots(figsize=(11.2, 5.3))
    x = list(range(len(items)))
    means = [item["mean_completed"] for item in items]
    stds = [item["std_completed"] for item in items]
    labels = [item["label"] for item in items]
    colors = [OVERALL_COLORS[str(item["run_name"])] for item in items]

    bars = ax.bar(
        x,
        means,
        yerr=stds,
        color=colors,
        edgecolor="#172033",
        linewidth=0.85,
        capsize=4,
        width=0.68,
        error_kw={"elinewidth": 1.25, "capthick": 1.25, "ecolor": "#172033"},
        alpha=0.94,
        zorder=2,
    )
    for idx, item in enumerate(items):
        values = seed_by_name[str(item["run_name"])]
        offsets = symmetric_offsets(len(values), width=0.42)
        ax.scatter(
            [idx + off for off in offsets],
            values,
            s=28,
            facecolor="white",
            edgecolor="#172033",
            linewidth=0.9,
            zorder=4,
        )
    best_idx = max(range(len(means)), key=lambda i: means[i])
    bars[best_idx].set_hatch("///")
    bars[best_idx].set_linewidth(1.5)
    ax.annotate(
        "best mean",
        xy=(best_idx, means[best_idx] + stds[best_idx]),
        xytext=(best_idx, means[best_idx] + stds[best_idx] + 92),
        ha="center",
        color="#172033",
        fontsize=9.5,
        arrowprops={"arrowstyle": "-|>", "lw": 1.0, "color": "#172033"},
    )

    ax.set_title("Overall Hyperparameter Tuning Comparison", pad=18)
    ax.text(
        0.5,
        1.01,
        "Bars show mean completed tasks; error bars show standard deviation over 5 seeds",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        color="#4A5568",
        fontsize=9.5,
    )
    ax.set_ylabel("Completed tasks in 5,000 steps")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=28, ha="right")
    ax.set_ylim(0, max(means[i] + stds[i] for i in range(len(means))) + 240)
    style_axis(ax)
    fig.tight_layout()
    save_figure(fig, out_dir, "overall_tuning_comparison")
    plt.close(fig)


def plot_planner_vs_ltfp(plt, planner_summary_path: Path, ltfp_summary, rows, out_dir: Path) -> None:
    planner_rows = read_planner_summary(planner_summary_path)
    ltfp_values = collect_seed_values(rows, metric="completed_tasks").get("tune_base", [])
    ltfp_blocked = collect_seed_values(rows, metric="blocked_moves").get("tune_base", [])
    if not planner_rows or not ltfp_values:
        return

    planner_completed = [float(row["completed_tasks"]) for row in planner_rows]
    planner_blocked = [float(row["blocked_moves"]) for row in planner_rows]
    groups = [
        {
            "label": "Planner\nbaseline",
            "completed": planner_completed,
            "blocked": planner_blocked,
            "color": "#9AA4B2",
            "edge": "#4B5563",
        },
        {
            "label": "LTF-PPO\nstandard",
            "completed": ltfp_values,
            "blocked": ltfp_blocked,
            "color": "#2F80ED",
            "edge": "#0B3D91",
        },
    ]

    fig = plt.figure(figsize=(10.6, 4.9))
    grid = fig.add_gridspec(
        1,
        2,
        left=0.08,
        right=0.985,
        bottom=0.13,
        top=0.76,
        wspace=0.30,
    )
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    plot_baseline_metric_panel(
        axes[0],
        groups,
        metric="completed",
        ylabel="Completed tasks in 5,000 steps",
        title="Task Completion",
        higher_is_better=True,
    )
    plot_baseline_metric_panel(
        axes[1],
        groups,
        metric="blocked",
        ylabel="Blocked moves",
        title="Congestion",
        higher_is_better=False,
    )
    fig.suptitle("Planner Baseline vs Standard LTF-PPO", y=0.96, fontsize=15, fontweight="bold")
    fig.text(
        0.5,
        0.895,
        "Mean over 5 fixed evaluation seeds; error bars show standard deviation and dots show individual seeds",
        ha="center",
        va="top",
        color="#4A5568",
        fontsize=9.5,
    )
    save_figure(fig, out_dir, "planner_baseline_vs_ltfp_standard")
    plt.close(fig)


def read_planner_summary(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    return list(payload.get("rows", []))


def plot_baseline_metric_panel(ax, groups, metric: str, ylabel: str, title: str, higher_is_better: bool) -> None:
    x = list(range(len(groups)))
    values = [group[metric] for group in groups]
    means = [mean(series) for series in values]
    stds = [pstdev(series) if len(series) > 1 else 0.0 for series in values]
    bars = ax.bar(
        x,
        means,
        yerr=stds,
        color=[group["color"] for group in groups],
        edgecolor=[group["edge"] for group in groups],
        linewidth=1.1,
        capsize=5,
        width=0.58,
        error_kw={"elinewidth": 1.35, "capthick": 1.35, "ecolor": "#172033"},
        zorder=2,
    )
    best_idx = max(range(len(means)), key=lambda idx: means[idx])
    if not higher_is_better:
        best_idx = min(range(len(means)), key=lambda idx: means[idx])
    bars[best_idx].set_hatch("///")
    bars[best_idx].set_linewidth(1.6)

    for idx, series in enumerate(values):
        offsets = symmetric_offsets(len(series), width=0.34)
        ax.scatter(
            [idx + off for off in offsets],
            series,
            s=38,
            facecolor="white",
            edgecolor=groups[idx]["edge"],
            linewidth=1.05,
            zorder=4,
        )
        ax.text(
            idx,
            means[idx] + stds[idx] + max(means + stds) * 0.035,
            f"{means[idx]:.1f}\n±{stds[idx]:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#172033",
        )

    ax.set_title(title, pad=10)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels([group["label"] for group in groups])
    ax.set_ylim(0, max(means[i] + stds[i] for i in range(len(means))) * 1.22)
    style_axis(ax)


def collect_seed_values(rows, metric: str) -> dict[str, list[float]]:
    result: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        result[str(row["run_name"])].append((int(float(row["seed"])), float(row[metric])))
    return {name: [value for _, value in sorted(values)] for name, values in result.items()}


def style_axis(ax) -> None:
    ax.grid(axis="y", color="#D8DEE9", linestyle="-", linewidth=0.8, alpha=0.85, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", colors="#172033")
    ax.spines["left"].set_color("#172033")
    ax.spines["bottom"].set_color("#172033")


def make_gradient(hex_color: str, n: int) -> list[str]:
    if n <= 1:
        return [hex_color]
    import matplotlib.colors as mcolors

    base = mcolors.to_rgb(hex_color)
    result = []
    for i in range(n):
        mix = 0.20 + 0.55 * (i / max(1, n - 1))
        rgb = tuple((1.0 - mix) * 1.0 + mix * channel for channel in base)
        result.append(mcolors.to_hex(rgb))
    return result


def symmetric_offsets(n: int, width: float) -> list[float]:
    if n <= 1:
        return [0.0]
    step = width / max(1, n - 1)
    return [-width / 2 + i * step for i in range(n)]


def save_figure(fig, out_dir: Path, stem: str) -> None:
    fig.savefig(out_dir / f"{stem}.png", bbox_inches="tight")


if __name__ == "__main__":
    main()
