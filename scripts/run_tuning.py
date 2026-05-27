from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


EXPERIMENTS = [
    {"name": "tune_base", "lr": "2.5e-4", "ent": "0.035", "wp": "0.03", "wait": "-0.02"},
    {"name": "tune_lr_low", "lr": "1.0e-4", "ent": "0.035", "wp": "0.03", "wait": "-0.02"},
    {"name": "tune_lr_high", "lr": "5.0e-4", "ent": "0.035", "wp": "0.03", "wait": "-0.02"},
    {"name": "tune_entropy_low", "lr": "2.5e-4", "ent": "0.015", "wp": "0.03", "wait": "-0.02"},
    {"name": "tune_entropy_high", "lr": "2.5e-4", "ent": "0.060", "wp": "0.03", "wait": "-0.02"},
    {"name": "tune_reward_mild", "lr": "2.5e-4", "ent": "0.035", "wp": "0.02", "wait": "-0.01"},
    {"name": "tune_reward_strong", "lr": "2.5e-4", "ent": "0.035", "wp": "0.05", "wait": "-0.03"},
]


EXTENDED_EXPERIMENTS = [
    {
        "name": "tune_steps_short",
        "lr": "2.5e-4",
        "ent": "0.035",
        "wp": "0.03",
        "wait": "-0.02",
        "total_steps": 150_000,
    },
    {
        "name": "tune_map_1layout",
        "lr": "2.5e-4",
        "ent": "0.035",
        "wp": "0.03",
        "wait": "-0.02",
        "training_layout_count": 1,
    },
    {
        "name": "tune_map_6layouts",
        "lr": "2.5e-4",
        "ent": "0.035",
        "wp": "0.03",
        "wait": "-0.02",
        "training_layout_count": 6,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all LTF-PPO tuning trainings, then evaluate each checkpoint on five seeds."
    )
    parser.add_argument("--total-steps", type=int, default=300_000)
    parser.add_argument("--agents", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=5000)
    parser.add_argument("--request-queue-size", type=int, default=10)
    parser.add_argument("--training-layout-count", type=int, default=4)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--runs-root", type=Path, default=Path("artifacts/agv_runs"))
    parser.add_argument(
        "--batch-name",
        default="tuning_batch",
        help="Prefix of the new directory that stores all tuning runs and logs.",
    )
    parser.add_argument(
        "--include-extended",
        action="store_true",
        help="Also run map-configuration and training-step tuning experiments.",
    )
    parser.add_argument("--no-gif", action="store_true", help="Skip rendering and GIF export.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    check_torch_available()
    batch_dir = create_batch_dir(args.runs_root, args.batch_name)
    logs_dir = batch_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    batch_log = logs_dir / "batch.log"
    summary_csv = batch_dir / "tuning_summary.csv"
    seed_summary_csv = batch_dir / "tuning_seed_results.csv"
    manifest_json = batch_dir / "tuning_manifest.json"
    write_manifest(args, batch_dir, manifest_json)
    log_message(batch_log, f"Batch directory: {batch_dir}")
    log_message(batch_log, f"Python interpreter: {sys.executable}")
    log_message(batch_log, f"Seeds: {args.seeds}")
    log_message(batch_log, f"Save GIF: {not args.no_gif}")
    log_message(batch_log, f"Include extended tuning: {args.include_extended}")

    summary_rows: list[dict[str, object]] = []
    seed_summary_rows: list[dict[str, object]] = []
    experiments = list(EXPERIMENTS)
    if args.include_extended:
        experiments.extend(EXTENDED_EXPERIMENTS)

    for experiment in experiments:
        name = experiment["name"]
        train_run_dir = batch_dir / name
        train_log = logs_dir / f"{name}_train.log"
        eval_log = logs_dir / f"{name}_eval.log"
        total_steps = int(experiment.get("total_steps", args.total_steps))
        training_layout_count = int(experiment.get("training_layout_count", args.training_layout_count))
        log_message(batch_log, "")
        log_message(batch_log, f"==== Training {name} ====")
        log_message(
            batch_log,
            "params: "
            f"learning_rate={experiment['lr']}, entropy_coef={experiment['ent']}, "
            f"waypoint_reward={experiment['wp']}, active_wait_penalty={experiment['wait']}, "
            f"total_steps={total_steps}, training_layout_count={training_layout_count}",
        )
        train_cmd = [
            sys.executable,
            "-m",
            "agv_drl.learning.train_ltf_ppo",
            "--total-steps",
            str(total_steps),
            "--agents",
            str(args.agents),
            "--horizon",
            str(args.horizon),
            "--request-queue-size",
            str(args.request_queue_size),
            "--training-layout-count",
            str(training_layout_count),
            "--regenerate-training-layout",
            "--learning-rate",
            experiment["lr"],
            "--entropy-coef",
            experiment["ent"],
            "--waypoint-reward",
            experiment["wp"],
            "--active-wait-penalty",
            experiment["wait"],
            "--rollout-steps",
            "128",
            "--minibatch-size",
            "1024",
            "--run-dir",
            str(train_run_dir),
        ]
        run(train_cmd, train_log, batch_log)

        checkpoint = train_run_dir / "checkpoints" / "ltfp_latest.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")

        log_message(batch_log, f"==== Evaluating {name} with seeds {args.seeds} ====")
        eval_cmd = [
            sys.executable,
            "-m",
            "agv_drl.experiments.evaluate_ltf_ppo",
            "--controller",
            "ltfp",
            "--checkpoint",
            str(checkpoint),
            "--horizon",
            str(args.horizon),
            "--agents",
            str(args.agents),
            "--request-queue-size",
            str(args.request_queue_size),
            "--seeds",
            args.seeds,
            "--run-name",
            f"eval_{name}_5seeds_gif",
        ]
        if not args.no_gif:
            eval_cmd.extend(["--render", "--render-every", "1", "--save-gif", "--gif-frame-ms", "80"])
        run(eval_cmd, eval_log, batch_log)

        eval_dir = latest_eval_dir(train_run_dir, f"eval_{name}_5seeds_gif")
        row, seed_rows = summarize_run(experiment, train_run_dir, eval_dir)
        row["total_steps"] = total_steps
        row["training_layout_count"] = training_layout_count
        for seed_row in seed_rows:
            seed_row["total_steps"] = total_steps
            seed_row["training_layout_count"] = training_layout_count
        summary_rows.append(row)
        seed_summary_rows.extend(seed_rows)
        write_summary_csv(summary_csv, summary_rows)
        write_seed_summary_csv(seed_summary_csv, seed_summary_rows)
        log_message(
            batch_log,
            "result: "
            f"mean_completed_tasks={row.get('mean_completed_tasks')}, "
            f"std_completed_tasks={row.get('std_completed_tasks')}, "
            f"mean_blocked_moves={row.get('mean_blocked_moves')}, "
            f"eval_dir={eval_dir}",
        )

    log_message(batch_log, "")
    log_message(batch_log, "==== All tuning runs finished ====")
    log_message(batch_log, f"Summary CSV: {summary_csv}")
    log_message(batch_log, f"Seed-level CSV: {seed_summary_csv}")


def check_torch_available() -> None:
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "The current Python interpreter cannot import PyTorch. "
            f"Interpreter: {sys.executable}\n"
            "Please activate the conda environment that contains PyTorch, or install it with "
            "`conda env update -f environment.yml` before running this script."
        ) from exc


def run(command: list[str], log_path: Path, batch_log: Path) -> None:
    printable = " ".join(command)
    log_message(batch_log, f"command: {printable}")
    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        log_file.write(f"command: {printable}\n\n")
        log_file.flush()
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
        return_code = process.wait()
        log_file.write(f"\nexit_code: {return_code}\n")
    if return_code != 0:
        log_message(batch_log, f"FAILED with exit_code={return_code}; see {log_path}")
        raise subprocess.CalledProcessError(return_code, command)


def create_batch_dir(runs_root: Path, batch_name: str) -> Path:
    runs_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in batch_name).strip("_")
    base = runs_root / f"{safe_name or 'tuning_batch'}_{stamp}"
    candidate = base
    suffix = 1
    while candidate.exists():
        candidate = runs_root / f"{base.name}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def latest_eval_dir(train_run_dir: Path, eval_prefix: str) -> Path:
    evals_root = train_run_dir / "evals"
    candidates = [
        path for path in evals_root.iterdir()
        if path.is_dir() and path.name.startswith(f"{eval_prefix}_")
    ]
    if not candidates:
        raise FileNotFoundError(f"Cannot find evaluation directory for {eval_prefix} under {evals_root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def summarize_run(
    experiment: dict[str, str],
    train_run_dir: Path,
    eval_dir: Path,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    summary_path = eval_dir / "summary_ltfp.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing evaluation summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    rows = summary.get("rows", [])
    completed = [str(row.get("completed_tasks", "")) for row in rows]
    blocked = [float(row.get("blocked_moves", 0.0)) for row in rows]
    mean_blocked = sum(blocked) / len(blocked) if blocked else ""
    aggregate: dict[str, object] = {
        "run_name": experiment["name"],
        "learning_rate": experiment["lr"],
        "entropy_coef": experiment["ent"],
        "waypoint_reward": experiment["wp"],
        "active_wait_penalty": experiment["wait"],
        "mean_completed_tasks": summary.get("mean_completed_tasks", ""),
        "std_completed_tasks": summary.get("std_completed_tasks", ""),
        "completed_tasks_by_seed": ";".join(completed),
        "mean_blocked_moves": mean_blocked,
        "train_dir": str(train_run_dir),
        "eval_dir": str(eval_dir),
    }
    seed_results: list[dict[str, object]] = []
    for row in rows:
        seed_results.append(
            {
                "run_name": experiment["name"],
                "learning_rate": experiment["lr"],
                "entropy_coef": experiment["ent"],
                "waypoint_reward": experiment["wp"],
                "active_wait_penalty": experiment["wait"],
                "seed": row.get("seed", ""),
                "completed_tasks": row.get("completed_tasks", ""),
                "delivered_to_goal": row.get("delivered_to_goal", ""),
                "generated_tasks": row.get("generated_tasks", ""),
                "mean_agent_reward": row.get("mean_agent_reward", ""),
                "blocked_moves": row.get("blocked_moves", ""),
                "policy_actions": row.get("policy_actions", ""),
                "policy_overrides": row.get("policy_overrides", ""),
                "planner_fallbacks": row.get("planner_fallbacks", ""),
                "forced_toggles": row.get("forced_toggles", ""),
                "invalid_toggle_fallbacks": row.get("invalid_toggle_fallbacks", ""),
                "gif_path": row.get("gif_path", ""),
                "train_dir": str(train_run_dir),
                "eval_dir": str(eval_dir),
            }
        )
    return aggregate, seed_results


def write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fields = [
        "run_name",
        "learning_rate",
        "entropy_coef",
        "waypoint_reward",
        "active_wait_penalty",
        "total_steps",
        "training_layout_count",
        "mean_completed_tasks",
        "std_completed_tasks",
        "completed_tasks_by_seed",
        "mean_blocked_moves",
        "train_dir",
        "eval_dir",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_seed_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fields = [
        "run_name",
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
        "policy_actions",
        "policy_overrides",
        "planner_fallbacks",
        "forced_toggles",
        "invalid_toggle_fallbacks",
        "gif_path",
        "train_dir",
        "eval_dir",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_manifest(args: argparse.Namespace, batch_dir: Path, path: Path) -> None:
    experiments = list(EXPERIMENTS)
    if args.include_extended:
        experiments.extend(EXTENDED_EXPERIMENTS)
    payload = {
        "batch_dir": str(batch_dir),
        "python": sys.executable,
        "total_steps": args.total_steps,
        "agents": args.agents,
        "horizon": args.horizon,
        "request_queue_size": args.request_queue_size,
        "training_layout_count": args.training_layout_count,
        "seeds": args.seeds,
        "save_gif": not args.no_gif,
        "include_extended": args.include_extended,
        "experiments": experiments,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def log_message(path: Path, message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(line + "\n")


if __name__ == "__main__":
    main()
