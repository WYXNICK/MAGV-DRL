from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List

import numpy as np
from PIL import Image

from ..config import COURSE_MAP_PATH, DEFAULT_RESULTS_DIR, WarehouseConfig
from ..env.task_env import AGVTaskEnv
from ..learning.action_fusion import ActionFusionStats, movement_intents_to_rware_actions
from ..learning.features import build_agent_features
from ..learning.policy import LocalFollowerPolicy, require_torch
from ..planning.astar import PrioritizedAStarController
from ..run_utils import create_run_dir, default_eval_root, find_latest_checkpoint


def run_episode(args, seed: int) -> Dict[str, float]:
    config = WarehouseConfig(
        n_agents=args.agents,
        horizon=args.horizon,
        seed=seed,
        request_queue_size=args.request_queue_size,
        sensor_range=args.sensor_range,
        dynamic_cost_weight=args.dynamic_cost_weight,
        layout_file=args.layout_file,
    )
    task_env = AGVTaskEnv(config, render_mode="rgb_array" if args.render else "human")
    task_env.reset(seed=seed)
    planner = PrioritizedAStarController(
        task_env,
        reserve_dynamic=args.controller == "planner",
    )
    follower = None
    if args.controller == "ltfp":
        if not args.checkpoint.exists():
            raise FileNotFoundError(
                f"Missing LTF-PPO checkpoint: {args.checkpoint}. "
                "Run python -m agv_drl.learning.train_ltf_ppo first, or use --controller planner for a diagnostic run."
            )
        follower = LocalFollowerPolicy(args.checkpoint, args.device)
        torch, _, _, _ = require_torch()
        torch.manual_seed(args.policy_seed + seed)
    total_reward = np.zeros(config.n_agents, dtype=np.float64)
    blocked = 0
    fusion_totals = ActionFusionStats()
    log_writer, log_file = make_step_logger(args, seed)
    gif_frames: List[Image.Image] = []
    for step in range(config.horizon):
        reference_actions = planner.reference_actions()
        fusion_stats = ActionFusionStats()
        if args.controller == "planner":
            actions = reference_actions
        else:
            reference_paths = planner.reference_paths()
            features = build_agent_features(task_env, reference_paths)
            learned_actions, _ = follower.act(features)
            actions, fusion_stats = movement_intents_to_rware_actions(
                task_env,
                reference_actions,
                learned_actions,
            )
            fusion_totals.add(fusion_stats)
        _, rewards, done, truncated, _ = task_env.step(actions)
        total_reward += np.asarray(rewards, dtype=np.float64)
        blocked += sum(int(event.blocked) for event in task_env.last_events)
        if log_writer is not None:
            write_step_log(log_writer, seed, step, task_env, actions, rewards, fusion_stats)
        if args.render and step % args.render_every == 0:
            frame = task_env.env.render()
            if args.save_gif:
                gif_frames.append(Image.fromarray(frame))
        if done or truncated:
            break
    if log_file is not None:
        log_file.close()
    gif_path = None
    if args.save_gif and gif_frames:
        gif_dir = args.run_dir / "gifs"
        gif_dir.mkdir(parents=True, exist_ok=True)
        gif_path = gif_dir / f"{args.controller}_seed{seed}.gif"
        gif_frames[0].save(
            gif_path,
            save_all=True,
            append_images=gif_frames[1:],
            duration=args.gif_frame_ms,
            loop=0,
        )
    result = {
        "seed": seed,
        "steps": step + 1,
        "completed_tasks": task_env.manager.completed_tasks,
        "delivered_to_goal": task_env.manager.delivered_to_goal,
        "generated_tasks": task_env.manager.generated_tasks,
        "mean_agent_reward": float(total_reward.mean()),
        "blocked_moves": blocked,
        "policy_actions": fusion_totals.policy_actions,
        "forced_toggles": fusion_totals.forced_toggles,
        "gif_path": str(gif_path) if gif_path is not None else None,
    }
    task_env.close()
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LTF-PPO AGV cooperative transport scheduling.")
    parser.add_argument(
        "--controller",
        choices=["ltfp", "planner"],
        default="ltfp",
        help="'ltfp' is the trained Learn-to-Follow PPO local follower. 'planner' is only a diagnostic/warm-start oracle.",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", default="0,1,2,3,4", help="Comma separated evaluation seeds.")
    parser.add_argument("--horizon", type=int, default=5000)
    parser.add_argument("--agents", type=int, default=30)
    parser.add_argument("--request-queue-size", type=int, default=10)
    parser.add_argument("--sensor-range", type=int, default=2)
    parser.add_argument("--dynamic-cost-weight", type=float, default=0.75)
    parser.add_argument(
        "--layout-file",
        type=Path,
        default=COURSE_MAP_PATH,
        help="RWARE custom layout. Defaults to the course example map in maps/course_map.txt.",
    )
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument(
        "--policy-seed",
        type=int,
        default=0,
        help="Base random seed used when stochastic policy sampling is enabled.",
    )
    parser.add_argument(
        "--no-step-log",
        action="store_true",
        help="Disable per-step CSV logs. Logs are enabled by default for reproducibility.",
    )
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--render-every", type=int, default=25)
    parser.add_argument(
        "--save-gif",
        action="store_true",
        help="Save rendered frames as gifs/<controller>_seed<seed>.gif under the evaluation run directory. Requires --render.",
    )
    parser.add_argument("--gif-frame-ms", type=int, default=110, help="Frame duration for saved GIF.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.controller == "ltfp" and args.checkpoint is None:
        latest = find_latest_checkpoint(args.runs_root)
        if latest is None:
            raise FileNotFoundError(
                f"No checkpoint found under {args.runs_root}. "
                "Pass --checkpoint explicitly after training."
            )
        args.checkpoint = latest
    results_root = args.results_root or default_eval_root(args.checkpoint, args.runs_root)
    args.run_dir = create_run_dir(f"eval_{args.controller}", results_root, args.run_name, args.run_dir)
    if args.save_gif and not args.render:
        raise ValueError("--save-gif requires --render so RWARE frames can be captured.")
    print(f"Evaluation run directory: {args.run_dir}")
    if args.checkpoint is not None:
        print(f"Checkpoint: {args.checkpoint}")
    if args.controller == "ltfp":
        print("Policy action selection: stochastic sampling")
        print(
            "RWARE load/unload handling: task state machine forces required TOGGLE_LOAD; "
            "the neural policy outputs only grid-move intents."
        )
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    rows: List[Dict[str, float]] = []
    for seed in seeds:
        row = run_episode(args, seed)
        rows.append(row)
        print(
            f"seed={seed} completed={row['completed_tasks']} "
            f"delivered={row['delivered_to_goal']} blocked={row['blocked_moves']}"
        )

    csv_path = args.run_dir / f"eval_{args.controller}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    completed = [float(r["completed_tasks"]) for r in rows]
    summary = {
        "controller": args.controller,
        "policy_seed": args.policy_seed,
        "checkpoint": str(args.checkpoint) if args.checkpoint is not None else None,
        "seeds": seeds,
        "mean_completed_tasks": mean(completed),
        "std_completed_tasks": pstdev(completed) if len(completed) > 1 else 0.0,
        "rows": rows,
    }
    summary_path = args.run_dir / f"summary_{args.controller}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    config_path = args.run_dir / "eval_config.json"
    config_path.write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {csv_path}, {summary_path}, and {config_path}")


def make_step_logger(args, seed: int):
    if args.no_step_log:
        return None, None
    log_dir = args.run_dir / "step_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = (log_dir / f"{args.controller}_seed{seed}_steps.csv").open(
        "w", newline="", encoding="utf-8"
    )
    fieldnames = [
        "seed",
        "step",
        "completed_tasks",
        "delivered_to_goal",
        "pending_tasks",
        "active_to_pickup",
        "active_to_deliver",
        "active_returning",
        "pickup_events",
        "deliver_events",
        "complete_events",
        "blocked_events",
        "mean_reward",
        "noop_actions",
        "forward_actions",
        "toggle_actions",
        "policy_actions",
        "forced_toggles",
    ]
    writer = csv.DictWriter(log_file, fieldnames=fieldnames)
    writer.writeheader()
    return writer, log_file


def write_step_log(
    writer,
    seed: int,
    step: int,
    task_env: AGVTaskEnv,
    actions,
    rewards,
    fusion_stats: ActionFusionStats,
) -> None:
    phases = Counter(task.phase.name for task in task_env.manager.active_by_agent.values())
    events = task_env.last_events
    writer.writerow(
        {
            "seed": seed,
            "step": step,
            "completed_tasks": task_env.manager.completed_tasks,
            "delivered_to_goal": task_env.manager.delivered_to_goal,
            "pending_tasks": len(task_env.manager.pending),
            "active_to_pickup": phases.get("TO_PICKUP", 0),
            "active_to_deliver": phases.get("TO_DELIVER", 0),
            "active_returning": phases.get("RETURNING", 0),
            "pickup_events": sum(int(event.pickup) for event in events),
            "deliver_events": sum(int(event.delivered) for event in events),
            "complete_events": sum(int(event.completed) for event in events),
            "blocked_events": sum(int(event.blocked) for event in events),
            "mean_reward": float(np.mean(rewards)),
            "noop_actions": sum(int(action == 0) for action in actions),
            "forward_actions": sum(int(action == 1) for action in actions),
            "toggle_actions": sum(int(action == 4) for action in actions),
            "policy_actions": fusion_stats.policy_actions,
            "forced_toggles": fusion_stats.forced_toggles,
        }
    )


if __name__ == "__main__":
    main()
