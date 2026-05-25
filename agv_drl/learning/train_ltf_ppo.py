from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from ..config import COURSE_MAP_PATH, DEFAULT_RESULTS_DIR, TRAIN_MAPS_DIR, WarehouseConfig
from ..env.task_env import AGVTaskEnv
from ..layouts import ensure_similar_training_layouts, layout_stats, read_layout
from ..planning.astar import PrioritizedAStarController
from ..run_utils import create_run_dir
from .action_fusion import MOVE_WAIT, reference_movement_intents, movement_intents_to_rware_actions
from .features import FEATURE_DIM, PATH_RADIUS, build_agent_features
from .policy import PPOConfig, make_actor_critic, require_torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the LTF-PPO local follower policy for RWARE AGV scheduling."
    )
    parser.add_argument("--total-steps", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=5000)
    parser.add_argument("--agents", type=int, default=30)
    parser.add_argument("--request-queue-size", type=int, default=10)
    parser.add_argument("--sensor-range", type=int, default=2)
    parser.add_argument("--dynamic-cost-weight", type=float, default=0.75)
    parser.add_argument(
        "--layout-file",
        type=Path,
        default=None,
        help=(
            "Optional single RWARE training layout for debugging. "
            "By default training uses a generated pool under --training-layout-dir."
        ),
    )
    parser.add_argument("--training-layout-dir", type=Path, default=TRAIN_MAPS_DIR)
    parser.add_argument("--training-layout-count", type=int, default=4)
    parser.add_argument(
        "--reference-layout-file",
        type=Path,
        default=COURSE_MAP_PATH,
        help="Reference target layout used only to derive training-map properties.",
    )
    parser.add_argument(
        "--regenerate-training-layout",
        action="store_true",
        help="Rebuild the default similar training layout before training.",
    )
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--gamma", type=float, default=0.985)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.20)
    parser.add_argument("--entropy-coef", type=float, default=0.035)
    parser.add_argument("--value-coef", type=float, default=0.50)
    parser.add_argument("--max-grad-norm", type=float, default=0.50)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--minibatch-size", type=int, default=1024)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument(
        "--waypoint-reward",
        type=float,
        default=0.03,
        help="Intrinsic reward for reaching the next planner waypoint, matching the original Learn-to-Follow training signal.",
    )
    parser.add_argument(
        "--active-wait-penalty",
        type=float,
        default=-0.02,
        help="Extra training-only penalty when an active AGV chooses WAIT.",
    )
    parser.add_argument("--log-file", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    training_layouts = resolve_training_layouts(args)
    torch, _, optim, Categorical = require_torch()
    torch.manual_seed(args.seed)
    ppo_cfg = PPOConfig(
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_ratio=args.clip_ratio,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        max_grad_norm=args.max_grad_norm,
        rollout_steps=args.rollout_steps,
        minibatch_size=args.minibatch_size,
        update_epochs=args.update_epochs,
        hidden_size=args.hidden_size,
    )
    run_dir = create_run_dir("train_ltfp", args.runs_root, args.run_name, args.run_dir)
    args.run_dir = run_dir
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_file, log_writer = make_train_logger(args)
    print(f"Training run directory: {run_dir}")
    print_training_layouts(training_layouts, args.reference_layout_file)

    env, planner = make_training_env(args, training_layouts[0], args.seed)
    model = make_actor_critic(ppo_cfg.hidden_size).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=ppo_cfg.learning_rate)

    global_steps = 0
    env_steps = 0
    update_idx = 0
    episode_idx = 0
    total_completed_tasks = 0
    total_delivered_to_goal = 0
    while global_steps < args.total_steps:
        obs_buf = []
        act_buf = []
        logp_buf = []
        val_buf = []
        rew_buf = []
        done_buf = []
        wait_action_count = 0
        reference_wait_count = 0
        reference_match_count = 0
        action_count = 0
        for _ in range(ppo_cfg.rollout_steps):
            reference_actions = planner.reference_actions()
            reference_paths = planner.reference_paths()
            reference_intents = reference_movement_intents(reference_paths)
            waypoints = [next_waypoint(path) for path in reference_paths]
            features = build_agent_features(env, reference_paths)
            obs_t = torch.as_tensor(features, dtype=torch.float32, device=args.device)
            logits, values = model(obs_t)
            dist = Categorical(logits=logits)
            sampled = dist.sample()
            movement_intents = sampled.detach().cpu().numpy()
            wait_action_count += int(np.sum(movement_intents == MOVE_WAIT))
            reference_wait_count += int(np.sum(reference_intents == MOVE_WAIT))
            reference_match_count += int(np.sum(movement_intents == reference_intents))
            action_count += env.n_agents
            movement_t = torch.as_tensor(movement_intents, dtype=torch.long, device=args.device)
            logp = dist.log_prob(movement_t)
            actions, _ = movement_intents_to_rware_actions(env, reference_actions, movement_intents)

            _, rewards, done, truncated, _ = env.step(actions)
            rewards = add_waypoint_rewards(env, rewards, waypoints, args.waypoint_reward)
            rewards = add_active_wait_penalties(env, rewards, movement_intents, args.active_wait_penalty)
            total_completed_tasks += sum(int(event.completed) for event in env.last_events)
            total_delivered_to_goal += sum(int(event.delivered) for event in env.last_events)
            obs_buf.append(features)
            act_buf.append(movement_intents)
            logp_buf.append(logp.detach().cpu().numpy())
            val_buf.append(values.detach().cpu().numpy())
            rew_buf.append(np.asarray(rewards, dtype=np.float32))
            done_buf.append(np.full(env.n_agents, float(done or truncated), dtype=np.float32))
            global_steps += env.n_agents
            env_steps += 1
            if done or truncated:
                episode_idx += 1
                layout = training_layouts[episode_idx % len(training_layouts)]
                env.close()
                env, planner = make_training_env(args, layout, args.seed + episode_idx)
                break

        next_features = build_agent_features(env, planner.reference_paths())
        with torch.no_grad():
            _, next_values_t = model(torch.as_tensor(next_features, dtype=torch.float32, device=args.device))
        next_values = next_values_t.cpu().numpy()
        batch = prepare_batch(
            np.asarray(obs_buf),
            np.asarray(act_buf),
            np.asarray(logp_buf),
            np.asarray(val_buf),
            np.asarray(rew_buf),
            np.asarray(done_buf),
            next_values,
            ppo_cfg,
        )
        stats = update(model, optimizer, Categorical, batch, ppo_cfg, args.device)
        update_idx += 1
        if update_idx % 5 == 0:
            save_checkpoint(ckpt_dir / "ltfp_latest.pt", model, ppo_cfg)
        print(
            f"update={update_idx} env_steps={env_steps} agent_steps={global_steps} "
            f"episode={episode_idx} ep_completed={env.manager.completed_tasks} "
            f"total_completed={total_completed_tasks} loss={stats['loss']:.4f} "
            f"reference_match={reference_match_count / max(1, action_count):.3f} "
            f"wait_rate={wait_action_count / max(1, action_count):.3f}"
        )
        log_writer.writerow(
            {
                "update": update_idx,
                "env_steps": env_steps,
                "global_steps": global_steps,
                "episode": episode_idx,
                "episode_completed_tasks": env.manager.completed_tasks,
                "episode_delivered_to_goal": env.manager.delivered_to_goal,
                "total_completed_tasks": total_completed_tasks,
                "total_delivered_to_goal": total_delivered_to_goal,
                "waypoint_reward": args.waypoint_reward,
                "active_wait_penalty": args.active_wait_penalty,
                "wait_action_rate": wait_action_count / max(1, action_count),
                "reference_wait_rate": reference_wait_count / max(1, action_count),
                "reference_action_match_rate": reference_match_count / max(1, action_count),
                "loss": stats["loss"],
                "policy_loss": stats["policy_loss"],
                "value_loss": stats["value_loss"],
                "entropy": stats["entropy"],
            }
        )

    save_checkpoint(ckpt_dir / "ltfp_latest.pt", model, ppo_cfg)
    metadata = vars(args)
    metadata["training_layout_files"] = [str(path) for path in training_layouts]
    metadata["ppo"] = ppo_cfg.__dict__
    metadata["feature_dim"] = FEATURE_DIM
    metadata["path_radius"] = PATH_RADIUS
    metadata["action_space"] = "ltf_grid_move_intents"
    metadata["movement_action_mask"] = False
    (run_dir / "training_config.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    log_file.close()
    env.close()


def resolve_training_layouts(args) -> list[Path]:
    if args.layout_file is not None:
        return [args.layout_file]
    return ensure_similar_training_layouts(
        args.reference_layout_file,
        args.training_layout_dir,
        args.training_layout_count,
        force=args.regenerate_training_layout,
    )


def print_training_layouts(layouts: list[Path], reference_layout: Path) -> None:
    reference_lines = read_layout(reference_layout)
    for idx, layout in enumerate(layouts):
        lines = read_layout(layout)
        stats = layout_stats(lines)
        same_as_reference = lines == reference_lines
        print(
            f"Training layout[{idx}]: {layout} "
            f"({stats.rows}x{stats.cols}, shelves={stats.shelves}, goals={stats.goals})"
        )
        if same_as_reference:
            print("WARNING: this training layout equals the held-out target layout.")


def make_training_env(args, layout_file: Path, seed: int):
    config = WarehouseConfig(
        n_agents=args.agents,
        horizon=args.horizon,
        seed=seed,
        request_queue_size=args.request_queue_size,
        sensor_range=args.sensor_range,
        dynamic_cost_weight=args.dynamic_cost_weight,
        layout_file=layout_file,
    )
    env = AGVTaskEnv(config)
    env.reset(seed=seed)
    return env, PrioritizedAStarController(env, reserve_dynamic=False)


def make_train_logger(args):
    path = args.log_file or (args.run_dir / "training_log.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    f = path.open("w", newline="", encoding="utf-8")
    fieldnames = [
        "update",
        "env_steps",
        "global_steps",
        "episode",
        "episode_completed_tasks",
        "episode_delivered_to_goal",
        "total_completed_tasks",
        "total_delivered_to_goal",
        "waypoint_reward",
        "active_wait_penalty",
        "wait_action_rate",
        "reference_wait_rate",
        "reference_action_match_rate",
        "loss",
        "policy_loss",
        "value_loss",
        "entropy",
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    return f, writer


def prepare_batch(obs, acts, logps, values, rewards, dones, next_values, cfg: PPOConfig):
    steps, n_agents = rewards.shape
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae = np.zeros(n_agents, dtype=np.float32)
    for t in reversed(range(steps)):
        next_nonterminal = 1.0 - dones[t]
        next_value = next_values if t == steps - 1 else values[t + 1]
        delta = rewards[t] + cfg.gamma * next_value * next_nonterminal - values[t]
        last_gae = delta + cfg.gamma * cfg.gae_lambda * next_nonterminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    flat = {
        "obs": obs.reshape(steps * n_agents, obs.shape[-1]),
        "acts": acts.reshape(steps * n_agents),
        "logps": logps.reshape(steps * n_agents),
        "advantages": advantages.reshape(steps * n_agents),
        "returns": returns.reshape(steps * n_agents),
    }
    adv = flat["advantages"]
    flat["advantages"] = (adv - adv.mean()) / (adv.std() + 1e-8)
    return flat


def update(model, optimizer, Categorical, batch, cfg: PPOConfig, device: str):
    torch, _, _, _ = require_torch()
    total = len(batch["acts"])
    indices = np.arange(total)
    sums = {
        "loss": 0.0,
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
    }
    batches = 0
    for _ in range(cfg.update_epochs):
        np.random.shuffle(indices)
        for start in range(0, total, cfg.minibatch_size):
            mb = indices[start : start + cfg.minibatch_size]
            obs = torch.as_tensor(batch["obs"][mb], dtype=torch.float32, device=device)
            acts = torch.as_tensor(batch["acts"][mb], dtype=torch.long, device=device)
            old_logps = torch.as_tensor(batch["logps"][mb], dtype=torch.float32, device=device)
            adv = torch.as_tensor(batch["advantages"][mb], dtype=torch.float32, device=device)
            returns = torch.as_tensor(batch["returns"][mb], dtype=torch.float32, device=device)
            logits, values = model(obs)
            dist = Categorical(logits=logits)
            logps = dist.log_prob(acts)
            ratio = torch.exp(logps - old_logps)
            clipped = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio) * adv
            policy_loss = -torch.min(ratio * adv, clipped).mean()
            value_loss = 0.5 * (returns - values).pow(2).mean()
            entropy = dist.entropy().mean()
            loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            sums["loss"] += float(loss.detach().cpu())
            sums["policy_loss"] += float(policy_loss.detach().cpu())
            sums["value_loss"] += float(value_loss.detach().cpu())
            sums["entropy"] += float(entropy.detach().cpu())
            batches += 1
    return {key: value / max(1, batches) for key, value in sums.items()}


def save_checkpoint(path: Path, model, cfg: PPOConfig) -> None:
    torch, _, _, _ = require_torch()
    torch.save(
        {
            "model": model.state_dict(),
            "hidden_size": cfg.hidden_size,
            "feature_dim": FEATURE_DIM,
            "path_radius": PATH_RADIUS,
            "action_space": "ltf_grid_move_intents",
            "runtime_movement_action_mask": False,
            "ppo": cfg.__dict__,
        },
        path,
    )


def next_waypoint(path):
    if len(path) >= 2:
        return path[1]
    return None


def add_waypoint_rewards(env: AGVTaskEnv, rewards, waypoints, reward_value: float):
    shaped = np.asarray(rewards, dtype=np.float32).copy()
    if reward_value <= 0:
        return shaped
    for agent in env.env.agents:
        waypoint = waypoints[agent.id - 1]
        if waypoint is not None and (agent.x, agent.y) == waypoint:
            shaped[agent.id - 1] += reward_value
    return shaped


def add_active_wait_penalties(env: AGVTaskEnv, rewards, movement_intents, penalty: float):
    shaped = np.asarray(rewards, dtype=np.float32).copy()
    if penalty == 0:
        return shaped
    for agent, intent in zip(env.env.agents, movement_intents):
        if int(intent) != MOVE_WAIT:
            continue
        if env.manager.task_for_agent(agent.id) is None:
            continue
        shaped[agent.id - 1] += penalty
    return shaped


if __name__ == "__main__":
    main()
