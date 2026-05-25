# 运行教程

本文只保留当前项目的标准运行方式。所有命令在项目根目录执行：

```powershell
cd E:\master\课程\深度强化学习与决策系统设计方法\group_project
conda activate agv-drl-rware
```

## 0. 外部环境代码准备

本仓库只提交 `agv_drl` 算法实现、地图、环境配置文件和中文说明文档，不提交外部 `robotic-warehouse` 源码。运行前需要在项目根目录下载 RWARE 官方代码，使目录结构如下：

```text
group_project/
  agv_drl/
  maps/
  robotic-warehouse/
```

下载命令：

```powershell
git clone https://github.com/semitable/robotic-warehouse.git robotic-warehouse
```

当前代码会在启动时自动把项目根目录下的 `robotic-warehouse` 加入 Python import path，因此不需要把它提交到本仓库，也不需要作为 submodule。

## 1. 不训练 follower 的 baseline 评测

该命令只运行任务状态机和 A* reference planner，不加载 PPO follower 模型。用于和训练后的 Learn-to-Follow PPO 方法对比。

```powershell
python -m agv_drl.experiments.evaluate_ltf_ppo `
  --controller planner `
  --horizon 5000 `
  --agents 30 `
  --request-queue-size 10 `
  --seeds "0" `
  --render `
  --render-every 1 `
  --save-gif `
  --gif-frame-ms 80 `
  --run-name eval_planner_baseline
```

输出会自动写入新的评测目录，包含：

```text
eval_planner.csv
summary_planner.json
eval_config.json
step_logs/
gifs/planner_seed0.gif
```

## 2. 标准训练 300000 steps

该命令训练 Learn-to-Follow PPO follower。A* 只提供 reference path，PPO policy 负责局部跟随和避碰。

```powershell
python -m agv_drl.learning.train_ltf_ppo `
  --total-steps 300000 `
  --agents 30 `
  --horizon 5000 `
  --request-queue-size 10 `
  --training-layout-count 4 `
  --regenerate-training-layout `
  --learning-rate 2.5e-4 `
  --entropy-coef 0.035 `
  --waypoint-reward 0.03 `
  --active-wait-penalty -0.02 `
  --rollout-steps 128 `
  --minibatch-size 1024 `
  --runs-root artifacts/agv_runs `
  --run-name train_ltfp_final_300k
```

训练输出目录示例：

```text
artifacts/agv_runs/train_ltfp_final_300k_YYYYMMDD_HHMMSS/
  checkpoints/ltfp_latest.pt
  training_config.json
  training_log.csv
```

## 3. 使用最终模型进行 LTF-PPO 评测并保存 GIF

当前最终模型路径：

```text
artifacts/agv_runs/train_ltfp_final_300k/checkpoints/ltfp_latest.pt
```

标准评测命令：

```powershell
python -m agv_drl.experiments.evaluate_ltf_ppo `
  --controller ltfp `
  --checkpoint artifacts\agv_runs\train_ltfp_final_300k\checkpoints\ltfp_latest.pt `
  --horizon 5000 `
  --agents 30 `
  --request-queue-size 10 `
  --seeds "0" `
  --render `
  --render-every 1 `
  --save-gif `
  --gif-frame-ms 80 `
  --run-name eval_ltfp_seed0_gif
```

输出会写入最终训练目录下的 `evals/` 子目录，包含：

```text
eval_ltfp.csv
summary_ltfp.json
eval_config.json
step_logs/ltfp_seed0_steps.csv
gifs/ltfp_seed0.gif
```

正式指标看 `summary_ltfp.json` 中的：

```text
completed_tasks
```

`completed_tasks` 表示完整闭环任务数：取货架、送到目标工作站、驮着货架返回原位、卸货完成。仅送到工作站但尚未返回原位卸货的任务不计入完成数。
