# 运行教程

本文只保留当前项目的标准运行方式。所有命令在项目根目录执行：

```powershell
cd E:\master\课程\深度强化学习与决策系统设计方法\group_project
```

环境创建完成后，后续运行命令前先激活环境：

```powershell
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

## 0.1 Python 环境安装

推荐使用 `environment.yml` 创建 conda 环境：

```powershell
conda env create -f environment.yml
conda activate agv-drl-rware
```

如果已经创建过环境，可以用下面的命令更新：

```powershell
conda env update -f environment.yml
conda activate agv-drl-rware
```

也可以使用 `requirements.txt` 安装依赖。该方式适合已经有 Python 3.11 环境、只想用 pip 安装项目依赖的情况。注意：必须先完成上一节的 `robotic-warehouse` 下载，因为 `requirements.txt` 中包含本地 RWARE 环境：

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

安装完成后建议先做一次依赖检查：

```powershell
python -c "import numpy, gymnasium, networkx, pyglet, six, PIL, torch, rware; print('dependency check ok')"
```

本项目已经用隔离的临时 conda 环境验证过 `requirements.txt`：可以正常安装 `numpy`、`torch`、`gymnasium`、`networkx`、`pyglet`、`six`、`Pillow` 和本地 editable 的 `robotic-warehouse/rware`，并且可以跑通最小训练与最小评测命令。验证环境未写入现有 conda 环境。

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

## 4. 一次性运行 5 个不同 seed 的稳定性评测

课程要求展示多次运行结果是否稳定时，需要在同一张正式地图、同样 AGV 数量和同样 5000 steps 下评测 5 个不同 seed。建议同时运行 planner baseline 和训练后的 LTF-PPO，便于对比。

不训练 follower 的 planner baseline：

```powershell
python -m agv_drl.experiments.evaluate_ltf_ppo `
  --controller planner `
  --horizon 5000 `
  --agents 30 `
  --request-queue-size 10 `
  --seeds "0,1,2,3,4" `
  --render `
  --render-every 1 `
  --save-gif `
  --gif-frame-ms 80 `
  --run-name eval_planner_5seeds
```

训练后的 LTF-PPO：

```powershell
python -m agv_drl.experiments.evaluate_ltf_ppo `
  --controller ltfp `
  --checkpoint artifacts\agv_runs\train_ltfp_final_300k\checkpoints\ltfp_latest.pt `
  --horizon 5000 `
  --agents 30 `
  --request-queue-size 10 `
  --seeds "0,1,2,3,4" `
  --render `
  --render-every 1 `
  --save-gif `
  --gif-frame-ms 80 `
  --run-name eval_ltfp_5seeds
```

这两个命令会为每个 seed 保存一个 GIF。文件名会自动用 controller 和 seed 区分，例如：

```text
gifs/planner_seed0.gif
gifs/planner_seed1.gif
gifs/planner_seed2.gif
gifs/planner_seed3.gif
gifs/planner_seed4.gif

gifs/ltfp_seed0.gif
gifs/ltfp_seed1.gif
gifs/ltfp_seed2.gif
gifs/ltfp_seed3.gif
gifs/ltfp_seed4.gif
```

输出目录中重点查看：

```text
eval_planner.csv 或 eval_ltfp.csv
summary_planner.json 或 summary_ltfp.json
step_logs/
gifs/
```

`eval_ltfp.csv` 每一行对应一个 seed，其中 `completed_tasks` 是该次 5000 steps 的完整闭环任务数。`summary_ltfp.json` 中的：

```text
mean_completed_tasks
std_completed_tasks
rows[].completed_tasks
```

分别表示 5 次运行的平均完成数、标准差，以及每个 seed 的完成数。报告稳定性时直接列出 5 个 `completed_tasks`，并补充平均值和标准差即可。

## 5. 一次性运行全部调参训练、5 seeds 评测并保存 GIF

下面的 PowerShell 脚本会依次训练 7 组参数，并在每组训练完成后自动找到该组最新 checkpoint，随后执行 5 seeds 评测并保存 GIF。所有实验固定使用：

```text
agents              30
horizon             5000
request_queue_size  10
total_steps         300000
training_layouts    4
eval seeds          0,1,2,3,4
```

运行前确认已经位于项目根目录，并已激活环境：

```powershell
cd E:\master\课程\深度强化学习与决策系统设计方法\group_project
conda activate agv-drl-rware
```

完整调参脚本已经写入：

```text
scripts/run_tuning.py
```

直接运行完整 300000-step 调参、5 seeds 评测并保存 GIF：

```bash
python scripts/run_tuning.py
```

该默认脚本覆盖三类主要训练参数：

```text
learning_rate          学习率
entropy_coef           探索强度
waypoint/wait reward   奖励设计
```

如果要更完整对应评分项中提到的 map configuration 和 training episodes/steps，再加入扩展实验：

```bash
python scripts/run_tuning.py --include-extended
```

`--include-extended` 会额外增加：

```text
tune_steps_short   total_steps=150000，用于和默认 300000 steps 对比训练轮数影响
tune_map_1layout   training_layout_count=1，用于比较训练地图数量较少时的泛化表现
tune_map_6layouts  training_layout_count=6，用于比较更多相似训练地图时的泛化表现
```

如果只是快速检查脚本流程，可以临时减少训练步数并不保存 GIF：

```bash
python scripts/run_tuning.py --total-steps 100000 --no-gif
```

推荐使用 Python 版脚本，因为它会通过 `sys.executable` 调用当前已激活 conda 环境中的 Python，避免嵌套 PowerShell 时 `python` 解析到其它解释器。

每组训练目录会形如：

```text
artifacts/agv_runs/tuning_batch_YYYYMMDD_HHMMSS/
  logs/
    batch.log
    tune_base_train.log
    tune_base_eval.log
    ...
  tuning_manifest.json
  tuning_summary.csv
  tune_base/
  tune_lr_low/
  tune_lr_high/
  tune_entropy_low/
  tune_entropy_high/
  tune_reward_mild/
  tune_reward_strong/
```

每组评测结果会写入该组训练目录下的 `evals/`，例如：

```text
artifacts/agv_runs/tuning_batch_YYYYMMDD_HHMMSS/tune_base/evals/eval_tune_base_5seeds_gif_YYYYMMDD_HHMMSS/
  eval_ltfp.csv
  summary_ltfp.json
  eval_config.json
  step_logs/
  gifs/ltfp_seed0.gif
  gifs/ltfp_seed1.gif
  gifs/ltfp_seed2.gif
  gifs/ltfp_seed3.gif
  gifs/ltfp_seed4.gif
```

调参对比时主要整理以下字段：

```text
run_name
learning_rate
entropy_coef
waypoint_reward
active_wait_penalty
mean_completed_tasks
std_completed_tasks
rows[].completed_tasks
mean blocked_moves
```

其中 `tuning_summary.csv` 会自动汇总每组参数和 5 seeds 评测结果，包括 `total_steps` 和 `training_layout_count`；`logs/batch.log` 记录完整批次流程；`logs/*_train.log` 和 `logs/*_eval.log` 分别保存每组训练和评测的完整终端输出。

注意：该脚本默认会训练 7 个 300000-step 模型，并为每个模型保存 5 个 5000-step GIF，运行时间和磁盘占用都比较大。如果只是快速筛选参数，可以先使用 `--total-steps 100000 --no-gif`；最终报告用的模型再按完整命令运行。
