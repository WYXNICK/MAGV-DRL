# MAGV-DRL: Multi-AGV RWARE Learn-to-Follow PPO

本文说明当前 `agv_drl` 的实现方案、任务定义、训练设置和结果文件。项目目标是在 `robotic-warehouse` / RWARE 环境中，让 30 台 AGV 在随机任务请求下完成协同调度和三阶段运输闭环：

```text
前往货架 -> 驮起货架并送到目标工作站 -> 驮着同一货架返回原始库位并卸货 -> 执行下一任务
```

## 文档导航

- 运行命令和环境安装步骤见 [TUTORIAL.md](TUTORIAL.md)。
- 本文重点解释实现思路、任务规则、训练地图、调度算法、PPO follower 和结果指标。
- 最终结果目录保存在 `artifacts/agv_runs/eval_planner_baseline/` 和 `artifacts/agv_runs/train_ltfp_final_300k/`。

## 外部依赖

本项目只提交自己的算法代码、地图、配置和结果文件，不提交外部 `robotic-warehouse` 源码。运行前需要在项目根目录单独下载官方 RWARE 仓库：

```powershell
git clone https://github.com/semitable/robotic-warehouse.git robotic-warehouse
```

`agv_drl.project_paths` 会自动把该目录加入 Python import path，后续模块通过 `from rware.warehouse import ...` 使用 RWARE 原生环境。

当前代码不直接 import `learn-to-follow` 仓库中的 SampleFactory/APPO 或 C++ Follower 模块，而是在 `agv_drl` 内实现一个面向 RWARE 三阶段搬运任务的自包含适配版。适配遵循 Learn-to-Follow 的核心结构：启发式 planner 为每个 agent 构造长期参考路径，learnable follower 根据局部观测和路径叠加信息输出局部移动意图，用于解决拥堵、等待和局部避碰问题。这样做的原因是课程任务不仅是 lifelong MAPF，还包含 RWARE 的货架装卸、送达工作站、返回原始库位卸货和任务调度逻辑，原始 `learn-to-follow` 的 Pogema 接口不能直接表达完整搬运闭环。

## 1. 总体结构

核心代码目录如下：

```text
agv_drl/
  config.py                    固定实验配置、奖励配置
  layouts.py                   训练地图生成
  env/
    rware_factory.py           创建 RWARE Warehouse
    task_env.py                三阶段任务状态机、任务队列、AGV-任务匹配
  planning/
    astar.py                   A* reference planner
  learning/
    features.py                Learn-to-Follow 风格局部特征
    action_fusion.py           PPO 移动意图到 RWARE primitive action 的转换
    policy.py                  Actor-Critic 网络与 checkpoint 加载
    train_ltf_ppo.py           PPO 训练入口
  experiments/
    evaluate_ltf_ppo.py        baseline / LTF-PPO 评测、日志、GIF
```

系统分成三层：

- `TaskManager` 负责随机任务流、AGV 与货架匹配、任务阶段切换和 `completed_tasks` 计数。
- `PrioritizedAStarController` 负责根据当前任务阶段生成 reference path。
- PPO local follower 只根据局部观测和 reference path，输出 `WAIT / UP / DOWN / LEFT / RIGHT` 五类移动意图。

其中 `TaskManager` 和 A* planner 是任务建模与参考路径层，负责把课程要求中的“识别任务、分配 AGV、取货架、送到目标、返回原位”转成可执行目标；PPO follower 是深度强化学习部分，负责在 RWARE 碰撞机制下学习如何更好地沿参考路径移动。PPO 不直接修改任务定义，也不决定一个任务是否完成。

## 2. RWARE 环境配置

正式评测固定使用课程地图：

```text
layout_file          maps/course_map.txt
n_agents            30
horizon             5000 environment steps
request_queue_size  10
sensor_range        2
dynamic_cost_weight 0.75
```

地图字符含义沿用 RWARE：

```text
.  通道 / highway，AGV 可通行
x  静止货架 storage cell
g  工作站 / goal
```

RWARE 原生运动规则保留：

- 空载 AGV 可以进入静止货架所在格子，即可以从货架底下穿过。
- 载货 AGV 不能穿过其它静止货架。
- 多 AGV 同格、对向交换等冲突由 RWARE movement graph 处理；失败的 `FORWARD` 会变成原地不动。
- RWARE primitive actions 包括 `NOOP / FORWARD / LEFT / RIGHT / TOGGLE_LOAD`。

当前实现没有设置最大同时执行任务数。`request_queue_size=10` 只表示 pending 随机请求队列大小；已分配任务从 pending 队列中移除，active task 的自然上限是 AGV 数量 30。

`request_queue_size` 是固定随机任务生成策略的一部分。它控制环境中同时暴露给调度器的 pending requested shelves 数量，而不是限制同时工作的 AGV 数。为了公平比较，planner baseline、LTF-PPO 和不同调参实验必须使用相同的 `request_queue_size`。

## 3. 一次完整任务的定义

任务对象是 `TransportTask`，包含：

```text
task_id          任务编号
shelf_id         被请求货架 ID
origin           该货架初始 storage cell
destination      随机选中的工作站 goal
assigned_agent   当前负责该任务的 AGV
phase            TO_PICKUP / TO_DELIVER / RETURNING
delivered        是否已经到达过 goal
```

完整任务必须经历三个阶段。

`TO_PICKUP`：

- AGV 空载前往 `shelf_id` 当前所在货架格。
- 到达货架格后执行 `TOGGLE_LOAD`。
- 只有当 `agent.carrying_shelf.id == task.shelf_id` 时，阶段切换到 `TO_DELIVER`。

`TO_DELIVER`：

- AGV 驮着同一个货架前往 `destination`。
- 当 `agent.carrying_shelf.id == task.shelf_id` 且 AGV 坐标等于 `destination` 时，只记一次 `delivered_to_goal`。
- 此时不卸货，不算完整任务完成，阶段切换到 `RETURNING`。

`RETURNING`：

- AGV 继续驮着同一个货架返回 `origin`。
- 到达 `origin` 后执行 `TOGGLE_LOAD` 放下货架。
- 只有同时满足以下条件，才 `completed_tasks += 1`：

```text
task.delivered == True
agent.carrying_shelf is None
agent 坐标 == task.origin
shelf 坐标 == task.origin
```

因此，`delivered_to_goal` 是“货架送到工作站”的中间指标，`completed_tasks` 才是课程要求比较的完整闭环任务数。

## 4. 随机任务队列与防重复分配

`TaskManager` 维护两个集合：

- `pending`：尚未分配给 AGV 的随机请求队列。
- `active_by_agent` / `active_shelves`：已经分配并正在执行的任务和货架。

一个货架可以进入 pending 队列需要满足：

```text
shelf_id 不在 active_shelves
shelf 当前坐标等于其 origin
shelf 未重复出现在 pending 队列
```

这保证了以下情况不会被再次分配：

- 货架已经被某个 AGV 拿走。
- 货架已经送到 goal，正在返回 origin。
- 货架还没有放回原始 storage cell。

每次分配后，`shelf_id` 会立即加入 `active_shelves`；只有完整返回并卸货后才移除。

## 5. AGV 与货架的匹配方法

当前实现使用匈牙利算法做 AGV-task 匹配。匹配发生在 `TaskManager.assign_idle_agents()` 中。

流程如下：

1. 刷新 pending 任务队列。
2. 找出所有空闲 AGV。
3. 如果有空闲 AGV 停在工作站或工作站下方排队区，优先给这些 AGV 分配任务，减少 goal 区域阻塞。
4. 对候选 AGV 和 pending tasks 构造 Manhattan 距离矩阵：

```text
cost[i, j] = distance(agent_i_position, task_j_shelf_position)
```

5. 使用纯 Python/Numpy 实现的最小代价分配算法 `min_cost_assignment()`，求当前批次总取货距离最小的匹配。
6. 分配成功的任务从 pending 中移除，对应货架加入 `active_shelves`。

相比之前“每轮选最近一对”的贪心策略，匈牙利算法优化的是当前所有空闲 AGV 到 pending 货架的总距离。它只影响任务分配层，不改变任务完成规则、A* 规划、PPO 策略或 RWARE 碰撞规则。

## 6. A* Reference Planner

`PrioritizedAStarController` 给每个 active AGV 生成当前阶段目标的 reference path：

```text
TO_PICKUP   -> 目标为任务货架当前位置
TO_DELIVER  -> 目标为任务 destination goal
RETURNING   -> 目标为任务 origin
```

路径规划中的障碍处理：

- 空载时允许经过静止货架格，因为 RWARE 允许空车从货架底下穿过。
- 载货时把其它静止货架视作障碍。
- 当前 AGV 正驮着的货架不视作障碍。

A* 还维护一个局部动态代价图。邻近其它 AGV 的格子会逐步增加 cost，用于让 reference path 稍微避开近期拥堵区域。

在 planner baseline 中，`reserve_dynamic=True`，planner 会做简单的一步 reservation，避免同一时刻多个 AGV 选择同一 next cell 或对向交换。  
在 LTF-PPO 中，`reserve_dynamic=False`，A* reference path 不硬性禁止穿过其它 AGV，更接近 Learn-to-Follow 的设定：planner 给参考路径，局部冲突由 learnable follower 和 RWARE 环境处理。

因此，`--controller planner` 是一个“不加载 PPO follower 的规则 baseline / 诊断基线”，不是 Learn-to-Follow 论文中的 neural follower。它保留任务状态机、匈牙利任务匹配、A* 和一阶 reservation，目的是提供一个偏强的非学习对照。如果训练后的 LTF-PPO 在相同地图、AGV 数、随机任务队列和 5000 steps 下仍优于该 baseline，说明学习型 follower 对局部拥堵处理确实有贡献。

## 7. Learn-to-Follow PPO Follower

PPO follower 不直接输出 RWARE primitive action，而是输出 Learn-to-Follow 风格的局部移动意图：

```text
0 WAIT
1 UP
2 DOWN
3 LEFT
4 RIGHT
```

输入特征由 `build_agent_features()` 构造，维度为 154，主要包含：

- 是否载货。
- 当前格是否 highway。
- AGV 朝向 one-hot。
- 任务阶段 one-hot。
- pending 队列占用比例。
- next waypoint 相对位置和距离。
- 前、左、右、后四个方向的局部占用信息：是否越界、是否有 AGV、是否有货架。
- 以 AGV 为中心、半径为 5 的 path overlay，其中下一 waypoint 权重更高。

为了避免 PPO 学成独立全局导航器，当前特征不提供绝对坐标和全局目标向量。策略主要学习“如何沿 planner path 在局部拥堵中跟随、等待、绕行或尝试前进”。

策略网络是一个 actor-critic：

```text
Linear(154, hidden_size=256)
LayerNorm
Tanh
Linear(256, 256)
Tanh
actor:  Linear(256, 5)
critic: Linear(256, 1)
```

正式评测使用 stochastic sampling：

```python
dist = Categorical(logits=logits)
actions = dist.sample()
```

没有 deterministic argmax，也没有 action mask、expert mix 或行为克隆参数。评测时每个 seed 会同时影响 RWARE 初始 AGV 位置/朝向、随机任务队列、随机 destination，以及 PPO stochastic policy 的采样序列。

## 8. RWARE Action 转换

`movement_intents_to_rware_actions()` 把 PPO 的局部移动意图转换成 RWARE primitive action：

- 如果 planner reference action 是必要的 `TOGGLE_LOAD`，直接强制执行 `TOGGLE_LOAD`。这保证装货和回 origin 卸货由任务状态机精确控制。
- 如果 PPO 输出 `WAIT`，转换为 `NOOP`。
- 如果 PPO 输出方向意图，且 AGV 当前朝向已经对齐，则转换为 `FORWARD`。
- 如果朝向不对齐，则转换为 `LEFT` 或 `RIGHT` 进行转向。

PPO 不负责决定何时装货或卸货；装卸货只在状态机允许的位置发生：

```text
TO_PICKUP 到达货架格且空载 -> TOGGLE_LOAD 取货
RETURNING 到达 origin 且载货 -> TOGGLE_LOAD 卸货
```

## 9. 奖励设计和训练目标

环境 shaped reward：

```text
step                -0.001
progress_delta      +0.015 * 到当前阶段目标的距离缩短量
pickup              +0.15
delivered_to_goal   +0.40
completed_return    +1.00
blocked_forward     -0.02
wrong_toggle        -0.01
```

Learn-to-Follow 风格辅助信号：

```text
waypoint_reward     +0.03   到达 planner 下一 waypoint
active_wait_penalty -0.02   有 active task 时主动 WAIT
```

PPO loss 形式：

```text
loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
```

因此训练日志中的 `loss` 可以为负值；这通常来自 entropy bonus 和 PPO surrogate objective，不表示任务得分为负。判断训练是否正常应重点看：

- `episode_completed_tasks` 是否增长。
- `wait_action_rate` 是否长期过高。
- `reference_action_match_rate` 是否逐渐提高或在合理范围。
- `value_loss` 是否爆炸。
- 评测中的 `completed_tasks`、`blocked_moves` 和 GIF 是否出现大面积停滞。

## 10. 训练地图构建

正式测试地图是 `maps/course_map.txt`。训练不直接使用目标地图，而是根据目标地图构建相似但不同的训练地图，位于：

```text
maps/train_maps/train_map_similar_00.txt
maps/train_maps/train_map_similar_01.txt
maps/train_maps/train_map_similar_02.txt
maps/train_maps/train_map_similar_03.txt
```

生成逻辑在 `agv_drl/layouts.py`：

1. 读取 `course_map.txt`，检查矩形地图和合法字符。
2. 统计行数、列数、货架数量和 goal 数量。
3. 识别课程地图的三个货架 band。
4. 识别每个 band 中五组两列货架 stack。
5. 保持全局布局类型不变：三层货架区、每层五组双列货架、底部工作站数量不变。
6. 每个 variant 只对一个货架 band 做小幅 `dx/dy` 平移。
7. 检查生成地图的 shelves/goals 数量与目标地图一致，并确保生成地图不等于目标地图。

训练时每个 episode 按顺序轮换这些训练地图。这样既符合“构建训练用地图”的要求，也避免直接在最终测试地图上训练。

如果需要展示“在给定地图上训练和调参”的额外对照，也可以用 `--layout-file maps/course_map.txt` 直接在课程地图训练；但标准设置把 `course_map.txt` 作为最终评测地图，把训练地图作为与其结构相似但不完全相同的构造地图，以减少只记住固定地图细节的风险。

## 11. 训练与评测参数

标准训练参数：

```text
total_steps         300000 agent steps
agents              30
horizon             5000
request_queue_size  10
training_layouts    4
learning_rate       2.5e-4
gamma               0.985
gae_lambda          0.95
clip_ratio          0.2
entropy_coef        0.035
value_coef          0.5
rollout_steps       128
minibatch_size      1024
update_epochs       4
hidden_size         256
```

评测固定使用：

```text
maps/course_map.txt
30 AGV
5000 environment steps
seed 0 或 0,1,2,3,4
stochastic policy sampling
```

`seed` 会影响 RWARE 初始 AGV 位置、朝向、随机任务队列和随机 destination 选择。为了课程要求中的固定随机生成策略，正式比较时应报告使用的 seed；为了满足“Results are stable across multiple runs”，建议同时报告 5 个 seed 的 `completed_tasks`、均值和标准差。

课程要求的核心评价指标是 `completed_tasks`，即 5000 steps 内完成完整闭环任务的数量。辅助分析指标包括：

```text
delivered_to_goal     只送到工作站的中间次数，不等于完整任务完成
generated_tasks       随机任务流产生过的任务数
blocked_moves         FORWARD 因碰撞/阻塞失败的次数
mean_agent_reward     训练用 shaped reward 的均值，仅作辅助观察
std_completed_tasks   多 seed 稳定性
```

正式比较时应优先看 `completed_tasks`；其它指标用于解释为什么某个策略更稳定或更容易拥堵。

## 12. 结果文件说明

本次准备上传两个结果目录：

```text
artifacts/agv_runs/eval_planner_baseline/
artifacts/agv_runs/train_ltfp_final_300k/
```

`eval_planner_baseline/` 是不加载 PPO follower 的 planner baseline，包含：

```text
eval_planner.csv
summary_planner.json
eval_config.json
step_logs/planner_seed0_steps.csv
gifs/planner_seed0.gif
```

当前该 baseline 的 `summary_planner.json` 中：

```text
seed              0
steps             5000
completed_tasks   518
delivered_to_goal 518
generated_tasks   558
blocked_moves     7315
```

`train_ltfp_final_300k/` 是 PPO follower 的最终训练结果，包含：

```text
checkpoints/ltfp_latest.pt
training_config.json
training_log.csv
evals/
```

其中 `checkpoints/ltfp_latest.pt` 是最终模型权重；`evals/` 中保存了使用该模型的评测 CSV、summary、step logs 和 GIF。单 seed 评测结果显示 LTF-PPO 比 planner baseline 完成更多闭环任务，说明局部 follower 对拥堵跟随和执行效率有提升。
