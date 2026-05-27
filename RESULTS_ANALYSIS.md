# 实验结果分析

本文档对当前项目保存的评测结果进行集中分析，重点回答三个问题：

1. 当前实现是否完成了课程要求的 5000 steps、30 AGV、多 seed 评测。
2. 加入 LTF-PPO follower 后，相比不训练 follower 的 planner baseline 是否有提升。
3. 不同训练参数对结果的影响是什么，哪些设置更合理。

结果数据来自以下目录：

```text
artifacts/agv_runs/eval_planner_5seeds/
artifacts/agv_runs/tuning_batch/
artifacts/figures/
```

其中 `artifacts/figures/` 保存论文风格 PNG 对比图，`tuning_batch` 保存调参训练与 5-seed 评测汇总表。

## 1. 评测设置

正式评测统一使用课程指定的核心条件：

| 项目 | 设置 |
| --- | --- |
| 仿真环境 | `robotic-warehouse` / RWARE |
| 测试地图 | `maps/course_map.txt` |
| AGV 数量 | 30 |
| 时间长度 | 5000 env steps |
| 随机任务队列 | 固定随机生成策略，不同 seed 改变随机初始化和任务序列 |
| 任务队列长度 | `request_queue_size=10` |
| 评测 seeds | 0, 1, 2, 3, 4 |
| 核心指标 | `completed_tasks` |

这里的 `completed_tasks` 是完整闭环任务数，不是单纯送到工作站的次数。一次任务只有在以下流程全部完成后才计数：

```text
空载 AGV 到达 requested shelf 原始储位
-> TOGGLE_LOAD 驮起货架
-> 载货前往 G/g 目标点
-> 到达目标点后继续驮着同一货架返回 origin
-> 在原始储位卸下货架
-> completed_tasks += 1
```

因此：

- `delivered_to_goal` 只表示货架已经到达过目标工作站，是中间指标。
- `completed_tasks` 才是课程要求比较的完整任务完成数。
- `blocked_moves`、`mean_agent_reward`、`generated_tasks` 用于解释拥堵、稳定性和任务供应情况，不作为唯一评分指标。

## 2. Planner Baseline 与 LTF-PPO 对比

本项目保留了一个不加载 PPO follower 的 planner baseline。它使用同样的任务生成、地图、AGV 数量和 5000 steps 设置，但控制动作主要由规则 A* planner 给出。LTF-PPO 则在 A* 提供参考路径的基础上，由学习到的局部 follower 根据局部观测、路径叠加信息和拥堵状态采样移动意图。

5 个 seed 的结果如下：

| 方法 | seeds completed_tasks | 平均 completed_tasks | 标准差 | 平均 blocked_moves |
| --- | --- | ---: | ---: | ---: |
| Planner baseline | 219, 510, 188, 645, 363 | 385.0 | 173.2 | 7286.2 |
| LTF-PPO standard | 1580, 1563, 1539, 1459, 1537 | 1535.6 | 41.5 | 5451.2 |

对比结论：

- LTF-PPO 平均完成任务数比 planner baseline 高 `1150.6` 个。
- 相对提升约为 `298.9%`，即约 `3.99x`。
- LTF-PPO 的标准差从 `173.2` 降到 `41.5`，稳定性明显更好。
- LTF-PPO 的平均 blocked moves 从 `7286.2` 降到 `5451.2`，下降约 `25.2%`。

这说明当前 PPO follower 并不是简单复制 A* planner 动作，而是在局部拥堵、等待和路径跟随中改善了执行质量。Planner baseline 在某些 seed 下会出现较多阻塞，导致完成数波动大；LTF-PPO 在 5 个 seed 中都维持在 1450 以上，说明 learned follower 对随机初始化和任务序列更鲁棒。

对应图：

```text
artifacts/figures/planner_baseline_vs_ltfp_standard.png
```

该图左侧展示 completed tasks，右侧展示 blocked moves。误差棒为 5 seed 标准差，散点为每个 seed 的单次结果。

## 3. 标准 LTF-PPO 结果是否稳定

标准参数为：

```text
learning_rate = 2.5e-4
entropy_coef = 0.035
waypoint_reward = 0.03
active_wait_penalty = -0.02
total_steps = 300000
training_layout_count = 4
```

其 5-seed completed tasks 为：

```text
1580, 1563, 1539, 1459, 1537
```

均值 `1535.6`，标准差 `41.5`，变异系数约 `2.7%`。对于多 AGV 随机任务系统，5 个随机种子的结果差异较小，说明该设置具备较好的稳定性。

训练日志中也能看到策略学习逐渐稳定：

- `reference_action_match_rate` 从训练早期约 `0.18` 上升到后期约 `0.90` 左右。
- `wait_action_rate` 从早期较高下降到后期接近 `0.2% - 0.8%`。
- 后期 `entropy` 仍保持非零，说明策略不是完全确定性的死板跟随，而是保留了随机性。

这与 Learn-to-Follow 的思想一致：planner 负责给出参考路径，learned follower 负责局部执行、等待、避碰和跟随。

## 4. 调参结果总览

`artifacts/agv_runs/tuning_batch/tuning_summary.csv` 中记录了不同参数组合的 5-seed 结果：

| 实验 | 主要变化 | 平均 completed_tasks | 标准差 | 平均 blocked_moves |
| --- | --- | ---: | ---: | ---: |
| tune_base | 标准参数 | 1535.6 | 41.5 | 5451.2 |
| tune_lr_low | learning rate = 1e-4 | 1452.0 | 21.3 | 5787.6 |
| tune_lr_high | learning rate = 5e-4 | 1361.6 | 107.8 | 7071.2 |
| tune_entropy_low | entropy = 0.015 | 521.8 | 326.8 | 61286.8 |
| tune_entropy_high | entropy = 0.060 | 1532.2 | 9.2 | 3978.2 |
| tune_reward_mild | waypoint 0.02, wait -0.01 | 1262.2 | 29.8 | 2963.2 |
| tune_reward_strong | waypoint 0.05, wait -0.03 | 1484.0 | 33.4 | 4399.0 |
| tune_steps_short | total steps = 150k | 1386.0 | 72.3 | 5980.2 |
| tune_map_1layout | 1 个训练地图 | 865.8 | 479.3 | 34880.0 |
| tune_map_6layouts | 6 个训练地图 | 1173.6 | 236.0 | 16737.0 |

总体结论：

- 标准参数 `tune_base` 的平均完成数最高。
- `tune_entropy_high` 的平均完成数几乎相同，但标准差最低，稳定性最好。
- 低 entropy、单训练地图会显著降低结果，说明局部 follower 需要足够探索和地图多样性。
- 学习率过高会增加不稳定性，学习率过低虽然稳定但性能略低。

对应总览图：

```text
artifacts/figures/overall_tuning_comparison.png
```

## 5. 学习率分析

学习率对比结果：

| learning rate | 平均 completed_tasks | 标准差 |
| --- | ---: | ---: |
| 1e-4 | 1452.0 | 21.3 |
| 2.5e-4 | 1535.6 | 41.5 |
| 5e-4 | 1361.6 | 107.8 |

分析：

- `1e-4` 学习较保守，稳定但平均完成数低于标准参数。
- `2.5e-4` 在当前训练步数下效果最好。
- `5e-4` 出现明显性能下降和更大波动，说明更新幅度过大时，局部 follower 容易学到不稳定动作偏好。

对应图：

```text
artifacts/figures/comparison_learning_rate.png
```

## 6. 探索率 entropy 分析

entropy 系数对比结果：

| entropy_coef | 平均 completed_tasks | 标准差 | 平均 blocked_moves |
| --- | ---: | ---: | ---: |
| 0.015 | 521.8 | 326.8 | 61286.8 |
| 0.035 | 1535.6 | 41.5 | 5451.2 |
| 0.060 | 1532.2 | 9.2 | 3978.2 |

分析：

- `0.015` 明显过低，策略过早变得确定，容易在拥堵处反复选择同类动作，形成长期堵塞。
- `0.035` 达到最高平均完成数。
- `0.060` 平均完成数几乎不低于标准参数，而且 blocked moves 更少、标准差更低。

因此，如果后续更重视稳定性，可以考虑使用 `entropy_coef=0.060`；如果更重视当前最高均值，则保留 `0.035`。

对应图：

```text
artifacts/figures/comparison_entropy.png
```

## 7. 奖励设计分析

奖励 shaping 对比结果：

| 设置 | waypoint_reward | active_wait_penalty | 平均 completed_tasks | 标准差 |
| --- | ---: | ---: | ---: | ---: |
| mild | 0.02 | -0.01 | 1262.2 | 29.8 |
| base | 0.03 | -0.02 | 1535.6 | 41.5 |
| strong | 0.05 | -0.03 | 1484.0 | 33.4 |

分析：

- mild 奖励较弱，follower 对参考路径推进的激励不足，完成数下降。
- strong 奖励比 mild 好，但没有超过 base。过强的 waypoint 奖励和等待惩罚可能让 agent 更倾向于持续推进，在局部拥堵中减少必要等待。
- base 在推进与等待之间取得更好的平衡。

对应图：

```text
artifacts/figures/comparison_reward.png
```

## 8. 训练步数分析

训练步数对比结果：

| total_steps | 平均 completed_tasks | 标准差 |
| --- | ---: | ---: |
| 150k | 1386.0 | 72.3 |
| 300k | 1535.6 | 41.5 |

分析：

- 150k 已经能学到有效 follower，明显优于 planner baseline。
- 300k 进一步提升平均完成数，并降低 seed 间波动。
- 当前课程实验中，300k 是较合适的折中：训练成本可接受，结果明显更稳定。

对应图：

```text
artifacts/figures/comparison_training_steps.png
```

## 9. 训练地图数量分析

训练地图数量对比结果：

| training_layout_count | 平均 completed_tasks | 标准差 | 平均 blocked_moves |
| --- | ---: | ---: | ---: |
| 1 | 865.8 | 479.3 | 34880.0 |
| 4 | 1535.6 | 41.5 | 5451.2 |
| 6 | 1173.6 | 236.0 | 16737.0 |

分析：

- 只使用 1 个训练地图时，性能极不稳定，说明 follower 容易过拟合某一种局部布局和拥堵模式。
- 4 个相似训练地图效果最好，既提供了足够多样性，又没有让训练分布过宽。
- 6 个地图没有继续提升，可能因为 300k 训练步数被分摊到更多地图上，每个地图上的有效学习样本变少；也可能是部分生成地图与正式测试图差异更大，使 follower 学到的局部行为更分散。

这说明“训练地图不是越多越好”，需要在地图多样性和训练样本密度之间平衡。

对应图：

```text
artifacts/figures/comparison_training_layouts.png
```

## 10. 对课程评分点的对应说明

### Functionality & Integration

当前结果表明：

- 可以创建 RWARE 仿真环境并运行 30 台 AGV。
- 随机任务队列持续刷新。
- `completed_tasks` 只在完整闭环后增加，符合“pick up shelf -> deliver to target -> return shelf -> proceed to next task”。
- 所有正式评测均运行满 5000 steps。

### Training & Parameter Tuning

当前调参覆盖了课程要求中提到的关键项：

- learning rate：`1e-4 / 2.5e-4 / 5e-4`
- reward design：mild / base / strong
- training episodes / steps：`150k / 300k`
- exploration rate：entropy `0.015 / 0.035 / 0.060`
- map configuration：`1 / 4 / 6` 个训练地图

这些结果可以说明训练参数对性能有明显影响，而不是只运行了一次固定参数。

### Transport Efficiency & Task Completion

核心结果为：

- Planner baseline：`385.0 ± 173.2`
- LTF-PPO standard：`1535.6 ± 41.5`

在相同地图、相同 AGV 数量、相同任务生成策略和相同 5000 steps 下，LTF-PPO 完成任务更多、波动更小，满足课程要求的对比分析。

## 11. 使用结果图时的建议

报告中建议按以下顺序展示：

1. `planner_baseline_vs_ltfp_standard.png`
   - 用于说明加入 PPO follower 相比 planner baseline 有明显提升。
2. `overall_tuning_comparison.png`
   - 用于展示所有调参结果总览。
3. `comparison_entropy.png`
   - 用于强调探索率对多 AGV 拥堵场景的重要性。
4. `comparison_training_layouts.png`
   - 用于说明训练地图构建数量会影响泛化和稳定性。
5. 其它参数图根据篇幅选择展示。

当前 `artifacts/figures/` 下只保留 PNG 文件，便于直接插入课程报告或展示文档。

## 12. 需要注意的解释边界

结果中的绝对 completed tasks 数值依赖以下因素：

- 任务队列长度 `request_queue_size`
- 随机任务生成策略
- AGV 初始位置和方向
- 训练地图构建方式
- 是否使用 stochastic follower
- planner baseline 是否启用动态预约或其它规则

因此，最公平的结论不是单独说某个绝对数值“高”或“低”，而是在同一套设置下比较不同方法。当前 planner baseline 与 LTF-PPO 使用相同地图、相同 seed 集合、相同 5000 steps、相同任务队列设置，因此两者对比是有效的。

如果后续修改 `request_queue_size`、地图、任务生成逻辑或评测 seed，需要重新生成 baseline 和 LTF-PPO 的成对结果，不能直接与本文档中的数值混用。
