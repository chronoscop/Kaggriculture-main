# PFSP 难度信号与分对局探索（v7-pfsp-5）

结论：上一版还不适合直接扩为一夜长训，本修订先处理已经观察到的两项问题。

## 依据

PFSP-4 到 1970，新增 85 轮、2720 局，冠军更新到 1925；收益比 league-3 稳定，但最新模型对当前冠军只有 2/8 胜。1951–1970 的历史对手训练局，探索执行赢 8/430，确定性执行赢 70/130、平 4 局。两组种子、对手不完全一致，仅是诊断。

进一步以 1925 模型对固定 1885 对手、种子 4000000000–4000000007 做只读对照，三种随机策略均统计同一组非探测种子对（12 局）：

| 随机采样方式 | 平均现金 | 平均现金差 | 胜局 |
|---|---:|---:|---:|
| 模型概率分布，无额外均匀提案 | 46606 | -4072 | 2/12 |
| 额外 5% 均匀提案 | 41324 | -9321 | 1/12 |
| 额外 20% 均匀提案 | 35696 | -19124 | 0/12 |

这是小样本对照，不证明最佳探索配比；它支持减少每局反复注入均匀随机动作。原始结果在 `/tmp/mixed_pfsp_exploration_audit.json`，未用于 PPO 或经验库。

## 两项修改

### PFSP 只用确定性训练探测衡量难度

对手抽样和困难历史模型保留，都使用 `greedy_training_probes` 的平滑得分率。沿用先验、半衰期、最低覆盖和抽样上限。缺少确定性证据时回到 0.5，不回退到已经接近全败的探索胜率。`sampled` 仍记录，不能参与难度排序。

正式验证局仍不进入 PFSP 或训练经验。训练探测虽然确定性执行，仍是训练分布的一部分，不替代独立评估。

### 按完整对局分配探索方式

每 32 局：

- **18 局 focused**：从模型自身概率分布随机抽样，额外均匀提案概率为 0；仍有探索，仍进入 PPO。
- **6 局 broad**：原来的 20% 生产类别均衡提案（取 `--exploration`），仍进入 PPO。
- **8 局 greedy**：确定性探测，用于经验保留和 PFSP 难度统计，不进入 PPO。

从随机种子对中随机挑选四分之一作为 broad，同种子交换座位的两局保持模式一致。整季固定模式，不在一条生产路线中途切换探索强度。角色预算仍为近期12/PFSP12/覆盖4/自对弈2/heuristic2。两种随机局的数量比例是实验起点，并非已证实最优。

两种随机模式每条 Sample 都记录实际 exploration 和 logp，PPO 重新计算相同混合分布，避免采样概率与训练比例不一致。未改变网络、现金奖励、等待合法性、路线生成和 PPO 超参数。

## 日志与恢复

- `training_revision = v7-pfsp-5`。
- `games[].exploration_regime`、`exploration` 记录实际执行方式和提案率。
- `exploration_summary` 汇总各组现金、现金差、胜负积分；组间对手/种子不同，不能当作严格 A/B。
- 对手统计新增 `focused`、`broad`，继续保留 `greedy`、随机汇总 `sampled`。
- `matchmaking_before[].difficulty_source / difficulty_score` 显示抽样依据。
- 从旧版本恢复时保留网络、Adam、随机状态、经验、对手权重、确定性战绩；清空旧的随机战绩，避免与新的探索配比混在一起。PFSP-5 自身续训保留所有统计。
- `--exploration 0.2` 现在对 league 的 broad 局生效；独立的 heuristic/selfplay 模式仍沿用所有随机局同一提案率。正式评估的纯策略采样仍是无额外均匀探索。

## 新实验命令

从 PFSP-4 的最佳完整检查点 **1925** 开始，先新增 **50 轮、1600 局采集**，评估另计。`iterations` 仍是累计上限。不要使用旧实验输出目录。

```bash
mkdir -p runs/mixed_v7_pfsp_focus_trial
native/target/release/mixed-train \
  --out runs/mixed_v7_pfsp_focus_trial \
  --resume runs/mixed_v7_pfsp_trial/best.json \
  --iterations 1975 \
  --games-per-update 32 --workers 7 --device cuda \
  --epochs 2 --batch-size 256 --seed 1200 --opponent league \
  --eval-every 5 --eval-games 8 --eval-seed 1000000000 \
  --exploration 0.2 --imitation-weight 0.05 \
  > runs/mixed_v7_pfsp_focus_trial/train.log 2>&1
```

先检查 best.json 的 iteration 仍为 1925；若用户后续继续了旧训练且最佳检查点变化，应据实际起点设置累计上限。修改代码或构建不会停止旧训练进程，由用户切换。修订验证通过不等于适合长训；下一轮重点是确定性能力能否保留/提升，以及 focused 对局能否提供更有效的生产样本。
