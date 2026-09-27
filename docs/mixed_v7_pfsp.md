# 近期冻结快照与动态历史对手（v7-pfsp-4）

> 最新修订：[确定性难度与分对局探索](mixed_v7_exploration.md)（v7-pfsp-5）。PFSP 使用确定性训练探测；每 32 局分为 18 局纯策略随机、6 局广泛探索、8 局确定性探测。

本修订针对 Kaggriculture 双人完整赛季，终局比较现金。调整训练对手的来源与调度；保持网络、现金奖励、PPO、探索概率、受保护经验库及冠军晋级的数学判据。本轮是调度实验，不保证收益提高。

## 为什么现在调整

旧 league 实验截至读取第 1954 轮：新增 69 轮、2208 局，13 次评估没有新入池或晋级。前五次固定对 heuristic 的现金均值 62977.775，最后五次 58525；冠军仍为 1885，训练池没有变化。“近期”只是在已获准入池的旧模型中选最新两个，不能持续推进课程。该趋势支持开展新实验，不能证明全部退步由对手造成。

## 调度与统计

每 32 局由 16 个种子各交换一次座位组成。初始预算：

| 来源 | 局数 | 方法 |
|---|---:|---|
| recent | 12 | 最新两个冻结版本交替 |
| pfsp | 12 | 在其余历史模型中按近期探索策略战绩抽样 |
| coverage | 4 | 对历史池按轮次循环，补充覆盖 |
| current | 2 | 当前策略自对弈 |
| heuristic | 2 | 基础经营对照 |

25% 的种子对采用确定性探测，每四个种子随机选一个；选定的两局都使用相同对手与执行模式。只有随机探索的当前模型决策进入 PPO。建议 games-per-update 为 32 的倍数；小批次按角色顺序截断。

- 每 10 个累计迭代冻结当前模型，不要求刷新冠军、不依赖 eval-every。入池不等于模型变强。
- 恢复时若池中没有所恢复的当前版本，先补入该版本；不会因恢复重复加入已有快照。
- 冻结模型始终确定性执行，整局固定。当前模型也在整批完整对局结束后才更新。
- 战绩按冻结模型的 iteration 保存，池内 slot 移动不会改变归属。探索执行（含 20% 探索混合）与确定性执行分别记录；自对弈和 heuristic 不用于历史 PFSP。
- 仅使用完整训练局（模拟器终态 step 719）的实际双方现金判定胜负，平局记 0.5。验证局不写入 PFSP 战绩，也不写入训练经验。
- 游戏胜负统计按 50 轮半衰期衰减；加 8 局、得分 0.5 的先验，缓和小样本。旧证据过期后向 0.5 回归，不把未观察对手当成必败对手。
- PFSP 权重为 `1 - 平滑得分率`，归一化后混入 20% 均匀分布。每个成员的条件抽样概率上限 40%；池少于三个成员时上限至少为 `1 / 成员数`。此上限只针对 PFSP 部分，不包括近期与覆盖预算。

12/12/4/2/2、10 轮快照、半衰期和概率限制均为可检验的初始设置，非来自金牌方案的通用最优值。配对局相关，先验和平滑也不代表统计置信区间。

## 保留谁

池容量仍为 8，避免大幅增加 GPU 权重分组和碎片推理：

1. 保护冠军及最新两个版本，重复身份只占一个位置。
2. 保护一个最旧的其他参考版本。
3. 最多保护两个近期仍难以战胜的历史模型。确定性训练探测的有效样本 >= 8 时使用该模式的平滑战绩，否则使用探索战绩；两者不合并。
4. 剩余位置尽量覆盖不同训练时期。

收入组合和混合路线画像仍是诊断资料，取消 `distinct_behavior` 入池，不再参与训练池淘汰排序。通过现有对战门槛的 competitive 候选仍可入池。验证对手组继续保留冠军、近期代表及行为有差异的参考（有画像时），属于诊断/选模而非训练采样。新冻结模型没有画像也可以参加训练；不为每次冻结额外跑画像校准。

## 结果与恢复

- `training_revision = v7-pfsp-4`。
- `matchmaking_before`：实际采集前各版本的战绩和平滑 PFSP 概率。概率仅指 PFSP 角色内。
- `opponent_outcomes`：本轮完整对局结束后的战绩，仍对应入池/淘汰前的版本。
- `games[].opponent_iteration`：实际冻结对手版本；配合 `opponent_role`、`greedy_probe` 阅读。自对弈/heuristic 为 null。
- `recent_snapshot_added`：本轮是否额外定期冻结。若该版本已通过评估入池，此值为 false，评估的 pool_admitted 为 true。
- `pool_iterations`：定期冻结与评估入池都处理后的池。
- 检查点 `league[].training_outcomes` 保存衰减累积量、累计局数、最后更新轮次及两种执行模式；恢复不会清空新版本战绩。旧 v7/retention/league 检查点缺少战绩时，从中性先验开始。
- `best.json` 仍只在正式评估晋级后保存；定期冻结只保存在 latest.json 的 league。没有 best 不等于没有保存模型。

网络、Adam、随机状态、成功经验和对手权重完整恢复。固定与轮换评估种子继续与训练隔离；评估数学门槛保持不变，但训练池变化会使轮换对手组改变，因此固定指标和独立共同对手仍是跨实验比较的依据。

## 切换实验

正在运行的进程使用已加载的旧二进制。修改源码、重新链接不会让它自动切换版本。先完成构建/测试，再由用户停止旧进程；保留旧目录作对照。

本机 Rust 环境：

```bash
export PATH="/tmp/route-rl-rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin:$PATH"
export CARGO_HOME=/tmp/route-rl-cargo
export RUSTUP_HOME=/tmp/route-rl-rustup
cargo build --manifest-path native/Cargo.toml --release --features train --offline -j 2
```

在 screen 中启动，仍从 **1885 的完整检查点**开始，到 1985 即新增 100 轮、3200 局采集，评估另计。先确认目录尚无已有训练输出。

```bash
mkdir -p runs/mixed_v7_pfsp_trial
native/target/release/mixed-train \
  --out runs/mixed_v7_pfsp_trial \
  --resume runs/mixed_v7_retention_long/best.json \
  --iterations 1985 \
  --games-per-update 32 --workers 7 --device cuda \
  --epochs 2 --batch-size 256 --seed 1200 --opponent league \
  --eval-every 5 --eval-games 8 --eval-seed 1000000000 \
  --exploration 0.2 --imitation-weight 0.05 \
  > runs/mixed_v7_pfsp_trial/train.log 2>&1
```

若需要 200 轮，将 iterations 改为 2085。每轮 32 局里 8 局是确定性探测，24 局为随机探索；训练步数/实际 PPO 样本数量取决于对局中的决策数量。

优先检查：近期版本是否按期推进；概率是否根据分版本战绩变化；确定性终局现金、对冠军及共同历史对手的胜负是否改善。路线条数增加、入池次数增加本身均不代表实力提升。保持奖励不变也意味着平均现金与比赛胜率的差异仍需另行研究。
