# 完整可执行方案集合学习（event-policy-iteration-v9）

## 为什么改

依据 `event_learning_audit.md`：已测 A/B 能拟合，不代表全候选 argmax 有证据；固定拟合后 23 个局面中 20 个实际首选未测。保留种子上的两选一泛化也没有改善。当前执行器已经支持批次、混合路线与转产；这次修改学习数据和实际选择的对应关系。

目标仍是终局比赛得分：胜 1、平 0.5、负 0。现金、产量和路线数量仅作诊断。DECEM 回放提供的是连续经营的行为参照，不是其训练算法证据。

## 实现

- `pipeline/plan_menu.rs`：从现有合法方案生成至多 4 个可执行选择；资源紧张时允许更少。冻结旧底座选择始终占第 0 项，不依赖新网络分数。可行时覆盖作物、畜牧和取消；候选排序使用当前公开资源和价格，不使用终局结果。候选不足时不编造动作。
- 保留显式 Keep（它与底座选择可能不同）。其他候选按执行后生产承诺去重，包含剩余轮次、启动状态、备料时间和资金底线；不以“同一种作物”或“参数相同”直接认定等价。
- 每个方案重新编码实际承诺变化：未来物料费用、当前资金余量、工时变化、下一批启动/备料时间、缺料和重置剩余轮次等。保留原归一化状态；新增座位和公开对手资金。模型能使用这些条件，不保证它必然学会。
- `event-train`：每个训练条件分别完整执行已验收策略和当前候选。轮换从两种真实轨迹中选状态；每个选中状态的**全部**菜单方案分别执行到 step=719，后续统一由同一冻结已验收策略接手。候选实际产生的后续局面也进入采集。
- 分支预算按整个集合预留；不足就跳过，不把未测候选填成失败/平局。
- `learning/event_sets.rs`：一整个集合是最小训练和经验保留单位。训练全部选择相对第 0 项的终局得分差，所有选择一起参与损失；胜、负、平样本都保留。不是只抽出赢者模仿，也没有现金辅助奖励。
- 经验仍分长期随机库和近期库，但保存的是完整集合；晋级前续接策略不变，晋级后旧续接标签作废。采集源、候选菜单、所有终局现金、后续策略版本均保存。
- 训练、评估、`event-agent` 使用同一菜单和编码。学习范围继续是已声明的经营事件，开局和其他范围沿用底座；本实验不扩大责任范围。
- `observations.jsonl`：每次评估检查**最新网络**，面对固定规则对手和冻结菜单底座，使用固定观察种子；不会因待复核候选冻结而一直显示旧网络。该面板不用于晋级；晋级仍使用另外的配对种子、冻结候选和独立复核。

### 边界

这是“小菜单内、可执行接法的条件选择”。菜单之外的策略目前不能被网络选择；菜单的启发式筛选也可能漏掉优秀方案。单局终局标签存在市场和对手条件带来的噪声。全候选证据覆盖只针对已采集的局面，不能证明未见状态泛化，也不能证明多个改动组合后一定有利；完整确定性对局与独立验收继续承担这部分检查。

保持现有 single/batch 责任范围，v9 实验不支持从 single 扩成 batch。没有新增高频独立买卖，也没有把未来销售款提前花掉。

## 初始化与兼容性

- 使用旧实验的 `best.json --init-from`：保留其中**已验收**经营策略作为冻结底座，新的菜单网络从零修正头开始；不载入旧的失败提议网络、Adam 或不完整对照。
- 新特征与菜单版本不同，旧 v8 不能 `--resume` / `--warm-start`。旧检查点仍能由 `event-agent` 按原协议执行。
- v9 用 `--resume latest.json` 恢复完整菜单经验、网络和优化器。v9 `--init-from` 若已验收菜单网络会保留其权重；不重新解释旧格式。
- `latest.json` 同时保存学习网络和已验收 deployment；提交入口只加载 deployment。`best.json` 在没有晋级时继续保持旧底座。

## 先跑 20 轮

在项目根目录执行。本机 cargo 不在 PATH 时先设置：

```bash
export PATH="/tmp/route-rl-rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin:$PATH"
export CARGO_HOME=/tmp/route-rl-cargo
cargo build --manifest-path native/Cargo.toml --release --features train --offline --bin event-train --bin event-agent -j 4

mkdir -p runs/event_policy_complete_sets_trial
native/target/release/event-train \
  --out runs/event_policy_complete_sets_trial \
  --init-from runs/event_policy_candidate_prefix_trial/best.json \
  --followup-scope single \
  --iterations 20 \
  --games-per-update 8 \
  --workers 7 \
  --device cuda \
  --branch-points 2 \
  --branch-steps-per-game 4320 \
  --max-branches-per-game 8 \
  --epochs 8 \
  --batch-size 64 \
  --learning-rate 0.0003 \
  --eval-every 5 \
  --eval-games 8 \
  --confirm-games 16 \
  --seed 380000 \
  --eval-seed 1960000000 \
  > runs/event_policy_complete_sets_trial/train.log 2>&1
```

`iterations` 是追加轮数；上述是 160 个训练条件、320 条完整基准/候选轨迹，另加终局分支和评估。每个条件最多 2 个完整集合、8 条分支、4320 个额外模拟步；预算不足实际会更少。`batch-size` 是集合数；历史 `--alternatives` 不控制新菜单。

无需 Python 采集。CPU 并行模拟，CUDA 更新网络；小批量网络更新不代表 GPU 会持续满载。完整集合避免无证据首选，但单个状态会花更多终局模拟，速度要看实际采集耗时。

## 看结果

1. `observations.jsonl`：最新策略在同种子、同对手下的 `mean_score_gain`；有限固定面板只显示趋势。
2. `evaluations.jsonl`：独立整局复核和是否晋级，仍是最终保留策略的依据。
3. `metrics.jsonl → complete_set_choices`：按首次安排/后续修订分开记录可区分胜负的集合数、实际首选正确率和遗憾值。这里是当轮训练集合表现，不是泛化成绩。`unsupported_argmax=0` 仅表示采集局面的所有菜单项都测过。
4. `comparisons.jsonl` / `sequences.jsonl`：`candidate_set_id`、`candidate_plans`、`all_terminal_cash`、候选/已验收轨迹的结果，支持追溯网络究竟选了什么。

检查的是：网络首选能否在新局面提高终局得分，以及组合成整局后是否仍有效。loss 下降和执行测试通过不能代替这个结论。

## 本次验证

- 134 项库测试、16 项 event-train 测试、5 项 event-agent 测试通过；最后的采样/特征调整再跑 5 项针对性测试通过。
- 真引擎逐步比较零修正头与旧策略的 720 回合动作一致；种养特征未被新字段覆盖。
- 完整集合缺项拒绝、预算原子预留、经验集合恢复、含不同宽度集合的条件化损失测试通过。合成条件相反的两组选择可以拟合，不当作真实经营泛化证据。
- `/tmp/event_menu_cuda_smoke`：实际导入旧 best 的一轮 CUDA 连通性检查，2 个条件、4 个完整集合、16 条终局分支；两座的初始候选与已验收轨迹结果一致。采集约 44.6 秒、更新约 0.35 秒，固定观察和晋级检查均完成。最后的采样排序调整由针对性测试覆盖。
- 这些检查证明协议和代码路径可运行，未证明收益提升。实际短实验由用户运行。
