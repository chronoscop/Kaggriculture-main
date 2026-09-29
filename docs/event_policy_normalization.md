> 本文记录历史v7。当前候选前缀采集、连续段证据及实验命令见 [event_policy_candidate_prefix.md](event_policy_candidate_prefix.md)。

# 事件策略：修复局面输入饱和

训练格式 `event-policy-iteration-v7`；原始事件行和实际执行契约仍为 `event-shared-policy-season-scope-v5`。权重自带 `_event_input_encoding=event-centered-bounded-v1`，无此字段的旧网络严格走旧编码。

## 为什么改

复核 competition_replay_review、plan_compare_trial_review、event_policy_evidence_trial_review 后，本轮集中解决已实测的输入尺度问题。前轮100轮并非零提升：第30轮候选于第40轮晋级，但后60轮没有进一步改善。最终网络在末期905条对照上第一层全部饱和，第二层最大变化仅0.000059；现金与承诺等条件难以进入比较。

目标仍是依据已有生产、现金、物料、工时和市场条件选择下一段安排。现有批次、连续生产、混合路线和后续修订能力保留；不能把饱和修复冒充新增这些能力，也不能推断DECEM使用哪种训练算法。

## 实现边界

- 统一变换位于 `Policy::forward` 的网络输入路径，CPU采集首选、GPU损失、更新后首选以及CPU部署使用同一实现。记录中的full_row仍保留原始特征，避免重复归一化；证据附带learner_input_encoding。
- 市场库存由原始库存/100恢复为相对基准变化 `(实际库存-10000)/10000`。10000为当前引擎初始库存，零表示基准、负表示减少、正表示增加。
- 对context和候选特征使用 `x/sqrt(1+x²)` 平滑有界变换，保留顺序和符号。固定规划器prior仍取原始候选第30维，不改变其加分语义。没有引入运行时估计的均值方差，因此不同线程、批次和恢复过程使用同一编码。
- 权重元数据随部署和完整检查点保存；读取未知编码会报错。旧v5/v6模型没有元数据，继续使用原始输入。旧检查点不得直接resume成v7。
- `--init-from`现为从共享事件策略v5/v6/v7读取**已验收部署和历史对手**，保留原编码；新学习器、Adam、经验库和待验收候选重新开始。不会继承旧提议权重。`--base-checkpoint`仍可用于v3底座的新实验。新模型完整验收通过才接管原有声明范围。
- 当前继续策略只在晋级后更新；胜负标签、均匀历史抽样、分支预算、事件范围、独立确认协议都不改变。未加入高频交易或现金奖励。

## 检查与指标

新增测试：旧输入及旧输出不变；新编码与权重恢复一致；同样候选、只有现金不同的两个状态能学出相反选择，且context分支收到非零梯度。此为表达与梯度通路测试，不是实际经营成绩。

每轮 `input_health` 记录输入范围、第一层饱和比例、第二层编码跨样本最大跨度、最后一批context梯度范数。最多使用本轮128条记录；这些是运行诊断，没有统计晋级含义。梯度非零和编码有变化也不能单独证明有效利用条件。

## 新开短实验命令

已验收的旧策略保留，新网络从头学习，另开目录20轮；先与旧冠军比，而不是继承失败提议权重：

```bash
export PATH="/tmp/route-rl-rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin:$PATH"
export CARGO_HOME=/tmp/route-rl-cargo
cargo build --manifest-path native/Cargo.toml --release --features train --offline \
  --bin event-train --bin event-agent -j 4

mkdir -p runs/event_policy_normalized_trial
native/target/release/event-train \
  --out runs/event_policy_normalized_trial \
  --init-from runs/event_policy_evidence_trial/best.json \
  --iterations 20 --games-per-update 16 --workers 7 --device cuda \
  --branch-points 2 --alternatives 2 \
  --branch-steps-per-game 1440 --max-branches-per-game 4 \
  --epochs 8 --batch-size 64 --learning-rate 0.0003 \
  --eval-every 5 --eval-games 8 --confirm-games 16 \
  --seed 240000 --eval-seed 1880000000 \
  > runs/event_policy_normalized_trial/train.log 2>&1
```

这不是与上一轮随机初始化实验的严格消融：参考策略已经更强。本轮短测与上述正式实验也使用不同种子。最终仍以独立终局比赛得分验收。修复数值缺陷不保证持续收敛，局部选择组合的相互影响仍需完整对局检验。

## 接着本次短测训练

如果继续本次 `runs/event_policy_normalized_check` 的5轮结果，可完整恢复新编码学习器及经验，另开目录再15轮（累计20轮）。`--resume`和`--init-from`二选一：

```bash
mkdir -p runs/event_policy_normalized_continue
native/target/release/event-train \
  --out runs/event_policy_normalized_continue \
  --resume runs/event_policy_normalized_check/latest.json \
  --iterations 15 --games-per-update 16 --workers 7 --device cuda \
  --branch-points 2 --alternatives 2 \
  --branch-steps-per-game 1440 --max-branches-per-game 4 \
  --epochs 8 --batch-size 64 --learning-rate 0.0003 \
  --eval-every 5 --eval-games 8 --confirm-games 16 \
  --seed 230000 --eval-seed 1870000000 \
  > runs/event_policy_normalized_continue/train.log 2>&1
```

## 本次实际复测结果

- 18项定向测试通过，release构建通过：2项输入与条件学习测试、12项训练协议测试、4项部署兼容测试。
- `runs/event_policy_normalized_check` 完成5轮CUDA训练，80局基础轨迹、223条终局对照；每轮第一层饱和比例均为0，第二层跨状态最大跨度约0.164–0.291，context梯度均非零。没有用这些数值作为晋级奖励。
- 新初始检查点中的deployment、previous_accepted与源best检查点逐字段一致。新学习器单独归一化并初始化，旧模型文件未改动。
- 第5轮初筛：4个种子、换座、3类对手，每个策略24局。候选得分率45.8333%，参考79.1667%；候选现金51895.33，参考55091.79。候选未晋级，无待复核候选。
- 结论：数值与编码兼容修复在短测中得到验证，当前没有经营/胜负改善证据。不能因输入健康就建议夜训；若继续，先用上面的15轮续训命令累计到20轮，再根据实际首选与整局结果判断。

## 单独检查最新学习器（不训练、不晋级）

训练中的第15/20轮评估可能仍在复核冻结的第10轮候选。为避免把它误当作最新网络，新增只读入口：

```bash
native/target/release/event-train \
  --evaluate-checkpoint runs/event_policy_normalized_continue/latest.json \
  --out runs/event_policy_normalized_eval20 \
  --eval-games 64 --eval-seed 1890000000 \
  --workers 7 --device cpu \
  > runs/event_policy_normalized_eval20.log 2>&1
```

`eval-games=64`表示每类对手64局，即32个种子换座；3类对手时每个策略192局、共384局。始终读取model中的当前学习器，忽略pending候选；接受策略、历史对手、作用范围与原训练保持一致。新种子须位于源检查点已使用的评估范围之后，且不得用于训练。此入口只写新目录下manifest和evaluation.json，绝不调用Adam或晋级；manifest保存实际候选权重快照。统计按独立种子聚合，采用预先固定的一次检查；不能把此前第10轮候选的样本混入本次结果。

本次使用1890000000–1890000031，未依据中途结果增加种子。目标是回答最新第20轮网络是否优于同场参考；不更改原晋级协议，也不直接将诊断结果晋级。
