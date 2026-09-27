# 独立策略与价值网络（v7-independent-6）

## 本次改动

- Actor 使用原 context → candidate/score/group 分支；critic 使用新增的独立 `critic_context` → value 分支。两者均可训练，不共享参数。
- Actor、critic 分别按范数 0.5 裁剪梯度，各参数继续保留自己的 Adam 状态，学习率仍为 1e-4。价值回归不再直接改变策略特征，也不再通过全局裁剪压缩策略梯度。
- 96 维观测、32 维候选、动作层级、现金奖励、探索分组、PFSP、经验筛选和辅助回放策略保持原配置。本次首先隔离网络干扰，不宣称已经解决经验保护和收益稳定性。
- 指标新增 `policy_loss`、`value_loss`（含 0.5 系数）、`policy_entropy`、`mean_ppo_kl`，均为已完成 PPO 小批次的均值；它们不是完整更新后或对冠军的 KL。
- 新 `policy_contract` 为 `mixed-routes-96x32-hierarchy-cash-independent-v4`，manifest 修订为 `v7-independent-6`。旧共享网络检查点明确报错，不自动迁移。新版本检查点完整保留两套参数、Adam、随机状态、对手池和经验库。

## 构建

```bash
export PATH="/tmp/route-rl-rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin:$PATH"
export CARGO_HOME=/tmp/route-rl-cargo
cargo build --manifest-path native/Cargo.toml --release --features train --offline -j 4
```

## 从零过夜训练

在已经进入的 screen 会话中执行。不要加 `--resume`；使用没有旧 metrics 的新目录。

```bash
mkdir -p runs/mixed_v7_independent_overnight
native/target/release/mixed-train \
  --out runs/mixed_v7_independent_overnight \
  --iterations 900 \
  --games-per-update 32 \
  --workers 7 \
  --device cuda \
  --epochs 2 \
  --batch-size 256 \
  --seed 1200 \
  --opponent league \
  --eval-every 5 \
  --eval-games 8 \
  --eval-seed 1000000000 \
  --exploration 0.2 \
  --imitation-weight 0.05 \
  > runs/mixed_v7_independent_overnight/train.log 2>&1
```

900 轮共 28800 局采集（含确定性探测），验证局额外计算。从零初始化模型、优化器、经验库和联盟；初始冻结对手为新初始化策略，后续自动产生近期快照，不加载旧冠军。

最近成熟策略的 focus 实验 50 轮采集/更新与验证约 33.5 分钟，以此粗估 900 轮约 10 小时；从零训练的经营复杂度、验证次数和新网络耗时会变化，不能保证定时结束。screen 内不需要 nohup 或末尾 `&`；Ctrl+A 后按 D 离开会话。

另一个终端看日志：

```bash
tail -f runs/mixed_v7_independent_overnight/train.log
```

`latest.json` 每轮保存；`best.json` 在通过晋级后保存。确定性胜率、现金与生产完成情况用于判断效果，loss 下降不代表策略更强。本次从零跑通只能说明实现可运行，不能预先保证一晚训练会改善收益。

## 后续续训

只可恢复本版本生成的检查点。保持原配置，加 `--resume runs/mixed_v7_independent_overnight/latest.json`，将 `--iterations` 改成希望达到的累计轮数。旧实验目录保留用于对照。

## 本机验证

- 38 项测试通过（37 项库测试、1 项 CLI 完整恢复测试），包含 CPU/CUDA 参数及概率隔离、独立裁剪与新检查点恢复。
- 从零 CUDA 短验证完成 32 局采集、40 次 PPO 小批次更新、评估和完整检查点保存。新增损失指标有限且可读；该验证不用于判断长期经济效果。
