> 历史版本，已由 [生产衔接对照学习](plan_comparison.md) 替代。当前 `plan-train` 不接受本文旧 PPO 参数或检查点，请使用新文档命令。

# 经营计划神经网络训练（plan-rl-v1）

## 当前入口与边界

**训练入口是 `plan-train`，不是 `plan-prototype search`。** 每轮完整对局产生当前网络的随机决策，随后在 CPU/CUDA 上更新神经网络权重。`plan-agent` 使用同一候选生成器和网络，确定性选择执行。

这是最小经营决策学习版，不是 DECEM 算法复现，也没有证明长训练一定持续提升。

### 网络具体决定什么

- 当前可行的作物／动物品种、1 格或最多 4 格的投入批次。
- 暂缓新增生产，继续兑现现有义务；以及满足条件时购地。
- 对已确认、现已空出、没有在途工作预约的地格，选择下一种生产。
- 候选包含真实采购请求、地块安排、成本、预计商品产出、时间和服务需求。现金、订单数量、季末和简化工人容量约束仍生效。

每天前 9 个回合提供逐回合经营机会，之后每 4 回合一次；已经承诺的工作持续执行。候选在执行器副本上构造，只有网络选择的方案提交。订单失败仍通过后续观察确认；预期回款不当成已到账现金。空地转产不能强制清掉正在生长的作物或仍在养的动物。

网络复用 320 维公共观察及自己的资源信息（含价格变化、双方可见生产／成熟信息），使用新的 32 维计划特征。所有计划同一类别，直接比较完整候选的概率。

底层路线、生产所需的采购／销售、基础雇工和备种仍由自建 Rust 规划器执行。工人容量仍是简化估计；尚未实现完整跨日预约求解、主动清退活体生产和独立可学习市场策略。因此不能把该版描述为所有经营环节都已可学习。

## 怎样训练

1. **可执行起点**：网络输出“在规划器偏好上的修正”，初始修正严格为零，因此确定性选择与规划器一致，不依赖近似模仿准确率。初始采样分布给规划器方案 75% 概率，其余可行方案合计 25%；随后这些概率由 PPO 学习改变，规划器方案可以被其他方案超过。这个偏好已进入真实 logp，采样与 PPO 使用同一个分布。默认用自己的规划器跑 8 局，仅收集初始化保护状态，不使用外部标签、不搜索参数、不做模仿预训练。
2. **PPO**：每轮默认 32 局，全部由当前网络随机采样，只使用我方样本。回报为终局胜 +1、平 0、负 −1；不奖励开工次数、预测利润、路线长度或收获数量。gamma=1，GAE 的 lambda=0.997 按实际经过的环境步数计算。
3. **稳定对手**：约各三分之一对自建固定规划器、近期冻结网络、冠军网络。每个种子交换座位；近期网络每次例行评估后刷新。当前没有继续重写 PFSP，也没有用外部强参考训练。
4. **能力保持**：最多 128 个初始化状态、64 个长期胜局状态、64 个近期胜局状态分别保留。长期名额在三类对手间分配，避免容易打赢的对手占满。每个 PPO 小批次后做独立的策略分布保持；胜局目标还包含小比例的实际成功动作，权重不由 critic 归零。
5. **完整更新检查**：检查 PPO 与辅助更新之后的整体 KL、成功状态上的偏移及确定性选择变化。超过阈值恢复网络与 Adam 动量；随机数与对局种子继续前进，避免反复重跑同一批游戏。

价值网络与策略网络独立。旧经验不混入 PPO 概率比或价值回归。CPU 工作线程负责仿真和网络推理；`--device cuda` 的 GPU 负责网络优化，低频小网络不会持续满载 GPU。

## 构建

```bash
export PATH=/tmp/route-rl-rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin:$PATH
export CARGO_HOME=/tmp/route-rl-cargo
cargo build --manifest-path native/Cargo.toml --release --features train --offline \
  --bin plan-train --bin plan-agent -j 4
```

上述 PATH 是本机工具链位置；已有 cargo 的机器不需要设置。运行不启动 Python，LibTorch 仅提供原生张量计算。

## 第一段训练：100 轮

在 screen 内执行：

```bash
mkdir -p runs/plan_rl_trial
native/target/release/plan-train \
  --out runs/plan_rl_trial \
  --iterations 100 \
  --games-per-update 32 \
  --workers 7 \
  --device cuda \
  --epochs 2 \
  --batch-size 256 \
  --learning-rate 0.0001 \
  --warmup-games 8 \
  --eval-every 10 \
  --eval-games 8 \
  --seed 1200 \
  --eval-seed 1100000000 \
  > runs/plan_rl_trial/train.log 2>&1
```

100 轮 = **3,200 局 PPO 采集**，另有 8 局初始化和验证对局。参数配置默认从 `native/configs/plan_prototype_v1.json` 读取，仅用于执行器固定预算，不会把参数搜索 JSON 当作网络权重加载。

`--eval-games 8` 表示每个对手 8 局（4 个种子交换座位）。初始化评估 16 局；每次例行评估分别运行当前模型和冠军、各面对固定规划器与冠军，共 32 局；有晋级希望时再用另一批种子复核 32 局。验证不产生训练样本。小样本晋级不等于统计意义上的实力保证。

## 后续训练

`--iterations` 始终表示**再训练多少轮**。必须使用新输出目录，保留前一段结果：

```bash
mkdir -p runs/plan_rl_long
native/target/release/plan-train \
  --resume runs/plan_rl_trial/latest.json \
  --out runs/plan_rl_long \
  --iterations 1000 \
  --games-per-update 32 \
  --workers 7 \
  --device cuda \
  --epochs 2 \
  --batch-size 256 \
  --eval-every 10 \
  --eval-games 8 \
  > runs/plan_rl_long/train.log 2>&1
```

续训恢复网络、Adam、RNG、后续对局种子、执行器配置、冻结对手和受保护经验；不重复初始化。从检查点恢复学习率、seed 和 eval seed。该命令是可用的续训方式，不代表已经验证值得跑整夜。

## 结果与判断

- `latest.json`：完整可恢复的 learner；每轮原子保存。
- `best.json`：冠军的真实网络和匹配的 Adam 状态。初始就存在；通过胜负初筛及新种子复核才替换。
- `checkpoint_XXXXXX.json`：评估轮的完整检查点。
- `metrics.jsonl`：`weights_changed`、`updates`、`full_update_kl`、`anchor_kl`、`anchor_argmax_change`、`reverted`、生产与各类对手成绩。
- `evaluations.jsonl`：确定性成绩；`candidate.by_opponent.0` 是固定规划器，`.1` 是当时冠军。训练指标中的 `.1` 是近期模型、`.2` 是冠军，具体含义不能混淆。
- `warmup.json`：初始保护状态样本量和零修正初始化说明。

先看实际生产能否维持、确定性对战得分是否改善，再看样本量和 loss。频繁 `reverted=true` 表示更新未被接受，应看偏移来源；不能把这种轮次当成已学进网络。`weights_changed=true` 仅证明权重更新，不证明变强。

推理接口（持久 JSONL，尚不是 Kaggle 提交压缩包）：

```bash
native/target/release/plan-agent --checkpoint runs/plan_rl_trial/best.json
```

## 实现检查记录

开发时发现近似行为模仿虽然训练集匹配率上升，在新局面仍会空转，因此最终入口已改为上述零修正初始化；`plan_rl_impl_smoke`、`plan_rl_ready_smoke` 是已被替代的开发检查，不应续训。最终检查包括：初始确定性动作与规划器逐步一致且实际开工、网络能学会覆盖初始偏好、采样概率正确、时间回报、资源副本隔离、独立 actor/critic、完整更新回退和检查点恢复。效果是否持续提升仍由正式训练确认。

最终 CUDA 连通检查保存在 `runs/plan_rl_residual_smoke`：6 局采集得到 379 条经营样本，完成 4 次 PPO 小批次更新及辅助更新，`weights_changed=true`、`reverted=false`，完整检查点已保存。初始确定性检查实际收获 1084 单位、峰值生产 63 格，平均现金 37744；这组仅用于检查开工与执行一致性，不能当成实力估计或训练提升。库测试 65 项、训练入口测试 2 项均通过。

## 采集性能修正（2026-09-28）

原 `plan_rl_trial` 前六轮每轮 32 局：平均采集 79.16 秒，其余更新处理约 0.57 秒。GPU 空闲主要因为等待 CPU 排路；本容器 CPU 配额为 7.65 核，7 个采集线程并不对应整台宿主机的全部逻辑 CPU。

已完成的等价优化：

- 路线模拟每个动作只比较实际受影响的地格、工人位置和该工人库存，移除整座农场及所有库存的复制。测试编译仍逐动作与完整状态比较核对。
- 经营候选只复制会修改的项目与计数，提交时保留正在执行的路线、回执和市场历史；每次观察共享预留资源和估价结果。
- 每个 Rust 采集线程设置单线程 LibTorch/OpenMP，避免嵌套计算线程膨胀；有效 worker 数不超过容器配额。
- 计划策略的所有候选同属一个类别，直接计算等价的 masked softmax，省去 19 类展开；确定性推理省去无用的探索分布计算。

相同检查点、相同六局、3 workers 的局部排路优化对照：采集 **37.48 → 28.03 秒**，减少约 **25.2%**。六局终局现金、胜负、437 条训练样本及生产/动作统计一致。旧训练同时运行，因此这不是独占资源的稳定吞吐基准；不能把它承诺为所有轮次固定的加速倍率。

证据：[性能对照](../runs/plan_rl_perf_check_local/comparison.json)。68 项库测试及 2 项训练入口测试通过；另做了 CPU/CUDA 概率等价检查。

新增 `update_seconds` 和每局 `candidate_seconds`、`inference_seconds`、`route_execution_seconds`、`fixed_opponent_seconds`、`engine_seconds`。各局计时可能并行重叠，累加值不能当成整轮墙钟时间。

二进制替换不会让已启动的训练进程自动变快。停止旧进程后，在新目录通过 `--resume runs/plan_rl_trial/latest.json` 继续；未保存的当前轮需要重跑，已保存网络、Adam、对手与经验正常恢复。检查点格式和候选/奖励定义不变。
