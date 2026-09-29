# Kaggriculture：Rust 混合生产半 RL

> **当前实验：[资源与市场条件驱动的生产提案](docs/event_policy_contextual.md)（event-policy-iteration-v12）**。保留已验收底座与收获衔接，完整比较续种、换作物、小批转养和延后；只按终局胜负学习和验收。夜间命令：`bash tools/run_contextual_overnight.sh runs/event_policy_contextual_overnight 10`。这是待验证的训练实验，尚未证明长训收益。此前证据见[执行审计](docs/conditional_execution_audit.md)。
> [v3固定窗口版](docs/plan_improvement.md)保留用于基准与检查点读取，下方v8命令为历史实验。

> 旧版：[可学习市场决策](docs/mixed_v8_market.md)（v8-market-3）。模型学习商品优先顺序、按实际预算选择交易数量，并结合对手成熟时间、种养变化响应市场与资金事件；独立策略/价值网络。支持规则交易对照和两个检查点直接对战。

对手与探索沿用 [PFSP](docs/mixed_v7_pfsp.md) 和 [分对局探索](docs/mixed_v7_exploration.md)；成功经验沿用 [受保护经验](docs/mixed_v7_league.md)。历史文档中的旧检查点续训命令不适用于当前版本。

旧训练 pipeline：**mixed-production-v8**。Rust 采集、LibTorch CPU/CUDA 学习；farm2945 仅用于独立评估。

## 本版改动

- **市场决策**：`--market-mode learned` 学买卖与持有；`rule` 为规则交易对照。新增市场历史、资金需求及对手公开产量特征。完整两组实验命令见 [v8 文档](docs/mixed_v8_market.md)。

- **分层策略**：先选经营/路线类别，再选具体方案；随机训练与确定性执行使用相同分类。
- **受控探索**：每 32 局含 18 局纯策略随机、6 局 `80% 当前策略 + 20% 生产类别均衡探索`、8 局确定性探测，PPO 使用实际采样概率。等待保留为合法经营选项。
- **投入与执行**：新建/转产项目每天最多 2 个、同时待启动最多 2 个，订单报价预算为当天开局现金的 25%。启动期保护最长 24 步，真实到货后调度。
- **自生成成功经验**：约 25% 采集局使用确定性当前策略；它们与冻结历史模型的盈利轨迹可供独立辅助学习，不混入 PPO。
- **真实现金回报**：按决策间隔记录实际现金变化和时间，用 GAE 计算目标；无开工、走路、采购次数奖励。
- **稳定对手**：历史模型确定性执行；混合规则对手、当前模型和历史模型，定期验证后晋级。

## 构建

需要 Rust、C++17 和匹配的 LibTorch；本机默认库目录为 `/usr/local/lib/python3.12/dist-packages/torch`。
训练不启动 Python 解释器；C++ 桥接调用 LibTorch。

本机 cargo 不在 PATH 时：

```bash
export PATH="/tmp/route-rl-rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin:$PATH"
export CARGO_HOME=/tmp/route-rl-cargo
```

```bash
cargo build --manifest-path native/Cargo.toml --release --features train --offline -j 4
ROUTE_RL_TEST_CUDA=1 cargo test --manifest-path native/Cargo.toml --release --features train --offline -j 4
```

## 短训练实验

**使用新目录，从头训练；旧 v7 检查点（包括独立网络 v7-independent-6）和 v8-market-1 / v8-market-2 检查点不兼容。只有本版本新训练生成的检查点可以续训。**

```bash
native/target/release/mixed-train \
  --out runs/mixed_v8_market3_learned_trial \
  --iterations 10 \
  --games-per-update 32 \
  --workers 7 \
  --device cuda \
  --epochs 2 \
  --batch-size 256 \
  --seed 1200 \
  --opponent league \
  --market-mode learned \
  --exploration 0.2 \
  --imitation-weight 0.05 \
  --eval-every 5 \
  --eval-games 8 \
  --eval-seed 1000000000
```

共 **10 轮、320 局采集**：每轮 24 局使用随机探索策略、8 局为确定性经验采集。确定性局不产生 PPO 样本。
当前模型自对弈的随机局可使用双方样本；历史对手样本只进入独立经验筛选。
第 5、10 轮进行验证，包含固定评估、轮换对手初筛及有条件的晋级复核；验证局数随对手池和初筛结果变化，不参与学习。
`iterations` 为累计轮数；`games-per-update` 为完整对局数，同种子交换座位，必须为偶数。

### 看哪些输出

- `metrics.jsonl`：新增 `policy_loss`、`value_loss`、`policy_entropy`、`mean_ppo_kl`；采集与更新耗时、每局现金/工作/收获、经营类别选择、等待概率、经验库和辅助更新统计。
- `evaluations.jsonl`：固定验证种子的确定性/随机策略收益及冠军挑战结果。
- `latest.json`：网络、Adam、训练随机状态、历史对手池、成功经验库，可直接续训。
- `best.json`：仅在通过晋级时生成或替换；未生成表示尚未达标。

重点关注：

1. `greedy_vs_heuristic.mean_cash / mean_margin`：确定性执行是否赚钱。
2. `sampled_vs_heuristic`：纯策略随机采样是否也改善；验证不添加 20% 探索分支。
3. `inactive_games`、实际收获与 `projects_started`：有没有完成真实生产。
4. `experience_episodes / imitation_samples`：是否产生了可用盈利经验并用于学习；经验库为空时辅助更新为 0。
5. `games[].policy_wait_probability`：网络本身的等待概率，按座位统计；`selected_groups` 为实际类别次数。`greedy_probe=true` 的局要与随机局分开看。

不能用 loss 或工作量代替实际收益。这里提供的是新训练机制，经济效果由短实验决定。

### 续训

确认短实验后，可以累计续到第 20 轮：

```bash
native/target/release/mixed-train \
  --out runs/mixed_v8_market3_learned_trial --resume runs/mixed_v8_market3_learned_trial/latest.json \
  --iterations 20 --games-per-update 32 --workers 7 \
  --device cuda --epochs 2 --batch-size 256 \
  --seed 1200 --opponent league \
  --market-mode learned \
  --exploration 0.2 --imitation-weight 0.05 \
  --eval-every 5 --eval-games 8 --eval-seed 1000000000
```

恢复时保留采集、探索、辅助学习和验证配置。辅助权重 0 可用于新 run 的关闭辅助学习实验，探索率 0 可用于新 run 的关闭额外探索实验。

## 独立评估

用未参与训练和晋级的新种子：

```bash
native/target/release/mixed-train \
  --mode evaluate --opponent heuristic \
  --out runs/mixed_v7_eval \
  --resume runs/mixed_v8_market3_learned_trial/latest.json \
  --seed 9001 --games-per-update 16 --workers 7 --device cuda
```

原版 farm2945 只通过独立评估客户端运行：

```bash
PYTHONPATH=src python -m route_rl.evaluate \
  --checkpoint runs/mixed_v8_market3_learned_trial/latest.json \
  --seeds 9001 9002 --out runs/mixed_v7_eval/farm2945.json
```

详细实现与限制见 [mixed_v7.md](docs/mixed_v7.md)，构建细节见 [native/README.md](native/README.md)。
