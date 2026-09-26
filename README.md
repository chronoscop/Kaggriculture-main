# Kaggriculture：独立 Rust 混合生产半 RL

当前唯一训练 pipeline：**mixed-production-v5**。

- 生产项目持续存在；项目生成任务，工人用连续路线服务多个动物与作物。
- Rust 规划器模拟沿途背包变化，预留种子、物料和地格，核对实际执行回执。
- RL 选择路线、生产项目、续种/转换、采购与雇工；销售结算目前使用规则。
- Rust 多局采集 → GPU 批量推理 → PPO 更新 → 完整检查点，全程不调用 Python baseline。
- `farm2945_resilient_response` 原版仅用于独立对战评估；旧 v4 和完整 baseline 移植链路已删除。

## 构建与训练

需要 Rust、C++17 编译器和匹配的 LibTorch；GPU 使用 CUDA 版 LibTorch。
本机原生库默认位于 `/usr/local/lib/python3.12/dist-packages/torch`，可用 `LIBTORCH` 覆盖。
没有 Python 训练进程；少量 C++ 桥接代码用于调用 LibTorch。

```bash
cargo build --manifest-path native/Cargo.toml --release --features train --offline -j 4
cargo test --manifest-path native/Cargo.toml --release --features train --offline -j 4

native/target/release/mixed-train \
  --out runs/mixed_v5_trial \
  --iterations 5 --games-per-update 32 --workers 7 \
  --device cuda --epochs 2 --batch-size 256 \
  --seed 1200 --opponent heuristic
```

`iterations` 是累计更新次数，`games-per-update` 是实际完整对局数（同种子双座位，必须为偶数）。
`--workers` 不传时自动读取容器 CPU 配额；本机约 7.65 核，默认 7。
`--opponent selfplay` 可改成同一冻结采样策略的自对弈；两席样本均参与更新。
当前默认轻量对手只是用于学习和调试，不代表强策略水平。

训练输出：`manifest.json`、`metrics.jsonl`、`latest.json`。

```bash
native/target/release/mixed-train \
  --out runs/mixed_v5_trial --resume runs/mixed_v5_trial/latest.json \
  --iterations 10 --games-per-update 32 --workers 7 \
  --device cuda --epochs 2 --batch-size 256 \
  --seed 1200 --opponent heuristic
```

继续训练必须保留 seed、对局数、对手、epochs 和 batch-size。
检查点包括网络、Adam 动量、迭代数和 Rust 随机状态；旧 v4 权重不可使用。

## 独立评估

先使用未见种子对轻量 Rust 对手评估：

```bash
native/target/release/mixed-train \
  --mode evaluate --out runs/mixed_v5_eval \
  --resume runs/mixed_v5_trial/latest.json \
  --seed 9001 --games-per-update 4 --workers 4 --device cuda
```

对原版 farm2945 的评估单独调用 Python 客户端，不计入训练吞吐：

```bash
PYTHONPATH=src python -m route_rl.evaluate \
  --checkpoint runs/mixed_v5_trial/latest.json \
  --seeds 9001 9002 --out runs/mixed_v5_eval/farm2945.json
```

原版位于 [agents/farm2945_resilient_response](agents/farm2945_resilient_response)，附来源和许可证。
环境固定使用 `third_party/kaggriculture-simulation` 的 1.32.7 Rust 规则实现。

## 目前验收范围

已验证混合路线的收肥→施肥、收获→续种、真实返仓销售、任务互斥、缺货和仓库溢出保护。
16 局 CUDA 并行验证已完成采集、更新和检查点保存，随后恢复训练再完成 16 局；CPU 自对弈与独立 farm2945 评估也已跑通。
这说明执行和训练闭环可运行，**尚不代表经济表现达标或已超过 farm2945**。

结构、约束和指标见 [pipeline 说明](docs/mixed_v5.md)；构建细节见 [native/README.md](native/README.md)。
