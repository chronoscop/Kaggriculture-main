# Kaggriculture Route RL

动态路线强化学习训练基座：同一个网络逐项选择工作、目标地格和种养投入，为所有工人构造路线。共享任务、库存、资金和工时预留；执行时确认实际到账，条件变化后滚动重排。Rust 引擎推进真实状态，冻结 baseline 作为对手。这个仓库**还没有训练好的获胜策略**。

## 项目结构

```text
.
├── agents/baseline.py          # 冻结的 farm2945_harvest_counter baseline
├── src/route_rl/              # 逐项规划、共享预留、执行器、PPO、评估与预检
├── tests/                     # 规划/PPO 测试及可选 Rust 集成测试
├── docs/architecture.md       # 动作、状态和训练设计
├── third_party/
│   └── kaggriculture-simulation/  # Rust 1.32.7 模拟器及其原始许可证
└── pyproject.toml
```

`agents/baseline.py` 的 SHA-256 为 `aa623df1a03567d4a1ac40fb7113295838189d19ebb4b9832e9374bb8fc86ff9`，与 `farm2945_harvest_counter.ipynb` 的校验值一致。训练不依赖本机其他目录。Rust 模拟器的源码、测试和许可证保留在 `third_party/`。

## 安装与预检

在仓库根目录运行，需要 Python 3.10+、稳定版 Rust 工具链和 PyTorch。租用 GPU 后，请确认当前 PyTorch 能识别 CUDA。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[train]"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
cargo build --release --manifest-path third_party/kaggriculture-simulation/src-rust/Cargo.toml -j 4
python -m unittest discover -s tests -v
route-rl-check --rust-steps 24 --takeover 0
```

可选的规则一致性检查需要官方环境。它会把同一动作流分别交给官方和 Rust 引擎，逐回合比较完整状态：

```bash
python -m pip install --no-deps kaggle-environments==1.32.7
python -m pip install jsonschema requests
route-rl-check --official-steps 24 --rust-steps 24 --takeover 0
```

## 训练与评估

```bash
route-rl-evaluate --mode baseline --seeds 9001 --out runs/baseline.json
route-rl-train --episodes 100 --seed 1234 --out runs/dynamic_v2
route-rl-evaluate --mode checkpoint --checkpoint runs/dynamic_v2/best.pt --seeds 9101 9102 --out runs/dynamic_v2/heldout.json
```

每次训练迭代用同一种子打双座位两局；`--episodes 100` 因而是 200 局训练。每 10 次迭代及最后一次迭代，用固定的 9001/9002 双座位选择 `best.pt`。最终请用未参与模型选择的种子评估 719 次行动后的现金和胜率。`latest.pt` 保存优化器，可接着训练：

```bash
route-rl-train --episodes 200 --seed 1234 --out runs/dynamic_v2 --resume runs/dynamic_v2/latest.pt
```

默认从第 0 步接管全部工人，最长规划到当日结束（24 个时段），每 6 回合滚动重排。可用 `--horizon` 和 `--replan-interval` 调整；`--takeover` 只用于暖启动对照实验。旧模板模型不兼容，必须重新训练。

查看独立种子下的规划依赖、实际动作与到账记录：

```bash
route-rl-evaluate --mode checkpoint --checkpoint runs/dynamic_v2/best.pt --seeds 9101 9102 --trace-dir runs/dynamic_v2/traces --out runs/dynamic_v2/heldout.json
```

训练输出默认位于 `runs/`，已被 Git 忽略。首次上传 GitHub 时提交源码、`Cargo.lock`、baseline 和文档即可。路线动作与已知局限见 [架构说明](docs/architecture.md)。

## 导出比赛提交包

训练完成后可导出根目录含 `main.py` 的 `submission.tar.gz`。包内只保留策略权重、动态路线执行器及单线程 CPU 推理模块，无 PyTorch / NumPy / Rust 运行依赖。

```bash
route-rl-export --checkpoint runs/dynamic_v2/best.pt --out runs/dynamic_v2/submission.tar.gz
route-rl-submission-check --archive runs/dynamic_v2/submission.tar.gz --seeds 9103 9104 --out runs/dynamic_v2/submission_check.json
```

导出时需要 Linux x86_64 和 C 编译器。检查器在独立单核进程验证完整对局与压力场景，输出体积、内存、耗时及现金结果。时间预算可能截断计划尾部，正式选模须评估最终导出的包。限制、入口与验证边界见 [提交包说明](docs/submission.md)。
