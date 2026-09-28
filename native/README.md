# 原生混合生产 pipeline

> **当前训练主线：[事件与下一批生产对照学习](../docs/event_plan_training.md)（event-plan-improvement-v4）**，入口 `event-train` / `event-agent`。支持加载第55轮v3已验收组合为冻结底座，新增可修订批次、共享资源约束和A/B/C真实对照；新实验命令见该文档。
> [v3固定窗口版](../docs/plan_improvement.md)保留用于基准与检查点读取，下方v8命令为历史实验。

> 旧版：[可学习市场决策](../docs/mixed_v8_market.md)（v8-market-3）。模型控制交易顺序和数量，新增对手潜在供给信息与事件触发，支持规则交易对照；320 维观测、19 类动作，旧 v7 / v8-market-1 / v8-market-2 检查点不兼容。

旧入口为 `mixed-production-v8`。旧 baseline 移植组件、组件测速命令及 v4 训练入口已删除。

## 构建

```bash
cargo build --manifest-path native/Cargo.toml --release --features train --offline -j 4
cargo test --manifest-path native/Cargo.toml --release --features train --offline -j 4
```

CUDA 测试需有 GPU：

```bash
ROUTE_RL_TEST_CUDA=1 cargo test --manifest-path native/Cargo.toml --release --features train --offline -j 4
```

不启用 `train` 时可只编译、测试 CPU 执行器，不依赖 LibTorch。
启用 `train` 时，Rust 通过小型 C++17 C ABI 调用匹配的 LibTorch；不启动 Python 解释器。
本机默认 `LIBTORCH=/usr/local/lib/python3.12/dist-packages/torch`，包含 `include/` 和 `lib/`。
可配置 `LIBTORCH`、`CXX`、`LIBTORCH_CXX11_ABI`（默认 1），ABI 必须与库一致。
GPU 支持来自 CUDA 版 LibTorch，不再依赖此前市场测速的 NVRTC 内核。

当前会话 Rust 工具链：
`/tmp/route-rl-rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin/`。
若 cargo 不在 PATH，把该目录加入 PATH。离线 Cargo 缓存位于 `/tmp/route-rl-cargo`。

## 入口

- `mixed-train --help`：独立训练、历史对手池、晋级验证、恢复与评估。
- `mixed-agent --checkpoint FILE`：逐行接收公开观测、输出动作；供外部对战评估。
- `mixed-agent --checkpoint heuristic`：同一规划器的轻量规则选择器。

完整命令见[主 README](../README.md)，实现与限制见 [mixed_v7.md](../docs/mixed_v7.md)。
生产训练不读取 `agents/`、farm2945 路线数据、原策略 Python 或旧运行轨迹。
原版 farm2945 仅通过仓库外部评估客户端加载。

## 经营计划实验原型

新增 `plan-prototype`：批量经营、商品价值路线评分及完整赛季参数搜索。已完成小规模配对实验；属于CPU参数搜索阶段，尚未接入新的神经网络PPO训练。结果、限制及可运行命令见 [原型实验](../docs/plan_prototype_results.md)。
