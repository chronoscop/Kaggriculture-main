# 原生混合生产 pipeline

只保留 `mixed-production-v5`。旧 baseline 移植组件、组件测速命令及 v4 训练入口已删除。

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

- `mixed-train --help`：独立训练、恢复与轻量对手评估。
- `mixed-agent --checkpoint FILE`：逐行接收公开观测、输出动作；供外部对战评估。
- `mixed-agent --checkpoint heuristic`：同一规划器的轻量规则选择器。

完整命令见[主 README](../README.md)，实现与限制见 [mixed_v5.md](../docs/mixed_v5.md)。
生产训练不读取 `agents/`、farm2945 路线数据、原策略 Python 或旧运行轨迹。
原版 farm2945 仅通过仓库外部评估客户端加载。
