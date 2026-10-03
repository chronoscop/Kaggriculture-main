# Rust 经营策略实验

`native/` 保留早期 mixed/plan/event 的模拟、执行、训练和审计代码。当前主线 BC 使用 Python/JAX，安装和训练不依赖本目录。当前入口与实验结论见 [主 README](../README.md) 和 [历史实验](../docs/experiment_history.md)。旧 Rust PPO 与全动作 BC 是不同管线，不能交换检查点。

## 仅构建执行器

需要 Rust 工具链，使用仓库内 `third_party/kaggriculture-simulation/src-rust/kagg-engine` 路径依赖：

```bash
cargo build --manifest-path native/Cargo.toml --release
cargo test --manifest-path native/Cargo.toml --release --lib
```

默认不启用 `train`，不需要 LibTorch。首次构建允许 Cargo 下载依赖；只有本地缓存齐全时才添加 `--offline`。无需使用旧文档中的临时工具链或缓存路径。

## 启用历史训练与审计

额外需要 C++17 编译器、归档工具 `ar` 和匹配的 LibTorch：

```bash
export LIBTORCH=/absolute/path/to/libtorch
export LIBTORCH_CXX11_ABI=1
cargo build --manifest-path native/Cargo.toml --release --features train
cargo test --manifest-path native/Cargo.toml --release --features train
```

将 `LIBTORCH` 替换为实际目录，其中应包含 `include/ATen/ATen.h` 和 `lib/`。也可指向兼容 PyTorch 安装的 `torch/` 目录。`LIBTORCH_CXX11_ABI` 必须与该库一致，可为0或1；编译器可通过 `CXX` 指定。Rust 通过 C++ C ABI 调用张量库，训练时不启动 Python 解释器。

有兼容 CUDA LibTorch 和 GPU 时，可额外运行：

```bash
ROUTE_RL_TEST_CUDA=1 cargo test --manifest-path native/Cargo.toml --release --features train
```

## 入口

| 程序 | 用途 |
| --- | --- |
| `mixed-train` / `mixed-agent` | 旧投资、路线与市场策略 |
| `plan-prototype` | 确定性经营原型与参数搜索 |
| `plan-train` / `plan-agent` | 经营计划比较和已验收组合 |
| `event-train` / `event-agent` | 事件菜单学习、配对验收及已验收部署 |
| `event-learning-audit` / `event-menu-audit` | 离线学习和实际菜单检查 |

编译后的程序位于 `native/target/release/`。先用对应 `--help` 检查当前参数，再根据检查点契约与历史记录定位匹配版本。历史文档中的旧 PPO 参数、`iterations` 语义和续训命令不能假定适用于当前入口。

agent 提供持久 JSONL 观察/动作接口；这不是直接可上传 Kaggle 的提交包。`latest` 包含学习状态，不代表新的学习权重已经获得部署资格；event/plan 使用检查点中的已验收部署，具体以对应实现为准。

源码及许可见 [Cargo.toml](Cargo.toml)、[LICENSE](LICENSE.txt)、[NOTICE](NOTICE.txt) 和模拟器目录。本地 farm2945 保留为外部评估参考，不是 DECEM 源码；历史事件训练不通过读取该参考源码生成训练标签。
