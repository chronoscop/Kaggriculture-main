# farm2945 Rust 移植

行为参照固定为 `agents/farm2945_resilient_response/main.py`，源码 SHA256：
`afae271aae1702a566066483ea7f1b173dad85a720578b806950df65b20424f3`。
来源、许可证和 notebook 核验记录见该目录的 `provenance.json`。
移植代码保留 Apache-2.0 LICENSE / NOTICE。

## 最新范围调整：baseline 仅用于评估

用户已调整方向：停止以完整移植 farm2945 为前提推进训练。
farm2945 原版保留为独立评估对手/参照，不再要求进入正式训练的逐步动作、
训练对手或奖励计算路径。已有 Rust 组件保留并按实际需要复用。
这是一项后续架构变更，当前 Python v4 命令尚未按此改造。

后续实现方向：

1. Rust 规则执行器维护生产项目、路线、库存与工时承诺，执行 RL 选择的经营安排；不依赖 farm2945 内部 hook。
2. Rust 环境多局并行，使用自对弈或轻量 Rust 对手产生真实样本，批量送入已有 GPU 网络。
3. 用实际对局结果构造训练奖励，移除训练时额外运行 farm2945 参照局的要求。
4. 独立评估保留原版 farm2945，可继续用 Python 运行；分别报告训练吞吐与评估耗时。
5. 重设经营候选、特征、奖励和检查点版本；现有 KEEP/CHANGE 网络不能直接代表独立经营策略。

下面的状态表记录已完成的组件工作，不表示仍要完成全部 baseline 移植。

## 当前状态（2026-09-26）

**本阶段收尾：已有通过对照的 Rust 策略组件和独立 GPU PPO 更新器；尚无完整 Rust baseline 和采集训练闭环。**
`PYTHONPATH=src python -m route_rl...` 目前仍走原 Python 流程，未接入这些组件。
下面的提速是组件测试结果，不能用作完整采集/训练提速。

| 部分 | 状态 |
|---|---|
| 新基线提取、默认路径、导出许可、检查点来源校验 | 完成；旧基线权重禁止混用 |
| Python KEEP 接线一致性 | seed 42，两个座位各 719 步通过 |
| 肥料市场计算 `_at_sim` | Rust CPU / CUDA，各 7,000 条对照通过 |
| 工人访问 `_ca_spawn` / `_ca_visits` / `_db_visits` | Rust；包括位置首次出现顺序、空槽、雇工与夜间重置 |
| 作物 `_ca_decays` / `_ca_yield_path` / `_db_schedule` | Rust；保持原模型的产量与排程规则 |
| 路线/产量/排程组件对照 | 1,100 条通过：320 条真实调用、780 条合成边界；当前真实调用覆盖 ca_visits / ca_yield_path，db 两函数由边界样本覆盖 |
| 对照采样对原策略的影响 | 已有 seed 42 输入序列的 719 步动作全部一致 |
| 路线数据、选择器、chassis 已启用修正层 | 719 条真实动作、状态和库存投影对照通过；开局修改保留原路线对象共享语义 |
| 自家实体投影与终局规划 | 845 条对照通过，含 42 个接受的终局计划及偏离后恢复 |
| 牛羊替换策略 | 1,319 条对照通过，其中 346 条修改动作 |
| 提前销售、报价排序、仓库清理 | 4,555 条对照通过 |
| 晚期番茄投入、采购预算、种子缩减、肥料保留 | 已写 Rust 实现，尚未完成行为对照，不能视作验收通过 |
| 网络前向、PPO、Adam、模型断点恢复 | 已实现；CPU 数值完全一致，CUDA 最大绝对误差约 2.4e-7；验证含三次更新和一次 KL 停止 |
| 独立 `train-rows` 命令 | CUDA 8 条合成验证样本、3 次更新及检查点保存已跑通；不是整局训练效果或吞吐测试 |
| 其余经营修正、完整对手预测模型 | 尚未完成迁移 |
| Rust 多局采集与真实样本批量推理接线 | 尚未接通 |

`routes.rs` 接收策略预定路线，预测的是策略的未来指令，不读取未来真实环境。
`all_visits` 保留位置首次出现顺序，避免后续候选平局选择变化。
每个并行对局最终需要独立策略状态；当前并行队列处理的是独立组件输入。

## 构建

需要 Rust 工具链；CPU 版本不需要 CUDA、Python 或网络依赖。

```bash
cargo build --manifest-path native/Cargo.toml --release --offline -j 4
cargo test --manifest-path native/Cargo.toml --offline -j 4
```

CUDA 版本要求 NVIDIA 驱动、CUDA Toolkit 的 NVRTC 库；
`CUDA_HOME` 默认 `/usr/local/cuda`。
Rust 直接调用 CUDA Driver / NVRTC，GPU 内核位于 `src/market.cu`。
上述 `cuda` 市场组件运行时不启动 Python、不载入 Python 或 LibTorch。
神经网络训练的 `train` 功能另行链接 LibTorch，见下节。

```bash
cargo build --manifest-path native/Cargo.toml --release --features cuda --offline -j 4
cargo test --manifest-path native/Cargo.toml --features cuda --offline -j 4
```

当前会话工具链在 `/tmp/route-rl-rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin/`；
若 `cargo` 不在 PATH，需要先把该目录加到 PATH。

## 原生神经网络训练组件

```bash
cargo build --manifest-path native/Cargo.toml --release --features cuda,train --offline -j 4

native/target/release/verify-learning runs/farm2945_rust_migration/learning_cpu.json cpu
native/target/release/verify-learning runs/farm2945_rust_migration/learning_cuda.json cuda

native/target/release/train-rows \
  --data runs/farm2945_rust_migration/learning_rows.jsonl \
  --out runs/farm2945_rust_migration/learning_checkpoint.json \
  --device cuda --epochs 3 --batch-size 8
```

最后一条是**合成数据的更新冒烟测试**，不能代替正式采集/训练命令。
输入 JSONL 必须携带新 baseline 哈希、采样时模型权重和 on-policy 样本；
`--resume` 要求数据来自恢复后的同一模型，不能反复拿旧 rollout 当作新样本。
检查点保存模型、Adam 动量/步数、学习率和 Rust 随机状态。

`train` 通过小型 C++ C ABI 桥调用本机匹配的 LibTorch；策略网络、PPO 和优化器调度在 Rust 中。
运行时不启动 Python、不调用 Python C API。不是所有依赖源码都改写为 Rust。
`LIBTORCH` 指向含 `include/`、`lib/` 的安装目录；本机默认复用
`/usr/local/lib/python3.12/dist-packages/torch` 的原生库（2.8.0+cu128）。
构建需要 C++17 编译器；`LIBTORCH_CXX11_ABI` 默认 1，必须匹配库。
Rust 初始化和采样使用 SplitMix64，不承诺与 Python 随机数流逐位一致；
数值对照加载同一组权重和样本。

新增策略对照命令：

```bash
native/target/release/verify-chassis runs/farm2945_rust_migration/chassis_reference_42.jsonl
native/target/release/verify-physics runs/farm2945_rust_migration/physics_reference.jsonl
native/target/release/verify-cattle runs/farm2945_rust_migration/cattle_reference.jsonl
native/target/release/verify-sales runs/farm2945_rust_migration/sales_reference.jsonl
```

这些是组件差分检查，部分输入含原 Python 层的进入状态；
不等同于完整 Rust 策略独立运行整局的动作一致性检查。

## 对照与测速

现有测试输入在 `runs/farm2945_rust_migration/`，不纳入 Git。
这两个 Rust 命令均会先核验基线哈希和逐条输出，一旦不一致立即失败。
GPU 模式直接对照 Python 预先保存的答案；hybrid 模式分别验证 CPU 和 GPU。

```bash
native/target/release/farm2945-native bench-market \
  --cases runs/farm2945_rust_migration/market_reference.jsonl \
  --backend hybrid --gpu-batch 32768 --repeats 200 \
  --out runs/farm2945_rust_migration/bench.json

native/target/release/verify-routes \
  runs/farm2945_rust_migration/routes_reference.jsonl 100 6
```

`--workers` 默认读取 CPU affinity 和 cgroup 配额；本容器可见 96 核，实际配额约 7.65 核，默认 7 个工作线程。
CPU 与 GPU 从同一队列领取不同任务；完成计数检查避免重复或漏算。
`--gpu-batch` 默认 32768，小批量启动开销较明显，可按实际候选批次规模调整。

### 本机实测（A40；仅组件）

肥料市场：相同 7,000 条输入重复 200 次，总共 140 万次，排除加载、CUDA 编译和首次校验：

| 执行方式 | 秒 |
|---|---:|
| Rust 单核 | 5.682 |
| Rust 6 核 | 1.007 |
| CUDA，批量 4096 | 1.264 |
| CUDA，批量 32768 | 0.473 |
| Rust 6 核 + CUDA，批量 32768 | 0.345 |

所有执行方式的校验和均为 `42799343400`。
协同比单核快约 16.5 倍；这是重复已捕获输入的吞吐，真实采集还要产生这些输入，收益须后续实测。

持续测试：2,100 万次，7 个 CPU 工作线程 + CUDA，计算 4.481 秒。
CPU 完成 7,242,880 次，GPU 完成 13,757,120 次。
含初始化全程 5.058 秒，CPU user+system 34.507 秒，平均约 6.82 个核。
200ms GPU 采样含初始化：中位数 81%，峰值 84%，均值 68.3%。
利用率是设备采样值；该测试没有执行神经网络训练。

路线组件：同一组 1,100 条输入重复 100 次，单核 5.553 秒、6 核 0.953 秒，约快 5.8 倍。
计时包括 Rust 结果对象构建，不含输入 JSON 解析和首次结果核验。

原始数据：`bench_*.json`、`routes_cpu_*.json`、`gpu_utilization.csv`、`cpu_timing.txt`。

## 重建离线参照

Python 仅在此处作为原策略行为参照；生成文件后，Rust 测试不需要 Python。
这些工具不用于正式采集或训练。

```bash
python tools/capture_farm2945_reference.py --seed 42 \
  --max-cases 5000 --random-cases 2000 \
  --out runs/farm2945_rust_migration/market_reference.jsonl \
  --actions-out runs/farm2945_rust_migration/action_reference_42_0.jsonl

python tools/capture_farm2945_routes.py \
  --actions runs/farm2945_rust_migration/action_reference_42_0.jsonl \
  --out runs/farm2945_rust_migration/routes_reference.jsonl
```

## 新方向的完成条件

1. Rust 经营执行器可以独立完成整局，不调用 Python baseline，不依赖固定回放。
2. 完成多局并行、真实决策样本、GPU 批量推理和 PPO 更新闭环。
3. 用固定种子检查合法性、项目执行和资金/库存回执；独立种子评估对 farm2945 的经济结果。
4. 同工作量报告完整采集吞吐、更新耗时和 CPU/GPU 利用率；组件测速不替代整体验收。

完整迁移 farm2945 剩余修正层和对手模型不再是训练链路的完成条件。
