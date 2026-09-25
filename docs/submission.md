# 训练模型导出为 Kaggle submission

按用户提供的约束制作：压缩包不超过 100 MiB，磁盘不超过 8 GiB，
内存不超过 6.5 GiB，CPU 配额 1.6 vCPU，文件挂载到
`/kaggle_simulations/agent/`。每天最多 5 次提交、仅最近 2 个生效属于平台提交管理限制，
本地导出和验证不会占用提交次数。

官方多文件格式要求根目录包含带 `agent` 函数的 `main.py`：
https://www.kaggle.com/competitions/kaggriculture/overview/citation

## 打包命令

导出端需要 PyTorch 和 Linux x86_64 C 编译器（`cc`）。
提交运行时使用 Python 标准库、系统 C/math 库和包内的推理模块，
不需要 PyTorch、NumPy、CUDA、Rust、pip 安装或网络。

```bash
PYTHONPATH=src python -m route_rl.submission \
  --checkpoint runs/dynamic_v2/best.pt \
  --out runs/dynamic_v2/submission.tar.gz

PYTHONPATH=src python -m route_rl.submission_check \
  --archive runs/dynamic_v2/submission.tar.gz \
  --seeds 9103 9104 \
  --out runs/dynamic_v2/submission_check.json
```

安装项目后同样可用 `route-rl-export` 和 `route-rl-submission-check`。

只支持本仓库 v2 且 `takeover=0` 的 checkpoint。
暖启动模型依赖 baseline 开局，导出器会拒绝，避免提交时悄悄改变策略。
旧模板 checkpoint 也会被拒绝。

## 包内结构

```text
main.py
NOTICE.txt
_route_submission/
  __init__.py
  routes.py
  controller.py
  features.py
  deployment.py
  inference.so
  weights.bin
  metadata.json
```

入口优先按文件自身位置定位模块，无 `__file__` 的加载器使用官方挂载目录；
权重和动态库按包模块自身路径定位，不依赖当前工作目录或训练机器绝对路径。
每局重置控制器状态，并按玩家座位隔离。

权重只包含策略所需的 471,681 个 float32 参数，约 1.80 MiB。
critic、优化器、训练轨迹、baseline、参考 submission 和模拟器不进入归档。
参考包仅用于确认入口形式，没有将其策略或文本指令嵌入新模型。

C 推理实现使用普通 x86_64 指令、单线程，不依赖 BLAS/OpenMP 或 Python 扩展 ABI。
导出器核对参数形状和有限性，比较 PyTorch / C logits 及贪心选择后才生成归档。
运行时校验权重 SHA-256。验证失败不会覆盖已有输出文件。

## 时间预算与策略一致性

导出默认 `--planning-seconds 0.65`：每回合开始设置规划截止时间，
到时把后续路线扩展结束，已提交的前缀仍然执行。不会切回旧调度器。
截止时间在网络选择之间检查，并非硬实时中断；候选生成和正在执行的一次推理
仍需完成，因此实际 act 总耗时可高于 0.65 秒。

预算截断会改变未完成的计划尾部，应以**导出的最终包**重新评估现金差和胜率。
检查器报告 `planning_budget_cutoffs`；压力场景的截断单独记录。
`--planning-seconds 0` 关闭预算，仅用于数值/策略一致性诊断，不适合未经测速就提交。
最终不同权重和不同农场状态仍需重新做资源验证。

## 验证范围

检查器把归档解压到临时的 `kaggle_simulations/agent` 目录，从不同 cwd 启动。
提交进程使用 `python -I -S` 禁用外部 Python 包并绑定一个 CPU，
本地 Rust 引擎和对手在外部验证进程运行，不属于提交包。
每个种子跑两个座位、每局 719 次行动，记录启动耗时、最大和 P99 act 耗时、
进程峰值 RSS、预算截断以及最终现金。

额外压力场景使用 25 名工人、100 块已解锁空地和充足现金。
它在真实网络计算后强制选择合法采购来持续扩展路线，以覆盖繁忙规划；
这项测试衡量性能，不代表模型的真实经营策略或胜率。

2026-09-25 冒烟模型的本机验证结果：
- 压缩包：1,753,108 字节，约 1.67 MiB；
- 解压文件：1,929,211 字节，约 1.84 MiB；
- 峰值进程 RSS（包含压力测试）：108,568,576 字节，约 103.5 MiB；
- 两局普通对局最大 act：约 0.055 秒；
- 25 工人压力场景 act：约 0.827 秒，触发规划预算；
- 测试绑定单个 CPU，未导入 torch、numpy 或 kaggsim。

这些是本机实测值，不是 Kaggle 全状态或所有硬件上的性能保证。
二进制目标为 Linux x86_64，正式提交前仍应在 Kaggle 实际运行环境验证动态库、
路径、启动和时间限制。示例模型只训练过一轮，经营能力很弱，
`runs/dynamic_v2_smoke/submission.tar.gz` 是打包验证样品，不是已选定的参赛策略。
