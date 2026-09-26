# Kaggriculture：baseline + 局部经营 RL

当前公共 pipeline 为 **baseline-local-economy-v4**。

当前基线为 **farm2945_resilient_response**，从用户提供的提交包按 notebook 哈希核验后提取。
Rust 迁移以该策略为行为参照。此前 `local_v4_trial` 使用的是旧基线，检查点和成绩不能混用。

baseline 每回合实际运行，保留已有生产、工人路线、雇工与市场执行。
RL 只在作物续接机会选择 KEEP、换种或暂缓；执行层同步管理新增种子、后续服务和真实回执。
**不需要 BC，也不让随机网络重新接管整条工作系统。**

新基线的接口检查及 seed 42 双座位 KEEP 完整复现已通过。单次换种、经济效果和提交资源
需要在新基线下重新验收。Rust 市场与路线热点已通过组件对照；完整 Rust 采集/训练仍在迁移。

- [Rust 移植进度、测速与命令](native/README.md)
- [实施边界、配置与分阶段命令](docs/local_v4.md)
- 主实现：[src/route_rl/local](src/route_rl/local)
- 当前基线：[main.py](agents/farm2945_resilient_response/main.py)，来源与哈希见同目录 `provenance.json`。
- Rust 环境：third_party/kaggriculture-simulation，版本固定到 kaggle-environments 1.32.7。

在仓库根目录进行接口检查（不会自动跑训练或完整对局）：

```bash
PYTHONPATH=src python -m route_rl.check
```

KEEP 结果位于 `runs/farm2945_rust_migration/keep.json`；后续验收按文档逐项进行。

当前只保留半 RL 流程。旧版 BC、整路线接管实现及其训练产物已移除。
runs/ 和 replay/ 不纳入版本控制；baseline 和原始回放保留。
