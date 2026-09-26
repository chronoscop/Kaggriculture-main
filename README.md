# Kaggriculture：baseline + 局部经营 RL

当前公共 pipeline 为 **baseline-local-economy-v4**。

baseline 每回合实际运行，保留已有生产、工人路线、雇工与市场执行。
RL 只在作物续接机会选择 KEEP、换种或暂缓；执行层同步管理新增种子、后续服务和真实回执。
**不需要 BC，也不让随机网络重新接管整条工作系统。**

这一轮完成代码接线和小型接口检查。完整 KEEP 复现、单次换种闭环、经济效果、
CPU/RAM 和比赛平台验证尚未进行，按顺序逐项验收。

- [实施边界、配置与分阶段命令](docs/local_v4.md)
- 主实现：[src/route_rl/local](src/route_rl/local)
- 原基线：[agents/baseline.py](agents/baseline.py)，保持原文件不变。
- Rust 环境：third_party/kaggriculture-simulation，版本固定到 kaggle-environments 1.32.7。

在仓库根目录进行接口检查（不会自动跑训练或完整对局）：

```bash
PYTHONPATH=src python -m route_rl.check
```

下一步是运行 KEEP 完整复现，后续验收按文档逐项进行。

当前只保留半 RL 流程。旧版 BC、整路线接管实现及其训练产物已移除。
runs/ 和 replay/ 不纳入版本控制；baseline 和原始回放保留。
