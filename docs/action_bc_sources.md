# 完整动作 BC 来源与本项目适配

参考源码为 [msdsm/kaggriculture-solution，固定 commit 84057a0](https://github.com/msdsm/kaggriculture-solution/tree/84057a0fda4238ccdebc46f9bf5496c6c4b2e00d)。具体源文件与迁移时的 SHA256 见 [来源清单](action_bc_sources.json)。保留原作者来源；本记录不为对方原创代码赋予新的许可证。官方环境及依赖的许可仍由对应包提供。

这是一次代码整合：全动作 BC 的必要算法由本项目维护在 `src/route_rl/full_action/`，不是从忽略的 clone 动态导入、复制或启动。安装包包含 Python 模块与模型配置。clone 可用于继续阅读和比较，但运行时不需要它。

| 本项目组件 | 借鉴的公开组件 | 本项目适配 |
| --- | --- | --- |
| `features.py`、`inventory_tracker.py`、`tracker_constants.py` | 实体特征与公开库存估计 | 项目内导入；训练和推理共用 |
| `catalog.py`、`quantities.py`、`sell_quantity.py` | 单位词表及市场数量语义 | 同一动作 ID 与实际执行数量 |
| `labels.py`、`legality.py` | 完整动作标签与官方执行器有效性检查 | 独立回放入口；核对 1.32.7/1.33.0 规则及原始时间、seed |
| `model.py` | typed adapters、Transformer、同农场 2D RoPE、单位/市场头 | 保留基础 manual attention；移除 cuDNN、自定义 kernel 与搜索补丁路径 |
| `bc_objective.py`、`trainer.py`、`dataset.py` | CE、熵、初始策略 KL、固定 teacher、epoch 续训 | 项目配置与输入契约；当前单进程单设备 |
| `global_update.py`、`sharding.py`、`metrics.py` | JAX 梯度与指标汇总 | 只纳入 BC 所需的函数 |
| `inference.py` | 固定形状输入、单位与市场解码 | 使用本项目 checkpoint；保留最多 20 个己方单位及 40 个总单位的编码；超出的己方单位 PASS；无启发式/季末搜索 |
| `checkpoints.py`、`evaluation.py`、`agent_worker.py` | 参数校验、原子保存、独立进程对局思路 | 本项目 policy 契约；交换座位；完整终局得分 |

`replay_download.py` 使用官方 Kaggle API，参考 release 未提供历史下载器或历史回放。`action_bc.py` 管理本项目源码指纹、数据隔离、缓存审计、训练续跑与候选评估。

新的数据契约为 `public-full-action-bc-v3`，policy 契约为 `route-rl-full-action-policy-v1`。旧包装器缓存、旧动作 checkpoint、原生 event `.json` 权重均不会被静默续用；已下载的原始回放可复用到新准备目录。

这次没有接入公开方案的 PPO、自博弈采集、critic 拟合、启发式补课、模型扩深或部署晋级。后续需要在本仓库逐步实现这些环节。
