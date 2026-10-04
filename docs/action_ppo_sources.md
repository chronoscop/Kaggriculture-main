# 本项目 PPO 来源与交付边界

参考为 [msdsm/kaggriculture-solution，固定 commit 84057a0](https://github.com/msdsm/kaggriculture-solution/tree/84057a0fda4238ccdebc46f9bf5496c6c4b2e00d)。本次阅读的具体文件、内容 SHA256、对应本项目文件和借鉴范围见 [来源清单](action_ppo_sources.json)。已有编码、模型、词表和 BC 部件继续采用 [BC 来源记录](action_bc_sources.md)。本项目源码自行维护在 `src/route_rl/ppo/`，训练、推理、评估和打包均无需参考 checkout。

| 本项目组件 | 借鉴内容 | 自身执行合同 |
| --- | --- | --- |
| `sampling.py`、`inference.py` | `actions/masks.py`、`actions/sequential.py` 的合法支持与条件概率；神经推理的固定形状接口 | 使用已安装的官方 Python 规则处理己方动作前缀，保存实际合法集合和采样概率；推理执行同一顺序。对手隐藏库存不进入支持集合。 |
| `rollout.py` | 双座位独立历史、固定输入与完整赛季采集 | 从实际完整终局构造胜=1、平=0.5、负=0；cash 仅用于诊断。 |
| `native/action_engine`、`native_backend.py`、`build_action_native.py` | 从固定 commit 迁入 RustBatchEnv、并行环境/特征、public tracker 和 NumPy buffer 接口 | 改用本项目模块名与构建入口；增加官方-prefix 支持、NOOP 槽保留、精确绝对 SELL；保留 `2×games` 同步 JAX 输入。源码/二进制身份校验，保留 Python 校验 backend。 |
| `economic_rules.py`、`sampling.py` | Final B 的条件动作选择与生产窗口/仓库规则 | 保留己方真实前缀，强制夜间销售先占订单槽；存在 mask 与 policy mask 分开，强制/外部因素不计 actor/KL/熵。未移植 A 的置信度重排和自动种子补单。 |
| `season_search/`、`controllers.py` | Final A 的 C++ 搜索器与最后一天接管 | 自有资产、连续观察记忆、明确神经 fallback；内部评分映射为未校准的预计终局得分，不加入现金辅助 reward；二进制/配置进入 continuation 身份。 |
| `action_teacher.py`、`candidate.py` | 规划器示范、冻结策略配不同控制器的思路 | 实际完整官方执行标签、整 seed 留出、原 BC 分割兼容、公开示范合并/泄漏检查；候选不改权重、不自动训练或部署。 |
| `objective.py`、`trainer.py` | clipped PPO、冻结参考策略的完整分布 KL、分头熵、Huber critic、Adam 与学习率计划；GAE | 概率、KL 和熵使用实际采集的动作前缀合法集合；目标版本为 `terminal-match-score-1-0.5-0-v1`。 |
| `pipeline.py`、`provenance.py` | critic 独立预热、循环更新和候选快照思路 | 预热冻结 actor 与 encoder；固定 BC teacher；自身源码与种子审计；继续策略带版本；所有训练产物保持候选。 |
| `packaging.py`、`tools/package_pipeline.py` | 本地策略导出、tar 包、文件校验清单 | 独立 source release；policy 包按检查点动作合同选择 BC 或 PPO 执行器，保留推理计算 dtype，移除优化器和训练状态，不自动提交比赛。 |

固定参考源码的 PPO 同样使用终局胜/平/负，尺度为 `+1/0/−1`；本项目改为 `1/0.5/0` 并使用 sigmoid 得分 critic，两者的胜负排序相同。参考搜索的现金差预测不是其 PPO 奖励，本项目也没有引入现金辅助奖励。没有迁入分布式/多卡自定义内核、Net2Net、额外市场/土地辅助训练目标。现金仍可作为公开状态或预测模型输入；critic 学习终局比赛得分。季末搜索以公开现金差预测经离散 logistic 不确定性映射为预计胜/平/负得分；该模型未校准，固定尺度映射通常保持现金差排序，不能宣称改变排序或提供强度改进。规则与搜索开关没有本项目胜率消融结果。

现有 plan/event 已实现批量生产、经济路线和两阶段转换，本 PPO 分支不把早期 mixed 的规模限制当作当前原因。DECEM 回放只支持连续生产、条件续种/转产、共享材料与工作路线、现金周转四类行为检查，不揭示其训练算法。此次补入的是 BC 后的完整赛季自博弈、实际条件概率、终局信用分配和独立候选对局；连通性测试不能证明上述经营能力已经学会。

源码包包含本项目 Python/Rust/C++ 源码、配置、工具、测试、说明与必要的 `kaggriculture-simulation` 源码依赖，含自己的两个 native 构建入口。保留 native、simulation、搜索 helper 和评估 baseline 随附的许可证及通知；不包含参考 clone、本地运行、回放、数据、模型、venv、凭据或编译物。已构建 wheel 可携带 Rust/C++ 二进制，标记解释器/平台兼容性；搜索策略包携带已核对 `.so`。策略包外部依赖 NumPy/JAX，PPO/经济推理依赖精确官方环境版本。目标比赛容器计时与正式表现需独立验收。

参考 revision 的根目录只有 `THIRD_PARTY_NOTICES.md`，没有仓库级 LICENSE；该 notice 明确不为其原创代码重新授权。本记录保留作者与文件来源，不把其他 Apache 依赖的许可证扩张为参考原创代码或整个交付包的许可证。
