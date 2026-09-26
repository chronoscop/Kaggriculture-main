# Baseline-local v4：先接线，再逐项验收

当前公共入口已经切换到 v4。baseline 在每个回合实际运行，RL 仅选择局部经营变更。
没有 BC 前置阶段。旧版 BC、整路线接管实现、检查点及日志已移除；当前仅维护此半 RL 流程。

## 本轮范围

已接线：
- 在 baseline 内部的 _db_plan / _db_execute 处接入独立的经营控制器，原 agents/baseline.py 不变。
- 已有短周期作物即将收获、路线内 24 回合内存在下一次 PLANT 时，生成一次续种机会。
- KEEP（继续 baseline）、换种、暂缓当前种植窗口。机会按地格/原品种/种植日去重。
- 调用 baseline 的服务预测生成完整候选。逐个考虑五种作物，不使用 _db_quote 的利润评分或 DB_MIN_GAIN 排序筛选。
- 保留每日浇水、成熟收获、长期作物退出的访问需求。共享地格、工人时段及新增种子预算。
- 选择后在原执行链中管理采购、实际种植和后续服务；baseline 继续执行其他生产。
- 真实状态回执、取消原因、错过服务日、未完成合同均记录，不把发出指令直接当成经营成功。
- 事件级 PPO、同种子双座位 baseline 对照、独立评估、完整 baseline 随提交包部署。

第一版的明确限制：
- 开局、动物、雇工和原有市场经营仍由 baseline 控制。
- 仅在已有路线服务窗口内修改作物项目，不支持自由生成新路线或任意转牧。
- 候选使用 baseline 既有养护周期模板；还没有搜索每种作物所有可能的持有期限。
- 未来访问是 baseline 路线预测，不能保证未来每次雇工/访问都实现。执行检查真实位置和库存；
  错过种植窗口会取消，已种作物在后续可用访问继续服务并报告缺口。
- baseline 自己已持有的 _DB_STATE 作物项目不被第一版抢占。
- 种子使用单独采购，第一版尚未消除所有被替换方案的冗余采购。
- 不能据“接线完成”宣称收益、路线一致性或平台资源限制已通过。

## 文件分工

- local/adapter.py：隔离加载 baseline、内部 hook、机会识别、合同执行与真实状态回执。
- local/contracts.py：候选项目和服务时段预留。
- local/encoding.py：市场、库存、需求、季节、候选工时特征。
- local/policy.py：先 KEEP/CHANGE 二选一，再选择替代候选。候选数量不改变初始总修改概率。
- local/runner.py：完整对局和配对比较，参照局信息不进入策略输入。
- local/training.py：只对事件决策更新 PPO。
- local/evaluate.py：baseline / KEEP / 随机 / 固定换种 / 模型五种对照。
- local/export.py、runtime.py：基线源码 + 局部控制器 + 小网络；推理只用标准库。
- local/resource_check.py：显式运行的单核、无第三方 Python 包测量；不是导出时自动通过的验收。

## 默认实验配置

start_day=6；每局最多1次被接受修改；最多1个未完成项目；全季新增种子承诺上限300；
初始修改总概率3%。预算检查还保留 baseline 本身的现金/饲料/既有订单预留。
这些是可配置的前期探索限制，不代表对手策略或最终最佳参数。

episodes 是累计更新次数；games-per-update 是每次策略对局数，必须为偶数。
每个策略种子、座位还要独立运行原 baseline 参照局，不能忽略这部分计算成本。
默认16局策略对局/更新，因此100次更新为1600局策略对局，另有参照及评估对局。

奖励：
M = 自己季末现金 - 对手季末现金；
delta_margin = M_local - M_baseline；
终局奖励 = delta_margin / 10000。
同一条事件轨迹连接到季末，gamma=1；不给走路少、任务多、修改次数等额外奖励。
同种子对照仍独立推进市场；不把参照局未来价格、对手私有信息喂给策略。

## 测试按顺序手动进行

本轮仅执行语法检查、baseline 接口加载和小型接口测试。
未运行下面的完整对局验收、长训练或提交资源验收。

0. 接口检查（默认不会打对局）：

```bash
PYTHONPATH=src python -m route_rl.check
```

1. KEEP 完整复现（下一步先做这一项）：

```bash
PYTHONPATH=src python -u -m route_rl.evaluate \
  --mode keep --verify-identity --seeds 42 \
  --out runs/local_v4_checks/keep.json
```

逐回合比较 wrapper 与独立 baseline 在相同观察下的完整动作，差一回合就失败。
同种子独立参照局进一步给出现金差。不以“现金超过3000”作为合格标准。

2. 单次固定换种，检查是否真正出现机会、是否落地和完成：

```bash
PYTHONPATH=src python -u -m route_rl.evaluate \
  --mode fixed --crop CARROT --max-changes 1 --seeds 42 \
  --trace-dir runs/local_v4_checks/fixed_traces \
  --out runs/local_v4_checks/fixed.json
```

没有发生修改时不能把该局计作换种执行通过。逐条看 decision、contract_start、unit_edit、
service_window_missed、contract_end，以及未完成合同。先解决执行问题，再判断经济效果。

3. 随机经营对照（仍保留相同预算和3%修改门）：

```bash
PYTHONPATH=src python -u -m route_rl.evaluate \
  --mode random --max-changes 1 --seeds 43 44 \
  --out runs/local_v4_checks/random.json
```

4. 完成上述检查后，才启动小规模事件训练。现在不建议跳过检查直接长跑：

```bash
PYTHONPATH=src python -u -m route_rl.training \
  --device cuda --episodes 5 --games-per-update 16 \
  --max-changes 1 --max-active 1 --max-extra-cash 300 \
  --change-probability 0.03 --lr 0.0001 \
  --seed 1234 --eval-every 1 --eval-seeds 9001 9002 \
  --out runs/local_v4_trial
```

这里默认2个评估种子仅用于接线后试跑，不够证明模型有效。
正式选模应另固定32个未训练种子，最终测试另留64个种子；参数支持 --eval-seeds / --seeds。
禁止训练种子与评估种子重叠。模式/预算改变应建立新的实验目录，不能伪装成同一断点续训。

```bash
PYTHONPATH=src python -u -m route_rl.training \
  --resume runs/local_v4_trial/latest.pt \
  --device cuda --episodes 10 --games-per-update 16 \
  --seed 1234 --eval-every 1 --eval-seeds 9001 9002 \
  --out runs/local_v4_trial
```

## 记录与选模

metrics.jsonl：实际策略样本数、策略/参照现金、配对经济改善、修改/完成/取消/回执缺口、采样与更新时间。
eval.jsonl：完整各局结果和汇总；默认使用与部署相同的 KEEP/CHANGE 贪心决策。
--stochastic 可另行测量采样策略，不能把采样成绩当成部署贪心成绩。
初期门概率只有3%，贪心模型可能一直KEEP，这是对照行为；看采样训练是否得到有效修改和经济信号。

经济指标：mean/median delta_margin、delta_cash、最差四分之一局损失、胜率、改善率。
双座位按种子聚合后报告标准误，不把两个座位当成完全独立样本。
执行指标与任务量不能替代经济指标。
best.pt 仅表示当前评估集合内最好，不附加“验收通过”或“强于baseline”的布尔标签。

## 后续导出

```bash
PYTHONPATH=src python -m route_rl.submission \
  --checkpoint runs/local_v4_trial/best.pt \
  --out runs/local_v4_trial/submission.tar.gz
PYTHONPATH=src python -u -m route_rl.submission_check \
  --archive runs/local_v4_trial/submission.tar.gz --seeds 9103 \
  --out runs/local_v4_trial/submission_check.json
```

新版包必须包含 baseline，它是运行时主体，不再只作为教师。
导出检查 schema、基线哈希、参数、标准库推理概率一致性、100 MiB压缩及8 GiB解压限制。
CPU/RAM/真实对局收益必须通过后续资源检查单独测量，不能用旧包测量结果替代新版验收。
当前资源检查覆盖所选真实对局；专门压力场景和Kaggle平台实测仍是后续步骤。
不会自动上传，也不会消耗平台提交次数。
