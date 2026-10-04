# 本项目的完整动作 PPO

本指南说明自有 `route_rl.ppo` pipeline；安装、从当前 BC 产物开始运行的命令见 [README](../README.md)。参考 clone 不参与运行。没有训练完成的 BC 权重时，先审查源码与配置；不要把初始化模型或测试权重当成最终策略。

## 迁移范围与证据

借鉴公开参考的 BC → 冻结 actor 的 critic 拟合 → 当前策略双座位自博弈 → clipped PPO、GAE、固定 BC KL 约束与单位/市场熵。算法来源及文件哈希见 [来源记录](action_ppo_sources.md)。本项目使用现有 124 维特征、Transformer 和绝对 SELL 头，新增完整在线反馈闭环；不是调用参考仓库的训练脚本，也不声称复制了它的历史训练量或比赛成绩。

[历史记录](experiment_history.md#replay-evidence)中的 DECEM 回放支持连续生产、条件续种/转产、共享材料与工作路线、现金周转四项行为诊断，不能揭示 DECEM 的训练算法。旧 mixed 的规模限制不能直接归因于当前模型；plan/event 已有批量生产、经济路线及两阶段转换。全动作模型仍有自身的 20 己方单位、40 总单位容量限制。PPO 采集超限会明确停止，不能默默丢弃单位后声称完整动作学习。

目前 BC 已有完整轨迹监督与冻结初始策略 KL，但没有在线采样概率、终局 score 标签和训练过的 critic。PPO 新增的能力是：按真实状态逐槽采样合法指令，存当时的条件支持集与联合概率；完整执行到终局；用同一支持集重新计算 PPO ratio、熵及 teacher KL；将新执行合同写入策略，评估和打包也按它推理。这让采集、标签、执行和交付有一致定义。

v2 默认接入 `native/action_engine` 的自有 Rust 批环境、特征编码、独立 public inventory tracker 和条件支持集。每步同步处理多局，把 `2×games` 个座位送给单设备 JAX；复用 host 缓冲，保存实际 float32 输入和完整概率标签。`official-python` 保留为显式校验 backend，两者共用经济规则及采样器。扩展源码、构建 receipt、二进制 hash 进入 run 身份；缺失或过期会停止，不静默换 backend。

完整采集的 `collection_seconds`、`collection_env_steps` 和 `collection_env_steps_per_second` 进入每轮诊断；步数按双方座位合计，耗时包含 setup/首次 JIT、特征、支持、推理、存储和可选搜索。未运行正式模型吞吐对比。

这次也接入了 B 方向的经济规则、可选最后一天 C++ 搜索、实际动作教师示范生成。分布式、多卡、Net2Net、特殊注意力 kernel 仍未迁入。默认每轮四局，不复刻参考服务器配置或历史训练量。入口检查 cgroup 主机内存与完整 rollout 预算；特征和支持集每局约 0.22 GiB，默认四局约 0.9 GiB，另有模型、优化器和 JAX 开销。增大 games 要建立新 run；主机预检不保证峰值显存。

## 目标与概率合同

唯一比赛回报是第 720 个状态的终局胜=1、平=0.5、负=0。参考的 +1/0/−1 奖励在本项目改为这个尺度。现金可作为公开观察输入或结果诊断，不能进入辅助奖励或晋级否决。critic 用 sigmoid 预测比赛得分；GAE 使用 gamma=1、lambda=0.97，最后一步正确终止，不跨局或座位传播优势。

每场自博弈有 719 次决策/座位，共 1,438 env steps。双方都是本轮候选，BC teacher 用于 KL，不是把全部采集改成对教师的比赛。双方 inventory tracker 独立；只更新已实际观察到的库存变化，不补造未来 worker 或私有对手库存。

合法采样使用官方 Python resolver 或等价的自有 native resolver，对己方单位动作和市场指令按前缀执行顺序核验。候选支持取自真实资源、位置和 worker 状态；多个动作共用的物料/现金/土地按已选前缀扣减。绝对 SELL 标签与实际请求数量一致，NOOP 保留实际槽位置；不能裁剪请求后继续记录原概率。

默认动作合同 `official-prefix-economic-full-action-ppo-v2` 在合法集合上增加可兑现生产窗口：FERTILIZE、CARE、PLANT/PLACE、种子/动物/土地投入和过晚雇工。夜间根据真实 post-unit 库存与口袋货物预测溢出，保留次日喂养小麦，强制 SELL 先占领先订单槽，再采样其余槽；入仓采购不能重新占掉回仓所需空间。717/718 清仓，718 仓门持货单位强制 DROP。它是按本项目前缀重写的经济合同，不是照搬参考的后置改写；未采用 A 的全局置信度重新排序或自动补种子。

保存每个 slot 的 action ID、支持集、存在 mask、独立 policy mask 和 log probability。强制动作在采样前确定、记录 singleton support、policy mask=False、概率因子为零。联合概率、PPO ratio、teacher KL 和熵仅使用网络选择的因素；完全由控制器决定的行也不进入 actor 分母或优势归一化。它们的真实状态、终局结果仍参与 critic/GAE。更新前检查实际概率重算及 execution/continuation 身份；不能用原网络概率更新修复或搜索替换后的动作。

PPO 推理按 checkpoint 中的合同使用相同支持集并取 argmax。旧 `official-prefix-full-action-ppo-v1` 继续原定义；未标记的 BC 保持 `GreedyPolicy`。BC 原代码和正在训练的源码指纹不变。经济/搜索候选使用 PPO 的评估入口或自己的 main.py，不能交给 BC evaluate 后混称同一策略。

## critic 拟合与 PPO

critic 阶段保持 BC actor 与 Transformer 主干完全冻结，只更新 value head，核对冻结部分 hash。每轮训练采集新局，验证使用固定独立 seeds 及固定采样随机种子；训练拟合 GAE 的 bootstrapped return，独立验证按真实终局 Monte Carlo score 的预测误差选择 best critic；两者的标签定义明确区分。拟合前先验证并保留 iteration 0，全部更新变差时继续保留原 head。patience 只决定拟合停止，不宣称比赛策略提升。保存的 `policy_with_critic.pkl` 来自 best head，latest state 保留可续训优化器。

PPO 从该 critic 策略建立新优化器，保留原 BC 作为固定 KL teacher。默认 clip=0.2，单位与市场熵系数各 0.0015，teacher KL=0.2，Huber value loss 权重 2，Adam epsilon=1e−5，梯度范数裁剪 5，学习率 5e−5；学习率预热与衰减使用优化器步数。critic 默认学习率 1e−4。对应 [PPO 配置](../src/route_rl/ppo/configs/ppo.json) 与 [critic 配置](../src/route_rl/ppo/configs/critic.json) 是本项目支持的设置。

每轮重新采集当前参数的完整对局，计算优势/return 后按成对座位打乱 minibatch。全局标准化优势只使用本轮训练样本；末尾 batch 的 padding 不参与损失或统计。critic/actor 共享主干在 PPO 阶段会一起更新；冻结 warmup 避免初始 critic 拟合扰动 BC，KL 和梯度裁剪不能保证以后完全没有梯度干扰或漂移。

运行设置、模型、初始化文件、teacher、源码、目标及执行合同共同决定 resume 身份。再次执行同一命令自动恢复最近完整 rollout-update，`--updates` 是累计目标且只能增加。改 batch/dtype/games/config/源码/teacher/初始化需要新 run；不要将 PPO state 当 BC state，也不要将 inference policy 当旧优化器原样续训。中途未提交的一轮会从上一检查点重新采集。已有指标/策略/完成记录而丢失 latest optimizer state 时拒绝原目录从零重启；需恢复原 state 或使用新 run。评估和 critic 交接同时核对 state 的 integration 身份。

## 数据隔离与评估

可选 `season_controller` 在 day=29 清晨接管；继承 `PolicyHistory` 的对手公开库存估计，不新建零历史 tracker。控制器明确要求最后一天、native expected-score objective、神经 fallback；默认预算 0.25 秒，上限 6 秒，启动需至少 10 秒 overage。动作与实际单位、词表检查不通过、预算不足、native 异常或不支持的市场参数都会回当前神经策略。controller 配置、源码、binary hash 与 continuation version 进入采集/检查点/候选/评估/打包。critic 和 PPO 使用同样的 continuation，否则拒绝交接。

搜索内部使用公开市场与对手库存预测，将预计终局现金差映射为未校准的胜/平/负期望。固定尺度映射保持单调，不是新的强度证据；预测量不是 PPO 辅助奖励。搜索仍需独立得分确认。`candidate` 创建新控制器产物而不改权重，`evaluate --comparison controller` 核对同一冻结模型及其 BC/PPO run 归属，以训练外换座初筛和另一批种子确认。限定接管遵循历史全局接管失败教训，不自动扩大到整季。

critic 训练、critic 验证、PPO 训练使用互不重叠的保留 seed ranges；所有已知 BC demonstration seeds 都从训练采集排除。评估拒绝 BC、critic、PPO 已用 seeds 和未来训练保留范围；整个训练范围保留也让续训不会意外采入早先的测试 seeds。缺失示范 seed 的数量随 run 和评估记录，无法保证这些未知 seed 与测试独立。

默认 critic 训练范围 [20,000,000, 29,000,000)，critic 验证 [30,000,000, 39,000,000)，PPO [1,000,000, 9,000,000)。修改起点/上界时必须建立新 run，并保持范围互不重叠。准备 BC 数据的 episode-hash holdout 不参与 PPO 训练。

评估开始时冻结 candidate、baseline 与 opponent entry 的实际文件字节，所有配对局均读取同一快照；保存对应文件哈希。对手入口的相对资源仍以原目录为上下文，应保留原依赖文件。每个 seed 让 candidate 与 baseline 分别对同一 opponent 换座，共四场；`--games 16` 表示每策略 16 场，总计 32 场。初筛报告只有 paired match-score delta > 0 才允许进入独立确认；确认必须使用另一批 seeds，且三份策略/对手及 run 身份完全不变。baseline 仅接受该 run 的原 BC teacher 或 critic 初始化文件：原 BC 文件检查整个 PPO 阶段的行为变化，`policy_with_critic.pkl` 比较同一 masked execution 下的学习变化。候选也须属于同一 run，防止跨 run 权重绕过 seed 记录；报告保留确切文件身份。

确认仍为 `candidate_only`，只产生可供人工审查的资格，不替换 accepted deployments。简单的两次正向均值比较不是总体胜率或显著性证明。不要用 cash、loss、critic MSE、BC accuracy、自博弈平均得分或测试连接成功代替独立比赛结果；对称自博弈双方的平均 score 固定为 0.5，也不能作为改进指标。

## 产物与交付

`route_rl.action_teacher generate` 使用自有规划器在完整官方环境双方生成示范，保存实际请求、720 状态 gzip 回放、双座位 index 和 immutable receipt。场景来自明确 seeds 或失败回放的已验证 seed；后者不能是用于选模或独立确认的 seed。`--exclude-run`/`--exclude-seeds` 保护既有数据与保留范围。先按 seed 留出，再选择兼容现有 BC hash split 的 episode IDs；不修改旧 BC 分割代码。

`action_teacher combine` 合并公开与生成 index，验证 teacher 完成 receipt、index/replay checksums、实际 seed 和整局 split，保留公开留出并拒绝跨 split 同 seed。输出保留实际绝对回放路径，再走本项目 `action_bc prepare → 新 BC adaptation run → critic → PPO`。生成与合并不自动启动训练，现有 cache/run/checkpoint 不覆盖。教师质量需要审阅真实回放，不能把参考曾出现的番茄或商店弱点直接当作本项目当前失败。

| 阶段 | 产物 |
| --- | --- |
| critic | `latest_critic_state.pkl`、`policy_with_critic.pkl`、`metrics.jsonl`、`receipt.json` |
| PPO | `latest_ppo_state.pkl`、`policy_latest_jax.pkl`、`policy-update-N.pkl`、`metrics.jsonl`、`receipt.json` |
| 两阶段共用 | `integration.json`、`run_config.json`，包含来源、teacher、目标、配置及数据隔离记录 |
| 配对评估 | screen/confirmation JSON 与对应 `.games.json`，包含逐局 score 与 frozen policy identity |
| 打包 | 不含训练数据/权重的源码包；有真实 checkpoint 后生成独立候选策略包 |

源码包是本次可审查的交付品；未来权重和对战结果不是本次编造的产物。策略打包入口检查 checkpoint checksum 和执行合同，包含自身推理模块、模型及入口，排除训练 state、data、run、参考 clone 和账号凭据。具体 runtime 依赖、验证边界和打包命令见 README；打包不等于自动比赛提交。

小模型的概率、合法性、value 冻结、resume、seed、整局及隔离包检查只证明实现一致性。这次没有启动正式 PPO 实验，也没有宣称学习后经营或对战得分提高。
