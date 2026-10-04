# 训练方法指南：从公开回放到完整 agent

本文介绍本仓库完整动作训练流程的使用顺序、模型结构、学习目标、指标和交付方式，适用于首次训练、续训及基于新示范开展独立实验的使用者。具体安装与运行命令见 [README](../README.md)；缓存和 BC 参数详见 [BC 指南](action_bc_pipeline.md)，在线学习的执行、恢复和评估约定详见 [PPO 指南](action_ppo_pipeline.md)。

主流程为 **公开回放 → 行为克隆（BC）→ 固定 BC teacher → critic 拟合 → 自博弈 PPO → 配对初筛 → 独立确认 → agent 打包**。季末搜索和搜索教师示范是可选步骤。本文的默认值来自仓库配置；示例路径、epoch 和更新轮数可按实验需要调整。

- [流程与运行准备](#1-流程与运行准备)
- [比赛目标](#2-比赛目标)
- [教师回放与数据划分](#3-教师回放与数据划分)
- [观察编码与模型结构](#4-观察编码与模型结构)
- [行为克隆](#5-行为克隆)
- [critic 拟合](#6-critic-拟合)
- [采集与动作执行](#7-采集与动作执行)
- [PPO 训练与指标](#8-ppo-训练与指标)
- [可选季末搜索](#9-可选季末搜索)
- [搜索教师与新一轮学习](#10-搜索教师与新一轮学习)
- [数据隔离与配对评估](#11-数据隔离与配对评估)
- [训练产物恢复与交付](#12-训练产物恢复与交付)
- [实验设计与回放诊断](#13-实验设计与回放诊断)
- [代码入口与来源](#14-代码入口与来源)

## 1. 流程与运行准备

![从公开回放到候选策略的完整训练流程](images/training_pipeline.svg)

蓝色表示数据，绿色表示模仿学习，橙色表示在线学习，紫色表示评估和交付。搜索教师循环使用单独的诊断种子，评估面板保持留出。

| 阶段 | 输入 | 主要工作 | 输出 |
| --- | --- | --- | --- |
| 下载 | Kaggle API、教师队伍及提交快照 | 识别实际教师座位，下载兼容的完整对局 | 原始回放、`teachers.json`、`teacher-seats.jsonl` |
| 准备 | 回放与座位索引 | 校验动作、整局划分、编码特征和标签 | 轨迹 `.npz`、缓存清单、准备记录 |
| BC | 缓存与初始化模型 | 监督学习 actor 和共享主干 | epoch 策略、final 策略、续训状态 |
| 固定 teacher | 所选 BC checkpoint | 复制到稳定路径，绑定文件 checksum | 固定 BC `.pkl` |
| critic | 固定 BC actor、完整自博弈 | 冻结 actor 和主干，拟合 value | `policy_with_critic.pkl` |
| PPO | best critic 策略、固定 teacher | 重新采集完整对局，更新整个网络 | `policy-update-N.pkl`、指标、续训状态 |
| 初筛与确认 | 冻结 candidate、baseline、opponent | 同种子换座比较，再用独立种子确认 | 汇总报告、逐局记录 |
| 打包 | 所选完整策略 | 导出推理入口、权重和依赖清单 | 可运行的 agent 归档 |

按 [README 的环境准备](../README.md#环境准备)安装依赖并检查设备。训练示例使用 Linux、Python 3.12 和单张 NVIDIA GPU；入口支持单进程、一个可见 JAX 设备，CPU 也可用于较小模型和流程检查。计划使用 Rust 批采集或季末搜索时，先按[构建运行组件](../README.md#1-构建运行组件)生成对应扩展；搜索构建记录会参与 PPO 源码身份，应在相关训练前准备。

本指南沿用 README 的示例目录：

| 示例路径 | 用途 |
| --- | --- |
| `data/action_bc/` | 教师快照、公开回放、索引及准备缓存 |
| `models/bc_initial.pkl` | BC 初始化模型 |
| `runs/bc/` | BC 状态、策略及指标 |
| `models/bc_teacher.pkl` | 选定后固定的 BC teacher |
| `runs/critic/`、`runs/ppo/` | critic 和 PPO 的独立运行目录 |
| `runs/ppo_eval/` | 初筛、确认及逐局记录 |
| `dist/` | agent 或源码归档 |

替换路径时同步修改下游输入。数据、权重、run 和编译产物需要自行生成并备份。

## 2. 比赛目标

每局记录 720 个状态，每个座位作出 719 次决策。强化学习只在真实终局取得比赛得分：

| 结果 | 回报 |
| --- | ---: |
| 胜 | 1 |
| 平 | 0.5 |
| 负 | 0 |

中间奖励为 0。现金、库存、土地和工人状态是观察或诊断信息，现金差不作为额外奖励或候选晋级条件。critic 预测预计终局得分。

双座位自博弈的双方终局得分之和为 1，因此双方平均得分恒为 0.5。判断实力变化需要训练外的固定对手、交换座位和独立评估；BC 拟合指标、critic 误差及 PPO loss 分别用于判断各阶段的学习情况。

## 3. 教师回放与数据划分

### 3.1 选择并保存教师快照

下载器根据官方 episode agent 元数据中的 submission ID 和座位 index 识别教师。同一局可能提供两条教师轨迹，但独立对局数仍为一局。

README 使用以下示例：

```bash
bash tools/download_public_bc.sh data/action_bc 20 100
```

首次调用从排行榜第一页选择有可用公开得分的 20 支队伍，每队选择公开分数最高的提交，保存教师快照。100 是每位教师单次调用最多选取的对局数。重复调用复用快照和已有回放，索引保留历史记录；公开对局增加后，累计数量可能超过单次限制。

下载和准备要求兼容版本、完整观察与动作、正常 `DONE` 终局，排除 seed=0 和异常回放，记录 checksum 及跳过原因。正常结束的胜、平、负均可作为示范。为复现实验，保留教师快照、回放、索引和下载记录；公开榜单和可访问回放会随时间变化。

### 3.2 对齐观察与实际动作

```text
observation[t] → action[t+1]
每个教师座位：719 行
每行：观察特征 + 20 个单位标签 + 10 个市场标签
```

[标签生成器](../src/route_rl/full_action/labels.py)依据官方规则核验请求是否执行。无效、未执行或 padding 动作使用忽略标签 `-100`。SELL 标签记录实际成交的绝对数量。

BC 通过上述机制过滤监督标签；PPO 则在采样前限定候选集合，并保存实际发出请求的概率。两者需要各自遵守对应的训练和执行定义。

### 3.3 按整局划分训练与留出

公开 BC 数据使用 `sha256("50:" + episode_id)` 的前 8 字节转为整数，按 `% 10 == 0` 划入留出集，比例约为 10%。同局双方保持同一 split；已知相同 seed 的不同 episode 若跨 split，准备入口会拒绝。

留出集用于监督验证和选模，整局双方及时间片保持在同一组。缺失 seed 的回放会单独计数，其与未来评估种子的独立性存在无法核验的部分。

运行 `prepare` 后执行 `audit-cache`，核对 split、轨迹及编码身份。数据或编码改变时使用新的准备目录。步骤见 [README 的缓存准备](../README.md#2-准备并检查缓存)。

## 4. 观察编码与模型结构

### 4.1 实体特征

每个样本使用固定的 `264 × 124` 特征矩阵。264 个 token 表示实体槽，124 维是统一特征容器；各实体 adapter 选择自身需要的字段。

| token 类型 | 槽数量 | 内容 | adapter 输入维数 |
| --- | ---: | --- | ---: |
| GLOBAL | 1 | 时间、双方公开现金和农场摘要 | 34 |
| CELL | 200 | 双方各 100 个格子的生产状态 | 34 |
| UNIT | 最多 40 | 双方单位、位置、状态及携带物 | 22 |
| PRODUCT | 12 | 商品、库存、市场及产出信息 | 19 |
| MEMORY | 1 | 对手公开库存估计及不确定性 | 50 |
| MARKET_SLOT | 10 | 十个订单槽的标识 | 10 |

空槽通过 token mask 忽略。编码容量为最多 20 个己方单位、40 个总单位；BC 超限轨迹拒绝编码，PPO 采集超限会停止。使用该容量检查数据和运行配置。

[库存 tracker](../src/route_rl/full_action/inventory_tracker.py)根据真实公开观察和历史动作更新库存估计。MEMORY 表示 tracker 状态；网络每次对当前实体矩阵做前向计算，输入不包含对手私有库存。

缓存特征以 float16 存储、标签以 int16 存储，读取时构造 float32 输入。网络计算 dtype 由训练参数选择；存储、输入和主要计算的 dtype 分别记录。

### 4.2 共享 Transformer

![实体 Transformer 的模型结构](images/model_architecture.svg)

图中展示 `bootstrap` 预设。单位动作、市场动作和 critic 共用实体编码与 Transformer 主干；动作支持集在 logits 输出后应用。

| 配置 | `bootstrap` | `10m` |
| --- | ---: | ---: |
| Transformer block | 6 | 12 |
| 宽度 | 256 | 256 |
| attention head / 每头维数 | 8 / 32 | 8 / 32 |
| FFN | 256 → 1,024 → 256 | 相同 |
| 激活 / dropout | GELU / 0 | 相同 |
| RoPE 维数 / base | 16 / 100 | 相同 |
| 参数量，含 value 分支 | 5,486,510 | 10,225,070 |

六类 typed adapter 线性投影后使用输入 LayerNorm。每个 block 使用 pre-LayerNorm 和残差：

```text
hidden → LayerNorm → self-attention → 加回 hidden
       → LayerNorm → Linear → GELU → Linear → 加回残差
最后 → final LayerNorm
```

同一农场的 CELL/UNIT token 对使用二维 RoPE，前 16 个 head 维度中 x/y 各占 8 维；跨农场和非空间 token 对使用未旋转的内积。`partitioned` 实现将 token 重排为己方 120、对方 120、全局 24 个，attention 仍覆盖全部 token。实现见 [model.py](../src/route_rl/full_action/model.py)。

模型通过初始化预设选择。改变架构需要新的初始化和运行目录，架构不同的 checkpoint 不能直接续接。

### 4.3 动作头与 value 分支

- 单位头读取己方单位 embedding，输出 `20 × 500` 个 logits。
- 市场头读取十个订单槽。原始每槽 1,075 类，与 `9 商品 × 100 绝对数量` 的 SELL 头组合后，得到 1,903 类：1,003 个非 SELL 选择及 900 个 SELL 选择。
- value 分支读取 GLOBAL embedding，经 `256 → 256 → 1` 的 GELU MLP 输出 raw value，再取 sigmoid，得到 `[0,1]` 的预计终局得分。

value 分支另有公开现金差/时间的 `linear_cost` 输入项。输出层和该项零初始化，初始预计得分为 0.5；BC 保持 value 参数冻结，critic 阶段开始拟合它。

一次网络前向生成各槽 logits，PPO 在这些 logits 上按动作前缀逐槽更新支持集。

## 5. 行为克隆

### 5.1 学习目标与训练参数

BC 以教师实际动作作监督，通过交叉熵拟合动作分布；熵项保留多样性，固定初始化策略的 KL 限制漂移。默认目标为：

$$
L_{BC}=CE_{unit}+CE_{market}
-0.10H_{unit}-0.10H_{market}
+0.05\left[KL(\pi_{initial}\Vert\pi_\theta)_{unit}
+KL(\pi_{initial}\Vert\pi_\theta)_{market}\right].
$$

CE 和准确率按有效标签计算，熵和 KL 按存在的动作槽计算。初始化 teacher 始终冻结：从随机模型开始时参考随机策略，从已有 BC 模型 adaptation 时参考该已有策略。

BC 更新 actor 和共享主干，value 不参与 loss。BC 使用完整动作监督和原 BC greedy 推理定义；经济支持集属于后续 critic/PPO 执行流程。

| 参数 | 配置默认 | README 示例 |
| --- | --- | --- |
| 累计 epoch | 2 | 30 |
| batch | 320 | 320 |
| 主要计算 dtype | BF16 | BF16 |
| 学习率 / Adam epsilon | `1e-4` / `1e-5` | 相同 |
| gradient norm clip | 5 | 相同 |
| shuffle seed | 51 | 51 |
| shuffle window | 16 条轨迹 | 16 条轨迹 |

epoch 和 batch 按数据规模与资源选择；示例训练量不表示最优值。batch 必须为不小于 2 的偶数。配置见 [bc.json](../src/route_rl/full_action/configs/bc.json)。

### 5.2 窗口读取与阶段输出

BC 流式读取 `.npz` 轨迹。默认最多驻留两个 16 条轨迹窗口，另外需要 batch、JAX 编译、模型及优化器空间。每轮打乱轨迹顺序和窗口内行顺序，跨窗口衔接尾批；每条样本访问一次，仅尾批 padding，padding 不计损失。验证顺序读取且不更新参数。

`audit-cache` 的 `decoded_bytes` 是全部缓存展开大小；实际驻留内存取决于轨迹窗口和额外运行开销。训练入口同时检查 cgroup 可用内存。窗口参数和读取方式参与恢复身份。

每个完整 epoch 后保存策略和续训状态，达到累计目标后保存 `final_student_jax.pkl` 及完成 receipt。已保存的 epoch 策略可用于完整对局评估。

### 5.3 读取 BC 指标

`runs/bc/metrics.jsonl` 按 epoch 记录 `train` 和 `validation`：

| 指标 | 读法 |
| --- | --- |
| `unit_ce`、`market_ce`，`unit_accuracy`、`market_accuracy` | 留出 CE 降低、准确率提高表示动作拟合改善；同时比较训练与留出差距 |
| `unit_entropy`、`market_entropy`，`unit_normalized_entropy`、`market_normalized_entropy` | 观察动作分布是否过早集中或一直过于分散 |
| `unit_teacher_kl`、`market_teacher_kl` | 相对初始化参考策略的漂移，需与熵和 CE 一起判断 |
| `loss` | 含 CE、熵和 KL 的复合目标，可能为负；绝对值不是胜率 |

训练 CE 持续下降而留出 CE 上升时，检查数据分布及过拟合。模型选择结合留出指标和训练外对局。指标实现见 [bc_objective.py](../src/route_rl/full_action/bc_objective.py)。

## 6. critic 拟合

### 6.1 固定 actor 与 teacher

从 BC 选择一份已保存的 epoch/final 策略，复制到固定路径，例如 `models/bc_teacher.pkl`。warmup 核对它与 BC run 的文件 checksum；后续 PPO 继续使用同一文件作为 KL teacher。BC 继续训练时，保留该固定文件。

critic 阶段冻结 actor 和 Transformer 主干，只更新 value 分支。其动作执行默认采用经济 v2 支持集，执行身份与固定权重一并记录。

### 6.2 训练标签与独立验证

每轮默认采集 4 场训练局，另以固定且独立的 2 场验证局检查 value：

| 项目 | 约定 |
| --- | --- |
| 训练标签 | 轨迹 GAE return，含中间 value bootstrap |
| 验证标签 | 实际终局 score，广播到各状态 |
| best 选择 | 独立验证的 `value_mse` |
| loss | Huber，delta=1，value 权重=2 |
| 学习率 / minibatch / 每轮 epoch | `1e-4` / 32 / 1 |
| patience | 连续 8 轮未改善 |

拟合前先保存 iteration 0。`policy_with_critic.pkl` 始终保存验证最优 value 与冻结 actor，`latest_critic_state.pkl` 保存最近训练状态。

从 `metrics.jsonl` 的 `critic_baseline` 和 `critic_update` 读取 `validation.value_mse`、`validation.value_mean`，从 `critic_update` 读取 `best_update`。验证 MSE 降低说明固定策略的价值预测改善；平均 value 还需结合该验证面板的真实结果解释。

达到更新目标或触发 patience 时阶段结束。已经触发 patience 的 run 不会仅因增加 `--updates` 而继续拟合。配置见 [critic.json](../src/route_rl/ppo/configs/critic.json)，运行步骤见 [README](../README.md#2-固定-bc-teacher拟合-critic)。

## 7. 采集与动作执行

### 7.1 每轮重新采集完整自博弈

PPO 每轮使用本轮策略进行完整双座位自博弈，双方分别维护公开观察历史。固定 BC teacher 用于 KL 约束。

默认 `rust-batch` backend 负责环境推进、特征、公开库存 tracker 及条件支持集，同一时刻将全部座位送入 JAX：

```text
1 局 = 719 决策 × 2 座位 = 1,438 seat env steps
默认 4 局 rollout = 5,752 行
推理批量 = 2 × games = 8
```

Rust 复用特征 buffer，并用 Rayon 并行部分环境与编码工作；网络推理、采样编排和部分数据整理由 JAX/Python 执行。`official-python` backend 可用于核对状态、特征、支持集、请求和概率语义。切换 backend 需显式配置，原生扩展缺失或身份失配时会停止。

`rollout_diagnostics.collection_seconds` 与 `collection_env_steps_per_second` 统计完整采集过程，包括准备、JIT、特征、支持集、推理、存储及可选搜索。比较吞吐时固定模型、backend、games、设备和计时范围。

### 7.2 条件支持集与经济规则

默认执行合同为 `official-prefix-economic-full-action-ppo-v2`。依据真实己方动作前缀检查合法条件，再按有效生产窗口、库存和时间约束投入：

- 按成长、产出和赛季剩余时间限制 FERTILIZE、CARE、PLANT、PLACE。
- 限制过晚的种子、动物、土地投入及会在当夜失效的雇工。
- 按 post-unit 仓库及携带物预测夜间回仓溢出，保留喂养需要的小麦。
- 必要 SELL 先占订单槽，再采样其余订单；采购保留回仓所需空间。
- 在 717/718 清仓，并在 718 对仓门持货单位安排必要 DROP。

市场支持集以公开市场、对手该步 NOOP 的条件核验己方前缀。实际同时下单仍会受对手影响；PPO 保存发出请求的概率，真实下一状态和终局反映成交后果。实现见 [sampling.py](../src/route_rl/ppo/sampling.py) 和 [economic_rules.py](../src/route_rl/ppo/economic_rules.py)。

### 7.3 保存实际概率与 mask

![采样、执行与 PPO 更新的对应关系](images/ppo_execution.svg)

每个动作槽保存 action ID、条件支持集、存在 mask、policy mask 和 log probability：

| mask | 含义 |
| --- | --- |
| 存在 mask | 槽存在且有实际动作记录，含强制因素 |
| policy mask | 该因素由网络选择，可计入 actor 概率、KL 和熵 |

强制动作在采样前确定，记录 singleton support、`policy_mask=False`，该因素 log probability 为 0。神经选择因素的联合概率为：

$$
\log\pi(a\mid s)=\sum_{j:\;policy\_mask_j=1}
\log\pi_j(a_j\mid s,a_{<j},\mathcal S_j).
$$

更新使用同一份已保存支持集重算概率，teacher KL 和熵也在这些支持集上计算。完全由规则或搜索决定的行不参与 actor loss 与优势标准化，真实转移仍参与 value/GAE。

## 8. PPO 训练与指标

### 8.1 从终局得分计算 GAE

critic 提供各时刻的预计得分 `V(s)`，最后一个实际转移取得终局 score：

$$
\delta_t=r_t+\gamma(1-d_t)V(s_{t+1})-V(s_t),
\qquad
A_t=\delta_t+\gamma\lambda(1-d_t)A_{t+1}.
$$

默认 `gamma=1`、`lambda=0.97`，学习 target 为 `return_t=A_t+V(s_t)`。终局下一状态 value 为 0，中间时刻使用 bootstrap；因此训练 return 与 critic 独立验证中广播的终局标签不同。

GAE 分别在每局、每座位内计算。有效神经行的优势在完整 rollout 上标准化，然后打乱成 minibatch，保留同局同一步的相邻座位配对；padding 不计损失。

### 8.2 Clipped PPO 与固定 teacher

概率比和策略项为：

$$
\rho_t=\exp\left[\log\pi_\theta(a_t\mid s_t)-\log\pi_{old}(a_t\mid s_t)\right],
$$

$$
L_{policy}=-\mathbb E\left[\min\left(\rho_tA_t,
\operatorname{clip}(\rho_t,0.8,1.2)A_t\right)\right].
$$

默认总目标为：

$$
L_{PPO}=L_{policy}+2L_{value}
+0.2KL(\pi_{BC}\Vert\pi_\theta)
-0.0015H_{unit}-0.0015H_{market}.
$$

value 使用 Huber loss。actor 平均覆盖有效神经行，critic 覆盖有效实际行；PPO 同时更新 actor、共享主干和 value，BC teacher 始终冻结。

### 8.3 参数、精度与更新计数

| 参数 | 配置默认 |
| --- | --- |
| games / 推理座位批量 | 4 / 8 |
| minibatch / 每轮 epoch | 32 / 1 |
| 学习率 / Adam epsilon | `5e-5` / `1e-5` |
| gradient norm clip | 5 |
| PPO clip / teacher KL | 0.2 / 0.2 |
| 单位 / 市场 entropy 系数 | 各 0.0015 |
| value Huber delta / 权重 | 1 / 2 |
| gamma / GAE lambda | 1 / 0.97 |
| LR warmup / decay / 最小比例 | 240 / 150,000 个优化器步 / 0.25 |
| 主要计算 dtype | BF16 |

[README 的 PPO 示例](../README.md#3-从-best-critic-训练-ppo)显式选择 `--compute-dtype float32` 和 `JAX_DEFAULT_MATMUL_PRECISION=highest`，以减少采样与更新图的数值差异。每轮更新前，入口重算整份 rollout 的行为概率，要求最大绝对 log probability 误差不超过 `0.0002`；超限时在梯度更新前停止。

checkpoint 保存 compute dtype，矩阵精度环境变量不进入 checkpoint 或 resume 身份。采用上述精度时，续训、评估和最终运行进程均在导入 JAX 前设置 `highest`；打包不会自动把该设置固化到 `main.py`。

默认 5,752 行、minibatch=32、epoch=1 对应 180 个 Adam 步，尾批为 24 行有效数据和 8 行 padding。CLI `--updates` 表示累计完整“采集→更新”轮数；`training.updates` 表示该轮优化器步数，学习率计划按优化器步计数。配置见 [ppo.json](../src/route_rl/ppo/configs/ppo.json)。

### 8.4 读取 PPO 指标

`metrics.jsonl` 的 `ppo_update` 同时记录运行进度、采集诊断和 `training`。先检查数值与更新稳定性，再结合训练外评估判断实力。

| 指标 | 含义与读法 |
| --- | --- |
| `initial_behavior_log_prob_max_error` | 更新前整个 rollout 的最大概率重算误差；检查采样与学习路径是否一致 |
| `clip_fraction` | 比率超出 `[0.8,1.2]` 的有效神经行比例；持续偏高表示大量比率已超过裁剪区间 |
| `approx_old_kl` | 旧/新策略联合动作 log probability 差的采样均值；观察每轮更新漂移，可因采样而为负 |
| `ratio_mean` | 新旧联合概率比的平均；接近 1 也可能伴随两侧大幅波动，需结合 clip fraction 和 KL |
| `teacher_kl` | 相对固定 BC teacher 的联合动作分布漂移，在相同支持集上计算 |
| `unit_entropy`、`market_entropy` | 有效神经动作因素的熵之和；受可选动作和有效槽数量影响 |
| `value_loss`、`value_mean` | 对 GAE return 的拟合及平均预计得分；与 critic 验证 MSE 的标签不同 |
| `policy_loss`、`loss` | 策略项与复合训练目标；绝对数值不直接代表比赛实力 |
| `gradient_norm` | 梯度裁剪前的全局范数；可以高于配置的裁剪上限 5 |
| `actor_sample_count` | 按有效样本数加权的平均每 minibatch 神经行数，可能为小数 |
| `max_log_prob_error` | 每个 minibatch 最大新旧 log probability 差的加权平均；训练期间策略已改变 |

除更新前的概率误差等单独统计项外，`training` 指标按各 minibatch 的有效 `sample_count` 加权汇总，包含轮内多个参数版本。`max_log_prob_error` 不等于整轮最大值，也不使用更新前概率校验的阈值。

联合概率、KL 和熵汇总多个单位及市场因素，不能直接套用单动作环境的通用阈值。比较同一合同、模型和采集设置下的多轮趋势。`samples` 包含每轮 epoch 的样本重复访问次数；`env_steps` 与 `rollout_env_steps` 使用座位决策步口径。

## 9. 可选季末搜索

季末控制器在第 30 天清晨（`day=29`）尝试接管，继承实际公开观察历史和库存估计。默认搜索预算 0.25 秒，首次接管至少需要 10 秒剩余 overage，reserve 为 3 秒；支持配置最多 6 秒搜索预算。

搜索库和构建记录必须在初始化时存在且匹配。成功初始化后，预算不足、错过接管清晨、不支持的输入或搜索执行异常会回到网络策略。

搜索根据公开现金流及终局现金差预测，通过离散 logistic 不确定性映射计算预计胜/平/负得分。该预测未经校准，正式质量判断仍使用真实终局比赛。

可以向已有 checkpoint 追加控制器，形成新的组合候选。此操作保持权重、改变最后一天执行，需对组合策略重新评估。比较时固定权重，以 `evaluate --comparison controller` 对带搜索和不带搜索的候选初筛并独立确认。

若将搜索纳入训练，两阶段 critic/PPO 使用相同 `--season-search` 或 controller 配置，并建立对应新 run。搜索动作排除 actor loss，实际结果仍通过 value/GAE 传回前面的网络决策。命令见 [README 的可选搜索](../README.md#可选季末搜索与搜索教师)，组合打包见[最终交付](../README.md#最终交付整理与策略打包)。

## 10. 搜索教师与新一轮学习

发现稳定的行为缺陷后，可使用规划器生成额外实际示范，再开展新的学习阶段：

1. 用独立诊断种子确认场景，通过 `--exclude-run`、`--exclude-seeds` 保护训练留出、选模及评估面板。
2. `action_teacher generate` 从明确 seed 或源回放核实的 seed 重开完整局，在默认官方规则下执行双方规划器。
3. 保存实际请求、720 个状态和正常终局，按整局 seed 留出；episode ID 与 BC hash split 兼容。
4. `combine` 核对来源、回放/index checksum、实际 seed 和 split，合并公开与教师示范，保留公开留出集。
5. 在新的准备目录编码，以冻结 BC 策略初始化新的 BC adaptation。
6. 固定新 BC teacher，进入新的 critic/PPO run，再进行训练外初筛和独立确认。

源回放用于提取 seed，生成器从初始状态重开默认规则的整局，原动作及中间状态不作为标签或重启点；源回放的自定义环境配置也不会自动复用。检查新回放中教师实际执行的行为和标签质量。

初筛与确认种子保持留出，发现问题后另选诊断和教师种子。新的缓存及 `runs/bc_adapt`、`runs/critic_adapt`、`runs/ppo_adapt` 与原运行目录独立，原 checkpoint 保留。步骤见 [README 的搜索教师](../README.md#2-生成搜索教师示范)和[混合示范](../README.md#3-混合示范并重新训练)。

## 11. 数据隔离与配对评估

### 11.1 各类数据的用途

| 数据/种子 | 用途 | 是否进入对应训练 |
| --- | --- | --- |
| BC training | 监督动作学习 | 是 |
| BC holdout | 监督验证与选模 | 否 |
| critic `[20,000,000,29,000,000)` | critic 训练采集 | 是 |
| critic validation `[30,000,000,39,000,000)` | best critic 选择 | 否 |
| PPO `[1,000,000,9,000,000)` | 策略在线采集 | 是 |
| screen 示例 `1,600,000,000` 起 | 候选初筛 | 否 |
| confirmation 示例 `1,700,000,000` 起 | 独立确认 | 否 |

上述在线训练范围来自默认配置。入口同时核对已知 BC seed、阶段种子和未来训练保留范围；修改种子配置时重新核查各面板。独立确认还要求与所引用初筛报告的种子不重叠。

### 11.2 同种子换座比较

每个 seed 下，candidate 和 baseline 分别对同一 opponent 换座，共四场，以控制地图和座位差异。`--games 16` 表示每策略 16 场，两策略合计 32 场。

初筛 paired match-score delta 为正后，再用未参与选模的种子确认同一组 candidate、baseline 和 opponent 文件。报告绑定文件 checksum、执行与 continuation 身份，确认后保留 `candidate_only` 结果。

学习比较可选择原 BC teacher 或初始 `policy_with_critic.pkl` 作为 baseline。BC baseline 比较整个阶段的行为变化，critic baseline 更接近相同 masked execution 下的学习变化。控制器比较使用同一冻结权重的不同包装；报告明确比较对象。

报告不能覆盖。读取汇总得分、paired delta 和 `.games.json`，结合逐局胜/平/负及回放解释结果；正向均值本身不等于统计显著性。评估步骤见 [README](../README.md#ppo-配对初筛与独立确认)。

## 12. 训练产物、恢复与交付

### 12.1 保存与恢复边界

| 文件 | 内容与用途 |
| --- | --- |
| `latest_bc_state.pkl` | BC 参数、优化器和已完成 epoch；用于原 run 续训 |
| `epoch-N-policy.pkl`、`final_student_jax.pkl` | BC 推理策略；用于评估或下一学习阶段 |
| `latest_critic_state.pkl` | 最近 critic 参数和优化器 |
| `policy_with_critic.pkl` | 验证 best value 与冻结 actor；PPO 初始化 |
| `latest_ppo_state.pkl` | PPO 参数、优化器、采样配置及更新进度 |
| `policy_latest_jax.pkl`、`policy-update-N.pkl` | 最近或指定完整更新边界的推理策略 |
| `integration.json`、配置、receipt | 数据、源码、teacher、执行及 continuation 身份 |
| screen/confirmation 及 `.games.json` | 冻结文件身份与实际逐局结果 |

相同目录与参数重跑，BC 仅增加累计 `--epochs`，critic/PPO 仅增加累计 `--updates`。恢复从最近完整 epoch/更新边界开始，未完成边界重新执行；critic 已耗尽 patience 时保持结束状态。

改变数据、初始化、teacher、batch/minibatch、games、dtype、窗口、配置、源码、执行或 continuation 时建立新的对应 run。新 PPO 的 initial 必须是指定 critic run 绑定的 best 策略，teacher 必须是其固定 BC 文件。以已有策略开展新的 BC adaptation 会建立新优化器。

### 12.2 打包完整策略

先选择实际要交付的 checkpoint；需要搜索时生成组合候选，并对该组合策略独立评估。checkpoint 已含搜索配置时，可直接用于打包。

`package_pipeline.py submission --policy` 明确选择模型，`--out` 仅指定归档路径。策略包包含 `main.py`、`policy.pkl`、运行代码、依赖及 manifest；搜索候选额外包含匹配的 `.so` 和构建记录。PPO 加季末搜索组成同一 agent：前 29 天由网络执行，第 30 天尝试搜索，运行期搜索失败时回网络。

生成的入口默认使用 CPU。按包内依赖准备目标环境，保留完整解压目录，并按训练时约定设置矩阵精度。目标环境核验依赖、原生库平台兼容性、完整对局和计时。

`package_pipeline.py source` 按脚本位置和固定规则收集仓库源码、配置、文档、测试及构建工具，排除训练数据、run、权重和编译二进制。解压后安装依赖并重新构建所需组件。

`verify` 校验归档路径、文件清单和 checksum。打包与评估均不自动替换已接受策略或提交比赛。命令见 [README 的最终交付](../README.md#最终交付整理与策略打包)。

## 13. 实验设计与回放诊断

确定要修改的具体行为，记录假设和当前证据，再选择对应的数据、执行或学习改动。一次实验保持比较对象、种子用途和文件身份清楚，新的 continuation 用其版本标识比较。

| 回放现象 | 检查方向 |
| --- | --- |
| 收获后长时间空地 | 生产窗口、资源、工人及实际续种请求 |
| 有产出却不能兑现 | 携带、回仓、仓库容量、销售槽及成交时点 |
| 多单位争抢资源 | 条件支持集、己方前缀资源扣减及共享路线 |
| 续种不随市场与资源改变 | 示范覆盖、观察字段及条件转产决策 |
| 季末搜索或新策略退化 | 同权重组合比较、接管时点、预算及 fallback |

完整执行检查和小模型 smoke 用于验证标签、概率、冻结边界及运行连接。策略效果通过冻结模型的训练外比赛判断。历史诊断需对照当前代码；回放体现的行为不能直接推断对手训练算法。

保留数据快照、源码身份、运行配置、checkpoint 和评估报告。详细案例见 [实验复盘](experiment_retrospective.md)及[历史证据](experiment_history.md)。

## 14. 代码入口与来源

| 内容 | 入口 |
| --- | --- |
| 安装、连续命令和打包 | [README](../README.md) |
| 下载、标签与缓存 | [BC 指南](action_bc_pipeline.md)、[replay_download.py](../src/route_rl/replay_download.py)、[labels.py](../src/route_rl/full_action/labels.py)、[dataset.py](../src/route_rl/full_action/dataset.py) |
| 实体编码与共享网络 | [features.py](../src/route_rl/full_action/features.py)、[model.py](../src/route_rl/full_action/model.py) |
| 条件动作与经济支持 | [sampling.py](../src/route_rl/ppo/sampling.py)、[economic_rules.py](../src/route_rl/ppo/economic_rules.py) |
| Rust 批环境 | [native README](../native/action_engine/README.md)、[native_backend.py](../src/route_rl/ppo/native_backend.py) |
| GAE、PPO、value 与冻结边界 | [objective.py](../src/route_rl/ppo/objective.py)、[trainer.py](../src/route_rl/ppo/trainer.py)、[pipeline.py](../src/route_rl/ppo/pipeline.py) |
| 季末控制器与教师 | [controllers.py](../src/route_rl/ppo/controllers.py)、[action_teacher.py](../src/route_rl/action_teacher.py) |
| 执行、恢复和评估细节 | [PPO 指南](action_ppo_pipeline.md) |

BC/PPO 和搜索部件的借鉴来源、固定 revision 及适配范围见 [BC 来源](action_bc_sources.md)、[BC 文件清单](action_bc_sources.json)、[PPO 来源](action_ppo_sources.md)和[PPO 文件清单](action_ppo_sources.json)。运行使用本仓库维护的代码，参考方案的数据、历史权重和训练规模由其来源记录说明。

三张插图为独立 SVG，GitHub 和支持 Markdown 图片的 IDE 可直接显示。修改插图时运行 `python tools/render_training_diagrams.py`；重新生成需要可选 Matplotlib，训练运行不依赖它。历史实现检查见 [pipeline_validation.json](pipeline_validation.json)。
