# Kaggriculture

本项目维护完整的 **BC → Rust 批采集 / critic 拟合 → 带经济规则的自博弈 PPO → 配对初筛/独立确认 → 候选策略打包** pipeline，并提供可选季末搜索与搜索教师示范生成。公开回放监督和在线学习共用本项目的特征、动作词表与 JAX Transformer，训练、评估和打包均无需 `kaggriculture-solution/`。

**当前交付为源码 pipeline 和打包入口，真实 BC/PPO 权重与比赛结果需等待训练。** BC 正在训练时，可先审查这套代码，完成后按下面命令衔接；候选不会自动接管 accepted 部署。仓库也保留早期 Rust mixed/plan/event 实验，它们使用独立数据和检查点格式。实现检查不代表已复现参考方案的比赛成绩。

## 当前支持

| 功能 | 状态 |
| --- | --- |
| 公开回放下载、教师座位识别、去重与版本检查 | 已实现 |
| 整局划分训练/留出集、动作有效性检查、特征缓存 | 已实现 |
| BC 训练、逐 epoch 留出验证、保存策略和恢复优化器 | 已实现，单进程、单设备 |
| 指定 epoch / 最终策略的双座位整局评估 | 已实现，手动触发 |
| 冻结 BC actor/主干的 critic 拟合、独立 critic 验证 | 已实现，单进程单设备 |
| 完整自博弈、真实合法采样概率、GAE 与 clipped PPO | 默认 Rust 多局批采集；官方 Python 路径用于校验 |
| 有效生产窗口、夜间仓库防溢出、终局 SELL / DROP | 采集/更新/推理统一的经济规则；强制部分不计 actor loss |
| 最后一天 C++ 搜索控制器 | 独立候选、限定 day=29 接管；可选纳入 critic/PPO continuation |
| 搜索教师 → 实际整局回放 → BC 缓存 | 已提供生成入口，按整局 seed 留出；不自动训练 |
| candidate/baseline 换座配对初筛与独立确认 | 已实现，候选不自动部署 |
| 独立源码交付包与 checkpoint 候选策略打包 | 已实现，权重生成后使用 |

## 环境准备

以下命令在仓库根目录执行，以 **Linux、系统 Python 3.12、单张 NVIDIA GPU** 为训练示例，无需创建 BC 虚拟环境。BC 不需要 Rust、LibTorch 或额外的参考源码仓库。虽然基础包声明 Python ≥3.10，固定版本的 NumPy/JAX 要求 Python ≥3.12；请确认 `python --version` 显示对应版本。已安装本项目 BC 依赖时，可跳过安装命令。

```bash
python --version
python -m pip install -e '.[bc-gpu]'

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=src
python -m route_rl.action_bc doctor
python -c 'import jax; print(jax.devices())'
```

GPU 训练前，最后一条命令应显示 `CudaDevice`。`bc-gpu` 使用 JAX 的 CUDA 12 依赖，需要兼容的 NVIDIA 驱动。`doctor` 列出依赖版本和项目源码指纹。

如果使用 CPU，将安装和设备选择改为：

```bash
python -m pip install -e '.[bc]'
export JAX_PLATFORMS=cpu
export PYTHONPATH=src
python -m route_rl.action_bc doctor
```

固定依赖见 [pyproject.toml](pyproject.toml)：NumPy 2.5.3、JAX 0.11.1、Optax 0.2.8、kaggle-environments 1.32.7。回放下载脚本会在独立的 `.venv-bc-tools` 中安装 Kaggle CLI 2.2.4；BC 准备、训练和评估使用当前系统 `python`。

## BC 复现步骤

### 1. 下载教师回放

本项目使用的下载命令是：

```bash
bash tools/download_public_bc.sh data/action_bc 20 100
```

三个参数依次是数据根目录、首次发现的教师队伍数、每位教师的对局上限。脚本自动创建 `.venv-bc-tools`、安装下载依赖并检查 Kaggle 认证；需要登录时会输出浏览器授权链接。账户需有比赛访问权限。这一步可以在安装 BC 训练依赖前单独执行。

首次运行冻结公开前 20 支队伍的教师快照，每队选择一个活跃提交，每位教师下载最多 100 局。重复运行会复用 `data/action_bc/teachers.json` 和已下载回放，并继续补足额度，不重新选择教师。100 是每位教师的独立对局上限，实际数量取决于公开且兼容的回放；一局可能贡献两个教师座位。

输出索引为 `data/action_bc/public/teacher-seats.jsonl`，后续准备步骤直接读取此文件。脚本先清理旧索引中的 seed=0 对局，下载时也排除零种子、异常结束和不兼容版本，保留正常结束的胜、平、负。原始回放、跳过原因和下载记录保存在同一下载目录中。

教师清单是首次下载时的排行榜快照，不能保证不同时间得到相同数据；复现同一次实验应保留原始清单、回放和索引。需要按 submission ID 单独下载或处理旧索引时，见 [BC 指南](docs/action_bc_pipeline.md#下载与数据选择)。

### 2. 准备并检查缓存

以下 `prepare` 用于首次生成缓存。当前 `data/action_bc/prepared_own/cache` 已生成时，跳过重复准备，直接训练；检查缓存可单独执行 `audit-cache`。重新编码时使用新的准备目录，保留原缓存。

```bash
PYTHONPATH=src python -m route_rl.action_bc prepare \
  --index data/action_bc/public/teacher-seats.jsonl \
  --out data/action_bc/prepared_own --workers 16

PYTHONPATH=src python -m route_rl.action_bc audit-cache \
  --cache data/action_bc/prepared_own/cache
```

`--workers` 是 CPU 预处理进程数。每个教师座位生成 719 条 `observation[t] → action[t+1]` 样本，按 episode 哈希约 10% 留出；同局双方保持同一 split，已知 seed 不允许跨训练和留出集。

**当前训练按窗口加载缓存。** 默认 `--shuffle-window 16`，最多驻留 32 条轨迹，float16 特征缓冲约 1.4 GiB，另需 batch、JAX 编译、优化器及显存开销。每轮仍访问全部样本，验证划分保持一致；`audit-cache` 的 `decoded_bytes` 是全部数据展开大小，不是当前驻留 RAM。入口同时检查容器 cgroup 内存上限，不能只看宿主机的 `free` 输出。旧全量加载器可能在 50 GB 内存容器中读取约 94 GB 数据时被 OOM kill。

### 3. 初始化并训练

初始化只执行一次。已有 `models/action_bc_own_initial.pkl` 时跳过 `init`，继续使用该文件：

```bash
PYTHONPATH=src python -m route_rl.action_bc init \
  --model bootstrap --seed 0 --out models/action_bc_own_initial.pkl
```

当前流式训练使用以下命令，保留原失败目录 `runs/action_bc_own_public_trial`：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m route_rl.action_bc train \
  --initial models/action_bc_own_initial.pkl \
  --cache data/action_bc/prepared_own/cache \
  --out runs/action_bc_own_public_stream \
  --epochs 30 --batch-size 320 --compute-dtype bfloat16 \
  --shuffle-window 16 --reuse-compatible-cache
```

`bootstrap` 是六层、宽度 256 的 Transformer。上述命令采用 30 个 epoch 的起步设置；默认 CLI 和 [BC 配置](src/route_rl/full_action/configs/bc.json)为 2 个 epoch。30 是参考训练谱系的起步遍数，并非本项目已验证的最优值，也不构成对参考全部早期超参数的复现。CPU 检查可在新 run 中使用较小的偶数 batch 和 `--compute-dtype float32`。

每个 epoch 后自动遍历留出集，记录单位/市场 CE、准确率、熵和 KL；整局对战需要下一步手动执行。训练只更新 actor 与主干，value head 不参与 BC。模型、损失、数据契约和容量限制见 [BC 指南](docs/action_bc_pipeline.md)。

流式读取使用明确记录的轨迹窗口乱序，与参考的全量行乱序不同。从旧全量加载器切换时使用新 run；当前缓存来自修复前的已核对 v3 代码，启动和续训都需保留 `--reuse-compatible-cache`，显式复用原特征/标签。此参数只允许已核对的兼容缓存，不能绕过编码检查；新代码首次生成的缓存无需兼容迁移，带此参数也不改变其标签。已有初始化策略可保留，旧优化器状态不能无缝续用。详细契约见 [BC 指南](docs/action_bc_pipeline.md#模型与训练)。

| 训练产物 | 用途 |
| --- | --- |
| `metrics.jsonl` | 每个 epoch 的训练与留出指标 |
| `epoch-N-policy.pkl` | 第 N 个 epoch 的推理策略，可在训练期间评估 |
| `latest_bc_state.pkl` | 最近完成 epoch 的参数与优化器状态，用于续训 |
| `final_student_jax.pkl` | 达到目标 epoch 后的最终推理策略 |
| `bc_config.json`、`integration.json`、`receipt.json` | 参数、源码/数据指纹和完成记录 |

### 4. 用新种子交换座位评估

训练完成后，以仓库内的 farm2945 参考 agent 为对手：

```bash
JAX_PLATFORMS=cpu PYTHONPATH=src python -m route_rl.action_bc evaluate \
  --run runs/action_bc_own_public_stream \
  --opponent agents/farm2945_resilient_response/main.py \
  --seed 1600000000 --games 16 \
  --out runs/action_bc_own_public_stream_eval/final-farm2945.json
```

16 局对应 8 个连续种子，每个种子交换座位，完整执行到第 720 个状态。报告中的 `score_rate` 按胜=1、平=0.5、负=0 汇总；现金用于诊断。输出还包括逐局记录 `final-farm2945.games.json`。

也可使用从 `agents/submission_farm2952_verified.tar.gz` 解压的 farm2952 对手；入口为 `agents/farm2952_verified/main.py`，许可证、NOTICE 和校验记录保存在同一目录：

```bash
JAX_PLATFORMS=cpu PYTHONPATH=src python -m route_rl.action_bc evaluate \
  --run runs/action_bc_own_public_stream \
  --opponent agents/farm2952_verified/main.py \
  --seed 1600000000 --games 16 \
  --out runs/action_bc_own_public_stream_eval/final-farm2952.json
```

此命令沿用 farm2945 评估的种子与交换座位设置，输出 `final-farm2952.json` 和 `final-farm2952.games.json`。

训练期间，可在 `epoch-1-policy.pkl` 出现后另开终端评估：

```bash
JAX_PLATFORMS=cpu PYTHONPATH=src python -m route_rl.action_bc evaluate \
  --run runs/action_bc_own_public_stream \
  --policy runs/action_bc_own_public_stream/epoch-1-policy.pkl \
  --opponent agents/farm2945_resilient_response/main.py \
  --seed 1600000000 --games 16 \
  --out runs/action_bc_own_public_stream_eval/epoch-1-farm2945.json
```

`--games` 必须为不小于 2 的偶数；评估文件不能覆盖。已知示范 seed 与评估 seed 重叠时会报错。用固定种子比较 epoch 后，再用另一批未参与选模的种子独立确认。部分公开回放缺失 seed，报告会列出这部分数据的独立性核查限制。当前评估只产生候选结果，不自动晋级或部署。

### 5. 续训

保留相同初始模型、缓存、代码和训练参数，重复 `train` 并增大累计 epoch 目标即可：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m route_rl.action_bc train \
  --initial models/action_bc_own_initial.pkl \
  --cache data/action_bc/prepared_own/cache \
  --out runs/action_bc_own_public_stream \
  --epochs 40 --batch-size 320 --compute-dtype bfloat16 \
  --shuffle-window 16 --reuse-compatible-cache
```

入口自动读取 `latest_bc_state.pkl`，从最近完成的 epoch 恢复；40 表示累计训练到第 40 个 epoch。保持原初始化、缓存、batch、dtype、窗口和兼容复用参数，仅增加 `--epochs`；没有已完成 epoch 的检查点时会从初始化重跑。改变 batch、dtype、窗口、数据、初始化或训练代码时需要新 run；数据或编码改变时也需要新准备目录。BC `.pkl` 不能接续旧 Rust `.json` 检查点。

## BC 完成后：critic → PPO

以下路径是后续产物约定，不表示当前已经有权重。等待 `runs/action_bc_own_public_stream/final_student_jax.pkl` 出现后运行；如选择已保存的 epoch 策略，critic 的 `--initial` 和 PPO 的 `--teacher` 必须都指向同一份 BC 文件。安装沿用 `.[bc-gpu]`；新 PPO 默认需要 Rust ≥1.88 构建自己的扩展，参考 clone 不参与构建或运行。当前 BC 无需切换采集器或重启。

先构建批采集扩展；`--jobs 2` 限制构建 CPU 并发，不使用训练 GPU。Cargo 首次构建需要获取锁定依赖，已有缓存可加 `--offline`：

```bash
python tools/build_action_native.py --jobs 2
export RAYON_NUM_THREADS=4
JAX_PLATFORMS=cpu PYTHONPATH=src python -m route_rl.action_ppo doctor
```

扩展位于 `src/route_rl/ppo/_native/`，身份包含源码、编译配置和 binary checksum。缺失或失配时 Rust 入口明确报错，不会静默降回 Python。新 run 默认 `rust-batch` + `official-prefix-economic-full-action-ppo-v2`；保留旧 `official-prefix-full-action-ppo-v1` 的原推理解释。

先固定选定的 BC 文件，避免之后 BC 续训覆盖 final 而改变 PPO teacher。只执行一次；已有目标文件时保留它：

```bash
mkdir -p models
cp --update=none runs/action_bc_own_public_stream/final_student_jax.pkl \
  models/action_bc_own_frozen.pkl
```

固定文件必须与 `--bc-run` 中的一份 epoch/final 策略 checksum 一致，入口会核验归属。然后拟合 value head，保持 actor 和 Transformer 主干冻结：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m route_rl.action_ppo doctor

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m route_rl.action_ppo warmup \
  --bc-run runs/action_bc_own_public_stream \
  --initial models/action_bc_own_frozen.pkl \
  --out runs/action_critic_own --updates 100
```

每轮默认采集 4 场训练局、2 场固定独立验证局，均使用完整 720 状态。训练使用 GAE return、验证使用真实终局 score 的预测误差；先保留未拟合的 iteration 0，再选择 best critic。默认 patience=8，可能在累计目标前停止。`policy_with_critic.pkl` 保留同一冻结 actor，只换 value head；它不是对战胜率提升的证据。

然后从 best critic 开始完整 PPO：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m route_rl.action_ppo train \
  --critic-run runs/action_critic_own \
  --initial runs/action_critic_own/policy_with_critic.pkl \
  --teacher models/action_bc_own_frozen.pkl \
  --out runs/action_ppo_own --updates 100
```

PPO 每轮默认 4 场当前策略双座位自博弈，即 5,752 env steps；同一步将 8 个座位一起送给 JAX，Rust 批量处理环境、特征、库存历史和合法前缀；minibatch=32、rollout epochs=1、BF16。默认设置见 [critic 配置](src/route_rl/ppo/configs/critic.json) 和 [PPO 配置](src/route_rl/ppo/configs/ppo.json)。`--games` 增大批量也会增加完整 rollout 的 RAM 和显存占用。100 轮是累计命令示例，不代表已测收益或耗时。

每轮 `metrics.jsonl` 的 `rollout_diagnostics` 记录 `collection_seconds` 和 `collection_env_steps_per_second`，按双方座位步数统计完整采集，包含环境准备、首次 JIT、特征、支持集、推理、存储和可选搜索；不是只测模拟器 step 的速度。

唯一奖励为终局胜/平/负 `1/0.5/0`。PPO 使用实际前缀支持集内的联合概率、GAE、clip=0.2、固定 BC teacher KL=0.2，actor 和 critic 同步更新。经济规则在采样前限制已无产出窗口的投入，夜间强制销售先占订单槽，末两回合清仓；强制动作的概率、KL、熵不计 actor loss，真实结果仍参与 value/GAE。现金不作为辅助奖励或晋级条件。BC 的 greedy 推理及当前训练源码指纹保持原定义。

官方校验路径可在新的 critic/PPO run 中显式选择 `--collection-backend official-python`，同一经济执行合同也用于 Python 推理。若要使用旧行为，两阶段同时传 `--execution official-prefix-full-action-ppo-v1`。切换 backend、规则或季末 continuation 都需新 run；已有 optimizer state 不自动迁移。

每轮保存 `latest_ppo_state.pkl`、`policy_latest_jax.pkl`、`policy-update-N.pkl` 和指标。重复原命令并仅增加 `--updates` 自动恢复完整更新边界；critic 同理。改变初始化、teacher、games、minibatch、dtype、配置或源码需新 run。详细概率、容量、恢复和数据隔离合同见 [PPO 指南](docs/action_ppo_pipeline.md)。

## PPO 配对初筛与独立确认

冻结一个已保存候选，与原 BC baseline 分别对 farm2945 在相同种子换座：

```bash
JAX_PLATFORMS=cpu PYTHONPATH=src python -m route_rl.action_ppo evaluate \
  --run runs/action_ppo_own \
  --policy runs/action_ppo_own/policy-update-100.pkl \
  --baseline models/action_bc_own_frozen.pkl \
  --opponent agents/farm2945_resilient_response/main.py \
  --seed 1600000000 --games 16 --phase screen \
  --out runs/action_ppo_own_eval/update-100-screen.json
```

`--games 16` 是每策略 16 场，candidate 加 baseline 共 32 场。也可用 `runs/action_critic_own/policy_with_critic.pkl` 作为 baseline，比较相同 masked execution 下的学习变化。必须保留 baseline 文件身份，不能混称两种比较。

初筛 paired match-score delta > 0 后，用另一批 seeds 确认同一候选、baseline 和 opponent：

```bash
JAX_PLATFORMS=cpu PYTHONPATH=src python -m route_rl.action_ppo evaluate \
  --run runs/action_ppo_own \
  --policy runs/action_ppo_own/policy-update-100.pkl \
  --baseline models/action_bc_own_frozen.pkl \
  --opponent agents/farm2945_resilient_response/main.py \
  --seed 1700000000 --games 32 --phase confirmation \
  --screen-report runs/action_ppo_own_eval/update-100-screen.json \
  --out runs/action_ppo_own_eval/update-100-confirmation.json
```

初筛/确认 seeds 不得用于 BC、critic 或 PPO 数据；入口也拒绝未来训练保留范围。确认输出仍为 `candidate_only`，不替换已接受部署。对称自博弈平均 score、loss 或现金增加不能作为晋级依据。此处示例面板不是统计显著性或总体胜率保证。

## 可选季末控制器与教师示范

使用 C++17 编译器构建本项目搜索库；策略包携带预编译库，运行时不编译：

```bash
python tools/build_season_search.py
```

先冻结同一份模型，分别创建“经济规则”和“经济规则 + 最后一天搜索”候选。下面用已经保存的 BC epoch 为例，两个新文件都不修改原 checkpoint：

```bash
JAX_PLATFORMS=cpu PYTHONPATH=src python -m route_rl.action_ppo candidate \
  --policy runs/action_bc_own_public_stream/epoch-1-policy.pkl \
  --out models/action_bc_epoch1_economic.pkl

JAX_PLATFORMS=cpu PYTHONPATH=src python -m route_rl.action_ppo candidate \
  --policy runs/action_bc_own_public_stream/epoch-1-policy.pkl --season-search \
  --out models/action_bc_epoch1_season.pkl

JAX_PLATFORMS=cpu PYTHONPATH=src python -m route_rl.action_ppo evaluate \
  --comparison controller --run runs/action_bc_own_public_stream \
  --policy models/action_bc_epoch1_season.pkl \
  --baseline models/action_bc_epoch1_economic.pkl \
  --opponent agents/farm2945_resilient_response/main.py \
  --seed 1600001000 --games 16 --phase screen \
  --out runs/action_controller_eval/epoch1-season-screen.json
```

初筛 paired match-score delta > 0 后，保留同样的模型与配置，用另一批 seeds 独立确认：

```bash
JAX_PLATFORMS=cpu PYTHONPATH=src python -m route_rl.action_ppo evaluate \
  --comparison controller --run runs/action_bc_own_public_stream \
  --policy models/action_bc_epoch1_season.pkl \
  --baseline models/action_bc_epoch1_economic.pkl \
  --opponent agents/farm2945_resilient_response/main.py \
  --seed 1700001000 --games 32 --phase confirmation \
  --screen-report runs/action_controller_eval/epoch1-season-screen.json \
  --out runs/action_controller_eval/epoch1-season-confirmation.json
```

入口核对权重相同、原策略属于该 run，并排除已知训练/验证 seeds。搜索仅从第 30 天清晨接管，继承真实观察历史，预算不足、输入不支持或异常时回网络。默认搜索预算 0.25 秒，可用 `--controller-config` JSON 调整至最多 6 秒；二进制和配置 hash 进入候选身份。

搜索内部将公开信息下预测的终局现金差映射为预计胜/平/负得分，模型尚未校准；固定不确定性尺度的映射是单调的，不能声称它已改变原现金差方案的排序或提高胜率。正式验收仍只看独立比赛终局得分。若把搜索纳入学习，critic 和 PPO 两阶段都传 `--season-search` 或相同 `--controller-config`；搜索接管动作不计 actor loss，终局结果沿真实 continuation 回传。

教师生成从明确 seed 或实际失败回放中的 seed 选择场景，规划器双方自博弈后保存真正执行的完整官方回放。先保留整局留出，再进入原 BC 准备接口：

```bash
JAX_PLATFORMS=cpu PYTHONPATH=src python -m route_rl.action_teacher generate \
  --out data/action_teacher_own --seed-start 60000000 --games 20 \
  --holdout-games 2 --seconds 0.25 \
  --exclude-run runs/action_bc_own_public_stream

PYTHONPATH=src python -m route_rl.action_bc prepare \
  --index data/action_teacher_own/teacher-seats.jsonl \
  --out data/action_teacher_own_prepared --workers 2
```

需要混合原公开示范时，使用已校验的合并入口，并以冻结 BC 权重开启新的 adaptation run：

```bash
PYTHONPATH=src python -m route_rl.action_teacher combine \
  --public-index data/action_bc/public/teacher-seats.jsonl \
  --teacher-index data/action_teacher_own/teacher-seats.jsonl \
  --out data/action_bc_teacher_mixed

PYTHONPATH=src python -m route_rl.action_bc prepare \
  --index data/action_bc_teacher_mixed/teacher-seats.jsonl \
  --out data/action_bc_teacher_mixed_prepared --workers 2

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m route_rl.action_bc train \
  --initial models/action_bc_own_frozen.pkl \
  --cache data/action_bc_teacher_mixed_prepared/cache \
  --out runs/action_bc_teacher_adapt \
  --epochs 2 --batch-size 320 --compute-dtype bfloat16 --shuffle-window 16
```

用重复的 `--source-replay path/to/failed.json.gz` 指定已核实失败场景，使用 `--exclude-seeds` 或 `--exclude-run evaluation-report.json` 额外保护选模/确认种子。已用于独立评估的 seed 保持留出，不能再拿它生成训练示范。生成后检查 `teacher_receipt.json`、`combine_receipt.json` 和回放。

等待 adaptation 的 `final_student_jax.pkl` 生成后，固定新 teacher，再进入新的 critic/PPO run：

```bash
cp --update=none runs/action_bc_teacher_adapt/final_student_jax.pkl \
  models/action_bc_teacher_frozen.pkl

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m route_rl.action_ppo warmup \
  --bc-run runs/action_bc_teacher_adapt \
  --initial models/action_bc_teacher_frozen.pkl \
  --out runs/action_critic_teacher_adapt --updates 100

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m route_rl.action_ppo train \
  --critic-run runs/action_critic_teacher_adapt \
  --initial runs/action_critic_teacher_adapt/policy_with_critic.pkl \
  --teacher models/action_bc_teacher_frozen.pkl \
  --out runs/action_ppo_teacher_adapt --updates 100
```

随后按上面的配对初筛/独立确认步骤评估；`--run` 改为 `runs/action_ppo_teacher_adapt`，候选改为其中保存的 `policy-update-N.pkl`，baseline 明确选择新 teacher 或该 run 的初始 critic，报告使用新的输出目录。需要将季末搜索纳入训练时，两阶段都加 `--season-search`。当前 BC 缓存、训练目录和 accepted 文件不覆盖；这些命令由你选择时机执行。

## 最终交付整理与策略打包

本次本地源码交付包位于 `dist/kaggriculture-pipeline-v0.7.0-documented.tar.gz`；可在新路径重新打包：

```bash
python tools/package_pipeline.py source \
  --out dist/kaggriculture-pipeline-review.tar.gz

python tools/package_pipeline.py verify \
  dist/kaggriculture-pipeline-review.tar.gz
```

包包含自身 Python/Rust/C++ 代码、配置、文档、测试、构建工具和必要依赖，排除参考 clone、data、runs、虚拟环境、缓存、编译物及凭据。解压后安装依赖并运行上述本项目构建工具即可；无需从 clone 复制文件。源码包不包含比赛权重。

实际 checkpoint 生成并完成审查后，打包指定候选：

```bash
PYTHONPATH=src python tools/package_pipeline.py submission \
  --policy runs/action_ppo_own/policy-update-100.pkl \
  --out dist/kaggriculture-policy.tar.gz
```

也支持本项目 BC、经济规则和季末控制器候选。入口按 checkpoint 的执行合同生成 `main.py`；搜索候选包含已核对的 `terminal_search.so`，不包含优化器或训练数据。策略包要求目标环境提供固定 NumPy/JAX；PPO/经济推理还要求官方 `kaggle-environments==1.32.7`。编译库须与目标 Linux 平台兼容，依赖与计时仍需目标环境验收；具体记录见 manifest。源码交付不伪造训练完成的比赛权重，也不执行比赛提交。

## 目录与文档

```text
src/route_rl/           BC/PPO 命令、回放下载、预处理与打包
  full_action/         保留 BC 特征、动作词表、JAX 模型、训练与推理
  ppo/                 critic、合法采样、自博弈、PPO、配对确认及配置
  season_search/       自有 C++ 搜索器、Python 适配与配置
tests/                 BC 数据、标签、恢复与执行契约检查
tools/                 下载封装与历史审计工具
agents/                本地评估参考 agent 及其来源/许可
native/action_engine/  完整动作 Rust 批环境、特征、追踪与前缀支持
native/                同时保留历史 Rust mixed/plan/event 实验
third_party/           Rust 模拟器依赖及其上游文档
docs/                  BC 指南、来源记录和实验历史
```

`data/`、`models/`、`runs/`、`replay/` 和构建产物被 Git 忽略；提交源码不会备份训练数据、权重或日志。

- [BC 详细指南与常见问题](docs/action_bc_pipeline.md)
- [完整训练方法与模型图解](docs/training_method.md)：从数据采集到 BC、critic、PPO、验收和交付，并比较参考 pipeline。
- [从知道 BC 到形成强解法：认知复盘](docs/experiment_retrospective.md)：对照四篇公开方案，练习问题建模、教师与数据选择、实验判断和预算取舍。
- [BC 代码来源与适配](docs/action_bc_sources.md)，[逐文件来源清单](docs/action_bc_sources.json)
- [PPO 详细指南](docs/action_ppo_pipeline.md)，[PPO 来源与适配](docs/action_ppo_sources.md)
- [本次实现验证记录](docs/pipeline_validation.json)：执行一致性与打包检查，包含尚未进行的正式实验。
- [历史实验与失败教训](docs/experiment_history.md)：旧实验已合并归档，历史建议不代表当前训练入口。
- [Rust 实验构建说明](native/README.md)

## 验证与后续开发

在安装 BC 依赖的环境中运行已有回归检查：

```bash
JAX_PLATFORMS=cpu PYTHONPATH=src python -m unittest discover -s tests -v
```

测试覆盖教师座位、标签对齐、实际 SELL 数量、split 隔离、缓存校验、窗口样本覆盖、容器内存检查、策略保存/恢复及 BC 参数更新边界。它们证明实现一致性，经营效果仍需独立完整对局。

新增检查覆盖合法前缀、真实行为概率、GAE 终止边界、critic 冻结、PPO 更新、恢复身份、种子隔离和独立打包。小模型完整局只证明连接和合同成立；正式训练与对战提升由后续独立实验判断。

## 来源

BC/PPO 必需组件适配与借鉴自 [msdsm/kaggriculture-solution](https://github.com/msdsm/kaggriculture-solution) 的固定 commit [84057a0](https://github.com/msdsm/kaggriculture-solution/tree/84057a0fda4238ccdebc46f9bf5496c6c4b2e00d)，运行时全部由本仓库维护。具体文件、改动和来源哈希见 [BC 来源](docs/action_bc_sources.md) 与 [PPO 来源](docs/action_ppo_sources.md)。Rust 模拟器和本地参考 agent 的来源及许可见各自目录内的 README、LICENSE 和 NOTICE。
