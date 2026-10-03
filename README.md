# Kaggriculture

用公开比赛回放训练 Kaggriculture 策略的实验项目。当前主线是 **完整动作行为克隆（BC）**：从教师回放中学习农场主、雇工和市场动作，使用 JAX Transformer 训练，并在官方环境中进行完整对局评估。

**项目仍在开发：BC 的下载、预处理、初始化、训练、续训和评估流程已可用；全动作 PPO、自博弈采集、critic 拟合和比赛提交打包尚未完成。** 仓库保留了早期 Rust mixed/plan/event 实验，它们与当前 BC 使用不同的数据和检查点格式。流程可运行不代表已复现参考方案的比赛成绩。

## 当前支持

| 功能 | 状态 |
| --- | --- |
| 公开回放下载、教师座位识别、去重与版本检查 | 已实现 |
| 整局划分训练/留出集、动作有效性检查、特征缓存 | 已实现 |
| BC 训练、逐 epoch 留出验证、保存策略和恢复优化器 | 已实现，单进程、单设备 |
| 指定 epoch / 最终策略的双座位整局评估 | 已实现，手动触发 |
| 全动作 PPO、critic 训练、自博弈与自动晋级 | 未完成 |
| 可直接上传 Kaggle 的 BC 提交包 | 未完成 |

## 环境准备

以下命令在仓库根目录执行，以 **Linux、Python 3.12、单张 NVIDIA GPU** 为训练示例。BC 不需要 Rust、LibTorch 或额外的参考源码仓库。虽然基础包声明 Python ≥3.10，固定版本的 NumPy/JAX 要求 Python ≥3.12；复现 BC 请使用 Python 3.12。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[bc-gpu,replays]'

export CUDA_VISIBLE_DEVICES=0
python -m route_rl.action_bc doctor
python -c 'import jax; print(jax.devices())'
```

GPU 训练前，最后一条命令应显示 `CudaDevice`。`bc-gpu` 使用 JAX 的 CUDA 12 依赖，需要兼容的 NVIDIA 驱动。`doctor` 列出依赖版本和项目源码指纹。

如果使用 CPU，在创建并激活虚拟环境后，将安装和设备选择改为：

```bash
python -m pip install -e '.[bc,replays]'
export JAX_PLATFORMS=cpu
python -m route_rl.action_bc doctor
```

固定依赖见 [pyproject.toml](pyproject.toml)：NumPy 2.5.3、JAX 0.11.1、Optax 0.2.8、kaggle-environments 1.32.7；下载使用 Kaggle CLI 2.2.4。

## BC 复现步骤

### 1. 登录 Kaggle 并下载教师回放

先确保账户有比赛访问权限，然后登录：

```bash
kaggle auth login --no-launch-browser
```

按命令输出的链接完成浏览器授权。已有可用凭据时可跳过登录。

冻结当前公开前 20 支队伍的教师清单，每队选择一个活跃提交，再下载每位教师最多 5 局：

```bash
python -m route_rl.replay_download discover \
  --top-teams 20 --out data/action_bc/teachers.json

python -m route_rl.replay_download download \
  --teachers data/action_bc/teachers.json \
  --out data/action_bc/public --limit-per-teacher 5
```

这是小规模起步数据。确认资源够用后，可重复下载命令扩大 `--limit-per-teacher`；下载器复用已有回放并去重。教师清单是下载时的排行榜快照，不能保证不同时间得到相同数据。复现同一次实验应保留原始清单、回放和索引。

也可以直接指定实际的 submission ID，重复 `--submission` 选择多个教师：

```bash
python -m route_rl.replay_download download \
  --submission 56722220 --limit-per-teacher 5 \
  --out data/action_bc/by_submission
```

这条命令使用独立数据目录；后续 `prepare --index` 应改为 `data/action_bc/by_submission/teacher-seats.jsonl`。submission 是策略提交，episode 是一场对局，下载器会自动列举 episode。

教师索引为 `public/teacher-seats.jsonl`；原始回放、跳过原因及下载记录也保存在下载目录中。教师座位从官方元数据确定，保留正常结束的胜、平、负回放；排除 seed=0、异常结束和不兼容版本。旧数据清理、认证和下载封装见 [BC 指南](docs/action_bc_pipeline.md#下载与数据选择)。

### 2. 准备并检查缓存

```bash
python -m route_rl.action_bc prepare \
  --index data/action_bc/public/teacher-seats.jsonl \
  --out data/action_bc/prepared --workers 4

python -m route_rl.action_bc audit-cache \
  --cache data/action_bc/prepared/cache
```

`--workers` 是 CPU 预处理进程数。每个教师座位生成 719 条 `observation[t] → action[t+1]` 样本，按 episode 哈希约 10% 留出；同局双方保持同一 split，已知 seed 不允许跨训练和留出集。

**训练会把全部缓存解压到主机内存。** 每条教师轨迹的特征约 47 MB，2,000 条约 94 GB，另需加载、拼接和训练开销。下载局数与座位轨迹数不同；先检查 `audit-cache` 的 `decoded_bytes`，不要按压缩文件大小估算 RAM。GPU batch 大小不会减少这部分内存。

### 3. 初始化并训练

```bash
python -m route_rl.action_bc init \
  --model bootstrap --seed 0 --out models/action_bc_initial.pkl

python -m route_rl.action_bc train \
  --initial models/action_bc_initial.pkl \
  --cache data/action_bc/prepared/cache \
  --out runs/action_bc_public \
  --epochs 30 --batch-size 320 --compute-dtype bfloat16
```

`bootstrap` 是六层、宽度 256 的 Transformer。上述命令采用 30 个 epoch 的起步设置；默认 CLI 和 [BC 配置](src/route_rl/full_action/configs/bc.json)为 2 个 epoch。30 是参考训练谱系的起步遍数，并非本项目已验证的最优值，也不构成对参考全部早期超参数的复现。CPU 检查可在新 run 中使用较小的偶数 batch 和 `--compute-dtype float32`。

每个 epoch 后自动遍历留出集，记录单位/市场 CE、准确率、熵和 KL；整局对战需要下一步手动执行。训练只更新 actor 与主干，value head 不参与 BC。模型、损失、数据契约和容量限制见 [BC 指南](docs/action_bc_pipeline.md)。

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
JAX_PLATFORMS=cpu python -m route_rl.action_bc evaluate \
  --run runs/action_bc_public \
  --opponent agents/farm2945_resilient_response/main.py \
  --seed 1600000000 --games 16 \
  --out runs/action_bc_eval/final-farm2945.json
```

16 局对应 8 个连续种子，每个种子交换座位，完整执行到第 720 个状态。报告中的 `score_rate` 按胜=1、平=0.5、负=0 汇总；现金用于诊断。输出还包括逐局记录 `final-farm2945.games.json`。

训练期间，可在 `epoch-1-policy.pkl` 出现后另开终端评估：

```bash
source .venv/bin/activate
JAX_PLATFORMS=cpu python -m route_rl.action_bc evaluate \
  --run runs/action_bc_public \
  --policy runs/action_bc_public/epoch-1-policy.pkl \
  --opponent agents/farm2945_resilient_response/main.py \
  --seed 1600000000 --games 16 \
  --out runs/action_bc_eval/epoch-1-farm2945.json
```

`--games` 必须为不小于 2 的偶数；评估文件不能覆盖。已知示范 seed 与评估 seed 重叠时会报错。用固定种子比较 epoch 后，再用另一批未参与选模的种子独立确认。部分公开回放缺失 seed，报告会列出这部分数据的独立性核查限制。当前评估只产生候选结果，不自动晋级或部署。

### 5. 续训

保留相同初始模型、缓存、代码和训练参数，重复 `train` 并增大累计 epoch 目标即可：

```bash
python -m route_rl.action_bc train \
  --initial models/action_bc_initial.pkl \
  --cache data/action_bc/prepared/cache \
  --out runs/action_bc_public \
  --epochs 40 --batch-size 320 --compute-dtype bfloat16
```

入口自动读取 `latest_bc_state.pkl`，从最近完成的 epoch 恢复；40 表示累计训练到第 40 个 epoch。改变 batch、dtype、数据、初始化或训练代码时需要新 run；数据或编码改变时也需要新准备目录。BC `.pkl` 不能接续旧 Rust `.json` 检查点。

## 目录与文档

```text
src/route_rl/           BC 命令、回放下载与预处理
  full_action/         特征、动作词表、JAX 模型、训练与推理
tests/                 BC 数据、标签、恢复与执行契约检查
tools/                 下载封装与历史审计工具
agents/                本地评估参考 agent 及其来源/许可
native/                历史 Rust mixed/plan/event 实验
third_party/           Rust 模拟器依赖及其上游文档
docs/                  BC 指南、来源记录和实验历史
```

`data/`、`models/`、`runs/`、`replay/` 和构建产物被 Git 忽略；提交源码不会备份训练数据、权重或日志。

- [BC 详细指南与常见问题](docs/action_bc_pipeline.md)
- [BC 代码来源与适配](docs/action_bc_sources.md)，[逐文件来源清单](docs/action_bc_sources.json)
- [历史实验与失败教训](docs/experiment_history.md)：旧实验已合并归档，历史建议不代表当前训练入口。
- [Rust 实验构建说明](native/README.md)

## 验证与后续开发

在安装 BC 依赖的环境中运行已有回归检查：

```bash
JAX_PLATFORMS=cpu python -m unittest discover -s tests -p test_action_bc.py -v
```

测试覆盖教师座位、标签对齐、实际 SELL 数量、split 隔离、缓存校验、策略保存/恢复及 BC 参数更新边界。它们证明实现一致性，经营效果仍需独立完整对局。

下一步计划是 critic 拟合、全动作自博弈采集和 PPO 更新，沿用当前观察、动作及检查点契约，再补独立评估与提交打包。以上环节尚未接入 BC；旧 Rust PPO 代码不等于当前全动作 PPO 已完成。

## 来源

BC 必需组件适配自 `msdsm/kaggriculture-solution` 的固定 commit，运行时全部由本仓库维护。具体文件、改动和来源哈希见 [来源记录](docs/action_bc_sources.md)。Rust 模拟器和本地参考 agent 的来源及许可见各自目录内的 README、LICENSE 和 NOTICE。
