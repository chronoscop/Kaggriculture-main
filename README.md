# Kaggriculture

本项目提供 **公开回放 → 行为克隆（BC）→ 冻结策略的 critic 拟合 → 自博弈 PPO → 配对评估 → agent 打包** 流程，并支持最后一天搜索与搜索教师示范。训练、评估和打包使用本仓库维护的 Python/Rust/C++ 代码，无需参考仓库 `kaggriculture-solution/`。

本指南面向首次使用项目的用户。所有命令在仓库根目录执行，文件名和训练轮数均为示例；替换路径时，同步修改后续命令的输入。`data/`、`models/`、`runs/` 和构建产物不随 Git 分发，需要自行生成。已有相应产物时，从后续步骤开始。

完整流程、模型结构、学习目标和指标解读见 [训练方法指南](docs/training_method.md)。

- [环境准备](#环境准备)
- [BC：下载、准备、训练与评估](#bc下载准备训练与评估)
- [critic 与 PPO](#critic-与-ppo)
- [PPO 配对初筛与独立确认](#ppo-配对初筛与独立确认)
- [可选：季末搜索与搜索教师](#可选季末搜索与搜索教师)
- [最终交付整理与策略打包](#最终交付整理与策略打包)
- [测试与文档](#测试与文档)

## 环境准备

训练示例使用 Linux、Python 3.12 和一张 NVIDIA GPU。基础包声明 Python ≥3.10，但固定版本的 NumPy/JAX 要求 Python ≥3.12。可在自己的虚拟环境中安装，已有兼容环境时也可直接使用。

```bash
python --version
python -m pip install -e '.[bc-gpu]'

unset JAX_PLATFORMS
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=src
python -m route_rl.action_bc doctor
python -c 'import jax; print(jax.devices())'
```

GPU 示例要求最后一条命令显示 `CudaDevice`。`bc-gpu` 安装 JAX 的 CUDA 12 依赖，需要兼容的 NVIDIA 驱动。训练入口支持单进程、单个可见 JAX 设备。

如使用 CPU，改为安装 CPU 依赖并选择 CPU 设备：

```bash
python -m pip install -e '.[bc]'
export JAX_PLATFORMS=cpu
export PYTHONPATH=src
python -m route_rl.action_bc doctor
```

切回 GPU 前执行 `unset JAX_PLATFORMS`。大型模型的 CPU 训练和对战可能较慢，可先使用较小模型检查流程。

固定依赖见 [pyproject.toml](pyproject.toml)：NumPy 2.5.3、JAX 0.11.1、Optax 0.2.8、kaggle-environments 1.32.7。回放下载脚本使用单独的 `.venv-bc-tools` 安装 Kaggle CLI 2.2.4；准备、训练和评估使用当前 `python`。

## BC：下载、准备、训练与评估

### 1. 下载教师回放

```bash
bash tools/download_public_bc.sh data/action_bc 20 100
```

三个参数依次是数据根目录、教师队伍数、每位教师单次最多选取的对局数。脚本创建下载工具环境并检查 Kaggle 认证；需要登录时会输出浏览器授权链接，账户需有比赛访问权限。

首次运行从公开排行榜第一页选择有可用公开得分的 20 支队伍，每队选择公开分数最高的提交，并保存 `data/action_bc/teachers.json`。后续运行复用该快照和已有回放；对局按固定哈希顺序选择，保留兼容且正常结束的胜、平、负，排除 seed=0。单局可贡献两个教师座位。索引保留历史记录，公开对局增加后累计数量可能超过单次的 100 局限制。

输出索引为 `data/action_bc/public/teacher-seats.jsonl`。教师快照、原始回放、下载记录和跳过原因应一起保留；排行榜与可下载对局会变化，单靠相同命令不能重建同一份数据。按 submission ID 下载及索引维护见 [BC 指南](docs/action_bc_pipeline.md#下载与数据选择)。

### 2. 准备并检查缓存

```bash
PYTHONPATH=src python -m route_rl.action_bc prepare \
  --index data/action_bc/public/teacher-seats.jsonl \
  --out data/action_bc/prepared --workers 16

PYTHONPATH=src python -m route_rl.action_bc audit-cache \
  --cache data/action_bc/prepared/cache
```

`--workers` 是 CPU 预处理进程数，按机器资源调整。每个教师座位生成 719 条 `observation[t] → action[t+1]` 样本；按 episode 哈希约 10% 留出，同局双方保持同一 split，已知 seed 不允许跨训练和留出集。

已有缓存时可单独执行 `audit-cache`，不必重复准备。BC 按轨迹窗口加载缓存；默认 `--shuffle-window 16`，最多同时驻留 32 条轨迹，另外需要 batch、JAX 编译、优化器和显存空间。`decoded_bytes` 表示全部缓存展开后的大小，不是当前内存占用；训练入口会检查容器 cgroup 可用内存。

### 3. 初始化并训练

初始化只执行一次；已有初始化文件时保留它：

```bash
PYTHONPATH=src python -m route_rl.action_bc init \
  --model bootstrap --seed 0 --out models/bc_initial.pkl

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m route_rl.action_bc train \
  --initial models/bc_initial.pkl \
  --cache data/action_bc/prepared/cache \
  --out runs/bc \
  --epochs 30 --batch-size 320 --compute-dtype bfloat16 \
  --shuffle-window 16
```

`bootstrap` 是六层、宽度 256 的 Transformer；`smoke` 可用于小模型检查。示例采用 30 个 epoch，CLI 和 [BC 配置](src/route_rl/full_action/configs/bc.json) 默认是 2 个 epoch，均不表示已验证的最优训练量。batch 必须为不小于 2 的偶数；CPU 检查可使用较小 batch 和 `--compute-dtype float32`。

每个 epoch 后自动遍历留出集，记录单位/市场 CE、准确率、熵和 teacher KL。BC 只更新 actor 与主干，value head 不参与训练；整局对战需单独执行。模型、损失和容量限制见 [BC 指南](docs/action_bc_pipeline.md)。

| 文件 | 用途 |
| --- | --- |
| `runs/bc/metrics.jsonl` | 每个 epoch 的训练与留出指标 |
| `runs/bc/epoch-N-policy.pkl` | 第 N 个 epoch 的推理策略 |
| `runs/bc/latest_bc_state.pkl` | 参数、优化器及续训状态 |
| `runs/bc/final_student_jax.pkl` | 达到累计 epoch 目标后的推理策略 |
| `bc_config.json`、`integration.json`、`receipt.json` | 配置、运行身份和完成记录 |

### 4. 用新种子交换座位评估

```bash
JAX_PLATFORMS=cpu PYTHONPATH=src python -m route_rl.action_bc evaluate \
  --run runs/bc \
  --opponent agents/farm2945_resilient_response/main.py \
  --seed 1600000000 --games 16 \
  --out runs/bc_eval/final-farm2945.json
```

默认读取该 run 的 `final_student_jax.pkl`。16 局使用 8 个连续种子，每个种子交换座位，完整执行 720 状态。`score_rate` 按胜=1、平=0.5、负=0 汇总，现金用于诊断；逐局记录写入 `final-farm2945.games.json`。

也可把对手改为仓库中的 `agents/farm2952_verified/main.py`，并为报告使用新名字，例如 `runs/bc_eval/final-farm2952.json`。对手的来源及许可文件保存在相应目录中。

训练期间可指定已经保存的 epoch 策略：

```bash
JAX_PLATFORMS=cpu PYTHONPATH=src python -m route_rl.action_bc evaluate \
  --run runs/bc \
  --policy runs/bc/epoch-1-policy.pkl \
  --opponent agents/farm2945_resilient_response/main.py \
  --seed 1600000000 --games 16 \
  --out runs/bc_eval/epoch-1-farm2945.json
```

`--games` 必须为不小于 2 的偶数，报告不能覆盖。入口拒绝已知示范 seed 与评估 seed 重叠；缺失示范 seed 的限制会写入报告。固定面板用于比较候选，另选未参与选模的种子独立确认；评估不自动部署策略。

### 5. 续训

保留初始化、缓存、源码和训练参数，重复命令并增大累计 epoch 目标：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m route_rl.action_bc train \
  --initial models/bc_initial.pkl \
  --cache data/action_bc/prepared/cache \
  --out runs/bc \
  --epochs 40 --batch-size 320 --compute-dtype bfloat16 \
  --shuffle-window 16
```

入口恢复 `latest_bc_state.pkl`，40 表示累计训练到第 40 个 epoch。改变 batch、dtype、窗口、初始化、数据或训练代码需要新 run；改变数据或编码时也需要新的准备目录。旧缓存仅在入口明确支持兼容编码时使用 `--reuse-compatible-cache`，具体条件见 [BC 指南](docs/action_bc_pipeline.md#模型与训练)。

## critic 与 PPO

### 1. 构建运行组件

Rust 批采集扩展需要 Rust ≥1.88；季末搜索需要 C++17 编译器。计划使用搜索时，在 critic/PPO 训练前构建搜索库：

```bash
python tools/build_action_native.py --jobs 2
python tools/build_season_search.py
export RAYON_NUM_THREADS=4
JAX_PLATFORMS=cpu PYTHONPATH=src python -m route_rl.action_ppo doctor
```

`--jobs` 限制 Rust 构建并发，`RAYON_NUM_THREADS` 限制批采集线程数。首次 Cargo 构建需要获取依赖，已有缓存可给构建命令加 `--offline`。不使用季末搜索时可跳过第二条命令；已有与源码匹配的扩展时可跳过相应构建。

Rust 扩展位于 `src/route_rl/ppo/_native/`；搜索库为 `src/route_rl/season_search/terminal_search.so`，构建记录为同目录的 `terminal_search.build.json`。源码、配置、构建记录及 binary checksum 用于身份核验，缺失或失配会报错。训练、评估和打包期间保持这些身份一致；首次添加搜索构建记录或更换构建可能改变 PPO run 身份。

### 2. 固定 BC teacher，拟合 critic

选择 BC 的 final 或已保存 epoch 文件，将它复制到固定路径。下面的 GNU `cp` 命令保留已存在的目标；已有文件时应确认它就是所选 BC 策略，否则使用新的文件名：

```bash
mkdir -p models
cp --update=none runs/bc/final_student_jax.pkl models/bc_teacher.pkl

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m route_rl.action_ppo warmup \
  --bc-run runs/bc \
  --initial models/bc_teacher.pkl \
  --out runs/critic --updates 100
```

固定 teacher 必须与所选 BC run 的一份 epoch/final 策略 checksum 一致。critic 阶段冻结 actor 和 Transformer 主干，只拟合 value head。每轮默认采集 4 场训练局、2 场固定独立验证局；训练使用 GAE return，验证使用真实终局 score 的预测误差，保留 best critic。

默认 `patience=8`，可在目标轮数前结束。`runs/critic/policy_with_critic.pkl` 是 best critic 对应的策略，`latest_critic_state.pkl` 用于恢复，`receipt.json` 标志该阶段完成。已经触发 patience 的 run 不会仅因增加 `--updates` 而继续拟合。

### 3. 从 best critic 训练 PPO

```bash
CUDA_VISIBLE_DEVICES=0 JAX_DEFAULT_MATMUL_PRECISION=highest PYTHONPATH=src \
python -m route_rl.action_ppo train \
  --critic-run runs/critic \
  --initial runs/critic/policy_with_critic.pkl \
  --teacher models/bc_teacher.pkl \
  --compute-dtype float32 \
  --out runs/ppo --updates 100
```

每轮默认 4 场当前策略双座位自博弈，共 5,752 个座位决策步；同一步批量推理 8 个座位。minibatch 默认 32，rollout epochs 默认 1。`--updates 100` 表示累计 100 轮采集与更新，每轮含多个优化器步；训练量是示例，可按资源调整。

[critic 配置](src/route_rl/ppo/configs/critic.json) 和 [PPO 配置](src/route_rl/ppo/configs/ppo.json) 默认使用 BF16；本指南的 PPO 命令显式使用 FP32 加 `highest` 矩阵精度，以减少采样图和更新图的数值差异。每轮更新前都会校验采样概率，`collected behavior log probability mismatch` 表示最大绝对误差超过 `0.0002`；先检查采样和重算的数值路径、dtype 及精度设置，保持该校验。

`compute_dtype` 会写入 checkpoint，`JAX_DEFAULT_MATMUL_PRECISION` 当前不会写入 checkpoint 或 resume 身份。按上述设置训练时，续训、评估和最终策略运行都应保持 `highest`；改变 dtype 要使用新 run。

唯一奖励是第 720 个状态的终局比赛得分：胜=1、平=0.5、负=0。PPO 使用 GAE、clip=0.2、固定 BC teacher KL=0.2，更新 actor、共享主干和 critic。默认经济合同在采样前约束生产窗口，并处理夜间销售、季末 SELL/DROP；强制动作不计 actor 的概率、KL 和熵，真实结果仍参与 value/GAE。现金是诊断指标，不是辅助奖励或晋级条件。

每轮保存 `latest_ppo_state.pkl`、`policy_latest_jax.pkl`、`policy-update-N.pkl` 和 `metrics.jsonl`。`rollout_diagnostics` 中的耗时和吞吐统计完整采集过程，包含准备、JIT、支持集、推理和存储。训练 loss、自博弈平均 score 或现金变化不能单独证明对战实力提高。

重复相同命令并仅增加 `--updates`，可恢复完整更新边界。改变初始化、teacher、games、minibatch、dtype、配置、源码或执行合同需要新 run。可在新的 critic/PPO run 中选择 `--collection-backend official-python`；两阶段的经济规则和季末 continuation 必须一致。概率、容量、恢复和种子隔离合同见 [PPO 指南](docs/action_ppo_pipeline.md)。

## PPO 配对初筛与独立确认

先冻结一个已保存 checkpoint，与原 BC baseline 分别对同一对手换座：

```bash
JAX_PLATFORMS=cpu JAX_DEFAULT_MATMUL_PRECISION=highest PYTHONPATH=src \
python -m route_rl.action_ppo evaluate \
  --run runs/ppo \
  --policy runs/ppo/policy-update-100.pkl \
  --baseline models/bc_teacher.pkl \
  --opponent agents/farm2945_resilient_response/main.py \
  --seed 1600000000 --games 16 --phase screen \
  --out runs/ppo_eval/update-100-screen.json
```

`--games 16` 是每策略 16 场，candidate 与 baseline 合计 32 场。也可用 `runs/critic/policy_with_critic.pkl` 作为 baseline，比较相同 masked execution 下的学习变化；报告应明确 baseline 的文件身份。

初筛的 paired match-score delta > 0 后，用另一批种子确认同一份 candidate、baseline 和 opponent：

```bash
JAX_PLATFORMS=cpu JAX_DEFAULT_MATMUL_PRECISION=highest PYTHONPATH=src \
python -m route_rl.action_ppo evaluate \
  --run runs/ppo \
  --policy runs/ppo/policy-update-100.pkl \
  --baseline models/bc_teacher.pkl \
  --opponent agents/farm2945_resilient_response/main.py \
  --seed 1700000000 --games 32 --phase confirmation \
  --screen-report runs/ppo_eval/update-100-screen.json \
  --out runs/ppo_eval/update-100-confirmation.json
```

种子数值是示例。入口拒绝评估面板与已知 BC/critic/PPO 数据或未来训练保留范围重叠；独立确认还要求与所引用初筛报告的种子不重叠。报告使用新路径，结果保持 `candidate_only`，不会自动替换 accepted 部署。确认对象是实际要交付的完整策略；给 PPO 追加搜索后，需要对新的组合候选单独比较。

## 可选：季末搜索与搜索教师

### 1. 为已训练策略追加季末搜索

季末控制器从第 30 天清晨（`day=29`）尝试接管，继承整局公开观察历史。默认搜索预算 0.25 秒，首次接管要求至少 10 秒剩余 overage；可用 `--controller-config` JSON 调整支持的预算，上限为 6 秒。已成功初始化的 agent 遇到搜索执行异常、不支持的输入或预算不足时回到网络。缺失或失配的搜索库、构建记录会在初始化阶段报错。

已有 PPO 添加搜索、生成组合候选和打包的命令统一见[最终交付整理与策略打包](#最终交付整理与策略打包)。追加控制器保持权重不变，但改变最后一天的执行方式；它不表示原 PPO 训练已包含搜索。若希望训练时也使用搜索，两阶段都传 `--season-search` 或相同的 `--controller-config`，并使用对应的新 run。

也可用冻结 BC 策略比较控制器。下面生成不带搜索和带搜索的两个候选，并初筛：

```bash
JAX_PLATFORMS=cpu PYTHONPATH=src python -m route_rl.action_ppo candidate \
  --policy models/bc_teacher.pkl \
  --out models/bc_economic.pkl

JAX_PLATFORMS=cpu PYTHONPATH=src python -m route_rl.action_ppo candidate \
  --policy models/bc_teacher.pkl --season-search \
  --out models/bc_season.pkl

JAX_PLATFORMS=cpu PYTHONPATH=src python -m route_rl.action_ppo evaluate \
  --comparison controller --run runs/bc \
  --policy models/bc_season.pkl \
  --baseline models/bc_economic.pkl \
  --opponent agents/farm2945_resilient_response/main.py \
  --seed 1600001000 --games 16 --phase screen \
  --out runs/controller_eval/bc-season-screen.json
```

初筛通过后，用另一批种子确认相同文件：

```bash
JAX_PLATFORMS=cpu PYTHONPATH=src python -m route_rl.action_ppo evaluate \
  --comparison controller --run runs/bc \
  --policy models/bc_season.pkl \
  --baseline models/bc_economic.pkl \
  --opponent agents/farm2945_resilient_response/main.py \
  --seed 1700001000 --games 32 --phase confirmation \
  --screen-report runs/controller_eval/bc-season-screen.json \
  --out runs/controller_eval/bc-season-confirmation.json
```

`--comparison controller` 核验相同冻结权重、原策略的 run 归属和种子隔离。对 PPO 组合候选比较时，使用对应的 `runs/ppo` 与不带搜索的同权重候选，保持 FP32 精度环境设置。

搜索根据公开信息预测终局现金差，再映射为未校准的胜/平/负期望。它不是 PPO 辅助奖励，开启搜索也不自动证明得分提高。

### 2. 生成搜索教师示范

```bash
JAX_PLATFORMS=cpu PYTHONPATH=src python -m route_rl.action_teacher generate \
  --out data/search_teacher --seed-start 60000000 --games 20 \
  --holdout-games 2 --seconds 0.25 \
  --exclude-run runs/bc \
  --exclude-run runs/ppo
```

搜索教师双方执行完整对局，保存实际动作和官方回放；20 局中 2 局按整局 seed 留出。需要不同场景时可重复传 `--seed` 或 `--source-replay path/to/failed.json.gz`；源回放只用于提取已核实 seed，不复用其动作标签。

用重复的 `--exclude-run` 指定需要保护的训练 run 或已有评估报告。`--exclude-seeds` 接受 JSON 整数列表或 `{"seeds": [...]}`。用于独立评估的 seed 保持留出，不再生成训练示范。生成完成后检查 `teacher_receipt.json`、`games.json` 和回放；该命令不自动启动训练。

### 3. 混合示范并重新训练

下面以原公开示范与搜索示范混合为例；只使用搜索示范时，可直接准备 `data/search_teacher/teacher-seats.jsonl`：

```bash
PYTHONPATH=src python -m route_rl.action_teacher combine \
  --public-index data/action_bc/public/teacher-seats.jsonl \
  --teacher-index data/search_teacher/teacher-seats.jsonl \
  --out data/action_bc_mixed

PYTHONPATH=src python -m route_rl.action_bc prepare \
  --index data/action_bc_mixed/teacher-seats.jsonl \
  --out data/action_bc_mixed_prepared --workers 2

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m route_rl.action_bc train \
  --initial models/bc_teacher.pkl \
  --cache data/action_bc_mixed_prepared/cache \
  --out runs/bc_adapt \
  --epochs 2 --batch-size 320 --compute-dtype bfloat16 --shuffle-window 16
```

`combine` 核验教师生成已完成、索引与回放 checksum、实际 seed 和 split，保留公开留出集并拒绝跨 split 冲突。输出包含 `teacher-seats.jsonl` 和 `combine_receipt.json`；保留其引用的原始回放。

adaptation 完成后固定新 teacher，并进入新的 critic/PPO run：

```bash
cp --update=none runs/bc_adapt/final_student_jax.pkl models/bc_adapt_teacher.pkl

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m route_rl.action_ppo warmup \
  --bc-run runs/bc_adapt \
  --initial models/bc_adapt_teacher.pkl \
  --out runs/critic_adapt --updates 100

CUDA_VISIBLE_DEVICES=0 JAX_DEFAULT_MATMUL_PRECISION=highest PYTHONPATH=src \
python -m route_rl.action_ppo train \
  --critic-run runs/critic_adapt \
  --initial runs/critic_adapt/policy_with_critic.pkl \
  --teacher models/bc_adapt_teacher.pkl \
  --compute-dtype float32 \
  --out runs/ppo_adapt --updates 100
```

评估和打包时替换为 adaptation 的 run、checkpoint、teacher 及新的报告/候选路径。需要训练季末 continuation 时，两阶段保持相同搜索配置。

## 最终交付整理与策略打包

### 1. 打包 PPO + 季末搜索 agent

下面从不带搜索控制器的第 100 轮 PPO checkpoint 追加季末搜索。搜索库与 `terminal_search.build.json` 应已按[构建运行组件](#1-构建运行组件)准备，并保持与源码匹配：

```bash
JAX_PLATFORMS=cpu JAX_DEFAULT_MATMUL_PRECISION=highest PYTHONPATH=src \
python -m route_rl.action_ppo candidate \
  --policy runs/ppo/policy-update-100.pkl \
  --execution official-prefix-economic-full-action-ppo-v2 \
  --season-search \
  --out models/ppo_update100_season.pkl

python tools/package_pipeline.py submission \
  --policy models/ppo_update100_season.pkl \
  --out dist/ppo_update100_season.tar.gz

python tools/package_pipeline.py verify \
  dist/ppo_update100_season.tar.gz
```

选择其他轮次时替换 checkpoint，并为组合候选、压缩包使用新的名字。`--execution` 应匹配源策略合同；示例使用默认经济合同。已有组合候选时从 `submission` 开始；若 PPO checkpoint 本身已包含搜索控制器，跳过 `candidate`，直接把它作为 `submission --policy`。候选和压缩包均拒绝覆盖同名文件。

`--policy` 明确指定要交付的策略，脚本不自动选择最好或最新模型；`--out` 仅指定输出路径。包内包含 `main.py`、`policy.pkl`、项目运行代码、搜索 `.so`、构建记录、`requirements.txt` 和 `bundle-manifest.json`。同一 agent 前 29 天使用 PPO，第 30 天尝试搜索，搜索执行失败时回 PPO。

解压后保持完整目录，按 `requirements.txt` 安装依赖，并在导入 JAX 前设置 `JAX_DEFAULT_MATMUL_PRECISION=highest`。生成的 `main.py` 默认使用 CPU，当前不会自动保存矩阵精度环境变量；仅在打包命令前设置该变量也不会把它写入入口。依赖版本和打包验证范围见 manifest，原生库的平台及编译信息见构建记录；目标 Linux 环境的依赖、完整对局和计时需要另行验证。打包不会自动晋级、部署或提交比赛。

纯 BC/PPO 也可直接作为 `submission --policy`；只有带搜索配置的候选才会包含搜索二进制。`verify` 检查归档文件清单、安全路径及 checksum，不衡量比赛效果。

### 2. 归档项目源码

```bash
python tools/package_pipeline.py source \
  --out dist/kaggriculture-source.tar.gz

python tools/package_pipeline.py verify \
  dist/kaggriculture-source.tar.gz
```

`source` 根据脚本位置确定仓库根目录，按固定规则收集源码、配置、文档、测试及构建工具；排除训练数据、run、模型权重、凭据及编译二进制。源码归档不包含可直接参赛的模型，解压后需安装依赖并重新构建所需扩展。

## 测试与文档

运行完整回归前，先按[构建运行组件](#1-构建运行组件)准备 Rust 批采集扩展和季末搜索库。缺失组件可能使相关检查跳过或报错；跳过的检查不算通过：

```bash
JAX_PLATFORMS=cpu PYTHONPATH=src python -m unittest discover -s tests -v
```

检查覆盖标签与缓存、split/seed 隔离、窗口读取、概率一致性、GAE、critic 冻结、策略恢复、执行合同和隔离打包。小模型 smoke 或完整局检查只证明连接与实现一致性，不证明大型模型训练效果或比赛得分提高。

| 路径 | 内容 |
| --- | --- |
| `src/route_rl/full_action/` | BC 特征、模型、训练和推理 |
| `src/route_rl/ppo/` | critic、合法采样、自博弈、配对评估 |
| `src/route_rl/season_search/` | C++ 搜索器及 Python 适配 |
| `native/action_engine/` | Rust 批环境、前缀支持与特征编码 |
| `tools/` | 下载、构建与打包入口 |
| `agents/` | 本地评估对手及其来源/许可 |
| `tests/`、`docs/` | 回归检查与详细文档 |

早期 mixed/plan/event 实验保留在 `native/` 等目录中，使用独立数据和 checkpoint 格式，不用于续接本指南的 BC/PPO 状态。

- [BC 指南](docs/action_bc_pipeline.md)：数据、模型、缓存和恢复合同。
- [PPO 指南](docs/action_ppo_pipeline.md)：概率、执行、seed 隔离和控制器合同。
- [训练方法指南](docs/training_method.md)：完整流程、模型图解、训练指标、恢复与交付。
- [实验复盘](docs/experiment_retrospective.md)、[历史实验与失败教训](docs/experiment_history.md)：历史证据及其适用范围。
- [实现验证记录](docs/pipeline_validation.json)：历史实现检查快照，不是实时训练或比赛报告。
- [Rust 实验说明](native/README.md)：早期实验入口。

## 来源

BC/PPO 部件借鉴自 [msdsm/kaggriculture-solution](https://github.com/msdsm/kaggriculture-solution) 的固定 commit [84057a0](https://github.com/msdsm/kaggriculture-solution/tree/84057a0fda4238ccdebc46f9bf5496c6c4b2e00d)，适配后由本仓库维护。具体文件、改动及来源哈希见 [BC 来源](docs/action_bc_sources.md)、[BC 清单](docs/action_bc_sources.json)、[PPO 来源](docs/action_ppo_sources.md) 和 [PPO 清单](docs/action_ppo_sources.json)。Rust 模拟器和参考 agent 的来源及许可见对应目录的 README、LICENSE 和 NOTICE；实现检查不构成对参考比赛成绩的复现声明。
