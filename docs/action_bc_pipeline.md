# 完整动作 BC 指南

BC 主流程已实现：公开回放 → 教师座位索引 → 特征/标签缓存 → JAX 训练 → 留出验证 → 官方环境完整对局评估。安装与连续运行命令见 [README](../README.md#bc-复现步骤)。本指南补充数据契约、资源要求、续训和故障处理。

实现位于 `src/route_rl/action_bc.py`、`replay_download.py`、`replay_prepare.py` 与 `full_action/`，不需要 `kaggriculture-solution/` 参考目录。critic 拟合、自博弈、PPO 更新和策略打包由独立的 `route_rl.ppo` / `route_rl.packaging` 提供，见 [PPO 指南](action_ppo_pipeline.md)；当前 BC 行为与指纹保持原定义，候选不自动部署。代码来源见 [来源记录](action_bc_sources.md)。

## 下载与数据选择

与 README 一致，使用以下脚本下载回放：

```bash
bash tools/download_public_bc.sh data/action_bc 20 100
```

它在 `.venv-bc-tools` 中安装 Kaggle CLI 2.2.4，检查认证，必要时提示浏览器登录；先清理旧索引中的零种子局，再发现或复用教师快照并下载。三个参数依次是数据根目录、首次发现的教师队伍数、每位教师的对局上限。此处每位教师最多 100 局，重复执行会复用原快照和回放，继续补足额度。脚本默认第三个参数是 5；复现这里的下载规模必须显式传入 `100`。

下载不需要 JAX，也不要求预先在训练环境安装 Kaggle CLI。脚本用 `python3` 创建下载环境；请确保该解释器可创建 venv。后续预处理、训练和评估使用 README 中的系统 Python 3.12，安装 `.[bc]` 或 `.[bc-gpu]`，并在仓库根目录设置 `PYTHONPATH=src`。

如需绕过排行榜快照、直接选择 submission，可使用脚本已创建的下载环境：

```bash
PYTHONPATH=src .venv-bc-tools/bin/python -m route_rl.replay_download download \
  --submission 56722220 --limit-per-teacher 100 \
  --out data/action_bc/by_submission
```

`--submission` 可以重复；也可以用 `--teachers` 传入显式教师清单，两种方式二选一。此例后续 `prepare --index` 应改为 `data/action_bc/by_submission/teacher-seats.jsonl`。submission 是策略提交，episode 是一场对局；额度计算每个教师的独立对局数，一局两个教师座位可以产生两条轨迹。

下载器：

- 从官方 episode agent 元数据中的 submission ID 与座位 index 识别教师，不猜赢家或固定座位。
- 保留正常完成的胜、平、负；元数据省略 agent 状态时，仍需回放终局确认双方 `DONE`。
- 只接受已核对的 1.32.7 / 1.33.0 Kaggriculture 回放，必须含双方完整观察、动作和 720 个状态。
- 排除 seed=0；缺失 seed 不视作 0。跳过零种子局不占额度，并保存其 ID 供恢复时复用。
- 保存 replay 校验值，去重教师座位；版本、异常结束等跳过原因写入 `skipped.jsonl`。无权限会停止，暂时错误有限重试。

对旧下载索引原地排除零种子局：

```bash
PYTHONPATH=src python -m route_rl.replay_download filter-zero-seed \
  --out data/action_bc/public
```

操作保留原索引备份和原始 replay，写入 `seed-zero-filter.json` 并更新已有下载记录。直接 `download` 和封装脚本也会先清理旧索引。准备入口再次核验真实 seed，拒绝重新引入零种子数据。

公开参考项目没有提供完整历史教师清单、回放集或下载器。本项目取得的是当前可访问的数据；从相同模型配置训练不等于精确重现其历史实验。

## 特征、标签与规则

| 项目 | 当前约定 |
| --- | --- |
| 时间对齐 | `observation[t] → action[t+1]`，每轨迹 719 行 |
| 特征 | `features[719,264,124]`，包括实体和公开库存估计 |
| 标签 | `labels[719,30]`，20 个己方单位槽 + 10 个市场槽 |
| 忽略标签 | `-100`，用于 padding、无效或未执行动作 |
| SELL | 按官方执行器确认的实际成交绝对数量生成标签 |
| 数据契约 | `public-full-action-bc-v3` |
| policy 契约 | `route-rl-full-action-policy-v1` |

训练和推理共用特征编码、动作词表及库存 tracker；准备时使用教师此前实际动作推进估计。官方 1.32.7 执行器检查动作有效性，不能把未执行请求当作成功标签。

当前固定容量为最多 20 个己方单位、40 个总单位。超限教师轨迹拒绝编码；推理会截取编码，超出的己方单位输出 PASS。模型不支持任意规模的完整农场。推理使用 `GreedyPolicy`，未接入参考方案的启发式后处理或季末搜索。

版本兼容逻辑位于 [replay_rules.py](../src/route_rl/replay_rules.py)。先前核对记录表明 1.32.7 与 1.33.0 的 Kaggriculture 规则文件一致；代码保留原始 `module_version`，准备/评估仍要求安装的 1.32.7 文件匹配固定哈希：

```text
规则 .py: bc8a54879ef02c7ea64b8b333d6a976f0ea65c4949149d01f463f23bccee653e
配置 .json: a82c89c1a2315b93f39775d8e025471a01b738647c9772658368ee6b1b6f4867
```

无 `step` 的观察使用实际 `day*24+hour` 检查时间；seed 从真实 `configuration.seed` 或 `info.seed` 读取。不要改写版本号或伪造观察字段绕过核验。

## 数据隔离与复现记录

按 `sha256("50:" + episode_id)` 确定约 10% 的留出集。同一 episode 的双方始终同组；已知相同 seed 的不同 episode 若跨 split，准备会拒绝。两组必须非空，不能用同一局复制出验证集。seed 缺失的对局保留计数，无法承诺它们与评估种子完全独立。

准备目录保存 `preparation.json`、逐教师 `manifests/`、`cache/index.json`、`cache/pipeline.json` 和逐轨迹 `.npz`/记录。缓存审计检查形状、有限特征、动作 ID、split、去重及数组哈希。

复现同一次训练需保存：

1. 代码 commit、`doctor` 输出与依赖版本。
2. 教师快照、原始回放、`teacher-seats.jsonl` 及下载记录。
3. 准备记录、完整缓存、初始 `.pkl`、训练配置和运行目录。
4. 评估所用 policy、对手文件、种子、座位和逐局输出。

BC 指纹只覆盖相关 Python 源码和模型配置，README/文档更新不会使已有 BC 缓存失效。源码、数据、缓存或训练参数改变时，入口拒绝原目录复用；旧 v1/v2 缓存与原生 event 检查点不能静默转为当前契约。仅对本次流式读取修复前的已核对 v3 缓存，可以显式使用 `--reuse-compatible-cache`：入口同时核对原源码指纹、当前编码/标签/划分指纹及完整缓存，保留原缓存记录，在新 run 中记录复用来源。其他编码变化仍需要重新准备。

## 模型与训练

| 初始化预设 | 层数 | 宽度 | 用途 |
| --- | ---: | ---: | --- |
| `bootstrap` | 6 | 256 | README 的起步训练 |
| `10m` | 12 | 256 | 更深模型，需另建实验 |
| `smoke` | 1 | 32 | 实现检查 |

预设随 Python 包分发，使用 manual attention 和从 features 恢复元数据的 partitioned 路径。未接入参考的实验 kernel、Net2Net 或模型权重。

训练损失为：

```text
CE_unit + CE_market
− 0.10 × H_unit − 0.10 × H_market
+ 0.05 × [KL(initial || current)_unit + KL(initial || current)_market]
```

学习率 `1e-4`，Adam epsilon `1e-5`，梯度裁剪 5，训练 shuffle seed 51；README 单独把模型初始化 seed 固定为 0。初始策略始终冻结作为 KL 参考，只训练 actor 和主干，不更新 value head，不使用现金辅助奖励。每个 epoch 的留出 pass 不更新参数。

配置来自参考公开的最后一阶段 BC 设置，默认 `epochs=2`、batch 320、BF16。README 起步命令设置 30 个 epoch，借鉴其 BC1 的公开遍数；BC1 的完整早期超参数未公开，这不是逐项历史复现。扩大数据、模型或遍数的收益需要单独实验确认。

当前入口要求一个进程仅看到一个 JAX 设备；多卡机器使用 `CUDA_VISIBLE_DEVICES=0`。`prepare --workers` 只控制 CPU 编码，不提供 GPU 分布式训练。

当前使用 `streaming-npz-v1` 读取器。初始化只读 `.npz` 内的数组头；训练按 `--shuffle-window`（默认 16）条轨迹加载，最多缓存两个窗口，及时释放数组。每轮随机打乱轨迹顺序、窗口内样本顺序，跨窗口衔接尾批，每条训练样本恰好访问一次；验证按原顺序访问。最后一个 batch 才补齐，padding 不参与损失。该读取乱序契约为 `trajectory-window-v1`，与参考的全量行乱序不同，会记录进配置和检查点，旧优化器状态不能无缝恢复。

当前缓存每条 float16 特征约 47 MB；默认最多 32 条驻留，轨迹缓冲约 1.4 GiB，另有 batch、JAX 编译、优化器和设备开销。整个缓存的 `decoded_bytes` 仍会打印，但不再一次性加载。内存预检同时读取 `/proc/meminfo` 与 cgroup v1/v2 限额。缓冲预算不是整个进程峰值内存保证。

本次 A40 运行的宿主机有约 503 GiB RAM，容器实际只允许约 46.6 GiB；1,986 条缓存展开需要 87.1 GiB，原全量读取器触发了 OOM kill。GPU batch 无法解决全量主机内存加载。保留原缓存和初始化模型，使用新 run 启动流式读取：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m route_rl.action_bc train \
  --initial models/action_bc_own_initial.pkl \
  --cache data/action_bc/prepared_own/cache \
  --out runs/action_bc_own_public_stream \
  --epochs 30 --batch-size 320 --compute-dtype bfloat16 \
  --shuffle-window 16 --reuse-compatible-cache
```

原失败目录 `runs/action_bc_own_public_trial` 保留。这里只验证了读取与训练通路；320 的设备端占用仍由实际训练观察。如果随后明确报 GPU 显存不足，需要减小偶数 batch 并换新 run，不能把主机 RAM 与显存问题混为一谈。

## 保存、恢复与评估

每个 epoch 完成后保存 `latest_bc_state.pkl`（参数、优化器、epoch）和 `epoch-N-policy.pkl`，追加 `metrics.jsonl`。全部完成后保存 `final_student_jax.pkl` 与 `receipt.json`；源码、初始文件、缓存和训练设置记录在 `integration.json`。

再次运行同一 `train` 命令会自动恢复最近完成的 epoch，无 `--resume` 参数。`--epochs` 是累计目标，只允许增加，不能下降；中途未完成的 epoch 会重跑。续训时 `--initial` 仍指向该 run 原来的初始化文件，不能改为 `latest_bc_state.pkl` 或新生成文件。当前 `runs/action_bc_own_public_stream` 每次启动都需保留 `--shuffle-window 16 --reuse-compatible-cache`，具体 30→40 epoch 命令及该 run 的评估命令见 README；评估不需要缓存复用参数。

改变 batch、dtype、窗口、初始化、训练设置或代码需要新 run。扩大教师数据还需要新准备目录。已有推理 `.pkl` 可作为新 BC 阶段的 `--initial`，但这会建立新的 KL 参考与优化器，不是原 run 的无缝恢复。

完整对局评估：

- `evaluate --run ...` 默认读取 `final_student_jax.pkl`；`--policy` 可指定已保存的 epoch 策略。
- 双方在独立进程运行，复用当前编码/tracker，使用固定官方环境；每种子交换座位，检查双方 `DONE` 和 720 个状态。
- 报告保存 policy 和对手哈希、逐局 seed/seat/rewards、`score_rate` 及缺失示范 seed 的计数。
- 评估输出文件已存在时拒绝覆盖；已知示范 seed 重叠时拒绝运行。源码变化后，需要恢复训练时的 BC 代码再评估。
- 所有 BC 输出标记 `candidate_only`。留出 CE/accuracy 改善不能代替经营能力，阶段性固定种子比较不能代替另一批种子的独立确认。

参考 farm2945 是本地评估对象，不是 DECEM 源码。连续生产、条件续种/转产、共享材料与路线、现金周转用于行为诊断；最终衡量终局比赛得分。对 DECEM 行为与旧实验的证据边界见 [历史记录](experiment_history.md#replay-evidence)。

## 常见问题

| 现象 | 处理 |
| --- | --- |
| Python 3.10/3.11 安装固定依赖失败 | BC 使用 Python 3.12，并确认 `python` 指向已安装固定依赖的解释器 |
| JAX 只看到 CPU | 检查是否安装 `bc-gpu`、NVIDIA 驱动和设备可见性；不要以 `doctor` 输出代替设备检查 |
| JAX 看到多张 GPU，训练拒绝启动 | 运行前设置 `CUDA_VISIBLE_DEVICES=0` |
| 无 Kaggle 权限或认证失败 | 在比赛页面确认访问权限，使用安装了 Kaggle CLI 的环境完成认证 |
| 准备报无训练/验证 split | 小数据可能没有哈希留出局，补下载兼容对局 |
| 已知 seed 跨 split | 从教师索引排除冲突对局，再使用新准备目录；不能关闭隔离检查 |
| 旧索引含 seed=0 | 对下载目录执行 `filter-zero-seed`，已生成缓存时重新准备 |
| 缓存/源码/训练设置不匹配 | 使用新准备目录或 run，保留原产物 |
| 训练输出 Killed / RAM 不足 | 检查容器 cgroup 限额及 OOM 计数；当前读取器按窗口加载，可减小 `--shuffle-window` 并换新 run；已核对的旧缓存可显式复用 |
| 显存不足 | 新 run 使用更小的偶数 batch；原目录不能改 batch 续训 |
| 尚无 final policy | 等训练完成，或通过 `--policy` 指定已完成 epoch 的文件 |

本文使用系统 `python`。排查时确认命令运行在仓库根目录，`PYTHONPATH=src`，且当前解释器已安装 BC 依赖；另开终端评估时也应保留这些设置。

BC 完成后的 PPO 使用本项目独立入口：固定 BC actor/主干拟合 critic → 当前策略双座位自博弈及真实合法采样概率 → PPO 更新与固定 BC KL → 训练外配对初筛与独立确认。命令见 [README](../README.md#bc-完成后critic--ppo)，执行语义与来源见 [PPO 指南](action_ppo_pipeline.md)。不要把 PPO 策略交给旧 BC greedy 评估入口；两者的 execution contract 不同。
