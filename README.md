# supermimic_rt：SuperMimic 实时因果跟踪改造

在官方 SuperMimic 上增加 `rt_4cam` 任务槽：策略只看过去 8 帧参考姿态和机器人自身传感器历史，用干净动捕数据训练，训练时向输入端注入模拟 4 相机单帧 IK 的降质，奖励与终止仍按干净参考计算。设计依据是仓库根目录的 `supermimc改造.md`。

本目录目前只有仿真训练与仿真评测。未接入 `g1_teleop`，未做实机验证。

## 目录与来源

| 路径 | 内容 | 来源 |
|---|---|---|
| `supermimic/` | 训练框架，改动在分支 `rt_4cam`（未提交） | github.com/HybridRobotics/supermimic @ `abc124c`（2026-11-10），Apache-2.0 |
| `mjlab_research/` | 仿真环境，分支 `rt_4cam`，目前无改动 | github.com/HybridRobotics/mjlab_research @ `dd51a9f`（2026-11-10） |
| `data/motions_raw/lafan1_g1_csv/` | 40 段 LAFAN1→G1 重定向 CSV，30 Hz | huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset @ `ce15729`，`g1/` |
| `data/motions/lafan1_g1_50hz/` | 上面转成的 mjlab 动作文件，40 段共 441 140 帧（147 分钟） | `tools/convert_csv_motions.py` |
| `data/motions_raw/mjlab_demo/` | mjlab 官方演示片段，sha256 与官方一致 | storage.googleapis.com/mjlab_beta；旧格式，当前 mjlab_research 读不了 |
| `tools/` | 数据转换、ONNX 校验、评测矩阵与汇总脚本 | 本项目 |
| `outputs/videos/` | 闭环评测视频 | `scripts/evaluate.py --video-file` |
| `REPORT.md` | 首轮验收结果（含与官方配置的对比和负结果） | 本项目 |

`data/`、`outputs/`、`supermimic/logs/`、`supermimic/wandb/`、`.venv/` 不入库。`supermimic/`（基于 `HybridRobotics/supermimic @ abc124c` 并包含 `rt_4cam` 改动）和 `mjlab_research/`（基于 `HybridRobotics/mjlab_research @ dd51a9f`）随本仓库一并托管，保留各自上游 `LICENSE`。

官方默认动作数据在 W&B 私有 registry（`supermimic/wandb-registry-motions/...`），本机无凭据，未下载。AMASS 需要注册并接受许可，未下载。

## 环境

Python 3.13 + uv。官方 `supermimic/uv.lock` 比公开的 `mjlab_research` 旧（mjlab 1.5.0 对 1.6.0），`uv sync` 会重新解析；不加约束会装上 `mjviser 0.0.17`，它新依赖的 `mujoco-core 4.0.1` 与 `mujoco 3.16.0` 写进同一个目录，`import mujoco` 直接失败。`supermimic/pyproject.toml` 里已把 `mjviser`、`torch`、`warp-lang` 钉到 `mjlab_research/uv.lock` 的版本。

```bash
cd supermimic_rt/supermimic
uv sync --group dev
uv run pytest tests -q
```

## 命令

都在 `supermimic_rt/supermimic/` 下执行。没有 W&B 账号时加 `WANDB_MODE=offline`。

```bash
# CSV → 50 Hz 动作文件
uv run python -I ../tools/convert_csv_motions.py \
  --input-dir ../data/motions_raw/lafan1_g1_csv --output-dir ../data/motions/lafan1_g1_50hz

# 训练（新任务 ID；默认 motion 0.3 / rt_4cam 0.7，proprio 传感器历史 8 帧）
WANDB_MODE=offline uv run src/supermimic/scripts/train.py SuperMimic-RT4Cam-G1 \
  --motion-file "../data/motions/lafan1_g1_50hz/walk*.npz" --runner.run_name <name>

# 仿真闭环评测（本地检查点，不需要 W&B）
WANDB_MODE=offline uv run src/supermimic/scripts/evaluate.py \
  --run-dir logs/SuperMimic-RT4Cam-G1/<时间戳> \
  --motion-file ../data/motions/lafan1_g1_50hz/walk1_subject2.npz \
  --slot rt_4cam --reference degraded
../tools/eval_matrix.sh logs/<task>/<时间戳> <motion.npz> ...   # 所有槽位 × 干净/降质
uv run python -I ../tools/summarize_evals.py logs/<task>/<时间戳> ...   # 汇总成表

# 评测的其他开关：--noise-scale 3（降质幅度倍率）、--shuffle-goal True（对照：
# 打乱目标，给出"不跟踪"的误差地板）、--video-file out.mp4（配 --num-envs 2，需 MUJOCO_GL=egl）

# 校验导出的 ONNX 与检查点一致
uv run python ../tools/check_rt4cam_onnx.py --run-dir logs/SuperMimic-RT4Cam-G1/<时间戳>
```

官方任务 `SuperMimic-GenCoord-G1` / `SuperMimic-MaxCoord-G1` 的配置和默认行为没有改，用作对照基线。

## `rt_4cam` 槽位契约

- 输入（部署时）：最近 `K=8` 个控制周期（50 Hz）的参考帧，最新在后。每帧为 `joint_pos (29,) rad`（机器人关节顺序）、参考骨盆 `root_pos (3,) m`、`root_quat (4,) wxyz`，坐标系为任意重力对齐、z 向上的世界系；外加机器人实测 `joint_pos (29,)`。
- 输出：317 维目标向量，`[参考关节角 8×29 | 参考骨盆重力投影 8×3 | 参考骨盆高度 8×1 | 相对窗口最旧帧的 (x, y, yaw) 8×3 | 最新参考关节角 − 实测关节角 29]`。不含未来帧，不含任何速度。
- 水平位置和朝向只以窗口内增量出现，所以参考的绝对水平位置和朝向对策略不可观测：相对参考的水平漂移不会被纠正（不是"零漂移"）。
- 缺帧必须由上游重复上一帧填充（训练里的掉帧模型就是这样）。空窗口不是合法输入。
- 训练和部署共用 `supermimic/tasks/rt_4cam.py::build_causal_goal`。ONNX 为 `policy_rt_4cam.onnx`，输入 `obs_history`（term-major 传感器历史）和 `goal`，输出 `actions`；窗口长度和列布局写在模型元数据里。

训练时的降质（`Rt4CamTaskCfg`，可用 `--runner.tasks.rt_4cam.*` 覆盖）：逐帧关节高斯噪声 0.05 rad；整段回合固定的骨盆高度偏置 ±5 cm；水平位置随机游走 1 cm/帧；每帧 15% 概率掉帧重复上一帧；每帧 2% 概率某一肢体整体偏 25°，之后每帧 50% 概率持续。噪声是因果且持久的：一帧进入窗口时带的污染，在它滑出窗口前不变。

## 改动清单

`supermimic/`（相对 `abc124c`）：

- `tasks/rt_4cam.py`（新）：目标向量、降质流 `Rt4CamStream`、配置。
- `tasks/task_spec.py`：`TaskSpec` 增加可选的有状态钩子 `init_goal_state` / `advance_goal_state` 和 `GoalContext`；`make_realtime_4cam_task`；`TasksCfg.rt_4cam`。
- `envs/world_model_env.py`、`envs/multi_task_world_model_env.py`：想象环境里按回合重抽、逐步推进降质流；目标在合成传感器之后组装。
- `runners/supermimic_runner.py`：真实仿真采集端的降质流状态；`evaluate()` 闭环评测；`run_config.json`；奖励配置从环境奖励项读取。
- `utils/tracking_reward.py`：`TrackingRewardCfg`（默认值等于官方常数）和带远场斜坡的 `tracking_kernel`。
- `tasks/env_cfgs.py`、`tasks/__init__.py`：`rt4cam_env_cfg` 和任务 ID `SuperMimic-RT4Cam-G1`。
- `runners/exporter.py`：`rt_4cam` 导出；超大数据集只跳过内嵌动作片段的图。
- `runners/run_config.py`、`scripts/train.py`、`scripts/evaluate.py`（新）。
- `tests/test_rt_4cam.py`、`tests/test_tracking_kernel.py`（新，27 项）。

`SuperMimic-RT4Cam-G1` 相对官方 gencoord 环境的差别（数值均为待验证假设，常量在 `env_cfgs.py`）：四个位姿跟踪项用 `0.7·高斯 + 0.3·远场斜坡`；骨盆高度误差平方乘 0.25；动作变化率惩罚 −0.1 → −0.5；保留基础任务的速度推搡，间隔 3–5 s（仅训练）。

## 与 `supermimc改造.md` 不一致的地方

1. **`-SmoothL1` 没有按字面实现。** 带负号的无界项会让单步奖励为负，而官方回报是 `累积存活概率 × 奖励`：远离目标时"结束回合"比"活着"回报更高，梯度会推着策略去摔倒，与"保命优先"相反。改为截断到 0 的斜坡 `max(0, 1 − huber(x)/huber(4))`：4 个标准差以内保持恒定有界的牵引斜率，奖励不为负，单步奖励上限不变。
2. **`latent_bc` 在 `models.py::MultiEncoderAE._compute_aux_loss`**，不在 `policy_learner.py`；机制与文档描述一致，未改。
3. **官方 gencoord 训练环境本来就开着摩擦（0.3–1.2）、躯干质心、编码器零偏随机化**，关掉的只有推搡。本改造把推搡加回来。推搡是世界模型看不到的外部输入，会成为它回归目标里的离群点；影响未测。
4. **电机延迟随机化未做。** 世界模型的输入只有当前状态和当前动作，延迟后的动作对它是不可观测输入；要做需要改世界模型输入，超出"不动世界模型"的前提。
5. **"调低骨盆高度奖励权重"已按文档实现，但理由不成立**：奖励对比的是干净参考的高度，输入端的高度偏置不会让奖励去追错误高度。保留为可调项。
6. **导出的 ONNX 约 31 MB**，不是"几 MB"：官方默认解码器 2048→1024→512→256，加 8 帧传感器历史。
7. **上下半身混拼、在线切换目标未实现**（文档第二部分）。按字面"直接把目标切到另一帧"不可行：参考根位置会瞬间离机器人数米，全局根位置/朝向两项奖励永远够不着。要做需要给参考加一个逐环境的水平重锚定（xy + yaw），贯穿 `MotionCommand`、回放缓冲和想象环境；混拼还需要第二个时间索引。

## 验证状态

见 `REPORT.md`。
