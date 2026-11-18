**项目目标与核心思路总结：**

我们希望基于官方开源的 **SuperMimic** 可微世界模型框架，改造并训练出一个**通用、高鲁棒的 50Hz 实时闭环全身跟踪控制策略（Whole-Body Tracking Controller）**，用于承接 **4 个 RGB 相机无标记动捕 + 单帧逆运动学（IK）重定向** 输出的实时人体参考动作，实现“人给什么动作，机器人在保持全身平衡的前提下就实时柔顺地跟什么动作”。

与原版 SuperMimic 依赖“偷看未来轨迹窗口、离线复现固定动作片段”不同，我们的方案要求**训练阶段 100% 仅使用干净的标准动捕数据集（如 AMASS），而在线推理阶段无需任何手工数据滤波或单段微调，直接实时消化粗糙、不平滑的 4 相机重定向裸流**。为此，我们复用官方 `MultiEncoderAE` 多任务架构，在机制层面集中解决原版在**毛躁数据跟踪、动作泛化性与真机鲁棒性**上的三大短板：

1. **抗毛躁实时伺服（解决粗糙数据抖动、穿地与滑步）**：彻底砍掉未来参考窗口与极易放大毛刺的参考速度项，控制器仅接收**机器人自身历史状态**与**过去几帧因果参考姿态历史**；训练时采用**“脏进净出”机制**——向策略输入端人为注入模拟 4 相机的帧间毛刺、整段高度偏移、滑步与遮挡掉帧，而在 32 步 BPTT 推演与隐空间对齐（`latent_bc`）中始终拿干净原版真值打分，配合动作平滑阻尼约束，让控制器自身形成具备物理惯性的柔顺滤波特性。
2. **零样本任意动作跟随（解决泛化能力不足）**：引入当前显式关节误差输入，利用 `tanh` 锁死有界隐空间防止异常输入越界，并在仿真采样中引入**上下半身跨片段随机混拼**与**在线随机突变目标姿态**，彻底打碎对固定舞蹈套路的背诵，逼迫策略学到底层通用的闭环姿态伺服控制律。
3. **保命优先与零漂移部署（解决真机鲁棒性）**：开启大范围地面摩擦、电机延迟随机化（通过 `sim_context` 明牌喂给世界模型以保梯度精确、对策略隐藏以逼出在线防滑自适应）及随机外力推搡；采用**梯度封顶跟踪损失（`SmoothL1`）配合累积存活概率门控**以及**窗口局部水平原点归零**，确保长时运行无累积漂移，且在遇到相机遮挡跳变或人体超限不可达姿势时严格遵循“站稳保命优先于动作模仿”，最终导出极轻量的单任务 ONNX 模型直连真机 50Hz 运行。

---

结合官方刚开源的 **`HybridRobotics/supermimic`** 仓库（及其配套环境仓库 **`HybridRobotics/mjlab_research`**），我们的整套算法不仅可以无缝落地，而且你会发现：**官方源码里现成内置了一套 `MultiEncoderAE` + `TaskSpec` 架构，简直就像专门为我们的改造预留好插槽一样**！

下面我完全基于官方仓库的真实代码结构，为你重新整理这份**可直接动手改代码的完整算法方案**。

> *注：动手拉代码时有个小坑先提醒你——官方 `README` 里写的克隆地址 `kevinzakka/mjlab_research` 会报不存在，实际公开地址是同组织下的 `git clone https://github.com/HybridRobotics/mjlab_research.git`（与 `supermimic` 放在同一级目录，使用 `uv` 和 Python 3.13 运行）。*

---

## 一、 先摸清官方 `supermimic` 的真实代码架构

打开官方仓库 `src/supermimic/`，它的核心训练流由以下文件组成：
1. **`models.py`（核心惊喜：`MultiEncoderAE` 多编码器策略）**：
   官方把策略网络（Actor）拆成了两截：
   * **前端：任务专属编码器（Goal Encoder）** + 独立的归一化器，输出经过一个 **`tanh` 激活函数**，死死锁在 `[-1, 1]` 的 **32 维有界隐空间**里！（这正好原生实现了我们前面说的“锁死隐变量边界防越界”！）
   * **后端：全任务共享解码器（Shared Decoder，默认宽度 `2048→1024→512→256`）**，接收 `32维隐变量 + 机器人自身历史观测（关节角、角速度、IMU重力与角速度、上一帧动作）`，输出 29 维关节 PD 目标。
2. **`task_spec.py`（任务插槽契约 `TaskSpec`）**：
   官方用 `TaskSpec` 管理不同控制模式（目前已内置了看未来完整动作的 `"motion"` 任务，和只看 3 维速度指令的 `"velocity"` 任务）。
3. **`learners/wm_learner.py`（世界模型学习器 `WorldModelLearner`）**：
   3 层 2048 宽的 MLP，吃 `机器人状态 + 动作 + 仿真物理上下文 sim_context（摩擦、质心偏移、关节零偏）`，在 **8 步自回归窗口**上用名义误差加权损失学习物理演化。
4. **`learners/policy_learner.py` & `envs/multi_task_world_model_env.py`（策略 BPTT 学习器）**：
   冻结世界模型权重，展开 **32 步想象轨迹（Imagined Rollouts）**，乘上预测的**累积存活概率（不摔倒权重）**，加末尾 Critic（`256→128`）的远期价值做 BPTT 反向传播；同时自带一条 **`latent_bc` 辅助损失**，能把其他任务编码器学到的隐特征拉向 `"motion"` 锚点任务。
5. **`exporter.py`（`_SingleEncoderOnnx` 部署导出器）**：
   训练完后，把你指定的某一个编码器 + 共享解码器 + 归一化统计量，单独切出来打包成一个极轻量的单任务 `.onnx` 文件直接上真机！

---

## 二、 结合官方框架的总体算法设计（改哪儿、怎么运作）

我们要训的是一个**“给什么动作就实时跟什么动作”的 50Hz 闭环跟踪控制器**，训练 **100% 只用干净标准动捕库（AMASS / LAFAN1）**，推理直连毛躁的 **4 相机单帧 IK**。

在官方框架上，我们**完全不动 `wm_learner.py`（世界模型）和 `supermimic_runner.py` 的主循环**，只在 **4 个具体位置**植入我们的核心机制：

### 1. 在 `task_spec.py` 中新增任务槽：`make_realtime_4cam_task()`（解决实时跟踪 + 毛躁数据去噪）

利用官方的多任务路由机制，训练时设置 `slot_mix = {"motion": 0.3, "rt_4cam": 0.7}`（30% 样本跑官方原版锚点任务稳固物理底座，70% 样本跑我们的实时 4 相机任务）：

* **输入定义（绝不看未来帧、绝不含参考速度）**：
  截取**过去 $K=8$ 帧到当前第 $t$ 帧**的参考历史（包含：29 维目标关节角 $\tilde{q}$、目标上身重力倾角、骨盆高度、当前帧与机器人的关节角度差 $\tilde{q}_t - q_t$、以及相对于第 $t-7$ 帧重设 `(0,0)` 原点的局部水平位移增量）。
* **`extract_training_goal`（训练时的“毛躁输入生成器”）**：
  从干净动作库取出这 8 帧历史 `g_clean[t-7 : t+1]` 后，**当场复制并弄脏成 `g_noisy` 喂给 `"rt_4cam"` 编码器**：
  1. **模拟单帧 IK 毛刺**：叠加 $\pm 3^\circ$ 关节高斯噪声；
  2. **模拟穿地 / 悬空**：整段 8 帧的骨盆高度统一随机偏移 $\pm 5\text{cm}$；
  3. **模拟原地滑步**：水平位移叠加轻微随机慢漂移；
  4. **模拟相机遮挡跳变与掉帧**：以低概率随机把单个肢体掰歪 $25^\circ$，或以 15% 概率重复上一帧。
* **为什么放进官方框架后，它能学得又快又稳？（双重“脏进净出”机制）**：
  1. **第一重（BPTT 干净目标打分）**：在 `multi_task_world_model_env.py` 的 32 步推演里，环境底层算跟踪 Reward 对比的本来就是**未加噪的干净动作真值 `g_clean[t]`**！策略吃 `g_noisy`，打分对 `g_clean`，天然逼出看 8 帧趋势去噪、无视穿地高度偏差的本领。
  2. **第二重（官方自带 `latent_bc` 蒸馏加速）**：官方 `policy_learner.py` 里的 `_compute_aux_loss` 会自动拿 `"motion"` 锚点编码器（看干净完整动作）算出的理想隐向量 $z_{\text{motion}}$ 当老师，让 `"rt_4cam"` 编码器（只看毛躁过去 8 帧）输出的 $z_{\text{rt\_4cam}}$ 直接向它看齐！

---

### 2. 在 `mjlab_research` 动作采样层打破固定套路（解决泛化能力）

官方原版只在固定剪辑里按难易度重置回放，所以只会“背课文”，换新动作泛化弱。在环境动作库采样（`motion_library` / `commands`）中加入两项随机组合：
1. **上下半身跨片段混拼（Upper/Lower Body Mixing）**：以 50% 概率将片段 A 的上肢/腰部关节目标与片段 B 的下肢迈步目标拼成新目标，覆盖训练集中不存在的手脚组合；
2. **在线随机切换目标（Random Goal Switching）**：在真实仿真交互中，每步以 `2%` 概率（约每 1 秒一次）**不重置机器人物理身体，直接把目标切换到另一个动作帧**——把策略彻底训成“不管当前差多少度，给什么新姿势都能平稳追过去”的伺服控制器。

---

### 3. 在 `mjlab_research` 的奖励与环境配置层加固底线（解决真机鲁棒性与防硬跟摔倒）

官方默认使用 `SuperMimic-GenCoord-G1` 任务配置，评测时关闭了域随机化。针对你的真机实时场景做三处调整：

1. **改进跟踪奖励梯度形态（远距可追赶 + 超限不扳倒）**：
   * 官方原版采用类 BeyondMimic 的纯高斯指数核奖励 $r = \exp(-\|e\|^2 / \sigma^2)$。注意一个小陷阱：当随机切目标或相机遮挡恢复导致当前姿势差了 $40^\circ$ 时，$\exp(-e^2)$ 趋近于 0、导数也趋近于 0（拉不动）；反之若用平方误差 $e^2$，导数无穷大又会把机器人拉摔倒。
   * **改法**：将关节/姿态跟踪项改为 **`0.7 * exp(-||e||²/σ²) + 0.3 * (-SmoothL1(e))`**（近处精准贴合，远处保持恒定温和的封顶牵引力），同时**大幅调低骨盆绝对离地高度 $z$ 的奖励权重**（彻底杜绝因 4 相机高度算低 4cm 而拼命往地里踩）。
2. **加重动作阻尼 + 保留官方“累积存活概率门控”**：
   * 调高相邻动作变化率惩罚 $\|a_t - a_{t-1}\|^2$（形成液压阻尼手感，吸收残余毛刺）；
   * 原样保留官方 `policy_learner.py` 中极其精妙的**累积存活概率加权**（一旦预测某一步有摔倒风险，后续所有跟踪得分权重瞬间归零），从机制上锁死 **“站稳保命永远高于模仿动作”**。
3. **打开 `sim_context` 物理域随机化 + 周期性外力推搡**：
   * 在环境配置中开启地面摩擦系数（`0.4 ~ 1.2`）、机身质心偏移、关节零偏与电机动作延迟随机化（这些参数会自动通过 `sim_context` 明牌喂给 `WorldModelLearner`，而策略靠自身历史观测在线感知打滑）；
   * 开启每 3~5 秒一次的躯干随机速度/外力推搡事件（Push Event），消除策略钻世界模型空子的娇贵毛病。

---

### 4. 用官方 `exporter.py` 一键导出单任务 ONNX 上真机

训练完成后，运行官方 `exporter.py` 中的 `_SingleEncoderOnnx`，指定槽位名称 `"rt_4cam"`：
* 它会自动丢掉世界模型、Critic 和训练用的 `"motion"` 编码器，只把 **`rt_4cam` 编码器（含 `tanh` 边界）+ 共享解码器 + 观测/目标归一化层** 烘焙成一个独立的 `rt_4cam_policy.onnx`（仅几 MB）。

---

## 三、 对接官方代码结构的实现伪代码

### 1. 在 `src/supermimic/task_spec.py` 中注册实时 4 相机任务槽
```python
# 对应官方 src/supermimic/task_spec.py 中的 TaskSpec 契约
def add_4cam_degradation(g_hist_clean: torch.Tensor) -> torch.Tensor:
    """仅在训练提取 goal 时调用：往干净 8 帧历史上撒 4 种模拟相机瑕疵"""
    g = g_hist_clean.clone()                              # [B, K=8, D_g]
    g[..., Q_SLICE]  += torch.randn_like(g[..., Q_SLICE]) * 0.05       # 1. ±3度 IK 毛刺
    g[..., Z_SLICE]  += torch.empty(g.size(0), 1, 1, device=g.device).uniform_(-0.05, 0.05) # 2. ±5cm 穿地/悬空
    g[..., XY_SLICE] += random_walk_drift(g.size(0), steps=8, std=0.01)# 3. 原地滑步慢漂
    g = apply_random_frame_hold_and_limb_spike(g, hold_p=0.15, spike_p=0.02) # 4. 掉帧与遮挡跳变
    return g

def build_causal_goal_tensor(raw_hist_8, current_robot_q):
    """训练与真机推理共用：去房间绝对坐标、不含任何求导速度、拼入当前角度差"""
    # 以这 8 帧起点的水平位置和航向角归零 (防长时程累积漂移)
    local_xy_yaw = align_horizontal_to_first_frame(raw_hist_8.root_pos, raw_hist_8.root_quat)
    q_err_now    = raw_hist_8.joint_pos[:, -1] - current_robot_q       # 当前单帧关节差
    return torch.cat([
        raw_hist_8.joint_pos.flatten(1),     # 过去8帧目标关节角 (8 * 29)
        raw_hist_8.grav_proj.flatten(1),     # 过去8帧目标上身重力倾角 (8 * 3)
        raw_hist_8.root_height.flatten(1),   # 过去8帧目标骨盆高度 (8 * 1)
        local_xy_yaw.flatten(1),             # 过去8帧局部水平位移增量 (8 * 3)
        q_err_now,                           # 显式当前关节误差 (29) -> 提升泛化响应
    ], dim=-1)

def make_realtime_4cam_task() -> TaskSpec:
    return TaskSpec(
        name="rt_4cam",
        goal_dim=8 * (29 + 3 + 1 + 3) + 29,  # 317 维纯位置/角度历史，0 速度项！
        extract_training_goal=lambda env_state, clean_clip_hist_8: build_causal_goal_tensor(
            add_4cam_degradation(clean_clip_hist_8), env_state.joint_pos
        ),
        build_inference_goal_obs=lambda live_4cam_queue_8, robot_q: build_causal_goal_tensor(
            live_4cam_queue_8, robot_q
        ),
    )
```

### 2. 训练启动与 `SuperMimicRunner` 内部自动执行的闭环
配置好任务槽 `tasks = [make_motion_task(), make_realtime_4cam_task()]` 并设置 `slot_mix = {"motion": 0.3, "rt_4cam": 0.7}` 后，直接用官方命令启动训练：
```bash
CUDA_VISIBLE_DEVICES=0 uv run src/supermimic/scripts/train.py \
  SuperMimic-GenCoord-G1 --motion-file /path/to/clean_amass_g1.npz --runner.run_name rt_4cam_tracker
```
在每次迭代中，官方 `SuperMimicRunner` 会自动完成三步闭环：
1. **仿真器交互（`mjlab` / MuJoCo Warp）**：并行跑随机摩擦/延迟、随机外力推搡、上下半身混拼与随机切动作，把真实物理状态转移塞进 `ReplayBuffer`；
2. **`WorldModelLearner.update`**：抽 8 步真实序列，连同 `sim_context` 一起把世界模型更新得更准（此时策略不动）；
3. **`ActorCriticPolicyLearner.update`**：锁死世界模型，展开 32 步想象轨迹——`rt_4cam` 槽位吃进毛躁过去 8 帧，穿过共享解码器和世界模型，拿**干净未加噪的 `AMASS` 目标算饱和跟踪回报 + 动作阻尼平滑 + 累积存活权重**，叠加 `latent_bc` 辅助对齐损失，一次 BPTT 反向传播更新策略网络！

### 3. 真机宇树 G1 50Hz 实时运行（加载 `exporter.py` 导出的单任务 ONNX）
```python
import onnxruntime as ort

# 加载官方 exporter.py 切出的单任务轻量 ONNX (内含归一化层 + rt_4cam编码器 + tanh + 共享解码器)
sess = ort.InferenceSession("rt_4cam_actor.onnx", providers=["CUDAExecutionProvider"])

obs_queue  = RingBuffer(maxlen=history_length) # 机器人自身传感器历史队列
goal_queue = RingBuffer(maxlen=8)              # 4相机单帧粗糙 IK 历史队列

# 线程 1：4相机视觉 + 单帧快速 IK（异步运行，不做任何滤波！）
def on_4cam_frame(rgb_views_4):
    kp_3d = triangulate_4cam(rgb_views_4)
    raw_ik_pose = fast_single_frame_ik(kp_3d)  # 仅求当前单帧关节角与根位姿
    goal_queue.push(raw_ik_pose)

# 线程 2：机器人 50Hz (20ms) 硬实时控制主循环
def control_step_50hz():
    robot_state = g1_robot.read_imu_and_motors()
    obs_queue.push(format_proprioception(robot_state, last_action))
    
    # 组装 317 维目标历史特征 + 本体历史特征，一次推理耗时 ~0.2ms
    goal_obs = build_causal_goal_tensor(goal_queue.get_all(), robot_state.q)
    pd_targets = sess.run(None, {
        "obs_history": obs_queue.get_flat(),
        "goal": goal_obs
    })[0]
    
    g1_robot.send_pd_targets(pd_targets)
```

---

## 四、 总结：为什么基于官方仓库这样改是“最小改动、最大收益”？

1. **零重复造轮子**：物理仿真底座（`mjlab`）、8 步世界模型学习器、32 步可微想象环境、累积防摔存活门控、隐空间 `tanh` 边界、甚至连多任务 `latent_bc` 蒸馏和单任务 ONNX 导出，**全都是官方仓库现成写好的**。
2. **精准根治三大痛点**：
   * **治毛躁数据**：靠 `make_realtime_4cam_task` 的“毛躁 8 帧输入 + 删参考速度 + 干净 BPTT 打分 + 动作阻尼平滑”；
   * **治泛化能力**：靠砍掉未来剧本只做因果伺服 + 环境侧“上下半身混拼 & 在线随机切目标”；
   * **治真机鲁棒性**：靠开启官方 `sim_context` 摩擦/延迟随机化 + 随机外力推搡 + 远距封顶 `SmoothL1` 配合官方存活概率门控。