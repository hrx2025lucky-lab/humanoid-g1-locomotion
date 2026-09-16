## sim2sim

### 1. sim2sim 目标

本次 sim2sim 的目标是把 IsaacSim 中训练得到的 G1 parkour policy 放到 MuJoCo 中运行。由于两个仿真器的模型、传感器和数据接口不同，sim2sim 代码需要显式处理以下一致性问题：

- 关节顺序一致：MuJoCo/SDK 顺序和 policy/IsaacSim 顺序不同，需要用索引映射。
- 关节符号一致：部分关节在 MuJoCo 模型和训练模型中的正方向可能不同，需要用 `JOINT_SIGNS` 修正。
- 默认关节角一致：policy 输出的是相对默认姿态的动作，需要加上 `DEFAULT_JOINT_POS`。
- 动作尺度一致：policy 输出需要乘以 `ACTION_SCALES`。
- PD 控制参数一致：MuJoCo 中需要使用训练或部署配置对应的 stiffness、damping 和 torque limit。
- 观测顺序一致：本体观测、历史观测和深度图特征必须按照训练时的顺序拼接。
- 深度图预处理一致：resize、crop、inpaint、blur、clip 和归一化必须与训练或部署时保持一致。

### 2. 安装 sim2sim 依赖

可以继续使用训练时的 conda 环境。激活环境并进入仓库：

```Bash
conda activate env_isaaclab
cd /path/to/instinctlab
```

前面执行 `python -m pip install -e source/instinctlab` 时已经安装了 NumPy、OpenCV 等项目依赖。如果当前环境尚未安装 MuJoCo，执行：

```Bash
python -m pip install mujoco
```

如果已经安装过 MuJoCo，可以跳过这一步。然后安装 ONNX Runtime：

```Bash
python -m pip install onnxruntime
```

使用以下命令检查依赖是否安装成功：

```Bash
python -c "import mujoco, onnxruntime; print('MuJoCo:', mujoco.__version__)"
```

### 3. sim2sim 代码结构与关键实现说明

`sim2sim/sim2sim.py` 中主要类如下：

- `RingBuffer`：保存本体观测历史。
- `DepthImagePipeline`：保存深度图历史，并完成深度图预处理。
- `Sim2simInstance`：封装 MuJoCo 环境、ONNX policy、观测构造、动作解码和 PD 控制。

从执行流程上看，`sim2sim.py` 可以概括为四步：观测获取、策略推理、动作解码、环境推进。代码主循环正是按这个顺序运行：先更新观测，再调用 ONNX policy 推理，接着将策略输出转换为 MuJoCo 的控制量，最后通过多次物理步推进仿真状态。

这四步构成了完整的 RL 闭环（observation → action → next_observation），总结如下：

| 步骤 | 做什么 | 产物 |
|------|--------|------|
| ① 观测获取 | 读 sensordata，计算 projected_gravity、joint_pos_rel 等，写入历史 buffer | 构造好的 observation vector |
| ② 策略推理 | depth encoder 编码深度图特征 + actor 输出 raw_action | 29 维归一化动作 |
| ③ 动作输出 | 关节顺序映射 × `action_scale` + `default_joint_pos` × `joint_signs`，再经 PD 控制生成力矩 | 各关节目标力矩 τ |
| ④ 环境推进 | `mj_step()` × `DECIMATION` 次，更新 qpos、qvel、sensordata | 下一帧的仿真状态（回到步骤①） |

#### 3.1 第一步：观测获取

`Sim2simInstance.update_observation()` 中构造的本体观测包括：

```Plaintext
base_angle_vel
projected_gravity
velocity_commands
joint_pos_rel
joint_vel_rel
last_action
```

每个观测项会写入历史 buffer，随后在 `get_history_obs()` 中展开并拼接。本体观测之外，深度图也会维护独立的历史队列，并在后续送入 depth encoder。

在这一步中，需要特别注意以下几个关键实现点。

`projected_gravity` 是世界系重力方向投影到机身坐标系的结果。sim2sim 中通过 IMU 四元数和 `quat_apply_inverse()` 得到：

```Python
projected_gravity = quat_apply_inverse(
    root_quat, np.array([0, 0, -1], dtype=np.float32)
)
```

这里需要注意四元数格式。当前代码注释说明使用的是 `wxyz` 格式。

`last_action` 必须使用 policy 顺序保存，并在下一帧作为观测输入。它表示上一时刻 actor 输出的原始动作，用来帮助策略感知自身刚刚做过什么，从而保持动作连续和平滑。

`DepthImagePipeline` 中不是直接使用最新一帧深度图，而是维护一个长度为 60 的队列，并从中抽取 8 帧作为输入：

```Python
self.indices = np.linspace(
    -1 - down_sample_factor * (self.nb_frames - 1), -1, self.nb_frames
).astype(np.int32)
```

训练侧在 `parkour_env_cfg.py` 中配置了 `delayed_frame_ranges=(0, 1)`，即每次获取深度图观测时，在 [0, 1] 帧范围内随机延迟，模拟真实传感器延迟。sim2sim 中 `DepthImagePipeline` 通过以下方式实现相同效果：

- 每帧 `update()` 处理完后，调用 `resample_delay()` 在 `delay_ranges` 范围内随机采样一个延迟值 `self.delay`。
- `get_depth_obs()` 从历史队列取帧时，将索引减去 `self.delay`，即取更早的历史帧，模拟延迟效果。

对应参数 `delay_ranges` 默认为 `(0, 1)`，与训练侧一致；设为 `(0, 0)` 可关闭延迟。

#### 3.2 第二步：策略推理

`get_history_obs()` 会先将各个本体观测历史展开并拼接，再将深度图历史送入 depth encoder 提取视觉特征，最后把两部分拼接成 actor 的输入。也就是说，actor 并不是直接接收原始深度图，而是接收“本体历史观测 + 深度编码特征”的组合输入。

这一阶段可以概括为：

- `RingBuffer` 提供最近若干帧的本体观测历史。
- `DepthImagePipeline` 提供经过预处理、带有历史和延迟效果的深度图输入。
- `depth_encoder` 将深度图历史编码为视觉特征。
- `actor` 根据完整历史观测输出当前时刻动作 `raw_action`。

#### 3.3 第三步：动作输出

策略输出的 `raw_action` 仍然是 policy 顺序下的归一化动作，并不能直接作为 MuJoCo 控制输入。sim2sim 需要先把它变换回 robot 顺序，再结合动作缩放、默认关节位置和 PD 控制生成最终力矩命令。

首先需要处理关节顺序映射。

`config.py` 中同时定义了：

```Python
UNITREE_G1_29DOF_SDK_JOINT_NAMES
POLICY_JOINT_NAMES
```

MuJoCo 传感器读出的关节顺序使用 SDK/robot 顺序，而 policy 输入和输出使用 IsaacSim 训练顺序。因此代码中需要两组映射：

```Python
self.robot_to_policy = self.compute_joint_indices(robot_joint_name, policy_joint_name)
self.policy_to_robot = self.compute_joint_indices(policy_joint_name, robot_joint_name)
```

使用方式：

- 构造 `joint_pos_rel` 和 `joint_vel_rel` 时，从 robot 顺序转成 policy 顺序。
- 将 policy 输出动作下发到 MuJoCo 前，从 policy 顺序转回 robot 顺序。

在完成顺序变换后，策略输出会先经过 `action_scale` 和 `default_joint_pos` 还原为目标关节位置，再结合 `joint_signs` 映射到 MuJoCo 侧的关节定义。随后，sim2sim 不直接输出目标位置，而是通过 PD 控制生成力矩：

- 比例项根据目标位置和当前关节位置的误差计算。
- 微分项根据当前关节速度提供阻尼。
- 最终力矩经过 `torque_limit` 截断后写入 `self.sim_env.model_data.ctrl[:]`。

此外，代码中还使用 `DECIMATION`，即一次策略输出对应多个 MuJoCo 物理步。这使得策略控制频率与底层仿真积分频率解耦，更接近训练时的控制设置。

#### 3.4 第四步：环境推进

当第三步已经得到当前控制周期的关节力矩 `tau` 后，sim2sim 并不会立刻回到策略网络重新推理，而是先让 MuJoCo 在这组控制输入下连续推进若干个物理步：

```Python
for _ in range(decimation):
    tau = (
        self.stiffness * (action - self.sim_env.model_data.sensordata[:29])
        - self.damping * self.sim_env.model_data.sensordata[29:58]
    )
    tau = np.clip(tau, a_min=-self.torque_limit, a_max=self.torque_limit)
    self.sim_env.model_data.ctrl[:] = tau
    self.sim_env.physical_step()
```

这里的 `self.sim_env.physical_step()` 可以理解为执行一次 MuJoCo 的底层积分更新。每调用一次，仿真器都会根据当前关节力矩、接触、重力和系统动力学，更新机器人状态，包括：

- 关节位置 `qpos`
- 关节速度 `qvel`
- IMU、关节传感器等对应的 `sensordata`
- 刚体位姿、碰撞接触和下一时刻的动力学状态

因此，MuJoCo 中“环境如何变化”并不是额外写一套状态转移逻辑，而是通过反复调用物理步函数，让仿真器自己根据动力学方程推进系统。这正对应 RL 闭环中的状态转移过程：

```Plaintext
observation_t -> action_t -> DECIMATION x MuJoCo physical_step() -> observation_{t+1}
```

第四步的作用，就是把策略给出的动作真正施加到仿真环境中，并得到下一时刻的状态。

这里的 `DECIMATION` 表示：策略每输出一次动作，不是只执行一个 MuJoCo 物理步，而是会连续执行多个物理步。比如当 `SIM_DT = 0.005`、`DECIMATION = 4` 时，MuJoCo 每个物理步前进 `0.005 s`，而同一个动作会连续执行 4 个物理步，也就是总共推进 `0.02 s`。因此，底层物理仿真频率是 `200 Hz`，策略控制频率是 `50 Hz`。

这样设计的好处是：

- 物理积分可以保持较高频率，接触和动力学更稳定。
- 策略不需要每个最小积分步都重新推理，计算量更低。
- 控制频率更容易对齐训练环境中的设置，减少 sim-to-sim 偏差。

完成这若干个物理步之后，代码会调用 `viewer.sync()` 刷新可视化，并通过：

```Python
time_to_sleep = SIM_DT * DECIMATION - (time.perf_counter() - time_start)
```

尽量让每个控制周期的墙钟时间接近 `SIM_DT * DECIMATION`。随后程序回到第一步，重新读取新的传感器数据，构造下一帧观测，形成完整闭环。


### 4. 需要补全的内容

本次项目要求补全：

```Plaintext
/path/to/instinctlab/sim2sim/sim2sim.py
```

重点补全 `DepthImagePipeline.update()` 中的深度图预处理流程。依据下面的功能要求、`config.py` 中的参数，以及 OpenCV API 自行补全。

`DepthImagePipeline.update()` 按照 `resize → crop → inpaint → Gaussian blur → clip → normalize → append` 的顺序处理深度图，需要补全以下内容：

1. **调整图像尺寸**
   - `self.shape` 按 `(height, width)` 保存，因此先解包为 `h, w`。
   - 调用 `cv2.resize` 时，`dsize` 的顺序是 `(width, height)`，即 `(w, h)`。
   - 插值方式 `interpolation` 为 `cv2.INTER_NEAREST`（最近邻插值）。
   - resize 后 NumPy 图像的形状应为 `(h, w)`。

2. **裁剪图像**
   - `self.crop_region` 的顺序为 `(y1, y2, x1, x2)`，分别表示从上、下、左、右裁掉的像素数。
   - NumPy 二维图像按 `[行, 列]` 索引，因此高度方向使用 `y1 : h - y2`，宽度方向使用 `x1 : w - x2`。

3. **修补无效深度**
   - 使用 `(image < 0.2)` 找出过近或无效的深度像素，并转换为 `np.uint8` mask。
   - 调用 `cv2.inpaint` 时传入深度图、mask、修补半径 `3` 和算法 `cv2.INPAINT_NS`。
   - mask 中非零位置是需要根据周围有效像素进行修补的区域。

4. **平滑深度图**
   - 使用 `cv2.GaussianBlur` 降低深度噪声。
   - 高斯核大小设为 `(3, 3)`，`sigmaX=1`，`sigmaY=1`。
   - 处理后的图像尺寸和数据类型应与输入保持一致。

5. **裁剪并归一化深度值**
   - 使用 `np.clip` 将深度限制在 `[self.near_clip, self.far_clip]`。
   - 使用线性归一化将深度映射到 `[0, 1]`：

   ```Python
   normalized = (image - near_clip) / (far_clip - near_clip)
   ```

6. **写入历史队列**
   - 将处理后的 `np.float32` 深度图写入 `self.que`，供 `get_depth_obs()` 按时间索引抽取。

相关 OpenCV 官方教程：

- [`cv2.resize`：Image Resizing](https://docs.opencv.org/4.x/da/d6e/tutorial_py_geometric_transformations.html)
- [`cv2.inpaint`：Image Inpainting](https://docs.opencv.org/4.x/df/d3d/tutorial_py_inpainting.html)
- [`cv2.GaussianBlur`：Smoothing Images](https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html)

补全后，`DepthImagePipeline.update()` 的输出尺寸应与 `self.obs_depth_image_shape` 一致。根据当前配置：

```Plaintext
RESIZED_DEPTH_IMAGE_SHAPE = (36, 64)  # (高度, 宽度)
OBS_DEPTH_IMAGE_CROP_REGION = (18, 0, 16, 16)  # (上, 下, 左, 右)
```

裁剪后的单帧深度图尺寸为：

```Plaintext
(36 - 18 - 0) x (64 - 16 - 16) = 18 x 32
```

depth pipeline 会从历史图像中采样 `nb_frames=8` 帧，因此送入 depth encoder 的形状应为：

```Plaintext
(1, 8, 18, 32)
```

### 5. 运行 sim2sim

进入 sim2sim 目录：

```Bash
cd /path/to/instinctlab/sim2sim
```

运行 parkour policy：

```Bash
python sim2sim.py --task parkour
```

运行站立 policy：

```Bash
python sim2sim.py --task stand
```

parkour 模式支持方向键或数字小键盘增量调整速度命令：

```Plaintext
↑ / 8：增加前向速度
↓ / 2：减小前向速度
← / 4：增加左转角速度
→ / 6：增加右转角速度
空格 / 5：速度命令归零
R：重置机器人、速度命令、上一帧动作和观测历史
```

每次按键分别按 `COMMAND_STEP` 增减速度，并使用 `COMMAND_RANGES` 限制命令范围。如果数字小键盘没有响应，请确认键盘 NumLock 状态，也可以直接使用方向键。