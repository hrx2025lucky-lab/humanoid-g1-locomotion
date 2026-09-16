# 实践9候选提交包（2026-09-10）

作者：黄若轩。包含 P1 MJLab 路线实现、原版测试真实终端截图、P2 七条 TensorBoard 曲线截图及 29.8 秒连续回放。尚未上传课程平台或公开仓库。

## P1 实现与修复

P1_commands_todo.py 来自 bymic_commands_mjlab_todo.py。修复整除 bin_width 导致尾段永远无法被采样的问题，改为按浮点相位映射到 [0, T-1]，最后转整数；与当前安装的 MJLab 参考实现保持一致。原课程 MJLab/IsaacLab 两份测试各 3 项通过，新增边界回归 4 项（均覆盖两条实现）通过。截图针对包内同一份 P1 文件，实际退出码 0。此修复属于 P1 项目函数；P2 实际训练的 MJLab 采样器本来就使用浮点相位，因此不需要为这个 P1 修复重训 P2。

## P2 训练来源与差异

P2 使用本地 hw6_distill 的 MJLab teacher 路线，最终模型 model_29999.pt；属于课程目标的工程适配，不能描述为原封不动运行课程包。实际训练有未来参考观测、历史观测和残差动作，采样分布含 80% 均匀成分。相位采样日志细分为 sampling_phase_entropy、sampling_phase_top1_bin，不能与 P1 单层 bin 的原始标签数值直接等同。

P2_scalars_all.png 是本机 TensorBoard 真实截图，橙色 initial 为 0–7065，蓝色 resume 为 7000–29999；保留恢复训练时的瞬态。页面平滑为 0.6，纵轴自动忽略极端点，原始事件没有修改。七条展示项是 episode length、body position reward、time_out、joint position error、phase entropy、phase top1 bin、policy std。sampling_phase_top1_bin 是归一化相位指标，不是整数 bin 编号。time_out 为框架记录项，不能当成 0–1 成功率。

## 回放与验收边界

P2_play.mp4 是完整回放的前 29.8 秒，25 fps、640×480，正常速度、连续片段，未删除片段内失败。完整证据 p9_final_full_2026-09-10 保留 6574 帧参考序列（约 131.46 秒）。单环境 seed 42 整段回放中，除初始化缓存帧外，相对身体位置平均误差约 3.80 cm；全局锚点位置平均误差约 1.048 m、最大 3.294 m。可以支持相对动作跟踪效果较好，不能支持高精度全局路径跟踪或真机验证。

P1_commands_test_output.png 已目视核对为真实测试终端，三个测试通过；P2_scalars_all.png 已目视核对七条曲线均到最终训练阶段；P2_play.mp4 已完整解码且时长 29.8 秒。包中不重复放入大型 checkpoint、课程 PDF 和全部 logs。

## 本地证据索引

详细材料审查、训练原始标签、独立动作数据核对见 evidence/；具体本地路径与哈希见 manifest.json。若课程要求 P2 必须采用指定原版入口，应提交时明确上述适配差异，再按课程口径处理；不能将适配说明省略后声称完全同版。
