# 实践 07 · GMR 人体动作到 G1 的重定向 — 我的改动清单

对照基准：上游 `YanjieZe/GMR`

本目录只收录**我自己写的和改的部分**，不含 GMR 上游全量代码。

## 我新增的文件

- `code/smplx_to_robot_npz.py` — SMPL-X 序列到 G1 的单段重定向导出
- `code/smplx_to_robot_dataset_npz.py` — 批量数据集重定向
- `code/build_segments.py` — 从完整重定向产物裁出走 / 跑 / 转弯片段，
  并生成来源哈希与 provenance 记录
- `code/motion_cfg.py` — 片段的采样权重配置，供实践 08 的 AMP 加载器读取

## 我修改的上游文件

- `general_motion_retargeting/motion_retarget.py` — 重定向主流程
- `general_motion_retargeting/utils/smpl.py` — 适配 SMPL-X pkl 模型
- `scripts/vis_robot_motion.py` — 加 `--loops`，录像时默认只播一遍
- `setup.py` — 固定依赖版本

## 与实践 08 的衔接

本实践输出的 `.npz` 必须能被实践 08 的 `motion_dataset.py` 直接加载，
四元数顺序为 `wxyz`。校验结果见 `evidence/amp_loader_verification.json`。
