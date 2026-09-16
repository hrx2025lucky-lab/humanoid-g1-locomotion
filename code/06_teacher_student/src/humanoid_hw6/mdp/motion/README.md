# Motion Commands

Internal implementation of reference-motion commands used by Humanoid HW6:

- `JointRefAnchorRpMotionCommand` — single-step student command
- `FutureJointRefAnchorRpMotionCommand` — teacher future-stacked command
- `TeacherStudentJointRefAnchorRpMotionCommand` — dual student/teacher views for distillation

Representations live in `representations.py`. Adaptive clip/phase resampling lives in `sampling.py`.
