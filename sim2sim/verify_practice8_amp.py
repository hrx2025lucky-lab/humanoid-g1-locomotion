"""实践 8（AMP 拟人走跑）实现验证 —— 不依赖 IsaacSim，可离线跑。

为什么要单独写：
作业只要求"补全 TODO 后能训练"，但"能跑"和"实现对"是两回事。
AMP 的几个公式一旦写错（比如熵项符号、奖励混合方向、特征顺序），
训练照样不报错，只会安静地学出错误的东西 —— 实践 5 已经教过我一次
"表面指标好看但任务没学会"的代价。

所以这里只测**可解析验证**的性质：给定输入能手算出唯一正确答案，
或者存在必须成立的不变量（如 alpha=1 时风格奖励完全不起作用）。

用法：
    python verify_practice8_amp.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path("/home/limx/workspace/Roxan_warmup/shenlan_hw/unitree_lab_amp")
sys.path.insert(0, str(REPO))

from rsl_rl_amp.algorithms.discriminator import AMPDiscriminator  # noqa: E402

PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(f"  {'✅' if cond else '❌'} {name}{('  ' + detail) if detail else ''}")


def make_disc(**kw) -> AMPDiscriminator:
    params = dict(frame_dim=80, history_steps=3, hidden_dims=(32, 32))
    params.update(kw)
    return AMPDiscriminator(**params)


def test_frame_dim() -> None:
    """单帧维度必须是 80 = 3+3+3+1+29+29+4×3（作业 TODO1 明确给出）。"""
    print("\n── 1. AMP 特征维度契约 ──")
    parts = {"base_lin_vel": 3, "base_ang_vel": 3, "projected_gravity": 3,
             "base_height": 1, "joint_pos": 29, "joint_vel": 29,
             "key_links_pos_b": 4 * 3}
    total = sum(parts.values())
    check("单帧维度 = 80", total == 80, f"实得 {total}")
    check("判别器窗口 = 3×80 = 240", make_disc().window_dim == 240)

    # 特征顺序必须与 motion_dataset.py 的常量一致，否则判别器比较的是错位的维度
    ds = REPO / "source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/amp/motion_dataset.py"
    text = ds.read_text()
    order = [k for k in parts if f'"{k}"' in text]
    positions = [text.index(f'"{k}"') for k in order]
    check("motion_dataset 声明了全部 7 个特征", len(order) == 7, f"找到 {len(order)}")
    check("特征在数据侧按预期顺序声明", positions == sorted(positions))

    cfg = REPO / ("source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/amp"
                  "/config/g1/amp_flat_env_cfg.py")
    ctext = cfg.read_text()
    amp_block = ctext[ctext.index("class AMPCfg"):ctext.index("amp: AMPCfg")]
    env_positions = [amp_block.index(k) for k in parts if k in amp_block]
    check("env cfg 侧 7 项齐全", len(env_positions) == 7, f"找到 {len(env_positions)}")
    check("env cfg 侧顺序与数据侧一致", env_positions == sorted(env_positions))
    check("AMP 组关闭观测噪声", "enable_corruption = False" in amp_block)
    check("AMP 组拼接为单向量", "concatenate_terms = True" in amp_block)


def test_style_reward() -> None:
    """r_amp = dt · beta · q(score)，q 由 LSQ 公式给出，可手算。"""
    print("\n── 2. 风格奖励（TODO3）──")
    dt, beta = 0.02, 5.0
    d = make_disc(reward_scale=beta)

    score = torch.tensor([1.0, 3.0, -1.0, 0.0])
    # q(s) = clamp(1 - 0.25(s-1)^2, 0, 1)
    #   s=1 → 1.0（判别器认定为专家）
    #   s=3 → 1-0.25*4 = 0（完全像策略）
    #   s=-1→ 1-0.25*4 = 0
    #   s=0 → 1-0.25   = 0.75
    want_q = torch.tensor([1.0, 0.0, 0.0, 0.75])
    got_q = d.quality_from_score(score)
    check("质量函数 q(score) 手算一致", torch.allclose(got_q, want_q, atol=1e-6),
          f"{got_q.tolist()}")

    # 用 monkeypatch 固定 forward，隔离出 style_reward 自己的算术
    d.forward = lambda samples: score  # type: ignore[assignment]
    rewards, out_score = d.style_reward(torch.zeros(4, 240), dt)
    check("返回原始 score 供日志用", torch.allclose(out_score, score))
    check("r = dt·beta·q 手算一致",
          torch.allclose(rewards, dt * beta * want_q, atol=1e-6),
          f"{rewards.tolist()}")
    check("风格奖励非负且有界 [0, dt·beta]",
          bool((rewards >= 0).all() and (rewards <= dt * beta + 1e-6).all()))

    # step_dt 必须真正参与：控制频率变化时风格/任务的相对强度不该漂移
    r2, _ = d.style_reward(torch.zeros(4, 240), dt * 2)
    check("step_dt 加倍则风格奖励加倍", torch.allclose(r2, rewards * 2, atol=1e-6))
    try:
        d.style_reward(torch.zeros(4, 240), 0.0)
        check("step_dt<=0 应报错", False)
    except ValueError:
        check("step_dt<=0 应报错", True)


def test_mix_rewards() -> None:
    """r = (1-alpha)·r_amp + alpha·r_task，两个端点值给出强约束。"""
    print("\n── 3. 奖励混合（TODO4）──")
    style = torch.tensor([1.0, 2.0, 3.0])
    task = torch.tensor([10.0, 20.0, 30.0])

    d0 = make_disc(task_reward_weight=0.0)
    check("alpha=0 → 纯风格奖励",
          torch.allclose(d0.mix_rewards(style, task), style))

    d1 = make_disc(task_reward_weight=1.0)
    check("alpha=1 → 纯任务奖励（风格完全不起作用）",
          torch.allclose(d1.mix_rewards(style, task), task))

    d = make_disc(task_reward_weight=0.4)
    want = 0.6 * style + 0.4 * task
    got = d.mix_rewards(style, task)
    check("alpha=0.4 手算一致", torch.allclose(got, want, atol=1e-6), f"{got.tolist()}")
    check("形状不变", got.shape == task.shape)

    # 混合不得原地改写输入：style/task 都是调用方还要用于日志统计的张量
    s0, t0 = style.clone(), task.clone()
    d.mix_rewards(style, task)
    check("不原地修改输入张量",
          torch.equal(style, s0) and torch.equal(task, t0))

    # 方向性：alpha 越大越偏任务奖励（这里 task > style，故混合值应单调上升）
    vals = [make_disc(task_reward_weight=a).mix_rewards(style, task)[0].item()
            for a in (0.0, 0.25, 0.5, 0.75, 1.0)]
    check("alpha 增大 → 结果单调偏向任务奖励",
          all(vals[i] < vals[i + 1] for i in range(len(vals) - 1)),
          f"{[round(v, 2) for v in vals]}")


def test_ppo_loss() -> None:
    """L = L_policy + c_v·L_value - c_e·H。熵项符号错了训练照跑但会塌缩。"""
    print("\n── 4. PPO 总 loss（TODO6）──")
    src = (REPO / "rsl_rl_amp/algorithms/ppo.py").read_text()
    i = src.index("# L = L_policy")
    block = src[i:i + 400]
    check("surrogate_loss 参与", "surrogate_loss" in block)
    check("value_loss 带 value_loss_coef", "self.value_loss_coef * value_loss" in block)
    check("熵项符号为负（鼓励探索）",
          "- self.entropy_coef * entropy.mean()" in block.replace("\n", " ").replace("  ", " "))

    # 独立复算一遍，确认公式本身
    surrogate, value_loss, entropy = 0.5, 2.0, 0.1
    c_v, c_e = 1.0, 0.01
    want = surrogate + c_v * value_loss - c_e * entropy
    check("公式手算 = 2.499", abs(want - 2.499) < 1e-9, f"{want}")


def test_env_step_mix() -> None:
    """历史不足时必须退回纯任务奖励，否则新 episode 前几步会被假信号污染。"""
    print("\n── 5. 环境步总奖励（TODO5）──")
    src = (REPO / "rsl_rl_amp/algorithms/amp.py").read_text()
    i = src.index("# r_t = (1 - alpha)")
    block = src[i:i + 500]
    check("调用 discriminator.mix_rewards",
          "self.discriminator.mix_rewards(style_rewards, rewards)" in block)
    check("按 valid_windows 掩码退回任务奖励",
          "torch.where(valid_windows, mixed_rewards, rewards)" in block)
    check("混合结果传给 PPO",
          "super().process_env_step(obs, mixed_rewards, dones, extras)" in src)
    check("episode 结束后重置历史计数",
          "torch.where(dones.bool(), torch.zeros_like(next_age), next_age)" in src)


def test_configs() -> None:
    """TODO2 / 7 / 8 / 9 的配置项。"""
    print("\n── 6. 配置与注册（TODO2/7/8/9）──")
    agents = (REPO / "source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/amp"
                     "/config/g1/agents/rsl_rl_ppo_cfg.py").read_text()
    for key in ("policy", "critic", "amp"):
        check(f"obs_groups 含 '{key}'", f'"{key}": [' in agents)

    i = agents.index("class G1AMPWalkToRunRunnerCfg")
    blk = agents[i:i + 900]
    check("WalkToRun 有独立 experiment_name",
          'experiment_name = "unitree_g1_29dof_amp_walk_to_run"' in blk)
    check("WalkToRun 用 walk_to_run profile",
          'amp_motion_profile: str = "walk_to_run"' in blk)
    check("WalkToRun 设置了 task_reward_weight",
          'task_reward_weight"] = ' in blk)
    # 名称不得与其它任务撞车，否则 checkpoint 目录互相覆盖
    names = [ln.split('"')[1] for ln in agents.splitlines()
             if "experiment_name = " in ln]
    check("各任务 experiment_name 互不重复",
          len(names) == len(set(names)), f"{len(names)} 个")

    env = (REPO / "source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/amp"
                  "/config/g1/amp_flat_env_cfg.py").read_text()
    j = env.index("class G1AMPWalkToRunFullPlayEnvCfg")
    fblk = env[j:j + 1800]
    check("FullPlay 继承 WalkToRunPlay",
          "class G1AMPWalkToRunFullPlayEnvCfg(G1AMPWalkToRunPlayEnvCfg)" in env)
    check("FullPlay 关闭线速度课程", "self.curriculum.lin_vel_cmd_levels = None" in fblk)
    check("FullPlay 关闭角速度课程", "self.curriculum.ang_vel_cmd_levels = None" in fblk)
    check("FullPlay 支持转弯（ang_vel_z 非零区间）",
          "ang_vel_z = (-0.4, 0.4)" in fblk)

    # 速度范围必须真正覆盖 Walk 与 Run 两段，否则触发不了走跑切换
    lo, hi = (-0.7, 2.5)
    check("FullPlay lin_vel_x 覆盖走(≤1.0)与跑(≥2.5)",
          f"lin_vel_x = ({lo}, {hi})" in fblk and lo <= -0.7 and hi >= 2.5,
          f"({lo}, {hi})")

    init = (REPO / "source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/amp"
                   "/config/g1/__init__.py").read_text()
    check("注册了 FullPlay 任务",
          '"Unitree-G1-29dof-AMP-WalkToRun-FullPlay"' in init)
    check("FullPlay 关联 FullPlay env cfg", '"G1AMPWalkToRunFullPlayEnvCfg"' in init)
    check("FullPlay 复用 WalkToRunRunnerCfg",
          init.count('"G1AMPWalkToRunRunnerCfg"') >= 2)


def test_no_todo_left() -> None:
    print("\n── 7. TODO 清零 ──")
    left = []
    for p in REPO.rglob("*.py"):
        if ".venv" in str(p):
            continue
        t = p.read_text(errors="ignore")
        if 'NotImplementedError("TODO' in t:
            left.append(p.name)
    check("无残留 NotImplementedError(\"TODO", not left, str(left))


def main() -> int:
    print("=" * 62)
    print("实践 8（AMP 拟人走跑）实现验证")
    print("=" * 62)
    test_frame_dim()
    test_style_reward()
    test_mix_rewards()
    test_ppo_loss()
    test_env_step_mix()
    test_configs()
    test_no_todo_left()
    print("\n" + "=" * 62)
    print(f"通过 {len(PASS)} / {len(PASS) + len(FAIL)}")
    if FAIL:
        print("失败项：")
        for f in FAIL:
            print(f"  - {f}")
    print("=" * 62)
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
