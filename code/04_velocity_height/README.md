# mjlab · 实践 04 蹲姿行走（速度 + 骨盆高度 MDP）

> **这个文件是环境配置时补建的，不是上游原始内容。**
>
> `pyproject.toml` 第 10 行声明了：
> ```toml
> readme = { file = "README.md", content-type = "text/markdown" }
> ```
> 但上游分发的压缩包里没有 README.md。uv 以可编辑模式安装本包时会调用
> `uv_build.build_editable` 读取该文件，因此报错：
> ```
> × Failed to build `mjlab @ file:///.../hw4_mjlab`
> ╰─▶ Call to `uv_build.build_editable` failed
>     Error: failed to open file `.../README.md`: No such file or directory
> ```
> 补建此文件即可通过构建。**不影响任何功能逻辑**——它只被打包元数据读取。

## 这是什么

mjlab 1.4.0 源码仓库，基于 MuJoCo + MuJoCo-Warp 的 GPU 并行强化学习框架。
与实践 1/2 用的 IsaacLab 是同类工具，但底层物理引擎不同（MuJoCo vs PhysX）。

需实现的要点用 `TODO` 标记，共 10 处。

## 环境说明

同步命令：

```bash
uv sync --extra cu128
```

### 已做的两处环境修复

1. **mujoco 来源从 nightly 改回 PyPI**（`pyproject.toml` 的 `[tool.uv.sources]`）

   原配置 `mujoco = { index = "mujoco" }` 强制从 `py.mujoco.org` 取包，
   而该 index 只保留最近的 nightly 构建。`uv.lock` 里锁定的
   `mujoco==3.8.1.dev907177387` 已被服务器清理，同步时报 404。
   实测该 index 现存最早版本是 3.10.1，已无任何 3.8.x。

2. **在 `constraint-dependencies` 里钉死 `mujoco==3.8.1`**

   仅注释掉 source 还不够：`override-dependencies = ["mujoco>=3.8.0.dev0"]`
   为了接受 dev 版把上界一并解除了，uv 会直接解析到最新的 3.12.0，
   比原 lockfile 跨了四个 minor 版本。钉到 PyPI 正式版 3.8.1 后，
   既贴近原始意图，也满足 `[project.dependencies]` 里 mjlab 自己声明的
   `mujoco~=3.8.0`。`mujoco-warp` 的 git pin 在此版本下解析无冲突。

要还原成原始配置，见同目录的 `pyproject.toml.orig`。

## 自检

```bash
uv run list-envs
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest tests/ -q
```
