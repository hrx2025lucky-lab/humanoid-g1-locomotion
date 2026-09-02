#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# 编译 MuJoCo raycaster 传感器插件（实践 2 sim2sim 的硬依赖）
# ═══════════════════════════════════════════════════════════════════════════
#
# sim2sim 侧的 height_scanner 由 MuJoCo 插件 mujoco.sensor.ray_caster 提供，
# 该插件必须在 MuJoCo **源码树内**编译，且版本与 Python 运行时严格一致，
# 否则报 "plugin mujoco.sensor.ray_caster not found" 或 ABI 不兼容。
#
# 插件（Albusgive/mujoco_ray_caster，最后更新 2026-04-27）是按旧版 MuJoCo 写的，
# 与 3.12.0 之间有三处上游改动需要适配，本脚本一并处理：
#
#   ① mjtnum.h 被改名为 mjtype.h          → 加转发头，插件源码不动
#   ② mjthread.h 与旧线程 API 整体删除     → 加串行语义兼容层，插件源码不动
#   ③ mjPLUGIN_LIB_INIT 变成带参宏         → 只能改插件源码（1 处）
#
# 另外本机 gcc 11.4 对 MuJoCo 自身碰撞代码报 -Wuninitialized 误报，
# 被 -Werror 升级成错误，需要放宽为 -Wno-error。
#
# 用法：
#   ./build_raycaster_plugin.sh          # 全流程（打补丁 → 配置 → 编译 → 验证）
#   ./build_raycaster_plugin.sh verify   # 只验证已有产物能否被 Python 加载
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail

ROOT="${ROXAN_ROOT:-/home/limx/workspace/Roxan_warmup}"
MJ_SRC="$ROOT/repos/mujoco_src"
PLUGIN_SRC="$ROOT/repos/mujoco_ray_caster"
PLUGIN_DST="$MJ_SRC/plugin/mujoco_ray_caster"
PY="${ROXAN_PYTHON:-$ROOT/envs/isaaclab/bin/python}"
SCENE="$ROOT/shenlan_hw/hw2_sim2sim/sim2sim/assets/scene_rough.xml"
SO="$MJ_SRC/build/lib/libsensor_raycaster.so"
JOBS="${JOBS:-12}"          # 训练同时在跑，别把 28 核吃满

# ── 版本一致性：编译用的源码版本必须等于 Python 运行时版本 ────────────────
check_versions() {
  local src_ver rt_ver
  src_ver=$(grep -oP 'VERSION \K3\.[0-9]+\.[0-9]+' "$MJ_SRC/CMakeLists.txt" | head -1)
  rt_ver=$("$PY" -c "import mujoco; print(mujoco.__version__)")
  echo "  mujoco_src   : $src_ver"
  echo "  python 运行时: $rt_ver"
  [[ "$src_ver" == "$rt_ver" ]] || {
    echo "❌ 版本不一致，插件会 ABI 不兼容。请让两者对齐后重试。" >&2; exit 1; }
  echo "  ✅ 一致"
}

# ── ① mjtnum.h → mjtype.h 转发头 ─────────────────────────────────────────
patch_mjtnum() {
  local f="$MJ_SRC/include/mujoco/mjtnum.h"
  [[ -f "$f" ]] && { echo "  ⏭  mjtnum.h 转发头已存在"; return; }
  cat > "$f" <<'EOF'
// 兼容性转发头（非上游文件）。
// MuJoCo 3.12 把 <mujoco/mjtnum.h> 改名为 <mujoco/mjtype.h>，mjtNum / mjMINVAL
// 等定义整体搬了过去。mujoco_ray_caster 有 12 处仍 include 旧路径，
// 加转发头即可，无需改插件源码。mjtype.h 是 mjtnum.h 的超集，转发安全。
#ifndef MUJOCO_INCLUDE_MJTNUM_H_
#define MUJOCO_INCLUDE_MJTNUM_H_
#include <mujoco/mjtype.h>
#endif  // MUJOCO_INCLUDE_MJTNUM_H_
EOF
  echo "  ✅ 写入 mjtnum.h 转发头"
}

# ── ② mjthread.h 兼容层（串行语义）────────────────────────────────────────
patch_mjthread() {
  local f="$MJ_SRC/include/mujoco/mjthread.h"
  [[ -f "$f" ]] && { echo "  ⏭  mjthread.h 兼容层已存在"; return; }
  cat > "$f" <<'EOF'
// 兼容性补齐头（非上游文件）。
//
// MuJoCo 3.12 删除了 mjthread.h 与整套旧引擎线程 API（changelog:
// "The header file ``mjthread.h`` was removed along with the old engine
// threading API"）。新的 mju_threadpool 只能设置引擎内部工作线程数，
// 不再向插件暴露通用任务队列，因此无法 1:1 迁移。
//
// mujoco_ray_caster 用它做射线分段并行：把 [0,nray) 切成互不相交的区间，
// 每段一个 task 入队，主线程算首段，然后逐个 join。各 task 之间无共享写，
// 所以「入队时同步执行」与「并行执行后 join」语义完全等价，只是变成串行。
//
// 本项目 height_scanner 是 17×11 = 187 根射线，串行开销可忽略，
// 换来零并发风险且插件源码零改动。若将来需要真并行，
// 把 mju_threadPoolEnqueue 换成 std::thread 实现即可，接口不变。
#ifndef MUJOCO_INCLUDE_MJTHREAD_H_
#define MUJOCO_INCLUDE_MJTHREAD_H_

#include <stddef.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void* (*mjfTask)(void*);

struct mjTask_ {
  mjfTask func;
  void* args;
  volatile int status;
};
typedef struct mjTask_ mjTask;

struct mjThreadPool_ {
  int nworkers;
};
typedef struct mjThreadPool_ mjThreadPool;

enum mjtTaskStatus_ {
  mjTASK_NEW = 0,
  mjTASK_QUEUED,
  mjTASK_COMPLETED
};

static inline void mju_defaultTask(mjTask* task) {
  if (!task) return;
  task->func = NULL;
  task->args = NULL;
  task->status = mjTASK_NEW;
}

static inline mjThreadPool* mju_threadPoolCreate(size_t number_of_threads) {
  mjThreadPool* pool = (mjThreadPool*)malloc(sizeof(mjThreadPool));
  if (pool) pool->nworkers = (int)number_of_threads;
  return pool;
}

// 串行语义：入队即执行完毕，因此 mju_taskJoin 无需等待。
static inline void mju_threadPoolEnqueue(mjThreadPool* pool, mjTask* task) {
  (void)pool;
  if (!task) return;
  if (task->func) task->func(task->args);
  task->status = mjTASK_COMPLETED;
}

static inline void mju_taskJoin(mjTask* task) {
  (void)task;
}

static inline void mju_threadPoolDestroy(mjThreadPool* pool) {
  free(pool);
}

#ifdef __cplusplus
}
#endif

#endif  // MUJOCO_INCLUDE_MJTHREAD_H_
EOF
  echo "  ✅ 写入 mjthread.h 兼容层"
}

# ── ③ mjPLUGIN_LIB_INIT 带参宏（唯一必须改插件源码之处）──────────────────
patch_register() {
  local f
  for f in "$PLUGIN_SRC/register.cc" "$PLUGIN_DST/register.cc"; do
    [[ -f "$f" ]] || continue
    "$PY" - "$f" <<'PYEOF'
import sys, pathlib
p = pathlib.Path(sys.argv[1]); s = p.read_text()
if "mjPLUGIN_LIB_INIT(" in s:
    print(f"  ⏭  {p} 已适配"); raise SystemExit
old = "mjPLUGIN_LIB_INIT {"
if old not in s:
    print(f"  ⚠️  {p} 未匹配到旧写法，请人工确认"); raise SystemExit
new = ("// MuJoCo 3.12 起 mjPLUGIN_LIB_INIT 需要一个名字参数，用于区分同一进程内\n"
       "// 多个插件库的初始化函数符号（changelog: \"macro now requires a name\n"
       "// argument to avoid initialization function\" 名字冲突）。旧的零参写法\n"
       "// 不会被展开，报 expected '}' before '::'。\n"
       "mjPLUGIN_LIB_INIT(ray_caster) {")
p.write_text(s.replace(old, new, 1))
print(f"  ✅ 已适配 {p}")
PYEOF
  done
}

# ── ④ 放宽 -Werror（gcc 11.4 对 MuJoCo 自身碰撞代码的误报）────────────────
patch_werror() {
  local f="$MJ_SRC/cmake/MujocoOptions.cmake"
  if grep -q -- "-Wno-error" "$f"; then echo "  ⏭  -Werror 已放宽"; return; fi
  sed -i 's/^      -Werror$/      -Wno-error/' "$f"
  grep -q -- "-Wno-error" "$f" && echo "  ✅ -Werror → -Wno-error" \
    || { echo "  ❌ 未能修改 $f" >&2; exit 1; }
}

# ── 挂载插件到 MuJoCo 源码树 ─────────────────────────────────────────────
install_plugin() {
  if [[ ! -d "$PLUGIN_DST" ]]; then
    cp -r "$PLUGIN_SRC" "$PLUGIN_DST"
    echo "  ✅ 已复制插件到 $PLUGIN_DST"
  else
    echo "  ⏭  插件已在源码树内"
  fi
  if ! grep -q "mujoco_ray_caster" "$MJ_SRC/CMakeLists.txt"; then
    echo "  ❌ CMakeLists.txt 未注册插件。请在 MUJOCO_BUILD_PLUGIN 段落内加：" >&2
    echo "     add_subdirectory(plugin/mujoco_ray_caster)" >&2
    exit 1
  fi
  echo "  ⏭  CMakeLists.txt 已注册 add_subdirectory"
}

# ── 验证：能否被 Python 运行时加载并驱动真实场景 ──────────────────────────
verify() {
  echo "════ 验证 ════"
  [[ -f "$SO" ]] || { echo "❌ 未找到 $SO" >&2; exit 1; }
  echo "  产物: $SO"
  "$PY" - "$SO" "$SCENE" <<'PYEOF'
import sys, mujoco, numpy as np
mujoco.mj_loadPluginLibrary(sys.argv[1])
m = mujoco.MjModel.from_xml_path(sys.argv[2])
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "height_scanner")
if sid < 0:
    sys.exit("❌ 场景里找不到 height_scanner 传感器")
dim = m.sensor_dim[sid]
print(f"  ✅ 插件注册成功，场景加载成功")
print(f"  height_scanner sensordim = {dim}  （187 根射线 × xyz = 561）")
PYEOF
  echo
  echo "接下来把这一行填进 sim2sim/config.py："
  echo "  RAYCASTER_PLUGIN_LIBRARY = \"$SO\""
}

main() {
  echo "════ 版本一致性 ════"; check_versions
  echo "════ 挂载插件 ════";   install_plugin
  echo "════ 适配 3.12 ════"
  patch_mjtnum; patch_mjthread; patch_register; patch_werror
  echo "════ 编译（-j$JOBS）════"
  cmake -S "$MJ_SRC" -B "$MJ_SRC/build" -DCMAKE_BUILD_TYPE=Release \
        -DMUJOCO_BUILD_SIMULATE=OFF -DMUJOCO_BUILD_TESTS=OFF \
        -DMUJOCO_BUILD_EXAMPLES=OFF > /dev/null
  cmake --build "$MJ_SRC/build" --target sensor_raycaster -j "$JOBS"
  echo
  verify
}

case "${1:-all}" in
  all)    main ;;
  verify) verify ;;
  *) echo "用法: $0 {all|verify}" >&2; exit 2 ;;
esac
