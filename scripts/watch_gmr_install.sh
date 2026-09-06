#!/usr/bin/env bash
# 看护 GMR 安装，装完自动验证并把结论写进日志。
#
# CUDA 依赖有 2~3 GB，按实测 283 KB/s 要跑几小时。与其守着看，
# 不如让它装完自己验一遍 —— 光看 pip 退出码不够，得确认真能 import、
# 且 numpy 是官方要求的 1.26.4（中断的那次就装成了 2.2.6）。
#
# 用法：nohup bash scripts/watch_gmr_install.sh > /dev/null 2>&1 &
set -uo pipefail

ROOT="/home/limx/workspace/Roxan_warmup"
PY="$ROOT/envs/gmr/bin/python"
LOG="$HOME/humanoid_logs/p7_retarget/gmr_install_watch.log"
mkdir -p "$(dirname "$LOG")"

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "════ 看护 GMR 安装 ════"

while pgrep -f "envs/gmr/bin/pip" >/dev/null 2>&1; do
  # 下载中的 wheel 落在 /tmp/pip-unpack-*，不是 pip cache ——
  # 盯 cache 会误判成"卡死"（踩过一次）
  cur=$(find /tmp/pip-unpack-* -name '*.whl' -printf '%f %s\n' 2>/dev/null \
        | sort -k2 -rn | head -1)
  log "  下载中: ${cur:-（解压/安装阶段）}"
  sleep 600
done

log "pip 进程已结束，开始验证"

out=$("$PY" -c "
import general_motion_retargeting, smplx, torch, numpy, scipy
print(f'gmr=ok torch={torch.__version__} numpy={numpy.__version__} scipy={scipy.__version__}')
print(f'cuda={torch.cuda.is_available()}')
" 2>&1)
rc=$?

if [ "$rc" -ne 0 ]; then
  log "❌ 验证失败："
  echo "$out" | tail -8 | tee -a "$LOG"
  log "  安装日志最后 15 行："
  tail -15 "$HOME/humanoid_logs/p7_retarget/gmr_install.log" | tee -a "$LOG"
  exit 1
fi

log "✅ $out"

# 官方 §4.1 要求装依赖前把 numpy 钉到 1.26.4；没钉住说明 setup.py 的改动没生效
if echo "$out" | grep -q "numpy=1.26.4"; then
  log "✅ numpy 锁在官方要求的 1.26.4"
else
  log "⚠️ numpy 不是 1.26.4，检查 setup.py 的版本固定是否生效"
fi

# 脚本能跑 --help 才算真的装好（这正是发现环境缺失的那个检验）
if timeout 120 "$PY" "$ROOT/repos/GMR/scripts/smplx_to_robot_dataset_npz.py" --help \
     > /tmp/gmr_help.out 2>&1; then
  log "✅ smplx_to_robot_dataset_npz.py --help 可运行"
else
  log "⚠️ --help 失败，最后 10 行："
  tail -10 /tmp/gmr_help.out | tee -a "$LOG"
fi

log "剩下只等两个下载包：SMPL-X 模型(7mq2) 与 ACCAD 数据(kh5i)"
log "════ 看护结束 ════"
