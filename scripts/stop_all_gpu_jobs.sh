#!/usr/bin/env bash
# 一键停掉所有 GPU 训练任务，把显卡还给桌面。
#
#   bash scripts/stop_all_gpu_jobs.sh
#
# 为什么需要它：单卡机器上训练会占满 24 GB 显存并把 GPU 打到 100%，
# 桌面合成器、浏览器都要用同一张卡，于是整机卡到没法用。
# 注意这不是内存问题——内存可能还剩几十 GB，看 free 会误判。
# 判断依据是 nvidia-smi 的显存占用，以及 /proc/pressure/memory（内存压力为 0 就不是内存的锅）。
#
# 停止顺序很重要：先停看护/排队脚本，再停训练进程。
# 反过来的话，看护脚本会检测到训练退出，立刻把下一个任务拉起来，白停。
set -uo pipefail

echo "── 1/3 停排队与看护脚本 ──"
# 模式漏一个，停止就不彻底：录像脚本是个 for 循环，
# 只杀掉它当前启动的 play 子进程，它会立刻起下一段，看着像杀不死。
# 所以这里必须覆盖所有会自己拉起 GPU 进程的脚本。
mapfile -t PIDS < <(pgrep -f "run_p1[01]|retrain_p9|watch_p11|run_p5p6|overnight_p7p8|run_overnight_queue|record_all_videos|record_p7|record_p11" 2>/dev/null)
for pid in "${PIDS[@]}"; do
    [ -z "$pid" ] && continue
    [ "$pid" = "$$" ] && continue          # 别把自己杀了
    echo "  停 $pid  $(ps -o args= -p "$pid" 2>/dev/null | cut -c1-56)"
    kill "$pid" 2>/dev/null
done
sleep 2

echo "── 2/3 停训练进程 ──"
mapfile -t GPU_PIDS < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')
if [ ${#GPU_PIDS[@]} -eq 0 ] || [ -z "${GPU_PIDS[0]:-}" ]; then
    echo "  GPU 上没有计算进程"
else
    for pid in "${GPU_PIDS[@]}"; do
        [ -z "$pid" ] && continue
        echo "  停 $pid  $(ps -o args= -p "$pid" 2>/dev/null | cut -c1-56)"
        kill "$pid" 2>/dev/null
    done
    # IsaacSim 对 SIGTERM 响应很慢（实测等 60 s 仍占着 15 GB），
    # 给足时间再上 SIGKILL。
    for i in $(seq 1 12); do
        sleep 5
        left=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' \n')
        [ -z "$left" ] && break
        echo "  等待退出… (${i}/12) 仍在: $left"
    done
    for pid in "${GPU_PIDS[@]}"; do
        [ -z "$pid" ] && continue
        if ps -p "$pid" >/dev/null 2>&1; then
            echo "  SIGTERM 无效，强制终止 $pid"
            kill -9 "$pid" 2>/dev/null
        fi
    done
    sleep 8
fi

echo "── 3/3 释放 GPU 文件锁 ──"
# 排队脚本被杀后，flock / sleep 子进程可能成为孤儿继续占着锁，
# 下次启动队列会一直卡在"等待 GPU 锁"。
for pid in $(fuser /tmp/humanoid_gpu.lock 2>/dev/null); do
    echo "  停锁持有者 $pid  $(ps -o args= -p "$pid" 2>/dev/null | cut -c1-40)"
    kill "$pid" 2>/dev/null
done
sleep 2

# 复查一轮：上面按"当时那份名单"逐个杀，但父脚本被杀的瞬间
# 可能刚好又拉起了下一个 play 进程，那个新进程不在名单里。
# 不复查就会出现"报告已清空、其实还占着显存"的假象。
for round in 1 2 3; do
    left=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')
    [ -z "$left" ] && break
    echo "  复查第 $round 轮，仍有: $(echo "$left" | tr '\n' ' ')"
    for pid in $left; do kill -9 "$pid" 2>/dev/null; done
    sleep 8
done

echo
echo "── 结果 ──"
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
echo "剩余计算进程: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr '\n' ' ')(空=已清干净)"
fuser /tmp/humanoid_gpu.lock >/dev/null 2>&1 && echo "⚠️ 锁仍被占用" || echo "锁已释放"
echo
echo "恢复训练：bash scripts/run_overnight_queue.sh"
