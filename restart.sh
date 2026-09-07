#!/bin/bash
# 一键重启 idphoto 后端（RMBG 3-worker 池）
# 用途：改完 server.js / matte_biref.py 后，用它彻底重置 Node 服务 + Python worker，
#       避免"改了代码进程却还在跑旧版"的坑。健康检查通过才提示成功。
set -e

PLIST="$HOME/Library/LaunchAgents/com.idphoto.backend.plist"
PORT=8100
LOG_DIR="/Users/mac/idphoto-backend/logs"

echo "==> 重启 idphoto 后端 :$PORT ..."

# 1) 触发 launchd 重载（KeepAlive + RunAtLoad 会自动拉起，kill 旧进程也是让 launchd 重启）
launchctl unload "$PLIST" 2>/dev/null && echo "    已 unload" || echo "    (unload 跳过)"
sleep 1
# 2) 双保险：若还有残留 node server.js 在监听，主动 kill（避免 launchd 抢先拉起同一个）
if lsof -iTCP:$PORT -sTCP:LISTEN -P >/dev/null 2>&1; then
  PID=$(lsof -tiTCP:$PORT -sTCP:LISTEN -P | head -1)
  echo "    旧进程 PID=$PID 仍在，kill ..."
  kill "$PID" 2>/dev/null || true
  sleep 1
fi
# 3) 重新 load
launchctl load "$PLIST" 2>/dev/null && echo "    已 load"
# 4) 健康检查：最多等 15s，直到 /health 返回 pyReady=true && workers=3
echo "==> 等待健康检查 ..."
for i in $(seq 1 15); do
  sleep 1
  H=$(curl -s --max-time 3 "http://localhost:$PORT/health" 2>/dev/null || echo "")
  if echo "$H" | grep -q '"pyReady":true' && echo "$H" | grep -q '"workers":3'; then
    echo "==> ✅ 后端已就绪（3 worker 全部加载）：$H"
    echo "==> 最新日志："
    tail -4 "$LOG_DIR/daemon.out.log" 2>/dev/null || true
    exit 0
  fi
  echo "    ... 第 $i 秒，等待 worker 就绪"
done
echo "==> ⚠️ 15s 内未就绪，请检查 $LOG_DIR/daemon.err.log / daemon.out.log"
exit 1
