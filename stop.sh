#!/bin/bash
# 停止后端（仅杀本后端进程，不影响其它）
PIDS=$(lsof -tiTCP:8100 -sTCP:LISTEN 2>/dev/null)
if [ -n "$PIDS" ]; then
  kill $PIDS 2>/dev/null
  echo "已停止: $PIDS"
else
  echo "未在运行"
fi
