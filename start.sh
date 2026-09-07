#!/bin/bash
# 启动证件照抠图后端（常驻模型）
# 可移植版：路径由脚本所在目录推导，不写死本机绝对路径，任何机器 clone 后可直接用。
# 用法：bash start.sh   （可用环境变量 PORT 覆盖端口，默认 8100）
cd "$(dirname "$0")" || exit 1            # 进入仓库根(本脚本所在目录)
mkdir -p logs
PORT="${PORT:-8100}"

# 若已在运行则不重复启动
if lsof -iTCP:"$PORT" -sTCP:LISTEN -P >/dev/null 2>&1; then
  echo "后端已在运行 :$PORT"; lsof -iTCP:"$PORT" -P | tail -1; exit 0
fi

# 确保 Python 依赖已装（首次会提示）。PY=${PYTHON:-python3}：可用 PYTHON 环境变量指定解释器（同 server.js）
PY="${PYTHON:-python3}"
if ! "$PY" -c "import onnxruntime, numpy, PIL" 2>/dev/null; then
  echo "⚠️  缺少 Python 依赖，先执行：pip3 install -r requirements.txt"
fi

# 把解释器路径传给子进程（server.js 会读 PYTHON 环境变量）
nohup env PYTHON="$PY" node "$(pwd)/server.js" > logs/server.log 2>&1 &
echo "后端已启动 PID=$! :$PORT"
sleep 2
curl -s --max-time 5 "http://localhost:$PORT/health" || echo "(健康检查稍后再试)"