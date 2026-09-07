# 证件照 AI 抠图后端（可移植版）

微信小程序「证件照 AI 抠图」的后端服务：接收上传人像图 → **BiRefNet** 抠图 → 返回透明 RGBA PNG。
Node.js `:8100` + Python BiRefNet（3 路 worker 池，本文件默认 2）。

> 本仓库已做**跨机器可移植**：代码不写死本机绝对路径，任何电脑 `git clone` + 装依赖 + 放模型即可跑。
> 唯一因机器不同而需自行处理的是：**域名 / 反向代理 / systemd/launchd 常驻**（见下方「对外暴露」）。

---

## 一、另一台电脑部署步骤（照抄即可）

环境要求：**Node.js ≥ 18**、**Python 3.9+**、**git**（macOS / Linux 均可）。

```bash
# 1) 拉代码
git clone <你的仓库地址> idphoto-backend
cd idphoto-backend

# 2) 装 Python 依赖（AI 抠图）
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
#    🚨 解释器路径：默认用命令 `python3`；若用它必须已装 onnxruntime（venv 最稳）。
#    · macOS 系统自带 python3 常缺 onnxruntime → 请用 venv，并 **export PYTHON 环境变量** 指到 venv 里的 python：
#        export PYTHON=/your/path/idphoto-backend/.venv/bin/python3
#      之后再 `bash start.sh`（start.sh 与 server.js 都会读到 PYTHON，未设则回退 python3）。
#      这样**无需改任何代码**，解释器路径完全由环境变量控制（可移植收尾，2026-09-07）。

# 3) 放模型（关键！大模型不进 git，必须手动拷）
#    从「本机」把这两个文件拷到  models/ ：
#       models/birefnet_swin_tiny.onnx      (224MB)
#       models/retinaface-resnet50.onnx     (109MB)
#    （缺任一文件服务会起不来或推不出脸装配）

# 3.5) cache/ 无需准备：后端首次运行时自动创建，用于「按 MD5 暂存抠图结果 + 超 300MB 清最旧」。
#    它存的是用户照片的处理结果（隐私数据），已 gitignore **不会进仓库**；另一台机器首次部署后自动重建。

# 4) 启动（默认 :8100，可 PORT=xxxx 覆盖）
bash start.sh
#    查健康：
curl http://localhost:8100/health
#    期望看到类似： {"ok":true,"pyReady":true,"workers":2,...}

# 5) 常驻/自愈（二选一，按你的系统）：
#    macOS 用 launchd（参考本机 restart.sh）；
#    Linux 用 systemd（可放下面模板到 /etc/systemd/system/idphoto.service）。
```

### 关于 Python 解释器路径（重要）
`server.js` 与 `start.sh` 启动 Python worker 用的命令，默认是 `python3`，可用 **环境变量 `PYTHON` 覆盖**
（`const PY_BIN = process.env.PYTHON || 'python3'`）。
- macOS 系统 `python3` 常缺 onnxruntime → **务必用 venv**，且让 venv 生效：
  ```bash
  source .venv/bin/activate
  export PYTHON="$(which python3)"   # 或直接写 venv 绝对路径
  bash start.sh
  ```
- **无需改任何代码**——解释器路径只由环境变量控制。二次启动同样先 `export PYTHON=...` 再 `bash start.sh`。

---

## 二、对外暴露（因机器而异的部分）

小程序前端请求的是 **HTTPS 域名**（当前 `https://idphoto.hcxserver.xyz`），它背后是本机 `:8100`。
新的这台机器要对外提供服务，需要相当于本机现有链路的一套（具体方案你已有经验）：
- 一个**公网 HTTPS 域名**（或用你现有的域名服务）指向这台机器；
- **反向代理**（如 caddy/nginx）把 `443 → 127.0.0.1:8100`；
- 若上云/换 IP，配置**公网/隧道**（cloudflared / Nginx 转发）让域名可达。

代码层**不需要为域名改任何东西**——`server.js` 只监听 `:8100`，域名由反向代理负责。

---

## 三、常见配置项（server.js 顶部）

| 常量 | 说明 |
|---|---|
| `PORT` | 监听端口，默认 8100（也支持环境变量 `PORT` 覆盖） |
| `NUM_WORKERS` | worker 并发数。本机 18GB 用 2（BiRefNet 峰值 ~3GB/worker）。**机器内存决定**：16GB 建议 2，32GB 可 3~4，低内存就 1。 |
| `PY` | matte_biref.py 路径（已用 __dirname，可移植，勿改） |
| `PYTHON` | python 解释器命令/路径（**环境变量**），默认 `python3`。venv 场景务必 `export PYTHON=/path/.venv/bin/python3` 后再启动 |

---

## 四、模型说明

- `matte_biref.py` 用 **BiRefNet**（发丝级）+ **RetinaFace**（人脸，用于可选自动排版，当前前端未请求）。
- 两个大模型因超 GitHub 单文件 100MB 限制**不入库**，部署时手动拷贝（见上）。已被 `.gitignore` 排除。
- `models/` 里旧的 `rmbg*.onnx` 是历史遗留，本仓已从索引移除（可用但非当前使用）。

---

## 五、health 自检（别信定时脚本的 15s）

`curl http://localhost:8100/health` 返回 `"pyReady":true` 才代表 worker 真正就绪。
模型加载约 5~7s，某些脚本 15s 探活会误报"未就绪"，实际健康；以 `/health` 的 `pyReady:true` 为准。

---

### 系统 cron / systemd（Linux 常驻参考）
```ini
# /etc/systemd/system/idphoto.service
[Unit]
Description=idphoto matting backend
After=network.target

[Service]
WorkingDirectory=/your/path/idphoto-backend
ExecStart=/usr/bin/node /your/path/idphoto-backend/server.js
Environment=PORT=8100
# 若用 venv 的 python，需要的话再加一行：
# Environment=PYTHON=/your/path/idphoto-backend/.venv/bin/python3
Restart=always
User=you

[Install]
WantedBy=multi-user.target
```
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now idphoto
curl http://localhost:8100/health
```