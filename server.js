/**
 * 独立证件照抠图后端（idphoto-backend）
 * ───────────────────────────────────────────────
 * 模型：BiRefNet Swin-Tiny (224MB onnx, MIT 可商用) —— CPU 单张 ~6s，发丝级边缘更干净
 *        （2026-09-05 由 RMBG 切换，用户三底色对比确认 BiRefNet 质量更好）。
 * BiRefNet 只能 Python (onnxruntime) 跑，故本 Node 服务 spawn 常驻 Python 子进程，
 * 每次 /extract 请求经 stdin/stdout 管道喂 base64、取回透明 PNG。
 * Node 侧保留 HTTP 接口与商业级后处理之外的编排，前端接口不变(与云函数对齐)。
 *
 * 完全独立于原小程序项目：独立目录、独立端口 :8100，不影响其它服务。
 * 接口：
 *   POST /extract   body { pngBase64 } → { code:0, pngBase64, tookSec }
 *   GET  /health    → { ok:true }
 */
const http = require('http');
const { spawn } = require('child_process');
const fs = require('fs');
const crypto = require('crypto');
const path = require('path');

// ── 字节级缓存：同图不重算（2026-09-05 提速）──
// 前端重复上传同一张图(重试/换底/换规格)时, 用 MD5 命中磁盘缓存直接返, 跳过 BiRefNet 6s 推理。
const CACHE_DIR = path.join(__dirname, 'cache');
const CACHE_MAX_MB = 300;   // 磁盘缓存上限(超限清理最旧)
let cacheHits = 0, cacheMiss = 0;
function cacheKey(b64) { return crypto.createHash('md5').update(b64).digest('hex'); }
function cacheGet(key) {
  try {
    const f = path.join(CACHE_DIR, key + '.png');
    if (fs.existsSync(f)) { const png = fs.readFileSync(f); if (png && png.length > 100) return png.toString('base64'); }
  } catch (e) {}
  return null;
}
function cachePut(key, pngBase64) {
  try {
    if (!fs.existsSync(CACHE_DIR)) fs.mkdirSync(CACHE_DIR, { recursive: true });
    fs.writeFileSync(path.join(CACHE_DIR, key + '.png'), Buffer.from(pngBase64, 'base64'));
    // 超限清理: 若缓存目录总大小超 CACHE_MAX_MB, 删最旧的直到达标
    if (process.hrtime() && Math.random() < 0.05) { // 偶尔检查一次开销
      try {
        let files = fs.readdirSync(CACHE_DIR).map(n => ({ n, s: fs.statSync(path.join(CACHE_DIR,n)).size, t: fs.statSync(path.join(CACHE_DIR,n)).mtimeMs })).sort((a,b)=>a.t-b.t);
        let total = files.reduce((s,f)=>s+f.s,0);
        while (total > CACHE_MAX_MB*1024*1024 && files.length > 10) {
          const rm = files.shift(); total -= rm.s; fs.unlinkSync(path.join(CACHE_DIR, rm.n));
        }
      } catch (e2) {}
    }
  } catch (e) {}
}

const PORT = Number(process.env.PORT || 8100);
const MAX_BODY_BYTES = 20 * 1024 * 1024; // 20MB 输入上限

// ── 常驻 Python BiRefNet worker 池（并行推理，提升吞吐）──
// 3 worker 池，同一时刻并行算 ~3 张。BiRefNet 单张 ~6s，实测 3 worker 吞吐约 30 张/分钟。
// （发丝质量优于 RMBG，但慢 ~2.4 倍；RMBG 版备份在 matte_biref.py.bak_rmbg_*）
const PY_BIN = process.env.PYTHON || 'python3';   // 可移植：另一台机器 python 可执行名可能不同（python3.x / venv 绝对路径），用 PYTHON 环境变量覆盖，默认 python3
const PY = path.join(__dirname, 'matte_biref.py');   // 可移植：不写死绝对路径，脚本所在目录定位
const NUM_WORKERS = 2;        // worker 数(2026-09-06 由3降为2): BiRefNet推理峰值~3GB/worker,2个峰值~6GB,释放内存避免18GB机器被swap拖慢推理(19s→~6s)。吞吐20张/分但单请求等待更好。
let workers = [];             // [{proc, pending, buf, ready}]
let rrIndex = 0;              // 轮询分配指针

function makeWorker() {
  const w = { proc: null, pending: null, buf: '', ready: false };
  spawnWorker(w);
  workers.push(w);
  return w;
}

function spawnWorker(w) {
  w.proc = spawn(PY_BIN, [PY], { stdio: ['pipe', 'pipe', 'pipe'] });
  w.buf = '';
  w.proc.stderr.on('data', d => console.warn('[matte-worker]', String(d).trim()));
  // 单一 stdout 监听器：按行分帧恢复给本 worker 的 pending 请求
  w.proc.stdout.on('data', chunk => {
    w.buf += chunk;
    let nl;
    while ((nl = w.buf.indexOf('\n')) >= 0) {
      const line = w.buf.slice(0, nl);
      w.buf = w.buf.slice(nl + 1);
      if (!w.pending) continue;                 // 无挂起请求，丢弃
      const resolve = w.pending;
      w.pending = null;
      try { resolve({ ok: true, data: JSON.parse(line) }); }
      catch (e) { resolve({ ok: false, error: 'PY_BAD_JSON:' + line.slice(0, 200) }); }
    }
  });
  w.proc.on('exit', (c, s) => {
    console.warn('[matte-worker] 退出', c, s, '，重启…');
    if (w.pending) { const r = w.pending; w.pending = null; r({ ok: false, error: 'PY_EXITED' }); }
    w.ready = false;
    setTimeout(() => spawnWorker(w), 1000);
  });
  w.proc.on('error', e => {
    console.error('[matte-worker] 启动失败', e);
    if (w.pending) { const r = w.pending; w.pending = null; r({ ok: false, error: 'PY_SPAWN_ERR' }); }
  });
  w.ready = true;
}

function allReady() { return workers.length === NUM_WORKERS && workers.every(w => w.ready); }

// 调 Python BiRefNet：从池里轮询选一个空闲 worker，喂 base64，返回透明 PNG JSON。
// 同一 worker 同一时刻只喂一个请求（其 stdout 单帧分帧）；不同 worker 并行处理 → 真正多进程并行。
function callMatte(pngBase64, opts) {
  opts = opts || {};
  return new Promise((resolve, reject) => {
    if (!allReady()) return reject(new Error('PY_NOT_READY'));
    // 轮询找一个空闲 worker
    const n = workers.length;
    let chosen = null;
    for (let k = 0; k < n; k++) {
      const idx = (rrIndex + k) % n;
      if (!workers[idx].pending && workers[idx].ready) { chosen = workers[idx]; rrIndex = idx + 1; break; }
    }
    if (!chosen) return reject(new Error('PY_POOL_BUSY'));
    const w = chosen;
    let settled = false;
    let timer = null;
    timer = setTimeout(() => {
      if (!settled) { settled = true; w.pending = null; reject(new Error('PY_TIMEOUT')); }
    }, 120000);
    w.pending = (r) => {
      if (settled) return;
      settled = true;
      if (timer) clearTimeout(timer);
      if (r && r.ok) resolve(r.data);
      else reject(new Error((r && r.error) || 'PY_UNKNOWN'));
    };
    try {
      w.proc.stdin.write(JSON.stringify({ pngBase64, maskOnly: !!opts.maskOnly, compose: opts.compose || null }) + '\n');
    } catch (e) {
      if (!settled) { settled = true; if (timer) clearTimeout(timer); w.pending = null; reject(e instanceof Error ? e : new Error('PY_WRITE_ERR:' + e)); }
    }
  });
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let size = 0;
    req.on('data', c => {
      size += c.length;
      if (size > MAX_BODY_BYTES) { reject(new Error('BODY_TOO_LARGE')); req.destroy(); return; }
      chunks.push(c);
    });
    req.on('end', () => resolve(Buffer.concat(chunks)));
    req.on('error', reject);
  });
}

// ── 并发保护：允许最多 NUM_WORKERS 个并行处理（3 worker 池），超过进 MAX_QUEUE 排队。
//    排队超时（QUEUE_TIMEOUT_MS）未轮到 → 返回 429，避免高峰期无限堆积拖垮进程。
const MAX_PROCESSING = NUM_WORKERS; // 3 worker 并行
const MAX_QUEUE = 10;            // 额外最多 10 个排队（BiRefNet 单张~6s×3并行，饱和吞吐约30张/分钟）
const QUEUE_TIMEOUT_MS = 60000;  // 排队最长等 60s，超时 429
let activeCount = 0;            // 正在推理的请求数
let queueLen = 0;               // 正在排队的请求数
const waiters = [];             // FIFO 排队

/**
 * 尝试获取一个处理槽。
 *  @returns {Promise<boolean>} true=拿到槽（调用方必须 release()）；false=繁忙（429）
 *  实现：active<1 且队未满 → 直接拿到；队满 → 立即繁忙；否则入队等唤醒，超时→繁忙。
 */
function acquireSlot() {
  if (activeCount < MAX_PROCESSING) {
    activeCount++;             // 占用处理槽
    return new Promise(r => r(true));
  }
  if (queueLen >= MAX_QUEUE) { // 队已满
    return new Promise(r => r(false));
  }
  queueLen++;
  return new Promise(resolve => {
    const timer = setTimeout(() => {
      const i = waiters.indexOf(wake);
      if (i >= 0) waiters.splice(i, 1);
      queueLen = Math.max(0, queueLen - 1);
      resolve(false);           // 超时 → 繁忙
    }, QUEUE_TIMEOUT_MS);
    function wake() {
      clearTimeout(timer);
      const i = waiters.indexOf(wake);
      if (i >= 0) waiters.splice(i, 1);
      queueLen = Math.max(0, queueLen - 1);
      activeCount++;            // 占用处理槽
      resolve(true);
    }
    waiters.push(wake);
  });
}
function releaseSlot() {
  activeCount = Math.max(0, activeCount - 1);
  if (waiters.length) { const w = waiters.shift(); w(); }
}
// 供 /status 查询当前负载
function loadInfo() { return { processing: activeCount, queueLen }; }

const server = http.createServer(async (req, res) => {
  const t0 = Date.now();
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  req.method === 'OPTIONS' && (res.end(''), 0);

  // 健康检查
  if (req.method === 'GET' && req.url === '/health') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ ok: true, pyReady: allReady(), workers: workers.filter(w=>w.ready).length, pid: process.pid, model: 'biref' }));
    return;
  }

  // 队列/负载状态（前端用）
  if (req.method === 'GET' && req.url === '/status') {
    const l = loadInfo();
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ ok: true, processing: l.processing, queue: l.queueLen, maxQueue: MAX_QUEUE, busy: l.queueLen >= MAX_QUEUE }));
    return;
  }

  // 抠图
  if (req.method === 'POST' && req.url === '/extract') {
    let body;
    let key = '';
    let fromCache = false;
    try {
      // ① 先读 body + 算 key（缓存命中无需 acquireSlot 排队）
      body = await readBody(req);
      const json = JSON.parse(body.toString('utf8'));
      const b64 = json.pngBase64 || json.base64 || '';
      const maskOnly = !!json.maskOnly;
      // 可选自动排版规格 {w,h}（正整数，宽≤2000，高≤2500 防御）
      let compose = null;
      if (json.compose && typeof json.compose === 'object') {
        const w = json.compose.w, h = json.compose.h;
        if (Number.isInteger(w) && Number.isInteger(h) && w > 0 && h > 0 && w <= 2000 && h <= 2500) {
          compose = { w, h };
        }
      }
      if (!b64 || b64.length < 100) {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ code: 'NO_BASE64', message: '缺少 pngBase64' }));
        return;
      }
      // 缓存 key：同图不同规格结果不同 → compose 时带规格后缀。mask/transparent 沿用旧 key（不回归）。
      key = cacheKey(b64) + (maskOnly ? ':m' : ':t') + (compose ? ':c:' + compose.w + 'x' + compose.h : '');
      const cached = cacheGet(key);
      if (cached) {
        cacheHits++;
        res.writeHead(200, { 'Content-Type': 'application/json' });
        if (maskOnly) res.end(JSON.stringify({ code: 0, maskBase64: cached, tookSec: 0, model: 'biref', source: 'cache', mask: true, cached: true }));
        else if (compose) res.end(JSON.stringify({ code: 0, pngBase64: cached, tookSec: 0, model: 'biref', source: 'cache', compose: true, detectOk: true, cached: true }));
        else res.end(JSON.stringify({ code: 0, pngBase64: cached, tookSec: 0, model: 'biref', source: 'cache', cached: true }));
        console.log(`[idphoto] 🔁 缓存命中 ${key.slice(0,8)} (hit=${cacheHits} miss=${cacheMiss})`);
        return;
      }
      cacheMiss++;

      // ② 未命中 → 走并发/排队保护 + 推理
      const got = await acquireSlot();
      if (!got) {
        const l = loadInfo();
        res.writeHead(429, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ code: 'BUSY', message: '服务繁忙，请稍后重试', processing: l.processing, queue: l.queueLen, maxQueue: MAX_QUEUE }));
        return;
      }
      try {
        const srcLen = Math.round(Buffer.from(b64, 'base64').length / 1024);
        const li = loadInfo();
        console.log(`[idphoto] 收图 ${srcLen}KB，BiRefNet 推理…（processing=${li.processing} queue=${li.queueLen}）${compose?' [compose:'+compose.w+'x'+compose.h+']':''}`);
        const out = await callMatte(b64, { maskOnly, compose });
        const tookSec = +((Date.now() - t0) / 1000).toFixed(2);
        const respOk = maskOnly ? (out.code === 0 && out.maskBase64) : (out.code === 0 && out.pngBase64);
        if (!respOk) {
          console.error('[idphoto] BiRefNet 未成功:', JSON.stringify(out).slice(0,200));
          res.writeHead(502, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ code: 'ERROR', message: (out && out.message) || 'BiRefNet 失败' }));
          return;
        }
        // ③ 写入缓存(下次同图直接命中) —— 注意 compose 降级(detectOk:false)也缓存该排版图(同一 spec 幂等)
        cachePut(key, maskOnly ? out.maskBase64 : out.pngBase64);
        console.log(`[idphoto] ✅ BiRefNet 完成，${out.tookSec}s（HTTP ${tookSec}s）${maskOnly?' [mask]':''}${compose?' [compose '+(out.detectOk?'ok':'degrade')+']':''}`);
        res.writeHead(200, { 'Content-Type': 'application/json' });
        if (maskOnly) res.end(JSON.stringify({ code: 0, maskBase64: out.maskBase64, tookSec, model: 'biref', mask: true, mlSec: out.tookSec, source: out.source }));
        else if (compose) {
          const body = { code: 0, pngBase64: out.pngBase64, tookSec, model: 'biref', mlSec: out.tookSec, source: out.source, compose: true, detectOk: !!out.detectOk };
          if (out.photoError) body.photoError = out.photoError;
          res.end(JSON.stringify(body));
        }
        else res.end(JSON.stringify({ code: 0, pngBase64: out.pngBase64, tookSec, model: 'biref', mlSec: out.tookSec, source: out.source }));
      } finally {
        releaseSlot();  // 处理完成/异常都释放处理槽，唤醒下一个排队者
      }
    } catch (e) {
      console.error('[idphoto] ❌ 失败:', e && e.message, e && e.stack ? e.stack.split('\n').slice(0,3).join(' ') : '');
      res.writeHead(500, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ code: 'ERROR', message: (e && e.message) || '推理失败' }));
    }
    return;
  }

  res.writeHead(404, { 'Content-Type': 'text/plain' });
  res.end('not found');
});

server.listen(PORT, () => console.log(`[idphoto] 后端已启动 :${PORT}（模型 RMBG, ${NUM_WORKERS} worker 池）`));
for (let i = 0; i < NUM_WORKERS; i++) makeWorker();
setTimeout(() => { console.log(`[idphoto] ✅ 就绪 ${allReady() ? '' : '(worker 加载中)'} HTTP :${PORT}`); }, 2000);

// ── 周期健康自检：不只探「进程是否活着」，而是每隔一阵真打一次轻量推理
//    确认 worker 不是「spawn 成功但模型加载失败 / OOM 反复重启」的假活状态。
//    pyReady=false（连续探活失败）仅打日志；真机队列/前端仍走 /status。
{
  let probeInFlight = false;
  setInterval(async () => {
    if (probeInFlight) return;                     // 上一轮还没结束(可能有请求在跑)，跳过
    probeInFlight = true;
    try {
      // 拿当前空闲 worker 喂一张极小图验证能真跑（不占 MAX_QUEUE 名额）
      const ready = allReady();
      const li = loadInfo();
      if (!ready) {
        console.warn(`[idphoto] ⚠️ 自检: worker 未全部就绪 (${workers.filter(w=>w.ready).length}/${NUM_WORKERS})`);
      }
      // 简单探活：挑一个空闲 worker 瞬时打一次 run 验证非假活
      if (ready && li.processing === 0) {
        // 极小 1x1 图（能触发 onnx 执行即可，开销几乎为零）
        let ok = false;
        try {
          const probe = await new Promise((resolve, reject) => {
            const start = Date.now();
            const w = workers.find(w => !w.pending && w.ready);
            if (!w) return reject(new Error('no-idle'));
            const to = setTimeout(() => reject(new Error('probe-timeout')), 5000);
            w.pending = (r) => { clearTimeout(to); r && r.ok ? resolve(true) : reject(new Error('probe-fail')); };
            try { w.proc.stdin.write(JSON.stringify({ pngBase64: 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==' }) + '\n'); }
            catch (e) { clearTimeout(to); reject(e); }
          });
          ok = !!probe;
        } catch (_) { ok = false; }
        if (!ok) console.warn('[idphoto] ⚠️ 自检: worker 假活(探活推理失败)，后续请求可能 500');
      }
    } catch (e) { /* 自检不应影响主流程 */ }
    probeInFlight = false;
  }, 30000); // 每 30s 一次，开销极小
}