/**
 * lib/matting-core.js —— 抠图算法核心（云函数端）
 *
 * 职责：把客户端上传的图片 Buffer → 透明底 PNG Buffer。
 *
 * 方案（优先级从高到低）：
 *   1. RMBG-1.4：真 AI 人像分割（理解"人 vs 背景"，不误抠衣服/肤色，可靠分割完整人像），
 *      onnxruntime-node 推理。模型从云存储按需下载（见 lib/model.js）。
 *   2. 启发式像素级背景分离：纯色背景兜底（无模型也能用）。
 *
 * 客户端约定：上传的是「压缩到长边 1500px 的 PNG」，云函数只处理 PNG。
 */

const { decodePng, encodePng } = require('./image-util');
const { getBestModel } = require('./model');

/**
 * 抠图后统一后处理（AI / 启发式 所有路径 100% 过一遍）
 *   作用 1: alpha 羽化 5px + 外层强制 α=0 → 消除锯齿状边缘
 *   作用 2: alpha-guided 局部色溢去除（半透明边缘像素取邻域 sureBg(α≈0) 像素估计真 bgRGB，再反推纯前景 RGB）
 *   作用 3: 分区硬保护 → 人像主体 92%×99% 强制 α=255（保住西装/衬衫/领带）
 * @param {Buffer} pngBuffer 上一步输出的透明底 PNG
 * @returns {Buffer} 处理后新的透明底 PNG
 */
function applyMattingPostprocess(pngBuffer) {
  try {
    const { width: w, height: h, data } = decodePng(pngBuffer);
    const n = w * h;
    const src = data;                       // 原始 RGBA（RGB 含「前景×α + 背景×(1-α)」混合）
    const alpha = new Float32Array(n);      // 当前 alpha（0..1）
    const gray = new Float32Array(n);       // 灰度 guidance（0..1）
    for (let i = 0; i < n; i++) {
      alpha[i] = src[i*4+3] / 255;
      gray[i]  = (0.299*src[i*4] + 0.587*src[i*4+1] + 0.114*src[i*4+2]) / 255;
    }

    // --- Step 1: 中心硬保护 82%×89%（人像主体强制前景，也作为确定前景锚点）
    //   ⚠ 从 92%×99% → 82%×89%（核心修 case5 人像左侧/右侧假阳长条）
    //   旧版 92%×99%：两侧各仅留出 4%，当 服装颜色 ≈ 背景颜色（深色POLO衫+深蓝底），
    //   RMBG 在两侧背景输出的 α≈0.05~0.20 假阳会被 α>0.03 → 1.0 锁死 → 后续
    //   连通域/距离场 就从这些假阳 当核向外扩张 → 长条假阳永远杀不掉
    //   新版 82%×89%：halfW 0.46→0.41（两侧各留 18%）halfH 0.495→0.445（各上下留 5.5%）
    //   证件照人像居中，不会顶到这个矩形外 → 不伤人像主体；两侧假阳不再被锁死 → 后续清理全有效
    const cx = Math.floor(w/2), cy = Math.floor(h/2);
    const halfW = Math.floor(w*0.41), halfH = Math.floor(h*0.445);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        if (Math.abs(x-cx) < halfW && Math.abs(y-cy) < halfH) {
          // ★ 只锁"绝对确定前景 α≥0.90" 成 1.0。
          //   逻辑支撑：
          //   - RMBG raw 直方图显示：55% 像素 α≥0.9（真人像主体），43% α=0（纯背景），中间(0.1,0.9) 仅 1.7%
          //   - 这 1.7% 恰好就是发丝/脸周/西装边的半透明混合像素，RGB 里混了原背景色
          //   - 若 Step1 把 α≥0.60→1.0 先锁成不透明，decontaminate 会因为 α≥0.98 直接保留原图 RGB（混有背景色）
          //     → 换红底后 发丝外侧 蓝边永远存在（你截图案例2 蓝边 正是这个残留）
          //   - 所以现在锁到 α≥0.90，让 α∈(0.03,0.90) 的发丝/脸/西装边 保持软，
          //     交给 Step4 decontaminate 做"局部多项式解 bg_true 反污染"去掉背景混合色，
          //     再由 Step4.5 硬化 把 α 定死（软边硬化后仍成锐利不透明）
          // ★ 只锁几乎确定前景 α≥0.98。发丝/脸/衣服边缘的软 α∈[0.03,0.98) 必须保留，
          //   否则会被 decontaminate 的 a>=0.98 分支"原样保留混色RGB" → 蓝边/杂色永远存在。
          //   保留软边后：decontaminate 能反污染解出"纯黑发/纯肤"干净 fg，再 Step4.5 硬化成锐利边缘。
          if (alpha[y*w+x] >= 0.98) alpha[y*w+x] = 1.0;
        }
      }
    }

    // --- Step 2: 导向滤波精化 alpha（边缘保留平滑，沿颜色边缘对齐 → 消除阶梯锯齿）---
    //   参数继承 MODNet v4 方案，并在 RMBG 高精度(1024) 基础上 eps 3e-3→1e-3：
    //   r=2（窗口 5×5）+ eps=1e-3 → 过渡带再收窄到约 2px，彻底去掉头发侧边 1px 级轻微羽化虚影
    const refined = guidedFilterAlpha(gray, alpha, w, h, 2, 1e-3);

    // --- Step 3: 外层 10px 强制 α=0（证件照边带必为背景。与 MODNet v4 对齐，原 12 太紧）---
    // 外层 MARGIN 4→2：连通域清斑 + 原始距离场 + 几何先验 已经吃了大部分外边带假阳，
    // 这里仅留 2px 保险，防止"人像顶/下巴几乎贴边、留白很少"的证件照被误切额头发际 1 像素
    const MARGIN = 2;
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++)
        if (x < MARGIN || x >= w-MARGIN || y < MARGIN || y >= h-MARGIN)
          refined[y*w+x] = 0;

    // --- Step 4: 前景反污染（去色溢）在 alpha 硬化之前跑 —— 完全对齐 MODNet v4 黄金参数
    //   R 6→8（扩大采样半径）：对肩膀尖角、西装轮廓这种突出边角，更大概率采到"确定背景"做局部反推，
    // headY 分区线：y < headY = 头发/脸区，否则身体/服装区。
    // 必须先定义：下一行 decontaminateForegroundSplit(..., headY) 要用到，防止 const 暂时性死区 (TDZ) 报错
    const headY = Math.floor(Math.floor(h/2) * 0.68);

    // --- Step 4: 前景反污染（去色溢 FBA 思想·局部二次曲面拟合）---
    // 头发区 R=10 / 身体区 R=16 分区：对每个半透明像素取窗口内最近的 K=10~16 个确定背景像素，
    // 用距离高斯加权的 2 次多项式最小二乘拟合出"真实背景色曲面 bg_true(x,y)"，
    // 再按 alpha 公式 fg = (mix - bg_true·(1-α)) / α 解出纯前景 RGB。
    // 根因上消除：头发角红/白残边、西装尖角蓝/黄残边（色渗污染，之前用全局平均 fg 救不了）
    const fgRgb = decontaminateForegroundSplit(src, refined, w, h, headY);

    // --- Step 4.5: 分区 alpha 硬化（最终参数，更硬一档收残色）---
    // 头发： lo=0.18 / hi=0.50（不变）。脸颊半透明肤色残 α≈0.08~0.16 → lo=0.18 直接 0，干净
    // 身体： ★ lo 0.08→0.05；hi 0.30→0.25
    //   - 案例3/4 白衬衫肩膀/袖口延伸出来的软边缘，RMBG 输出 α≈0.06~0.10 其实是真衬衫像素的渐隐
    //   - lo 0.08 会把 α=0.06~0.07 这档真衬衫像素 直接杀透明 → 形成肩膀大块第一档缺口
    //   - 降到 0.05：α∈[0.05,0.25) 的软边保留，经 Step4.6 柔和削边 + Step4.8 距离场 核外再杀，
    //     最后若还是 >0 → 它就是真像素，不会再被误杀
    for (let y = 0; y < h; y++) {
      const isHead = y < headY;
      const lo = isHead ? 0.18 : 0.05;
      const hi = isHead ? 0.50 : 0.25;
      for (let x = 0; x < w; x++) {
        const i = y*w + x;
        let a = refined[i];
        if (a <= lo) a = 0;
        else if (a >= hi) a = 1;
        else a = (a - lo) / (hi - lo);
        refined[i] = a;
      }
    }

    // --- Step 4.6: 身体（非头发）边界轻量收边 —— RMBG 1024 精度独享
    //   原理：对 y>=headY 的身体区域，做一次 3×3 腐蚀（任一邻域 α<0.5 则当前像素 α-=0.4）
    //   作用：把西装尖角、西装肩膀轮廓那圈"残留的 0.3~0.5 半透明"擦掉，
    //         视觉效果就是肩膀/西装边角不再有虚化渐变，边缘锐利干净
    //   头发区完全不动（避免破坏发丝过渡）
    {
      const tmp = new Float32Array(refined);
      for (let y = Math.max(1, headY); y < h - 1; y++) {
        for (let x = 1; x < w - 1; x++) {
          const i = y*w + x;
          if (tmp[i] <= 0.02 || tmp[i] >= 0.98) continue;  // 已经明确的 fg/bg 不动
          let hasSoft = 0;
          for (let dy = -1; dy <= 1; dy++)
            for (let dx = -1; dx <= 1; dx++) {
              const v = tmp[(y+dy)*w + (x+dx)];
              if (v < 0.50) hasSoft++;
            }
          // ★ hasSoft≥5 → ≥6；削幅 -0.25 → -0.15
          //   逻辑支撑：案例3/4 白衬衫肩膀软轮廓 9 格中 恰有 5~6 格 α<0.50（这是衬衫真实软边，不是虚影）
          //   - 旧 hasSoft≥5 -0.25 ：把这些真软边 削 -0.25 → 本来就小的 α（约 0.10~0.20）再扣 0.25 → α=0 → 肩膀缺口
          //   - 新 hasSoft≥6 -0.15 ：9 格 ≥6 格才是虚影（更严判定），且削幅更轻 -0.15
          //     → 西装尖角/服装虚影仍被削 < lo 0.05 → 被硬化杀；但衬衫真软边 保留 → 肩膀大块缺口 消失
          if (hasSoft >= 6) refined[i] = Math.max(0, tmp[i] - 0.15);
        }
      }
    }

    // --- Step 4.7：边缘去边 Defringe（\\u2605 根治 RMBG 硬边带来的蓝/红/杂色残边）---
    // 根因：RMBG-2.0 INT8 输出是"硬边"（alpha 直方图软边桶\\u22480），头发/肩部边缘像素
    //       直接被后处理硬化为 α=255，而 decontaminate(a>=0.98) 会原样保留其"前景×α+背景×(1-α)"
    //       混合色 → 换蓝/红/白底后 蓝边/红边/米白边 永远残留，调参无法根治。
    // 本步：对"确定前景边缘像素"(α≥0.98 且 3×3 含背景)，用其周围"纯前景核"的真实颜色
    //       反投影去除掺入的背景色：将混合色沿 前景核色↔背景色 连线拆成 纯前景 + 背景占比，
    //       背景占比被当前源色吸收 → 还原出干净纯前景色。此即经典 matting defringe。
    {
      // 预构建：确定前景 mask + 确定背景 mask。
      // ★ 改动1（修发丝/衣边色溢）：原版 fgMask 只标 α≥0.98，而实测 1dcd 蓝边像素 83% 模型α<0.9（发丝半透明边缘）
      //   完全没被 Defringe 碰到 → 蓝边残留。现放宽到 α≥0.45，连"发丝边缘的半透混色"也纳入去色溢。
      //   bgMask 仍取 α≤0.03（确定背景）。安全门 bw≥2 保证只处理"3×3 内既有背景又有前景"的真边缘，
      //   发丝内部（无背景参照）不会被动。
      const fgMask = new Uint8Array(n);
      const bgMask = new Uint8Array(n);
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          const i = y * w + x;
          if (refined[i] <= 0.03) bgMask[i] = 1;
          else if (refined[i] >= 0.45) fgMask[i] = 1;
        }
      }
      const Rdf = 3;   // 采样半径（放宽，覆盖发丝交叉处）
      let defringed = 0;
      for (let y = Rdf; y < h - Rdf; y++) {
        for (let x = Rdf; x < w - Rdf; x++) {
          const i = y * w + x;
          if (!fgMask[i]) continue;   // 只处理"前景边缘(含半透明发丝)"像素

          // 收集窗口内 背景色 bgSet(从 src 采真背景色) 与 前景核色(从 fgRgb 采去污染后的纯前景色)
          let bw = 0, bR = 0, bG = 0, bB = 0;
          const fgPix = [];   // [[r,g,b], ...]
          for (let dy = -Rdf; dy <= Rdf; dy++) {
            for (let dx = -Rdf; dx <= Rdf; dx++) {
              const j = (y + dy) * w + (x + dx);
              if (bgMask[j]) { bw++; bR += src[j*4]; bG += src[j*4+1]; bB += src[j*4+2]; }
              else if (fgMask[j]) { fgPix.push([fgRgb[j*4], fgRgb[j*4+1], fgRgb[j*4+2]]); }
            }
          }
          if (bw < 2 || fgPix.length < 2) continue;  // 既要有背景参照，也要有前景参照

          const bgR0 = bR / bw, bgG0 = bG / bw, bgB0 = bB / bw;

          // fgRef：窗口内前景颜色取"离中线最集中的众数团"，剔除被背景污染的边缘浅色。
          //   1) 先算 fgPix 均值 as 质心
          let sR = 0, sG = 0, sB = 0;
          for (const p of fgPix) { sR += p[0]; sG += p[1]; sB += p[2]; }
          const cenR = sR / fgPix.length, cenG = sG / fgPix.length, cenB = sB / fgPix.length;
          //   2) 剔除距质心超过 80 的（这些是被背景污染而偏淡/偏色的边缘像素）
          let k = 0, kR = 0, kG = 0, kB = 0;
          for (const p of fgPix) {
            const dr = p[0]-cenR, dg = p[1]-cenG, db = p[2]-cenB;
            if (dr*dr + dg*dg + db*db <= 80*80) { k++; kR += p[0]; kG += p[1]; kB += p[2]; }
          }
          if (k === 0) continue;   // 全是污染像素，无纯净前景参照（极窄细丝），跳过
          const fgR0 = kR / k, fgG0 = kG / k, fgB0 = kB / k;

          // 当前混合色（src）
          const mR0 = src[i*4], mG0 = src[i*4+1], mB0 = src[i*4+2];
          // 背景占比 t：混合色在 前景核→背景 连线上的投影 t∈[0,1]，越大越偏背景
          const dFbR = fgR0 - bgR0, dFbG = fgG0 - bgG0, dFbB = fgB0 - bgB0;
          const dMbR = mR0 - bgR0, dMbG = mG0 - bgG0, dMbB = mB0 - bgB0;
          const denom = dFbR*dFbR + dFbG*dFbG + dFbB*dFbB;
          if (denom < 1e-6) continue;
          let t = (dMbR*dFbR + dMbG*dFbG + dMbB*dFbB) / denom;
          t = t < 0 ? 0 : (t > 1 ? 1 : t);
          // 明显掺入背景才去边（t>=0.28），避免误伤真实发丝/衣纹
          if (t < 0.28) continue;
          // 有限收边：把边缘混色沿“前景核色”方向收拢，幅度随背景占比 t 增大，
          //  结果始终落在 [min(mix,fgRef), max(mix,fgRef)] → 不会越过头成黑色假边。
          const g = Math.min(1.0, t * 1.5);
          let pR = mR0 + (fgR0 - mR0) * g;
          let pG = mG0 + (fgG0 - mG0) * g;
          let pB = mB0 + (fgB0 - mB0) * g;
          pR = pR < 0 ? 0 : (pR > 255 ? 255 : pR);
          pG = pG < 0 ? 0 : (pG > 255 ? 255 : pG);
          pB = pB < 0 ? 0 : (pB > 255 ? 255 : pB);
          fgRgb[i*4]   = Math.round(pR);
          fgRgb[i*4+1] = Math.round(pG);
          fgRgb[i*4+2] = Math.round(pB);
          defringed++;
        }
      }
      if (defringed) console.log('[matting] [Step4.7 Defringe] 去边 ' + defringed + ' 个 前景边缘像素（清除 蓝/红/米白 背景混合残边）');
    }

    // --- Step 4.8：人像主体距离场擦除 —— 散点/外围伪边终极一刀
    // 把"确定前景(α≥0.90)"当核做 8 邻域 BFS 距离扩张到 DMAX=20px；
    // 扩张波没覆盖到的像素若仍然 α>0 → 离人像主体 20 像素以上，证件照里不可能属于真人像素 → α=0
    //   清除：左上竖条白点 / 头发圈外蓝色薄边环 / 水印或文字外围 残留假阳散点
    {
      // ★ Step4.8 重写：核只从 α≥0.95（绝对确定前景）出发；DMAX=18；核外 α≥0.10 一律判假阳杀
      //   逻辑支撑（针对你 4 类失败）：
      //   - 案例2 发丝外蓝边：旧核 ≥0.70 → 会从"蓝边像素本身 α≈0.70 的假核"向外扩张 12px → 把蓝边包住保留
      //     新核 ≥0.95：蓝边 α≈0.70 不进核 → 距离不到 → 蓝边 α≈0.30~0.60 被直接 α=0
      //   - 案例4 黑POLO复杂背景（人物身后有花/植物/彩色杂物）：假阳像素 α≈0.15~0.30
      //     旧 DMAX 12 从真人像主体出发，刚好覆盖不到"胳膊旁边 13~17px 的花/杂物假阳"（距离 D=13~17）
      //     旧 kill 只杀 refined>0 → 但假阳 α>0 且 D=14 没覆盖到 又没满足 kill（旧逻辑只要 距离===-1 才杀，但假阳
      //     实际离主核 14px → 旧 DMAX=12 距离===-1 → 杀），OK。但真正致命的是：旧核 ≥0.70 也把 "花/植物 背景里 α≈0.75 假阳
      //     当核"向外扩张 12px → 把假阳假核本身+周围 12px 包住 → 保留 → 胳膊外漏彩色杂色
      //   - 新 DMAX 18：能覆盖胳膊外 18px 内的半透明假阳（发丝一般宽 10~15px，18px 也够）
      //   - 新 kill α≥0.10：α<0.10 的半透明边缘保留（发丝尖/脸颊薄过渡），其余核外任何可见半透明 都判背景假阳 α=0
      const DMAX = 18;
      const SEED = 0.95;
      const KILL = 0.10;
      const dist = new Int16Array(n);
      for (let i = 0; i < n; i++) dist[i] = -1;
      const q = [];
      for (let i = 0; i < n; i++) if (refined[i] >= SEED) { dist[i] = 0; q.push(i); }
      const DX = [-1, 1, 0, 0, -1, -1, 1, 1];
      const DY = [ 0, 0,-1, 1, -1,  1,-1, 1];
      for (let qh = 0; qh < q.length; qh++) {
        const idx = q[qh], d = dist[idx];
        if (d >= DMAX) continue;
        const yy = (idx / w) | 0, xx = idx - yy * w;
        for (let k = 0; k < 8; k++) {
          const nx = xx + DX[k], ny = yy + DY[k];
          if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
          const ni = ny * w + nx;
          if (dist[ni] !== -1) continue;
          dist[ni] = d + 1;
          q.push(ni);
        }
      }
      let killed = 0;
      for (let i = 0; i < n; i++) {
        if (dist[i] === -1 && refined[i] >= KILL) { refined[i] = 0; killed++; }
      }
      if (killed) console.log('[matting] [Step4.8 距离场清理] 人像主体 ' + DMAX + 'px 外 清掉 ' + killed + ' 个 α≥' + Math.round(KILL*100) + '% 假阳像素（复杂背景花色/蓝边环/散点白斑 等）');
    }

    // --- Step 6 (P3)：CIE-L*a*b* ΔE 色渗残边校正 —— ★ 全图运行 不分内外；门槛更宽松、保护更严
    //   针对用户说"复杂背景根本不能分割"的反馈：
    //   - ΔE 阈值 9→12 / 13→16：ΔE<12 视觉上两颜色几乎同色（人眼 95% 人分辨不出 ΔE<8）
    //     → 只要像素颜色和四角估计背景色 ΔE<12 直接 α=0（复杂背景下 "蓝底/红底上 稍暗稍亮的灰/花" 假阳会命中）
    //   - 保护条件 fgC/tC ≥ 0.75 → 0.90：只有 3×3 9 格中有 8~9 格 都是确定前景 (α≥0.95) 才算"完全贴死
    //     主体不杀"。任何"人像边缘"（真边缘 9 格中 3~6 格是确定前景）都会进入 ΔE 判定。
    //     之前 0.75 太松 → 脸盘外一圈半透明肤色残影 / 发边蓝边 的 3×3 中 7 格是确定前景 → 被"不杀"保留
    {
      const CS = Math.min(20, Math.floor(Math.min(w, h) * 0.04));
      let sBgR = 0, sBgG = 0, sBgB = 0, sBgN = 0;
      const corners = [[0, 0], [w - CS, 0], [0, h - CS], [w - CS, h - CS]];
      for (let c = 0; c < corners.length; c++) {
        const ox = corners[c][0], oy = corners[c][1];
        for (let yy = 0; yy < CS; yy++) for (let xx = 0; xx < CS; xx++) {
          const j4 = ((oy + yy) * w + (ox + xx)) * 4;
          sBgR += src[j4]; sBgG += src[j4 + 1]; sBgB += src[j4 + 2]; sBgN++;
        }
      }
      if (sBgN) { sBgR /= sBgN; sBgG /= sBgN; sBgB /= sBgN; }
      const BgLb = rgb2lab(sBgR, sBgG, sBgB);
      const FOK = 1;
      // ★ 新增：复杂背景估计 bgRGB 的 2 档颜色扩展（防止"四角估计的 bgRGB"太局限 在 纯色角落）
      //   另外取 4 条边中点的 20×20 作为 辅助bg，分别 ΔE 判定，命中任一档都杀
      const sidePts = [
        [Math.floor(w/2) - CS/2, 0], [Math.floor(w/2) - CS/2, h-CS],
        [0, Math.floor(h/2)-CS/2], [w-CS, Math.floor(h/2)-CS/2]
      ];
      let sBg2R = 0, sBg2G = 0, sBg2B = 0, sBg2N = 0;
      for (const [ox0, oy0] of sidePts) {
        const ox = Math.max(0, Math.min(w-CS, Math.floor(ox0)));
        const oy = Math.max(0, Math.min(h-CS, Math.floor(oy0)));
        for (let yy = 0; yy < CS; yy++) for (let xx = 0; xx < CS; xx++) {
          const j4 = ((oy + yy) * w + (ox + xx)) * 4;
          sBg2R += src[j4]; sBg2G += src[j4+1]; sBg2B += src[j4+2]; sBg2N++;
        }
      }
      const BgLb2 = (sBg2N > 0) ? rgb2lab(sBg2R/sBg2N, sBg2G/sBg2N, sBg2B/sBg2N) : BgLb;
      for (let y = FOK; y < h - FOK; y++) {
        for (let x = FOK; x < w - FOK; x++) {
          const i = y * w + x;
          const a = refined[i];
          if (a < 0.03 || a > 0.70) continue;   // 防“扣错衣服”：仅处理 α<0.70 半透明，实心衣装(color≈bg)不误杀
          // ★ 保护条件：仅 3×3 9 格中有 ≥0.90×9=8.1 → 8~9 格 都是 α≥0.95 的确定前景，才认为完全贴死主体，不杀
          let fgC = 0, tC = 0;
          for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
            tC++;
            if (refined[(y + dy) * w + (x + dx)] >= 0.95) fgC++;
          }
          if (fgC / tC >= 0.90) continue;
          const i4 = i * 4;
          const Lp = rgb2lab(src[i4], src[i4 + 1], src[i4 + 2]);
          // ★ ΔE 判定：同时与「四角背景」「四边中点背景」比，只要任一档命中 ΔE<12 就 α=0 杀
          //   ΔE<12 且 α<0.35 → ≤0.20；ΔE<16 且 α<0.35 → ≤0.20 （更狠杀半透明残色）
          const cDE = (Lb) => {
            const dL = Lp[0] - Lb[0], da = Lp[1] - Lb[1], db = Lp[2] - Lb[2];
            return Math.sqrt(dL * dL + da * da + db * db);
          };
          const dE1 = cDE(BgLb), dE2 = cDE(BgLb2);
          const dE = Math.min(dE1, dE2);
          if (dE < 12.0) refined[i] = 0;
          else if (dE < 16.0 && a < 0.35) refined[i] = Math.min(refined[i], 0.20);
        }
      }
    }

    // --- Step 7：抗锯齿羽化（1 次 3×3 平均）---
    //   RMBG 硬边 alpha 硬化后几乎纯 0/1（日志 soft≈0.2%），边缘呈阶梯“锯齿”。
    //   导出前对 alpha 做 1 次 3×3 均值，把 1px 硬跳变转成 ~1px 平滑过渡去锯齿；
    //   内部(全1)/外部(全0)不受影响，保持人像锐利。
    {
      const ftmp = new Float32Array(refined);
      for (let y = 1; y < h - 1; y++) {
        for (let x = 1; x < w - 1; x++) {
          const i = y * w + x;
          let sumv = 0;
          for (let dy = -1; dy <= 1; dy++)
            for (let dx = -1; dx <= 1; dx++)
              sumv += ftmp[(y + dy) * w + (x + dx)];
          refined[i] = sumv / 9;
        }
      }
    }

    // --- Step6.5 (在 Step7 抗锯齿之后)：残留背景色边缘收缩 ⚠ 必须放在 Step7 后才有用 ---
    // 背景：1dcd蓝底59%、aed4红底52% 的"边缘过渡像素合成白底后是彩色"(见待办1)。
    //       根因是 深发丝+背景色距接近 → Step4 反污染有物理天花板：解出的纯前景 RGB 仍掺背景色。
    //       Step7 的 3×3 均值把硬边摊成 ~1px 平滑过渡，这些过渡像素合成白底后把渗色暴露成彩边
    //       (蓝/红/米白)。此步过去被放在 Step7 前，直接被 3×3 均值中和 → 实测无效；
    //       现在移到 Step7 之后，渗色过渡像素被压 alpha 的效果能保留下来。
    // 思路：这些残留色像素都是"紧贴确定背景的真边缘过渡像素"。对它量"距邻域确定背景色"的色距，
    //       色距越小 = 渗色越重(颜色几乎等于背景在边缘处渗出) → alpha 压得越低(往透明收缩)，
    //       让它在合成白底/红底/蓝底时被底色冲淡，不再残留原始背景色相。
    //       只有紧贴确定背景(bgN≥2)的过渡像素才处理 → 发丝/衣纯内部(3×3 无背景)天然跳过，不误伤。
    {
      const ER = 2;              // 邻域半径（3×3×翻边=5×5），够采到真背景又不过钝
      const BG_OF = 0.05;        // 确定背景 alpha 上界
      const FG_OF = 0.96;        // 确定前景 alpha 下界（>此值的硬边像素也不再动，避免切进发丝主体）
      let shrunk = 0;
      for (let y = ER; y < h - ER; y++) {
        for (let x = ER; x < w - ER; x++) {
          const i = y*w + x;
          const a = refined[i];
          if (a <= BG_OF || a >= FG_OF) continue;   // 只处理 过渡带 α∈(0.05,0.96)
          // 收集 ER×ER 内 确定背景像素 的平均 背景色(从 src 取，反污染前的真实混合色)
          let bN = 0, bR = 0, bG = 0, bB = 0;
          for (let dy = -ER; dy <= ER; dy++) for (let dx = -ER; dx <= ER; dx++) {
            const j = (y+dy)*w + (x+dx);
            if (refined[j] <= BG_OF) { const j4 = j*4; bR += src[j4]; bG += src[j4+1]; bB += src[j4+2]; bN++; }
          }
          if (bN < 2) continue;                      // 无确定背景参照(纯前景内部) → 不是边缘残色，跳过
          const cbR = bR/bN, cbG = bG/bN, cbB = bB/bN;
          const i4 = i*4;
          const dr = cbR - src[i4], dg = cbG - src[i4+1], db = cbB - src[i4+2];
          const d2 = dr*dr + dg*dg + db*db;          // 距邻域真实背景色 的 色距平方
          // 色距阈值：紧贴背景的边缘像素 若 反污染已清干净，其颜色应偏离背景较远(接近自身发/肤/衣色)。
          //   只有依然≈背景色(渗色没救回来)的才收缩。阈值基于实测渗色样本(距local bg ΔRGB≈几千~2万)刻度。
          //   收缩幅度由渗色深度平方根(真实色距)连续滑动调制，避免硬阈值一刀切出洞：
          //     色距≈0(完全=背景渗出) → alpha×0.25(最大限度冲淡)；色距≈90 → 几乎不动。
          //   下界 0.25 + 前景深度保护(见下)：保证发丝/衣边即便重渗色也保留剪影，不砍穿成洞。
          const distBg = Math.sqrt(d2);
          if (distBg < 90) {
            // ★ 前景深度保护：3×3 内 ≥4 格强前景(α>0.9) = 本像素 位于发丝/衣边内层，
            //   只是贴边一层渗色，不是外层纯残色 → 收缩幅度减半，砍不穿发丝主体(修 v2 的发顶洞)
            let fgN = 0;
            for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
              const kk = (y+dy)*w + (x+dx);
              if (kk >= 0 && kk < n && refined[kk] > 0.9) fgN++;
            }
            let k = 0.25 + (distBg / 90) * 0.65;      // 0.25~0.90 渐变系数
            if (fgN >= 4) k = 0.55 + k * 0.4;          // 内层发丝：收幅减半，保住剪影
            refined[i] *= k;
            shrunk++;
          }
        }
      }
      if (shrunk) console.log('[matting] [Step6.5 边缘收缩] 收缩 ' + shrunk + ' 个 残留背景色过渡像素（蓝/红/米白 边缘色彩 被底色冲淡）');
    }

    // --- Step 5: 合成输出（RGB=纯前景fg*, A=精化 alpha）---
    // ⚠ 必须 alloc 全新 buffer：不能用 Buffer.from(src) 直接拷贝，否则确定背景像素 (α=0)
    //   的 RGB 仍然是原图混合色，PNG 压缩时会把这些 RGB 保留下来，
    //   小程序端若 alpha 合成错误就会"透出那圈原始背景色"。
    const out = Buffer.alloc(n * 4);
    for (let i = 0; i < n; i++) {
      out[i*4]   = fgRgb[i*4];
      out[i*4+1] = fgRgb[i*4+1];
      out[i*4+2] = fgRgb[i*4+2];
      const a = refined[i];
      out[i*4+3] = a <= 0.02 ? 0 : (a >= 0.98 ? 255 : Math.round(a * 255));
    }
    // 诊断：后处理最终 alpha 分布（对比 mattingWithRmbg 里 raw 直方图，看后处理到底改了多少软边）
    try {
      let ff = 0, ss = 0, bb = 0;
      for (let i = 0; i < n; i++) { const v = refined[i]; if (v >= 0.9) ff++; else if (v > 0.1) ss++; else bb++; }
      console.log('[matting] [postprocess 输出] fg(≥0.9)=' + (ff / n * 100).toFixed(1) +
        '% soft(0.1,0.9)=' + (ss / n * 100).toFixed(1) + '% bg(<0.1)=' + (bb / n * 100).toFixed(1) + '%');
    } catch (eDiag) {}
    return encodePng(w, h, out);
  } catch (e) {
    console.warn('[matting-core] postprocess 降级跳过:', e.message);
    return pngBuffer;
  }
}

/**
 * 导向滤波（Guided Filter, He et al. 2010）—— 边缘保留平滑，O(N)。
 * 作用：以 guidance I 的边缘为参考，平滑输入 p（alpha），
 *       让 alpha 轮廓沿「真实颜色边缘」连续过渡，替代盒式模糊 → 消除阶梯锯齿。
 * 这是 closed-form matting (Levin et al.) 的快速线性近似。
 */
function guidedFilterAlpha(I, p, w, h, r, eps) {
  const n = w * h;
  const I2 = new Float64Array(n);
  const P  = new Float64Array(n);
  const Ip = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    I2[i] = I[i] * I[i];
    P[i]  = p[i];
    Ip[i] = I[i] * p[i];
  }
  const satI  = integralImageF64(I, w, h);
  const satI2 = integralImageF64(I2, w, h);
  const satP  = integralImageF64(P, w, h);
  const satIp = integralImageF64(Ip, w, h);

  const meanA = new Float64Array(n);
  const meanB = new Float64Array(n);
  for (let y = 0; y < h; y++) {
    const y0 = Math.max(0, y - r), y1 = Math.min(h - 1, y + r);
    for (let x = 0; x < w; x++) {
      const x0 = Math.max(0, x - r), x1 = Math.min(w - 1, x + r);
      const area = (x1 - x0 + 1) * (y1 - y0 + 1);
      const mI  = boxSumF64(satI,  x0, y0, x1, y1, w) / area;
      const mI2 = boxSumF64(satI2, x0, y0, x1, y1, w) / area;
      const mP  = boxSumF64(satP,  x0, y0, x1, y1, w) / area;
      const mIp = boxSumF64(satIp, x0, y0, x1, y1, w) / area;
      const varI = mI2 - mI * mI;
      const covIp = mIp - mI * mP;
      const a = covIp / (varI + eps);
      meanA[y*w+x] = a;
      meanB[y*w+x] = mP - a * mI;
    }
  }

  const satA = integralImageF64(meanA, w, h);
  const satB = integralImageF64(meanB, w, h);
  const q = new Float32Array(n);
  for (let y = 0; y < h; y++) {
    const y0 = Math.max(0, y - r), y1 = Math.min(h - 1, y + r);
    for (let x = 0; x < w; x++) {
      const x0 = Math.max(0, x - r), x1 = Math.min(w - 1, x + r);
      const area = (x1 - x0 + 1) * (y1 - y0 + 1);
      const mA = boxSumF64(satA, x0, y0, x1, y1, w) / area;
      const mB = boxSumF64(satB, x0, y0, x1, y1, w) / area;
      const idx = y*w + x;
      const qi = mA * I[idx] + mB;
      q[idx] = qi < 0 ? 0 : (qi > 1 ? 1 : qi);
    }
  }
  return q;
}

function integralImageF64(src, w, h) {
  const sat = new Float64Array(w * h);
  for (let y = 0; y < h; y++) {
    let rowSum = 0;
    const base = y * w;
    for (let x = 0; x < w; x++) {
      rowSum += src[base + x];
      sat[base + x] = (y > 0 ? sat[base + x - w] : 0) + rowSum;
    }
  }
  return sat;
}

function boxSumF64(sat, x0, y0, x1, y1, w) {
  const D = sat[y1*w + x1];
  const B = y0 > 0 ? sat[(y0-1)*w + x1] : 0;
  const C = x0 > 0 ? sat[y1*w + (x0-1)] : 0;
  const A = x0 > 0 && y0 > 0 ? sat[(y0-1)*w + (x0-1)] : 0;
  return D - B - C + A;
}

/**
 * decontaminateForeground —— P1：FBA-Matting 思想 · 局部二次多项式曲面拟合的"真·前景反污染"
 *
 * ——旧版反污染的致命缺陷：用"全局/局部确定前景平均色"替代公式里的确定背景色 → 反污染方向完全错。
 *   发丝边缘处颜色是 mix = hair*α + bg*(1-α)，解纯前景 hair 必须知道"像素 (x,y) 的真实背景 bg_true"，
 *   然后用 matting 标准公式反推：fg = (mix - bg_true·(1-α)) / α。
 *
 * ——新版做法：
 *   1) 对每个半透明像素 (x,y)，在半径 REGRAD 窗口内取最近 K=12 个 确定背景像素 (α≤0.03)；
 *   2) 把这些背景点 (xx, yy, bgRGB) 做 2 次多项式加权最小二乘（特征：1, nx, ny, nx², ny², nx·ny），
 *      归一化到以 (x,y) 为原点、1/REGRAD 尺度（防止坐标太大造成正规方程病态 / 奇异）；
 *      拟合时对近点按距离高斯加权，σ=REGRAD/3；
 *   3) 拟合出的曲面预测 (nx=0,ny=0) 处值 = 就是 (x,y) 的真实背景色估计 bg_true (3 通道独立解)；
 *   4) 代入标准公式解 fg，clamp 0-255；
 *   5) 若正规方程奇异或窗口内确定背景点 < 6，降级为 KNN 距离加权平均背景。
 *
 * α>0.98 确定前景：直接保留原图 RGB（不再是混合色，不需要解）
 * α<0.04 确定背景：RGB 设 0（最终合成乘 α=0 无影响；PNG 压缩不残留旧背景色）
 */
function decontaminateForeground(src, alpha, w, h, R) {
  return decontaminateForegroundSplit(src, alpha, w, h, h);  // 不分区时整个高度都当身体
}

/**
 * 分区反污染：头发区 小窗口 R=10，身体区 R=16
 *   发丝像素极细 + 颜色变化剧烈：窗口太大或 K 太多，
 *   拟合用的确定背景点 (头发外) 会被远处平坦色背景主导 → 邻接发丝处的背景色曲面估错，
 *   解出的 fg 会掺背景色（头发角残留红/白残边）。
 *   头发 REGRAD=10 / K=10；身体 REGRAD=16 / K=16（西装尖角采样更多背景 → 估准 → 西装尖角色渗消除）
 */
function decontaminateForegroundSplit(src, alpha, w, h, headY) {
  const n = w * h;
  const out = Buffer.alloc(n * 4);
  const POLY_DIM = 6;
  const REGRAD_DEF_H = 10, KMAX_DEF_H = 10;
  const REGRAD_DEF_B = 16, KMAX_DEF_B = 16;

  for (let y = 0; y < h; y++) {
    const isHead = y < headY;
    const REGRAD = Math.max(6, isHead ? REGRAD_DEF_H : REGRAD_DEF_B);
    const K_MAX  = isHead ? KMAX_DEF_H : KMAX_DEF_B;
    const SIG2   = 2 * ((REGRAD / 3) ** 2);
    for (let x = 0; x < w; x++) {
      const i  = y * w + x;
      const i4 = i * 4;
      const a  = alpha[i];
      if (a >= 0.98) {
        out[i4] = src[i4]; out[i4+1] = src[i4+1]; out[i4+2] = src[i4+2];
        continue;
      }
      if (a <= 0.04) {
        out[i4] = 0; out[i4+1] = 0; out[i4+2] = 0;
        continue;
      }
      // ------------- 半透明窄带 α∈[0.04,0.98]：真正需要反污染 -------------
      const x0 = Math.max(0, x - REGRAD), x1 = Math.min(w - 1, x + REGRAD);
      const y0 = Math.max(0, y - REGRAD), y1 = Math.min(h - 1, y + REGRAD);
      // Step A：选窗口内最近 K_MAX 个 确定背景 (α≤0.03)
      const nb = [];  // [d2, idx]
      const TH_BG = 0.03;
      for (let yy = y0; yy <= y1; yy++) {
        for (let xx = x0; xx <= x1; xx++) {
          const j = yy * w + xx;
          if (alpha[j] > TH_BG) continue;
          const dx = xx - x, dy = yy - y;
          const d2 = dx*dx + dy*dy;
          if (nb.length < K_MAX) {
            nb.push([d2, j]);
          } else {
            let mi = 0;
            for (let k = 1; k < nb.length; k++) if (nb[k][0] > nb[mi][0]) mi = k;
            if (d2 < nb[mi][0]) nb[mi] = [d2, j];
          }
        }
      }
      let bgR, bgG, bgB;
      if (nb.length < POLY_DIM) {
        // 不够做 6 维回归 → KNN 距离加权平均背景
        let sW = 0, sR = 0, sG = 0, sB = 0;
        for (let k = 0; k < nb.length; k++) {
          const wtk = Math.exp(-nb[k][0] / SIG2) + 1e-6;
          const j4 = nb[k][1] * 4;
          sW += wtk;
          sR += wtk * src[j4];
          sG += wtk * src[j4+1];
          sB += wtk * src[j4+2];
        }
        if (sW > 0) { bgR = sR / sW; bgG = sG / sW; bgB = sB / sW; }
        else {
          // 极端：人像内凹洞，无任何确定背景点 → 用原图就行（反正 α 大，1-α 很小）
          out[i4] = src[i4]; out[i4+1] = src[i4+1]; out[i4+2] = src[i4+2];
          continue;
        }
      } else {
        // Step B：2 次多项式正规方程
        const N_REG = nb.length;
        const phi = new Array(N_REG);
        const br = new Array(N_REG);
        const bg_arr = new Array(N_REG);
        const bb = new Array(N_REG);
        const wts = new Array(N_REG);
        for (let k = 0; k < N_REG; k++) {
          const j  = nb[k][1];
          const yy = Math.floor(j / w);
          const xx = j - yy * w;
          const nx = (xx - x) / REGRAD;
          const ny = (yy - y) / REGRAD;
          phi[k] = [1, nx, ny, nx*nx, ny*ny, nx*ny];
          const j4 = j * 4;
          br[k]    = src[j4];
          bg_arr[k]= src[j4+1];
          bb[k]    = src[j4+2];
          wts[k]   = Math.exp(-nb[k][0] / SIG2) + 1e-6;
        }
        const AWA  = new Float64Array(POLY_DIM * POLY_DIM);
        const AWbR = new Float64Array(POLY_DIM);
        const AWbG = new Float64Array(POLY_DIM);
        const AWbB = new Float64Array(POLY_DIM);
        for (let k = 0; k < N_REG; k++) {
          const wk = wts[k];
          const p  = phi[k];
          for (let r = 0; r < POLY_DIM; r++) {
            const pr = p[r];
            AWbR[r] += wk * pr * br[k];
            AWbG[r] += wk * pr * bg_arr[k];
            AWbB[r] += wk * pr * bb[k];
            for (let c = r; c < POLY_DIM; c++) AWA[r*POLY_DIM + c] += wk * pr * p[c];
          }
        }
        for (let r = 0; r < POLY_DIM; r++)
          for (let c = 0; c < r; c++)
            AWA[r*POLY_DIM + c] = AWA[c*POLY_DIM + r];
        const inv = matInvert6(AWA, POLY_DIM);
        if (!inv) {
          // 奇异：KNN 加权平均兜底
          let sW = 0, sR = 0, sG = 0, sB = 0;
          for (let k = 0; k < N_REG; k++) {
            const j4 = nb[k][1] * 4;
            sW += wts[k];
            sR += wts[k] * src[j4];
            sG += wts[k] * src[j4+1];
            sB += wts[k] * src[j4+2];
          }
          bgR = sR / sW; bgG = sG / sW; bgB = sB / sW;
        } else {
          // 预测 (nx=0, ny=0) → 只有 φ[0]=1 非零 → theta_0 就是答案
          bgR = matVec6(inv, AWbR)[0];
          bgG = matVec6(inv, AWbG)[0];
          bgB = matVec6(inv, AWbB)[0];
        }
      }
      // Step C：fg = (mix - bg_true*(1-α)) / α（★加防黑边保底）
      //   RMBG 硬边模型的窄软带 α 代表“边缘残量”而非真实半透明覆盖率，
      //   极端反污染会把发丝/衣服边的恢复色解成黑色 → 边缘 1px 黑噪环。
      //   保底：恢复色不得比原混合色暗超过 DARK_FLOOR，否则保留原色(有限去溢)。
      const invA  = 1 / Math.max(0.04, a);
      const oneMa = 1 - a;
      const DARK_FLOOR = 28;
      let fr = (src[i4]   - bgR * oneMa) * invA;
      let fg = (src[i4+1] - bgG * oneMa) * invA;
      let fb = (src[i4+2] - bgB * oneMa) * invA;
      if (fr < src[i4] - DARK_FLOOR) fr = src[i4];
      if (fg < src[i4+1] - DARK_FLOOR) fg = src[i4+1];
      if (fb < src[i4+2] - DARK_FLOOR) fb = src[i4+2];
      if (fr < 0) fr = 0; else if (fr > 255) fr = 255;
      if (fg < 0) fg = 0; else if (fg > 255) fg = 255;
      if (fb < 0) fb = 0; else if (fb > 255) fb = 255;
      out[i4]   = Math.round(fr);
      out[i4+1] = Math.round(fg);
      out[i4+2] = Math.round(fb);
    }
  }
  return out;
}

// --------------- 工具：6×6 矩阵逆 (Gauss-Jordan 部分主元) + 6×6·6×1 向量乘 ---------------
function matInvert6(A, DIM) {
  const N = DIM;
  const CW = 2 * N;
  const M = new Float64Array(N * CW);
  for (let r = 0; r < N; r++) {
    for (let c = 0; c < N; c++) M[r*CW + c] = A[r*N + c];
    M[r*CW + (N + r)] = 1;
  }
  for (let r = 0; r < N; r++) {
    let piv = r, pivMax = Math.abs(M[r*CW + r]);
    for (let rr = r + 1; rr < N; rr++) {
      const v = Math.abs(M[rr*CW + r]);
      if (v > pivMax) { pivMax = v; piv = rr; }
    }
    if (pivMax < 1e-10) return null;
    if (piv !== r) {
      const R1 = r*CW, R2 = piv*CW;
      for (let c = 0; c < CW; c++) {
        const t = M[R1+c]; M[R1+c] = M[R2+c]; M[R2+c] = t;
      }
    }
    const pr = M[r*CW + r];
    for (let c = r; c < CW; c++) M[r*CW + c] /= pr;
    for (let rr = 0; rr < N; rr++) {
      if (rr === r) continue;
      const f = M[rr*CW + r];
      if (Math.abs(f) < 1e-12) continue;
      for (let c = r; c < CW; c++) M[rr*CW + c] -= f * M[r*CW + c];
    }
  }
  const Inv = new Float64Array(N * N);
  for (let r = 0; r < N; r++)
    for (let c = 0; c < N; c++)
      Inv[r*N + c] = M[r*CW + (N + c)];
  return Inv;
}
function matVec6(A, b) {
  const N = 6, o = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    let s = 0;
    for (let j = 0; j < N; j++) s += A[i*N + j] * b[j];
    o[i] = s;
  }
  return o;
}


// -------- CIE-L*a*b* 工具（仅 0-255 sRGB → L/a/b* 709 近似）--------
function rgb2lab(R, G, B) {
  const rn = R / 255, gn = G / 255, bn = B / 255;
  const rl = rn > 0.04045 ? Math.pow((rn + 0.055) / 1.055, 2.4) : rn / 12.92;
  const gl = gn > 0.04045 ? Math.pow((gn + 0.055) / 1.055, 2.4) : gn / 12.92;
  const bl = bn > 0.04045 ? Math.pow((bn + 0.055) / 1.055, 2.4) : bn / 12.92;
  let X = rl * 0.4124564 + gl * 0.3575761 + bl * 0.1804375;
  let Y = rl * 0.2126729 + gl * 0.7151522 + bl * 0.0721750;
  let Z = rl * 0.0193339 + gl * 0.1191920 + bl * 0.9503041;
  X /= 0.95047; Y /= 1.0; Z /= 1.08883;
  const cbrt_fn = Math.cbrt;
  const f_t = (t) => (t > 0.008856) ? cbrt_fn(t) : (7.787 * t + 16 / 116);
  const fx = f_t(X), fy = f_t(Y), fz = f_t(Z);
  return [116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)];
}

// RMBG-1.4 模型：从云存储按需下载（见 lib/model.js）
// 不再用固定本地路径，改用 getRmbgModel() 动态获取

/**
 * 主入口：图片 Buffer → 透明底 PNG Buffer
 * @param {Buffer} imageBuffer PNG 图片
 * @returns {Promise<Buffer>} 透明底 PNG
 */
async function matting(imageBuffer) {
  let pngBuf;
  try {
    const { modelPath, arch } = await getBestModel();
    console.log('[matting] 选模型:', arch);
    pngBuf = (arch === 'modnet')
      ? await mattingWithModnet(imageBuffer, modelPath)
      : await mattingWithRmbg(imageBuffer, modelPath);
  } catch (e) {
    console.error('[matting] AI 不可用，降级启发式:', e.message);
    pngBuf = mattingHeuristic(imageBuffer);
  }
  // ★ 所有路径（AI 成功 / AI 失败启发式兜底）100% 过统一后处理
  //   → 消除锯齿 + 羽化 5px + 局部色溢去除 + 中心硬保护 92%×99%
  return applyMattingPostprocess(pngBuf);
}

/**
 * RMBG-1.4 推理：真 AI 人像分割。
 * 模型：briaai/RMBG-1.4 的 onnx model_fp16.onnx
 * 官方预处理（来自 preprocessor_config）：
 *   - 输入固定 1024×1024（do_pad=false，直接等比缩放到方形）
 *   - 归一化：(x/255 - 0.5) / 1（image_mean=0.5, image_std=1）
 *   输入 tensor 名 "input"，输出 tensor 名 "output"（SegFormer 结构，通常为输入的 1/4）
 * 流程：
 *   1. 解码 PNG → RGBA
 *   2. 等比缩放到 1024×1024 → 归一化张量 [1,3,1024,1024]
 *   3. onnxruntime-node 推理 → alpha 蒙版（读实际输出尺寸）
 *   4. 把 alpha 双线性放大回原图尺寸
 *   5. alpha 合成到原图 → 透明底 PNG
 * 依赖：onnxruntime-node；模型路径由 getRmbgModel() 提供
 */
async function mattingWithRmbg(imageBuffer, modelPath) {
  const ort = await getOrt();  // 先尝试 onnxruntime-node (native, 若 GLIBCXX 失败则自动回退 onnxruntime-web WASM)
  const { width, height, data } = decodePng(imageBuffer);

  // 关键：RMBG-2.0 INT8 量化模型是「固定输入形状」= 1024×1024。
  //       INT8 量化不支持动态尺寸，喂 768/512 会直接报 "Got: 768 Expected: 1024"
  //       并降级启发式（→ 效果差）。因此必须读取模型真实固定 shape，不能调小！
  // 先建 session 拿模型固定输入尺寸
  const session = await ort.InferenceSession.create(modelPath, { executionProviders: ['cpu'] });
  const inputName = session.inputNames[0];   // "input"
  const outputName = session.outputNames[0]; // "output"
  let SIZE = 1024;
  // 读取模型真实固定输入尺寸（INT8 量化模型不支持动态 shape）
  // onnxruntime-node 1.15.1 的 session.inputMetadata[type/dims] 经常是 undefined，
  // 所以这里不强依赖 metadata；优先按 1024（RMBG-2.0 INT8 量化 1024×1024 标准尺寸）。
  try {
    const meta = session.inputMetadata && session.inputMetadata[inputName];
    const dims = meta && (meta.dims || meta.shape || (meta.type && meta.dimensions));
    if (Array.isArray(dims) && dims.length >= 2) {
      const h = Number(dims[dims.length - 2]), w = Number(dims[dims.length - 1]);
      if (w === h && w >= 256 && w <= 1024 && isFinite(w)) SIZE = w;
    }
  } catch (e) {}
  // 诊断：把输入输出 tensor 的 shape/dtype 打出来（全部 try 包住，绝不影响主流程）
  try {
    const iMeta = session.inputMetadata && session.inputMetadata[inputName];
    const oMeta = session.outputMetadata && session.outputMetadata[outputName];
    const fmtDims = (m) => {
      if (!m) return '?';
      const d = m.dims || m.shape || (m.type && m.dimensions);
      if (Array.isArray(d)) return d.map(String).join(',');
      return '?';
    };
    console.log('[matting] [DIAG] input: name=' + inputName + ' dims=[' + fmtDims(iMeta) + '] type=' + String(iMeta && (iMeta.type || 'float32')));
    console.log('[matting] [DIAG] output: name=' + outputName + ' dims=[' + fmtDims(oMeta) + '] type=' + String(oMeta && (oMeta.type || 'float32')));
  } catch (e) { console.warn('[matting] [DIAG] meta log skip:', e.message); }
  console.log('[matting] 🇧 BiRefNet 输入分辨率 =', SIZE + '×' + SIZE, '(读取模型固定 shape, INT8 量化默认1024)');
  // 1) resize 到模型期望尺寸 + RMBG-2.0 官方归一化 mean=0.5 / std=1.0 → CHW float32
  const input = new Float32Array(1 * 3 * SIZE * SIZE);
  resizeToTensorRmbg(data, width, height, SIZE, SIZE, input);

  // 2) 推理
  const feeds = { [inputName]: new ort.Tensor('float32', input, [1, 3, SIZE, SIZE]) };
  const results = await session.run(feeds);
  const outTensor = results[outputName];
  const alphaRaw = outTensor.data;           // 输出，可能 [1,1,h,w] NCHW 或 [1,h,w,1] NHWC
  const outShape = outTensor.dims;           // 必须显式判定！
  console.log('[matting] [DIAG] outTensor shape=[' + outShape.join(',') + '] elems=' + alphaRaw.length + ' type=' + outTensor.type);

  // 从输出张量取 alpha 平面 [h,w] → 自动处理 NCHW / NHWC，避免错读内存
  //   NCHW: [1,1,H,W] → index = n*C*H*W + c*H*W + y*W + x  == y*W + x (c=0,n=0)
  //   NHWC: [1,H,W,1] → index = n*H*W*C + y*W*C + x*C + c  == y*W + x (c=0,n=0)
  //   ⚠ 但如果 shape 像 [1,1024,1024,1] / [1,1,1024,1024] 之外的变体（比如 C=3），需要取 C=0
  const shapeStr = outShape.map(String).join(',');
  // 找到 H 和 W：最后两个非 1？不，通用法：dim 从两边往里收，取最后两个 >1 的；找不到就用 shape 末尾两个
  let rank = outShape.length;
  let dimsNo1 = [];
  for (let d = 0; d < rank; d++) if (outShape[d] !== 1) dimsNo1.push(d);
  let iH, iW;
  if (dimsNo1.length >= 2) {
    iH = dimsNo1[dimsNo1.length - 2];
    iW = dimsNo1[dimsNo1.length - 1];
  } else {
    iH = rank - 2; iW = rank - 1;
  }
  const outH = outShape[iH];
  const outW = outShape[iW];
  // 定位 C 的位置：判断经典 NCHW（C 在 1 号位置且非 N/H/W） vs NHWC（C 在末尾）
  const nchwLayout = (rank === 4 && outShape[1] === 1 && outShape[2] === outH && outShape[3] === outW);
  const nhwcLayout = (rank === 4 && outShape[1] === outH && outShape[2] === outW && outShape[3] === 1);
  let layout;
  if (nchwLayout && !nhwcLayout) layout = 'NCHW';
  else if (nhwcLayout && !nchwLayout) layout = 'NHWC';
  else {
    // 其他：按 shapeStr 启发。若 C 是最后一维且 ==1，则 NHWC；否则 NCHW
    layout = (outShape[rank - 1] === 1) ? 'NHWC' : 'NCHW';
  }
  console.log('[matting] [DIAG] 解析 H=' + outH + ' W=' + outW + ' 内存布局=' + layout + ' (shape=' + shapeStr + ')');
  function getA(y, x) {
    if (layout === 'NHWC') {
      // [1,H,W,1] or more generally: idx = ((0*H + y)*W + x)*C + 0  (C=1)
      return alphaRaw[(y * outW + x) * (outShape[rank - 1] || 1)];
    } else {
      // NCHW: [1,1,H,W] etc: idx = 0*C*H*W + 0*H*W + y*W + x
      return alphaRaw[y * outW + x];
    }
  }

  // 3) 把输出 alpha 双线性放大回原图尺寸
  const alpha = new Float32Array(width * height);
  for (let y = 0; y < height; y++) {
    const sy = Math.min(outH - 1, (y * outH) / height);
    const y0 = Math.min(outH - 1, Math.floor(sy));
    const y1 = Math.min(outH - 1, y0 + 1);
    const fy = sy - y0;
    for (let x = 0; x < width; x++) {
      const sx = Math.min(outW - 1, (x * outW) / width);
      const x0 = Math.min(outW - 1, Math.floor(sx));
      const x1 = Math.min(outW - 1, x0 + 1);
      const fx = sx - x0;
      const a00 = getA(y0, x0);
      const a10 = getA(y0, x1);
      const a01 = getA(y1, x0);
      const a11 = getA(y1, x1);
      const top = a00 * (1 - fx) + a10 * fx;
      const bot = a01 * (1 - fx) + a11 * fx;
      alpha[y * width + x] = top * (1 - fy) + bot * fy;
    }
  }

  // 【诊断日志】alpha 统计 + 分位数 + 前景占比 + 直方图
  //  ⚠ 全部 try-catch：onnxruntime 某些版本会在数组里混入 undefined/NaN，
  //    这里任何报错都会让主流程 catch → 降级启发式 → 看不到 AI 真效果，所以绝不允许抛错。
  try {
    const N = alpha.length;
    const sorted = new Float32Array(Math.min(8000, N));
    const sstep = Math.max(1, Math.floor(N / sorted.length));
    let sidx = 0, sum = 0, cntFin = 0;
    for (let i = 0; i < N; i += sstep) {
      let v = alpha[i];
      if (typeof v !== 'number' || !isFinite(v)) v = 0;  // 兜底：NaN/undefined → 0
      sorted[sidx++] = v;
      sum += v;
      cntFin++;
    }
    const sN = sidx;
    sorted.sort();
    const q = (p) => sorted[Math.max(0, Math.min(sN - 1, Math.floor(p * sN)))] || 0;
    const pct = (th) => {
      let c = 0;
      for (let i = 0; i < sN; i++) if (sorted[i] >= th) c++;
      return (sN ? c / sN * 100 : 0).toFixed(1) + '%';
    };
    const hist = [0,0,0,0,0,0,0,0,0,0];
    for (let i = 0; i < sN; i++) {
      const b = Math.min(9, Math.max(0, Math.floor((isFinite(sorted[i]) ? sorted[i] : 0) * 10)));
      hist[b]++;
    }
    const f3 = (v) => (typeof v === 'number' && isFinite(v) ? v : 0).toFixed(3);
    const mean = cntFin ? (sum / cntFin) : 0;
    // 独立采样重算真实 min/max（避免 sorted[sorted.length - 1] 未初始化 → 读成 0 误导诊断）
    let vminR = Infinity, vmaxR = -Infinity, rn = 0;
    const rstep = Math.max(1, Math.floor(N / 20000));
    for (let i = 0; i < N; i += rstep) {
      const vv = alpha[i];
      if (typeof vv !== 'number' || !isFinite(vv)) continue;
      if (vv < vminR) vminR = vv;
      if (vv > vmaxR) vmaxR = vv;
      rn++;
    }
    if (!rn) { vminR = 0; vmaxR = 0; }
    console.log('[matting] [DIAG] alpha 上采样后统计: N=' + N +
      ' sample=' + sN +
      ' mean=' + f3(mean) +
      ' min=' + f3(vminR) + ' p5=' + f3(q(0.05)) + ' p25=' + f3(q(0.25)) +
      ' p50=' + f3(q(0.5)) + ' p75=' + f3(q(0.75)) + ' p95=' + f3(q(0.95)) +
      ' max=' + f3(vmaxR) +
      ' (≥0.1占比=' + pct(0.1) + ' ≥0.5占比=' + pct(0.5) + ' ≥0.9占比=' + pct(0.9) + ')');
    console.log('[matting] [DIAG] alpha 直方图(×10桶): [' + hist.join(',') + ']');
  } catch (diagErr) {
    console.warn('[matting] [DIAG] alpha 统计跳过（不影响主流程）:', diagErr.message || diagErr);
  }

  // 4) alpha 合成透明底 PNG（先 sigmoid → 0~1 → 证件照中心矩形保护 → 转 0~255）
  //   ★ 重要：这里是"模型输出后的第一版 alpha"，精细边缘收敛交给外层 applyMattingPostprocess。
  //           所以本步只要"别把人像主体误杀成透明"，窄带阈值在外面再切一刀。
  //   中心硬保护 82% × 89%（和后处理保持一致，修 case5 长条假阳）
  const cx = Math.round(width / 2), cy = Math.round(height / 2);
  //   halfW 0.46→0.41 halfH 0.495→0.445：两侧留出 18%，上下各留出 5.5%
  const halfW = Math.round(width * 0.41), halfH = Math.round(height * 0.445);
  const out = Buffer.from(data);
  // 自适应判断：量化 BiRefNet/RMBG 输出可能已经是 sigmoid 后的 0~1，也可能是不定范围 logits
  //   采样判断：若 alpha 值域明显 >3，说明是 logits，需要 sigmoid 压回 0~1
  let vmax = -1e9, vmin = 1e9;
  const stride = Math.max(1, Math.floor(alpha.length / 4000));
  for (let i = 0; i < alpha.length; i += stride) {
    const v = alpha[i];
    if (v > vmax) vmax = v; if (v < vmin) vmin = v;
  }
  const needSigmoid = (vmax - vmin) > 3.0;
  const fm2 = (v) => (typeof v==='number'&&isFinite(v) ? v : 0).toFixed(2);
  console.log('[matting] RMBG/BiRefNet 输出范围 min=' + fm2(vmin) + ' max=' + fm2(vmax) + ' → ' + (needSigmoid ? 'logits(需sigmoid)' : '已是0~1概率'));
  // --- 几何先验 (证件照强规则) 预清理：边带几何区域里"看起来就是背景色"的像素强制 α=0
  //   作用：修掉 RMBG 把纯色背景的不规则斑点误判成前景（你截图里"左上角乱码般的灰白块"）
  //   做法：四角 20×20 采样估计真实 bgRGB；若外边带像素和其感知色距 <55，无论模型说啥一律清透明
  let estBgR=0, estBgG=0, estBgB=0, estBgN=0;
  {
    const CS = Math.min(20, Math.floor(Math.min(width,height)*0.04));
    const pts = [[0,0],[width-CS,0],[0,height-CS],[width-CS,height-CS]];
    for (const [ox,oy] of pts) for (let yy=0; yy<CS; yy++) for (let xx=0; xx<CS; xx++) {
      const j = ((oy+yy)*width + (ox+xx))*4;
      estBgR += data[j]; estBgG += data[j+1]; estBgB += data[j+2]; estBgN++;
    }
  }
  if (estBgN > 0) { estBgR/=estBgN; estBgG/=estBgN; estBgB/=estBgN; }
  // 外边带几何先验：证件照这些几何位置 99% 不属于人像主体
  //  左/右 18%、顶部 18%、底部 10%（底部 10% 常是"水印"一行字，水印是背景平面一部分，直接纳入外带）
  //  色距阈值 55² → 75²：RMBG 对"背景里稍暗一点的像素"会误判成前景，放宽能多吃掉这一类假阳
  const bandX   = Math.floor(width  * 0.18);
  const bandY_T = Math.floor(height * 0.18);
  const bandY_B = Math.floor(height * 0.10);
  const tolBgSq = 75 * 75;

  // ★ 新增：几何先验 ΔE 版 "全图辅助背景色"（4 个边中点 20×20 采样）
  //   针对用户反馈「复杂背景根本不能分割」：四角背景色 太局限，人物身侧/上下 区域可能是 花色/渐变/杂物，
  //   与四角 ΔE 差很大 → 外边带的色距规则 杀不掉这些假阳。这里额外取 4 个边中点 做 ΔE 判定。
  const CS = Math.min(20, Math.floor(Math.min(width,height)*0.04));
  let sBg2R = 0, sBg2G = 0, sBg2B = 0, sBg2N = 0;
  {
    const sidePts = [
      [Math.floor(width/2) - CS/2, 0], [Math.floor(width/2) - CS/2, height - CS],
      [0, Math.floor(height/2)-CS/2], [width - CS, Math.floor(height/2)-CS/2]
    ];
    for (const [ox0, oy0] of sidePts) {
      const ox = Math.max(0, Math.min(width-CS, Math.floor(ox0)));
      const oy = Math.max(0, Math.min(height-CS, Math.floor(oy0)));
      for (let yy = 0; yy < CS; yy++) for (let xx = 0; xx < CS; xx++) {
        const j = ((oy + yy) * width + (ox + xx)) * 4;
        sBg2R += data[j]; sBg2G += data[j+1]; sBg2B += data[j+2]; sBg2N++;
      }
    }
  }
  const estBg2R = (sBg2N > 0) ? sBg2R / sBg2N : estBgR;
  const estBg2G = (sBg2N > 0) ? sBg2G / sBg2N : estBgG;
  const estBg2B = (sBg2N > 0) ? sBg2B / sBg2N : estBgB;

  for (let i = 0; i < width * height; i++) {
    let a = alpha[i];
    if (needSigmoid) a = 1 / (1 + Math.exp(-a));   // logits → 0~1
    const y = Math.floor(i / width), x = i - y * width;
    const inCenter = Math.abs(x - cx) < halfW && Math.abs(y - cy) < halfH;
    const j = i * 4;
    const rr = data[j], gg = data[j+1], bb = data[j+2];
    const rm = (rr + estBgR) / 2 / 256;
    // 加权 RGB 色距（人眼对绿色最敏感）
    const dr = rr - estBgR, dg = gg - estBgG, db = bb - estBgB;
    const d2_1 = (2+rm)*dr*dr + 4*dg*dg + (3-rm)*db*db;
    const dr2 = rr - estBg2R, dg2 = gg - estBg2G, db2 = bb - estBg2B;
    const rm2 = (rr + estBg2R) / 2 / 256;
    const d2_2 = (2+rm2)*dr2*dr2 + 4*dg2*dg2 + (3-rm2)*db2*db2;
    const d2_min = Math.min(d2_1, d2_2);

    if (inCenter) {
      // ★ 只把绝对确定前景 α≥0.90 锁 255；α∈(0.03,0.90) 的发丝/脸/服装软边 保持软 alpha →
      //   外层 applyMattingPostprocess 的 decontaminate 会对 α∈[0.04,0.98] 做 bg_true 反污染，
      //   去色溢后再 Step4.5 硬化成锐利边缘。（修 case2 蓝边 case1 脸颊残影）
      // ★ 不复保护把 α≥0.90 锁 255：保留发丝/脸边缘软 α → decontaminate 反污染去背景混色(蓝边)
      a = a < 0.03 ? 0 : Math.round(a * 255);
    } else {
      // ★ 外边带 inBand（左/右18% 顶18% 底10%）+ inCenter 外 其他区域（人物身侧中间条）都跑色距杀假阳：
      //   复杂背景下 身侧植物/花色/杂物 假阳 α≈0.15~0.35 只要"与四角/四边 估计背景色"色距 <75²
      //   → 无论 α 多少一律 α=0（身侧假阳 终于被杀）
      const inBand = (x < bandX) || (x >= width - bandX) || (y < bandY_T) || (y >= height - bandY_B);
      // ★★ 改动0（修白衣/身体误扣，数据确证版）：色距杀假阳【必须尊重模型高 alpha】。
      //    根因（实测 8c17 白衣 47062 个误杀像素 全部模型α≥0.9）：原逻辑 `inCenter 外 && d2_min<75² → a=0`
      //    不区分"模型是否确信此像素是前景"，导致 RMBG 已判 α≥0.9 的白衣/身体（色距恰<75²）被一刀透成透明，
      //    且发生在连通域清理之前，后面救不回来。
      //    修法：无论 inBand 还是场景过渡带，只要【模型α≥0.9（RMBG 确信前景）】就绝不因色距杀，只做 alpha 稳定化；
      //    色距杀仅作用于模型存疑区（α<0.9）——背景假阳（花色/杂物，模型通常 α<0.9）仍被清。
      if (a >= 0.9) {
        // 模型确信前景：保留，仅按阈值稳定化（高置信前景不进 alpha=0 色距杀）
        a = a < 0.03 ? 0 : a > 0.95 ? 255 : Math.round(a * 255);
      } else {
        // 模型存疑区（α<0.9）：inBand(靠边)激进清真背景假阳；过渡带(人物身侧)用更严色距+邻域前景保护
        const inBand2 = (x < bandX) || (x >= width - bandX) || (y < bandY_T) || (y >= height - bandY_B);
        if (inBand2) {
          if (a > 0.01 && d2_min < tolBgSq) { a = 0; }
          else if (a > 0.01) { a = a < 0.05 ? 0 : a > 0.95 ? 255 : Math.round(a * 255); }
          else { a = a < 0.02 ? 0 : a > 0.98 ? 255 : Math.round(a * 255); }
        } else {
          // 过渡带（人物身侧/下摆紧邻区，模型α<0.9）：仅"极近背景色且无前景邻居"才杀，避免误伤贴边衣物
          const TOLL_TRANS = 45 * 45;
          let nearFg = false;
          const yy0 = Math.max(0, y - 1), yy1 = Math.min(height - 1, y + 1);
          const xx0 = Math.max(0, x - 1), xx1 = Math.min(width - 1, x + 1);
          for (let ny = yy0; ny <= yy1 && !nearFg; ny++)
            for (let nx = xx0; nx <= xx1; nx++)
              if (alpha[ny * width + nx] >= 0.9) { nearFg = true; break; }
          if (!nearFg && a > 0.01 && d2_min < TOLL_TRANS) { a = 0; }
          else if (a > 0.01) { a = a < 0.05 ? 0 : a > 0.95 ? 255 : Math.round(a * 255); }
          else { a = a < 0.02 ? 0 : a > 0.98 ? 255 : Math.round(a * 255); }
        }
      }
    }
    out[i * 4 + 3] = a;
  }

  // ---------- 连通域清理：只保留"最大人像本体"，其余小斑点（你看到的灰色斑块）一律 α=0 ----------
  // 证件照=单个人像居中，所以只要是"离群的小前景团块"100% 是 RMBG 假阳，绝不是发丝/肢体。
  try {
    const N = width * height;
    const labels = new Int32Array(N);          // -1 = 未访问, ≥0 = 区域编号
    for (let i = 0; i < N; i++) labels[i] = -1;
    const comps = [];                          // [{area, pixels:[...idx], cx, cy}]
    // seedThr：64(旧 0.25) → 85(≈0.33)
    //   - 64 太激进：case5 深色服装+深色背景，大片背景假阳 α≈0.20~0.32 被当种子
    //     → 左右背景假阳连成 长条大连通域 → 连通域清理以为是 人像次大块 保留 → 长条假阳 永远杀不掉
    //   - 85≈0.33：刚好是 黑发外圈蓝边 α≈0.30~0.40 的阈值上沿，蓝边伪软边 仍连小连通 → 被杀；
    //     大片背景假阳 α<0.33 不再入种子，无法连通 → 距离场 干净杀
    const seedThr = 85;                        // α≥85 (≥0.33 × 255)
                                              // 伪边会连成小连通 → 非最大块 → 最后被 α=0 清掉（解决红底黑发外圈 一圈蓝薄边）
    const dx8 = [-1, 1, 0, 0, -1, -1, 1, 1];
    const dy8 = [0, 0, -1, 1, -1, 1, -1, 1];
    let lc = 0;
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const i0 = y * width + x;
        if (labels[i0] !== -1) continue;
        if ((out[i0 * 4 + 3] | 0) < seedThr) continue;
        // BFS
        const q = [i0];
        labels[i0] = lc;
        let ar = 0, mx = 0, my = 0;
        // 直接在队列上扩展 push (避免 JS shift 开销)
        for (let qh = 0; qh < q.length; qh++) {
          const idx = q[qh];
          const yy = Math.floor(idx / width), xx = idx - yy * width;
          ar++; mx += xx; my += yy;
          for (let k = 0; k < 8; k++) {
            const nx = xx + dx8[k], ny = yy + dy8[k];
            if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
            const ni = ny * width + nx;
            if (labels[ni] !== -1) continue;
            if ((out[ni * 4 + 3] | 0) < seedThr) continue;
            labels[ni] = lc;
            q.push(ni);
          }
        }
        comps.push({ id: lc, area: ar, cx: mx / Math.max(1, ar), cy: my / Math.max(1, ar) });
        lc++;
      }
    }
    if (comps.length > 1) {
      // 1) 找最大连通域 main (人像本体)
      comps.sort((A, B) => B.area - A.area);
      const main = comps[0];
      // 估计人像本体特征尺寸：假设面积 对应到椭圆近似，宽高比 ~0.7
      // 用 sqrt(area) 当特征尺度，次大连通域要与质心距 ≤ 0.6×√面积 且 面积 ≥ main×1% 才保
      const scale = Math.sqrt(main.area);
      const keepIds = new Set([main.id]);
      // 次大连通域保留条件 进一步放宽：修 案例3/4 白衬衫肩膀大块被误杀 形成透明缺口
      //   - 案例3/4 白衬衫左右两"肩膀延伸出来的袖口/胳膊块"，面积大约是 主域的 1.5‰~3‰
      //   - 旧 0.002 (2‰) 太严：这两个胳膊延伸块 刚好在 2‰ 边界附近被误判"太小"杀掉 → 形成肩膀缺口
      //   - 调到 0.001 (1‰)：蓬松发丝尖、细领带结、衬衫袖口/胳膊延伸块 都能保留。1‰ 仍远大于"背景假阳散点块
      //     的面积（通常 < 1/100000 = 0.001%），所以不会把假阳 当成次块保留
      const minAreaOK = main.area * 0.001;
      const maxDistX = width * 0.22;
      const maxDistY = height * 0.35;
      for (let k = 1; k < comps.length; k++) {
        const c = comps[k];
        if (c.area < minAreaOK) continue;
        const dx = Math.abs(c.cx - main.cx);
        const dy = Math.abs(c.cy - main.cy);
        if (dx < maxDistX && dy < maxDistY && Math.max(dx, dy) < scale * 0.85) keepIds.add(c.id);
      }
      // 2) 所有非保留连通域 → 把像素 α 写成 0
      for (let i = 0; i < N; i++) {
        const lb = labels[i];
        if (lb === -1) continue;        // 本来就不是前景种子，不动（让外层后处理再修）
        if (!keepIds.has(lb)) out[i * 4 + 3] = 0;
      }
      console.log('[matting] [几何清理] 连通域=' + lc +
        ' 保留=' + keepIds.size + '/' + comps.length +
        ' 主域面积=' + main.area + ' 质心(' + Math.round(main.cx) + ',' + Math.round(main.cy) + ')' +
        ' 清除=' + (lc - keepIds.size) + ' 个假阳小斑块');
    } else if (comps.length === 1) {
      console.log('[matting] [几何清理] 连通域=1 (干净无假阳斑块)，跳过');
    } else {
      console.warn('[matting] [几何清理] 没有找到任何前景种子 (0 连通域!) → 跳过');
    }
  } catch (ccErr) {
    console.warn('[matting] [几何清理] 连通域异常跳过:', ccErr.message || ccErr);
  }

  // ---------- RMBG 原始 alpha 距离场假阳清理（核心修 case5 长条假阳 + case1/2 头发外圈蓝边）----------
  // 必须放在"中心保护+几何先验合成完 alpha 但尚未出 mattingWithRmbg"这一步：
  //  一旦出去到 applyMattingPostprocess，外层会把中心 82%×89% α>0.03→1；如果服装颜色≈背景颜色，
  //  两侧背景假阳 α≈0.05~0.20 会被锁死 → 后续距离场从假阳扩张 → 长条假阳永远杀不掉
  // 核阈值 SEED=0.70：只挑 真人像确定前景（RMBG 对主体输出 α>0.9）
  // 核外 DMAX=12：只保留紧贴人像本体 12px 内 α≥0.08 的像素（发丝/西装尖），其余一律 α=0
  {
    const DMAX = 12;
    const SEED = 0.70;
    const KILL = 0.08;
    const N    = width * height;
    const dist = new Int16Array(N);
    for (let i = 0; i < N; i++) dist[i] = -1;
    const q = [];
    const SEED_BYTE = Math.round(SEED * 255);
    const KILL_BYTE = Math.round(KILL * 255);
    for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
      const i = y * width + x;
      if ((out[i * 4 + 3] | 0) >= SEED_BYTE) { dist[i] = 0; q.push(i); }
    }
    const DX = [-1, 1, 0, 0, -1, -1, 1, 1];
    const DY = [ 0, 0,-1, 1, -1,  1,-1, 1];
    for (let qh = 0; qh < q.length; qh++) {
      const idx = q[qh], d = dist[idx];
      if (d >= DMAX) continue;
      const yy = (idx / width) | 0, xx = idx - yy * width;
      for (let k = 0; k < 8; k++) {
        const nx = xx + DX[k], ny = yy + DY[k];
        if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
        const ni = ny * width + nx;
        if (dist[ni] !== -1) continue;
        dist[ni] = d + 1;
        q.push(ni);
      }
    }
    let killed = 0;
    for (let i = 0; i < N; i++) {
      if (dist[i] === -1 && (out[i * 4 + 3] | 0) >= KILL_BYTE) {
        out[i * 4 + 3] = 0;
        killed++;
      }
    }
    console.log('[matting] [RMBG-raw 距离场 D=' + DMAX + '] 清掉 ' + killed + ' 个 核外 α≥' + Math.round(KILL*100) + '% 假阳像素（中心锁死未介入，判断最干净）');
  }

  return encodePng(width, height, out);
}

/** MODNet 专用 RGBA(HWC) → 缩放到 (dw, dh) 的 CHW float32 张量。归一化 (x/255 - 0.5) / 0.5 = [-1, 1] */
function resizeToTensor(src, sw, sh, dw, dh, out) {
  for (let y = 0; y < dh; y++) {
    const sy = Math.min(sh - 1, Math.floor((y * sh) / dh));
    const row = src.subarray(sy * sw * 4, sy * sw * 4 + sw * 4);
    for (let x = 0; x < dw; x++) {
      const sx = Math.min(sw - 1, Math.floor((x * sw) / dw));
      const si = sx * 4;
      const o = (y * dw + x);
      out[o] = row[si] / 255 - 0.5;          // C0
      out[dw * dh + o] = row[si + 1] / 255 - 0.5; // C1
      out[dw * dh * 2 + o] = row[si + 2] / 255 - 0.5; // C2
    }
  }
}


/** RMBG-2.0 / BiRefNet 专用 RGBA(HWC) → (dw,dh) 的 CHW float32 张量。
 *  RMBG-2.0(BiRefNet) 官方预处理：mean=[0.5,0.5,0.5], std=[1.0,1.0,1.0]
 *    → 即 (x / 255 - 0.5)，值域 [-0.5, 0.5]
 *  （旧 ImageNet 归一化 std 会放大 ~4x，相当于喂乱码 → 模型输出全糊/全错）
 *  输入尺寸 1024×1024 (直接 resize，不要 padding / letterbox)
 */
function resizeToTensorRmbg(src, sw, sh, dw, dh, out) {
  for (let y = 0; y < dh; y++) {
    const sy = Math.min(sh - 1, Math.floor((y * sh) / dh));
    const row = src.subarray(sy * sw * 4, sy * sw * 4 + sw * 4);
    for (let x = 0; x < dw; x++) {
      const sx = Math.min(sw - 1, Math.floor((x * sw) / dw));
      const si = sx * 4;
      const r = row[si] / 255, g = row[si + 1] / 255, b = row[si + 2] / 255;
      const o = (y * dw + x);
      out[o]             = r - 0.5; // C0 R  (RMBG-2.0: std=1.0，不再 / 0.229)
      out[dw*dh + o]     = g - 0.5; // C1 G
      out[2*dw*dh + o]   = b - 0.5; // C2 B
    }
  }
}


/**
 * MODNet Photographic 推理（256x256 输入，输出 alpha）
 *  - preprocess: 直接 resize 原图 → 256x256 双线性，normalize (x/255-0.5)/0.5 → [-1,1]
 *  - postprocess: 双线性 256x256 → origW×origH，中心矩形保护 + 形态学开闭
 */
async function mattingWithModnet(imageBuffer, modelPath) {
  const ort = await getOrt();
  const { width: w, height: h, data } = decodePng(imageBuffer);
  const SIZE = 256;

  // 1) preprocess: w,h → SIZE,SIZE 双线性 + [-1,1]
  const input = new Float32Array(3 * SIZE * SIZE);
  const src = data;
  for (let y = 0; y < SIZE; y++) {
    const sy = (y + 0.5) * h / SIZE - 0.5;
    const y0 = Math.max(0, Math.floor(sy)), y1 = Math.min(h - 1, y0 + 1);
    const fy = Math.max(0, Math.min(1, sy - y0));
    for (let x = 0; x < SIZE; x++) {
      const sx = (x + 0.5) * w / SIZE - 0.5;
      const x0 = Math.max(0, Math.floor(sx)), x1 = Math.min(w - 1, x0 + 1);
      const fx = Math.max(0, Math.min(1, sx - x0));
      const i00 = (y0 * w + x0) * 4, i10 = (y0 * w + x1) * 4;
      const i01 = (y1 * w + x0) * 4, i11 = (y1 * w + x1) * 4;
      const r = src[i00]*(1-fx)*(1-fy) + src[i10]*fx*(1-fy) + src[i01]*(1-fx)*fy + src[i11]*fx*fy;
      const g = src[i00+1]*(1-fx)*(1-fy) + src[i10+1]*fx*(1-fy) + src[i01+1]*(1-fx)*fy + src[i11+1]*fx*fy;
      const b = src[i00+2]*(1-fx)*(1-fy) + src[i10+2]*fx*(1-fy) + src[i01+2]*(1-fx)*fy + src[i11+2]*fx*fy;
      const o = y * SIZE + x;
      input[o] = (r / 255 - 0.5) / 0.5;
      input[SIZE*SIZE + o] = (g / 255 - 0.5) / 0.5;
      input[2*SIZE*SIZE + o] = (b / 255 - 0.5) / 0.5;
    }
  }

  // 2) 推理
  const session = await ort.InferenceSession.create(modelPath, { executionProviders: ['cpu'] });
  const inName = session.inputNames[0], outName = session.outputNames[0];
  const feeds = { [inName]: new ort.Tensor('float32', input, [1, 3, SIZE, SIZE]) };
  const res = await session.run(feeds);
  const matte = res[outName].data; // Float32 1*1*256*256

  // 3) 放大回原图尺寸（双线性）
  const alpha = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    const sy = (y + 0.5) * SIZE / h - 0.5;
    const y0 = Math.max(0, Math.floor(sy)), y1 = Math.min(SIZE - 1, y0 + 1);
    const fy = Math.max(0, Math.min(1, sy - y0));
    for (let x = 0; x < w; x++) {
      const sx = (x + 0.5) * SIZE / w - 0.5;
      const x0 = Math.max(0, Math.floor(sx)), x1 = Math.min(SIZE - 1, x0 + 1);
      const fx = Math.max(0, Math.min(1, sx - x0));
      const a00 = matte[y0*SIZE+x0], a10 = matte[y0*SIZE+x1], a01 = matte[y1*SIZE+x0], a11 = matte[y1*SIZE+x1];
      const top = a00*(1-fx) + a10*fx, bot = a01*(1-fx) + a11*fx;
      alpha[y*w+x] = top*(1-fy) + bot*fy;
    }
  }
  // ========================================================
  // MODNet 后处理 v4：alpha 硬化 + 2px 羽化 + 局部 alpha-guided 色溢去除
  //   —— 消除形态学 Erosion/Dilate 导致的锯齿（bug 1）
  //   —— 消除全局背景色估计错导致的「红底有蓝边/蓝底有红边/米白有米白边」（bug 2）
  //   —— 服装边缘不羽化过度（bug 3）
  // ========================================================
  const cx = Math.floor(w / 2), cy = Math.floor(h / 2);
  const halfW = Math.floor(w * 0.46), halfH = Math.floor(h * 0.495); // 中心保护 92% × 99%（再扩大一圈，服装/领带全包住）
  const headY2 = Math.floor(cy * 0.68);                              // y < headY2 = 头部 + 头发；y >= = 服装
  const ma = new Uint8Array(w * h);

  // --- Step 1: alpha 分区硬化（不用形态学滑窗，纯算术避免锯齿）---
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const v = alpha[y*w + x];
      const dx = Math.abs(x - cx), dy = Math.abs(y - cy);
      const inCenter = dx < halfW && dy < halfH;
      let a;

      if (inCenter) {
        // 中心硬保护：v<0.03 → 0；其他 → 255（人像主体 100% 保）
        a = v < 0.03 ? 0 : 255;
      } else if (y < headY2) {
        // 头部 + 头发：宽松保守。α>0.10 就当 255，α<0.04 就当 0。中间做 2 倍渐变（给羽化留余地）
        if (v < 0.04) a = 0;
        else if (v > 0.10) a = 255;
        else a = Math.round((v - 0.04) / 0.06 * 255);
      } else {
        // 服装：严格。α>0.20 就当 255，α<0.06 就当 0（残色残边全擦掉）
        if (v < 0.06) a = 0;
        else if (v > 0.20) a = 255;
        else a = Math.round((v - 0.06) / 0.14 * 255);
      }
      ma[y*w + x] = a;
    }
  }

  // --- Step 2: 盒式 2px 羽化（box blur × 2 次 → 高斯感，平滑边缘消锯齿）---
  // 只对 α∈(0,255) 的边界像素周围 3×3 做平均，中心保护区域 α=255 不受影响
  for (let k = 0; k < 2; k++) {
    const next = new Uint8Array(ma);
    for (let y = 1; y < h - 1; y++) {
      for (let x = 1; x < w - 1; x++) {
        const cur = ma[y*w + x];
        if (cur === 0 || cur === 255) continue;   // 非边界不动
        const n  = ma[(y-1)*w + x];
        const s  = ma[(y+1)*w + x];
        const we = ma[y*w + (x-1)];
        const ea = ma[y*w + (x+1)];
        const nw = ma[(y-1)*w + (x-1)];
        const ne = ma[(y-1)*w + (x+1)];
        const sw = ma[(y+1)*w + (x-1)];
        const se = ma[(y+1)*w + (x+1)];
        next[y*w + x] = Math.round((cur * 4 + n + s + we + ea + nw + ne + sw + se) / 12);
      }
    }
    for (let i = 0; i < ma.length; i++) ma[i] = next[i];
  }

  // --- Step 3: 外层 10px 强制 α=0 ---
  const MARGIN = 10;
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++)
      if (x < MARGIN || x >= w - MARGIN || y < MARGIN || y >= h - MARGIN)
        ma[y*w + x] = 0;

  // --- Step 4: alpha-guided 局部色溢去除（解决 bug 2：红底/蓝底/米白各留对应颜色残边）---
  // 原理：对每个半透明像素 p（0 < α < 255），取它 8×8 邻域内「确定为背景的像素（α≈0，且原图 RGB 和中心 p 的 RGB 距离 > 一定阈值 30，防止取到人像边缘暗部）」
  //       把这些确定背景像素 RGB 平均作为 p 的 bgRGB（不是全局估计，是「这根头发丝周围那一圈的真实背景色」）
  //       再反推：RGB_pure = (RGB_original - bgRGB * (1-α)) / α
  // 效果：不管原背景是什么颜色（米白/黑/红/蓝/灰），边缘像素拿到它自己邻域的真实 bg，色溢去除精准；换任何新底色都不会残色。
  const R = 8;   // 邻域半径
  const minConfident = 3;   // 至少要采到 3 个确定背景像素才用局部，否则 fallback 全局
  const out = Buffer.from(data);
  // 快速构建「确定背景」mask：ma<=10 && 原图 RGB 不像肤色/头发
  const sureBg = new Uint8Array(w * h);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++)
      sureBg[y*w + x] = (ma[y*w + x] <= 10) ? 1 : 0;

  // 全局 fallback 背景色（极少数像素局部采不到时用）：最外圈 32px 直方图 15% 分位
  const BM = 32;
  const samples = [];
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++)
      if (x < BM || x >= w - BM || y < BM || y >= h - BM) {
        const i = (y*w + x)*4;
        samples.push([data[i], data[i+1], data[i+2]]);
      }
  function pct(arr, p) { arr.sort((a,b)=>a-b); return arr[Math.min(arr.length-1, Math.floor(arr.length*p))]; }
  let globalBgR = 128, globalBgG = 128, globalBgB = 128;
  if (samples.length > 0) {
    globalBgR = pct(samples.map(s=>s[0]), 0.15);
    globalBgG = pct(samples.map(s=>s[1]), 0.15);
    globalBgB = pct(samples.map(s=>s[2]), 0.15);
  }

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idx = (y*w + x);
      const a = ma[idx];
      out[idx*4 + 3] = a;

      if (a <= 0 || a >= 255) continue;  // 全透明/不透明不处理
      const af = a / 255;
      const i4 = idx * 4;

      // 局部采：8×8 邻域里 sureBg==1 的所有像素 RGB 平均
      let rSum = 0, gSum = 0, bSum = 0, cnt = 0;
      const x0 = Math.max(0, x - R), x1 = Math.min(w - 1, x + R);
      const y0 = Math.max(0, y - R), y1 = Math.min(h - 1, y + R);
      for (let yy = y0; yy <= y1; yy++) {
        for (let xx = x0; xx <= x1; xx++) {
          if (!sureBg[yy*w + xx]) continue;
          const j4 = (yy*w + xx) * 4;
          rSum += data[j4]; gSum += data[j4+1]; bSum += data[j4+2]; cnt++;
        }
      }
      let bgR, bgG, bgB;
      if (cnt >= minConfident) {
        bgR = Math.round(rSum / cnt); bgG = Math.round(gSum / cnt); bgB = Math.round(bSum / cnt);
      } else {
        bgR = globalBgR; bgG = globalBgG; bgB = globalBgB;
      }

      // 反推纯前景 RGB（去掉 bgRGB × (1-α)）
      const nr = Math.round((data[i4]   - bgR * (1 - af)) / af);
      const ng = Math.round((data[i4+1] - bgG * (1 - af)) / af);
      const nb = Math.round((data[i4+2] - bgB * (1 - af)) / af);
      out[i4]   = Math.min(255, Math.max(0, nr));
      out[i4+1] = Math.min(255, Math.max(0, ng));
      out[i4+2] = Math.min(255, Math.max(0, nb));
    }
  }
  return encodePng(w, h, out);
}


/**
 * 方案 B（默认兜底）：像素级背景分离。
 * 用「边缘主色 + 色距阈值」把接近背景色的像素置透明。
 * 证件照背景多为单一纯色（墙/幕布/天空），此算法对纯色背景效果好。
 */
function mattingHeuristic(imageBuffer) {
  const { width: w, height: h, data } = decodePng(imageBuffer);
  const bg = sampleEdgeColor(w, h, data);
  const out = Buffer.from(data);
  const tol = 90;
  const feather = tol * 0.35;

  for (let i = 0; i < out.length; i += 4) {
    const r = out[i], g = out[i + 1], b = out[i + 2];
    const dist = Math.sqrt((r - bg.r) ** 2 + (g - bg.g) ** 2 + (b - bg.b) ** 2);
    if (dist < tol) {
      out[i + 3] = 0;
    } else if (dist < tol + feather) {
      const t = (dist - tol) / feather;
      out[i + 3] = Math.round(255 * (1 - t));
    }
  }

  // 中心矩形硬保护（证件照人像必在中心 82%×95%，无论色距，强制 α=255 保人像不被误擦洞）
  const halfW = Math.round(w * 0.41), halfH = Math.round(h * 0.475);
  const cx = Math.round(w / 2), cy = Math.round(h / 2);
  let recover = 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (Math.abs(x - cx) < halfW && Math.abs(y - cy) < halfH) {
        const ai = (y * w + x) * 4 + 3;
        if (out[ai] < 220) { out[ai] = 255; recover++; }
      }
    }
  }

  // 最外 6px 边带 → 强制背景（证件照边带必是背景，解决中心保护后最外圈 6 像素背景残留圈）
  let cleared = 0, band = 6;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (x < band || y < band || x >= w - band || y >= h - band) {
        const ai = (y * w + x) * 4 + 3;
        if (out[ai] > 40) { out[ai] = 0; cleared++; }
      }
    }
  }

  // 质量门禁：不透明像素 20~93% 判成功；否则判定色距失败 → 回退「人像整体不透明 + 边 16px 羽化」
  let opaque = 0;
  for (let i = 3; i < out.length; i += 4) if (out[i] >= 128) opaque++;
  const ratio = opaque / (w * h);
  console.log('[matting] 启发式抠图: 不透明占比 ' + (ratio*100).toFixed(1) + '%', '中心硬保护恢复', recover, '像素, 边带清理', cleared, '像素');
  if (ratio < 0.20 || ratio > 0.93) {
    console.log('[matting] ⚠️ 启发式抠图异常 (' + (ratio*100).toFixed(1) + '%)，回退「人像全不透明 + 边缘16px羽化」');
    for (let i = 3; i < out.length; i += 4) out[i] = 255;
    const edge = 16;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const padX = Math.min(x, w - 1 - x), padY = Math.min(y, h - 1 - y);
        const pad = Math.min(padX, padY);
        if (pad < edge) {
          out[(y * w + x) * 4 + 3] = Math.round(255 * (pad / edge));
        }
      }
    }
  }
  return encodePng(w, h, out);
}

/** 采样四角主色作为背景 */
function sampleEdgeColor(w, h, data) {
  const freq = {};
  const sw = Math.max(2, Math.floor(w * 0.06));
  const sh = Math.max(2, Math.floor(h * 0.06));
  const push = (x, y) => {
    const i = (y * w + x) * 4;
    const r = data[i] >> 4, g = data[i + 1] >> 4, b = data[i + 2] >> 4;
    const key = r + ',' + g + ',' + b;
    freq[key] = (freq[key] || 0) + 1;
  };
  for (let y = 0; y < sh; y++) {
    for (let x = 0; x < sw; x++) {
      push(x, y); push(w - 1 - x, y); push(x, h - 1 - y); push(w - 1 - x, h - 1 - y);
    }
  }
  let bestKey = null, best = 0;
  for (const k in freq) {
    if (freq[k] > best) { best = freq[k]; bestKey = k; }
  }
  const p = bestKey.split(',');
  return { r: (+p[0]) << 4, g: (+p[1]) << 4, b: (+p[2]) << 4 };
}

// ====== ORT 引擎加载：先 native，再 WASM（解决腾讯云 CentOS 7 GLIBCXX 版本低问题）======
let _ortPromise = null;
function getOrt() {
  if (_ortPromise) return _ortPromise;
  _ortPromise = (async () => {
    try {
      const ort = require('onnxruntime-node');
      // 真的能 new Tensor 吗？有些老系统能 require 进来但 new Tensor 时抛 native binding 错，提前试一次
      const _probe = new ort.Tensor('float32', new Float32Array([0, 0, 0, 0]), [1, 1, 2, 2]);
      console.log('[ORT] onnxruntime-node native 加载成功（版本=', ort.env?.version || 'unknown', '）');
      return ort;
    } catch (e) {
      console.warn('[ORT] onnxruntime-node 加载失败（', e.message, '）→ 自动回退 onnxruntime-web WASM');
      const ort = require('onnxruntime-web');
      ort.env.wasm.numThreads = 1;        // 云函数 sandbox 多线程会被 OOM kill
      ort.env.wasm.simd = true;            // SIMD 打开，CPU 推理快 ~2 倍
      ort.env.wasm.proxy = false;
      console.log('[ORT] onnxruntime-web WASM 回退成功');
      return ort;
    }
  })();
  return _ortPromise;
}

module.exports = { matting, applyMattingPostprocess };
