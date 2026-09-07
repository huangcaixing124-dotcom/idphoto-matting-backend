/**
 * lib/model.js —— 模型管理（从云存储按需下载 + 缓存）
 *
 * 支持两种模型（优先用精度高的，没有就用轻量 MODNet 兜底）：
 *   1) RMBG-1.4 INT8 量化版（推荐，社区常用版 ~80MB）
 *      - BRIA 官方 1024x1024 SegFormer 架构，精度高，头发/西装完整抠出
 *      - 需要用户自行把 rmbg_quant.onnx 上传到云存储，填 fileID
 *   2) MODNet Photographic (25MB) 已随包备用，立即可用
 *      - 256x256 输入，速度快，包体小
 *      - 已把模型复制到 models/modnet_photographic.onnx
 *      - 云存储 fileID 对应：MODNET_MODEL_FILEID
 *
 * 冷启动时：
 *   - 先读本地 os.tmpdir() 缓存（热实例无需每次下载）
 *   - 没缓存 → 从云存储下载到本地临时目录
 *
 * 选择优先级：RMBG fileID 已配且有效 → 用 RMBG；否则用 MODNet。
 */

const fs = require('fs');
const path = require('path');
const os = require('os');

// ========== 云存储 fileID 配置区 ==========
// 【两种至少填一个】
//  1) 把 .onnx 文件在微信开发者工具 → 云开发 → 云存储 → 上传文件
//  2) 右键刚上传的文件 → 详情 → 复制 "File ID" 填到下面
// =========================================

// RMBG-1.4 INT8（推荐，精度高。上传 .onnx 后填；没填就自动用 MODNet）
//  例: 'cloud://cloud1-d5guqvm8j968dc040.636c-xxx/rmbg_quant.onnx'
const RMBG_MODEL_FILEID = 'cloud://cloud1-d5guqvm8j968dc040.636c-cloud1-d5guqvm8j968dc040-1456232604/rmbg_quant.onnx';  // 用户已上传 RMBG-2.0 INT8 量化 42.35MB，精度优先

// MODNet Photographic (25MB) —— 立即可用的 AI 模型
// 【必填】把 cloudfunctions/matting/models/modnet_photographic.onnx 上传到云存储，填 File ID
const MODNET_MODEL_FILEID = 'cloud://cloud1-d5guqvm8j968dc040.636c-cloud1-d5guqvm8j968dc040-1456232604/modnet_photographic.onnx';

// 本地缓存路径
const RMBG_LOCAL = path.join(os.tmpdir(), 'rmbg.onnx');
const MODNET_LOCAL = path.join(os.tmpdir(), 'modnet.onnx');
// 代码包内打包好的 MODNet（随 matting/models/*.onnx 一起上传，省 10s 冷启动公网下载）
const MODNET_PACKED = path.join(__dirname, '..', 'models', 'modnet_photographic.onnx');
// 代码包内打包好的 RMBG（复制 Desktop/rmbg_quant.onnx → models/rmbg_quant.onnx，
// 随代码一起上传，省 52s 冷启动云存储下载，把整个 60s 留给 1024 推理）
const RMBG_PACKED = path.join(__dirname, '..', 'models', 'rmbg_quant.onnx');

async function downloadToLocal(fileID, localPath, label) {
  if (fs.existsSync(localPath)) return localPath;
  const cloud = require('wx-server-sdk');
  console.log('[model]', label, '云存储下载 fileID=', fileID, '→', localPath);
  const t0 = Date.now();
  let res;
  try {
    res = await cloud.downloadFile({ fileID });
  } catch (e) {
    console.error('[model] ❌ ' + label + ' 下载失败！请确认：\n'
      + '  ① 云存储中确实上传了模型文件（label=' + label + '），fileID=' + fileID + '\n'
      + '  ② model.js 中填的 FileID 字符串是否与云存储详情里「File ID」复制的完全一致？\n'
      + '  错误堆栈:', (e && e.stack) || (e && e.message) || e);
    throw e;
  }
  if (!res || !res.fileContent || res.fileContent.length < 1024 * 1024) {
    console.error('[model] ❌ ' + label + ' 下载文件太小 (' + (res ? res.fileContent.length : 'undefined') + ' bytes)，fileID 可能不正确或文件已被删除');
    throw new Error(label + '_FILE_TOO_SMALL');
  }
  fs.writeFileSync(localPath, res.fileContent);
  const mb = res.fileContent.length / 1024 / 1024;
  console.log('[model] ✅', label, '下载完成', ((Date.now() - t0) / 1000).toFixed(1) + 's,', mb.toFixed(1) + 'MB');
  return localPath;
}

function looksConfigured(fileID) {
  return fileID && typeof fileID === 'string'
    && fileID.startsWith('cloud://')
    && !fileID.includes('xxx')
    && fileID.length > 25;
}

/**
 * 自动选模型并返回本地路径
 * @returns {Promise<{modelPath: string, arch: 'rmbg'|'modnet'}>}
 * @description 方案 B：RMBG(BiRefNet 1024×1024) 优先（头发丝级高精度），MODNet 兜底。
 *   注意：RMBG 推理较慢（1024 分辨率），请把云函数「超时≥120s / 内存 2048MB」，
 *   否则会 FUNCTIONS_TIME_LIMIT_EXCEEDED。若超时，MODNet 会自动降级兜底。
 */
async function getBestModel() {
  // RMBG / BiRefNet 优先（方案 B，1024×1024 高分辨率 → 头发丝真正锐利）
  if (looksConfigured(RMBG_MODEL_FILEID)) {
    try {
      // 优先用代码包内的 RMBG（0s 加载，不走云存储下载）
      if (fs.existsSync(RMBG_PACKED)) {
        const st = fs.statSync(RMBG_PACKED);
        if (st.size > 30 * 1024 * 1024) {  // 42MB 模型
          console.log('[model] ✅ 已找到代码包内 RMBG:', RMBG_PACKED, (st.size/1024/1024).toFixed(1)+'MB（0s 加载，不走云存储）');
          return { modelPath: RMBG_PACKED, arch: 'rmbg' };
        }
      }
      const p = await downloadToLocal(RMBG_MODEL_FILEID, RMBG_LOCAL, 'RMBG');
      return { modelPath: p, arch: 'rmbg' };
    } catch (e) {
      console.warn('[model] RMBG/BiRefNet 失败，自动降级 MODNet:', e.message);
    }
  }
  // MODNet 兜底
  if (looksConfigured(MODNET_MODEL_FILEID)) {
    try {
      const fs = require('fs');
      if (fs.existsSync(MODNET_PACKED)) {
        const st = fs.statSync(MODNET_PACKED);
        if (st.size > 1024 * 1024) {
          console.log('[model] ✅ 已找到代码包内 MODNet:', MODNET_PACKED, (st.size/1024/1024).toFixed(1)+'MB（0s 加载，不走云存储）');
          return { modelPath: MODNET_PACKED, arch: 'modnet' };
        }
      }
      const p = await downloadToLocal(MODNET_MODEL_FILEID, MODNET_LOCAL, 'MODNet');
      return { modelPath: p, arch: 'modnet' };
    } catch (e) {
      console.warn('[model] MODNet 也失败:', e.message);
    }
  }
  throw new Error(
    '未配置任何 AI 模型 fileID！\n' +
    '   请在 cloudfunctions/matting/lib/model.js 填 RMBG_MODEL_FILEID 或 MODNET_MODEL_FILEID。'
  );
}

module.exports = { getBestModel, RMBG_MODEL_FILEID, MODNET_MODEL_FILEID };
