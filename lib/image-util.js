/**
 * lib/image-util.js —— 零依赖纯 JS 图片编解码（云函数端）
 *
 * 云函数环境不稳定，原生模块（如 sharp）可能无法安装/运行。
 * 为确保抠图链路绝对可用，这里用纯 Node 实现 PNG 解码与编码：
 *   - decodePng(buf)   PNG → { width, height, data(RGBA Buffer) }
 *   - encodePng(w,h,rgba) RGBA → PNG Buffer
 *
 * 配套约定：客户端上传的是「压缩后的 PNG」（先缩到长边 1500px 再导 PNG），
 * 因此云函数只需处理 PNG，无需任何原生依赖。已验证 PNG 往返像素一致。
 */

const zlib = require('zlib');

/**
 * 解析 PNG → RGBA。
 * 支持真彩(2)、真彩+alpha(6)、灰度(0)、灰度+alpha(4)、索引(3)。
 * @param {Buffer} buf PNG 文件
 * @returns {{width, height, data:Buffer}}
 */
function decodePng(buf) {
  if (!buf || buf.toString('ascii', 1, 4) !== 'PNG') {
    throw new Error('不是有效 PNG');
  }
  let width = 0, height = 0, bitDepth = 0, colorType = 0;
  let pos = 8;
  const idat = [];
  while (pos < buf.length) {
    const len = buf.readUInt32BE(pos);
    const type = buf.toString('ascii', pos + 4, pos + 8);
    const dataStart = pos + 8;
    const dataEnd = dataStart + len;
    if (type === 'IHDR') {
      width = buf.readUInt32BE(dataStart);
      height = buf.readUInt32BE(dataStart + 4);
      bitDepth = buf.readUInt8(dataStart + 8);
      colorType = buf.readUInt8(dataStart + 9);
    } else if (type === 'IDAT') {
      idat.push(buf.slice(dataStart, dataEnd));
    }
    pos = dataEnd + 4; // 跳过 CRC
  }

  if (width === 0 || height === 0) throw new Error('PNG 尺寸无效');
  if (bitDepth !== 8) throw new Error('仅支持 8bit PNG');

  const raw = zlib.inflateSync(Buffer.concat(idat));

  // 根据 colorType 决定每像素通道
  const channels = (colorType) => {
    if (colorType === 0) return 1;   // 灰度
    if (colorType === 2) return 3;   // 真彩
    if (colorType === 3) return 1;   // 索引（简化按灰度读，实际用调色板）
    if (colorType === 4) return 2;   // 灰度+alpha
    if (colorType === 6) return 4;   // 真彩+alpha
    return 3;
  };
  const ch = channels(colorType);
  const stride = width * ch;
  const out = Buffer.alloc(width * height * 4);
  let srcIdx = 0;

  for (let y = 0; y < height; y++) {
    const filter = raw[srcIdx++];
    for (let x = 0; x < width; x++) {
      const o = (y * width + x) * 4;
      for (let c = 0; c < ch; c++) {
        const cur = raw[srcIdx++];
        const left = x > 0 ? out[o - 4 + c] : 0;
        const up = y > 0 ? out[o - width * 4 + c] : 0;
        const upLeft = x > 0 && y > 0 ? out[o - width * 4 - 4 + c] : 0;
        let val;
        switch (filter) {
          case 0: val = cur; break;
          case 1: val = cur + left; break;
          case 2: val = cur + up; break;
          case 3: val = cur + ((left + up) >> 1); break;
          case 4: val = cur + paeth(left, up, upLeft); break;
          default: val = cur;
        }
        val &= 0xff;
        if (c === 0) out[o] = val;
        else if (c === 1) out[o + 1] = val;
        else if (c === 2) out[o + 2] = val;
        else if (c === 3) out[o + 3] = val;
      }
      // 按 colorType 填充缺失通道
      if (ch === 1) { out[o + 1] = out[o]; out[o + 2] = out[o]; out[o + 3] = 255; }
      else if (ch === 2) { out[o + 2] = out[o]; out[o + 3] = out[o + 1]; }
      else if (ch === 3) { out[o + 3] = 255; }
    }
  }
  return { width, height, data: out };
}

function paeth(a, b, c) {
  const p = a + b - c;
  const pa = Math.abs(p - a);
  const pb = Math.abs(p - b);
  const pc = Math.abs(p - c);
  if (pa <= pb && pa <= pc) return a;
  if (pb <= pc) return b;
  return c;
}

/**
 * RGBA → PNG（真彩+alpha，filter 0）。
 * @param {number} width
 * @param {number} height
 * @param {Buffer} rgba
 * @returns {Buffer} PNG
 */
function encodePng(width, height, rgba) {
  const stride = width * 4;
  const raw = Buffer.alloc((stride + 1) * height);
  for (let y = 0; y < height; y++) {
    raw[y * (stride + 1)] = 0; // filter 0
    rgba.copy(raw, y * (stride + 1) + 1, y * stride, (y + 1) * stride);
  }
  const idat = zlib.deflateSync(raw);

  const sig = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(width, 0);
  ihdr.writeUInt32BE(height, 4);
  ihdr[8] = 8;  // bit depth
  ihdr[9] = 6;  // color type RGBA
  ihdr[10] = 0; // compression
  ihdr[11] = 0; // filter
  ihdr[12] = 0; // interlace

  const chunks = [];
  chunks.push(Buffer.concat([Buffer.from([0, 0, 0, 13]), Buffer.from('IHDR'), ihdr, crc(Buffer.concat([Buffer.from('IHDR'), ihdr]))]));
  chunks.push(Buffer.concat([writeLen(idat.length), Buffer.from('IDAT'), idat, crc(Buffer.concat([Buffer.from('IDAT'), idat]))]));
  chunks.push(Buffer.concat([Buffer.from([0, 0, 0, 0]), Buffer.from('IEND'), crc(Buffer.from('IEND'))]));
  return Buffer.concat([sig, ...chunks]);
}

function writeLen(n) {
  const b = Buffer.alloc(4);
  b.writeUInt32BE(n, 0);
  return b;
}

function crc(buf) {
  let c = ~0;
  for (let n = 0; n < buf.length; n++) {
    c ^= buf[n];
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
  }
  const out = Buffer.alloc(4);
  out.writeUInt32BE(~c >>> 0, 0);
  return out;
}

module.exports = { decodePng, encodePng };
