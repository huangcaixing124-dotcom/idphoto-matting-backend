#!/usr/bin/env python3
"""
独立证件照抠图服务（stdio 管道，Node 后端调用）。
读 stdin: JSON {"pngBase64": "...", "compose": {"w":295,"h":413}|null}  ->
写 stdout: JSON {"pngBase64": "..."(透明PNG), "tookSec": n}
模型: BiRefNet Swin-Tiny (224MB onnx, MIT 可商用)，发丝级边缘更干净（用户 2026-09-05 三底色对比确认优于 RMBG）。
      实测单进程物理常驻 ~0.86GB（此前"18GB OOM不能并发"系 vmmap 虚存测法误判，实际可并发）。
      单张 CPU ~5.7s（比 RMBG 的 2.5s 慢，但发丝质量更好）。
后处理: 最小羽化(1.5)收边即达目标,不再叠加反污染/深发填充(实测副作用大)。
自动排版(2026-09-06): 当 compose 给出目标规格 {w,h} 时，在抠图后按
   HivisionIDPhotos 的「国标自动排版」算法用人脸框 + 3 参数(head_measure=0.2/
   head_height=0.45/head_top_range=(0.12,0.10))二次裁剪排进 spec 画幅，
   <<拿到图就能用>>。人脸检测用 RetinaFace-resnet50 onnx(与 Hivision 同模型同解码，
   保证 cfg 与模型一致)。0/多张脸 → detectOk:false 降级返回未排版图(HTTP200)。
模型固定加载一次（进程常驻）。
"""
import sys, os, json, time, base64, io
from itertools import product
from math import ceil
import numpy as np
from PIL import Image, ImageFilter
import onnxruntime as ort

MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'birefnet_swin_tiny.onnx')
FACE_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'retinaface-resnet50.onnx')
SIZE = 1024

# 国标构图参数（2026-09-06 定稿·人像填满法）
#   方案：把人像主体高度填满画幅（头顶留白 → 画幅底），头占画幅比例随源身材而定
#   （贴脸半身照自然≈37%）。统一 scale 等比缩放，绝不拉伸。头顶留白对齐 TOP_M。
TOP_M = 0.10   # 头顶留白(画幅高比，对齐目标模板~9-10%)

# 进程级常驻 session（首次调用后复用）
_session = None
_face_session = None


def get_session():
    global _session
    if _session is None:
        so = ort.SessionOptions()
        so.intra_op_num_threads = 4
        _session = ort.InferenceSession(MODEL, so, providers=['CPUExecutionProvider'])
    return _session


# =====================================================================
# 自动排版（Hivision adjust_photo 移植，纯 numpy/PIL，无 cv2）
# 坐标系约定：compose_photo 输入的 rgba 是 BiRefNet 输出的透明图（≤1024 长边），
# face_rect 必须是【同一坐标系】(即原图 face_rect × 抠图降采样比 s)。
# =====================================================================
def _prior_box(cfg, image_size):
    """RetinaFace PriorBox.forward 移植。"""
    min_sizes, steps = cfg["min_sizes"], cfg["steps"]
    feature_maps = [[ceil(image_size[0] / st), ceil(image_size[1] / st)] for st in steps]
    anchors = []
    for k, f in enumerate(feature_maps):
        for i, j in product(range(f[0]), range(f[1])):
            for min_size in min_sizes[k]:
                s_kx = min_size / image_size[1]
                s_ky = min_size / image_size[0]
                for cy, cx in product([(i + 0.5) * steps[k] / image_size[0]],
                                      [(j + 0.5) * steps[k] / image_size[1]]):
                    anchors += [cx, cy, s_kx, s_ky]
    out = np.array(anchors).reshape(-1, 4)
    if cfg["clip"]:
        out = np.clip(out, 0, 1)
    return out


def _decode(loc, priors, variances):
    boxes = np.concatenate(
        (priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
         priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])), axis=1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def _decode_landm(pre, priors, variances):
    pieces = [priors[:, :2] + pre[:, k:k + 2] * variances[0] * priors[:, 2:]
              for k in range(0, 10, 2)]
    return np.concatenate(pieces, axis=1)


def _py_cpu_nms(dets, thresh):
    x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(ovr <= thresh)[0] + 1]
    return keep


def get_face_session():
    global _face_session
    if _face_session is None:
        _face_session = ort.InferenceSession(FACE_MODEL, providers=['CPUExecutionProvider'])
    return _face_session


def retinaface_get_rect(rgb_image, fake_detect=None):
    """
    检测人脸，返回 (x, y, w, h) 于 rgb_image 坐标空间；0 或 >1 张脸返回 None。
    cfg 用 retrainface-resnet50 实际训练 cfg（与 Hivision 同栈，保证模型一致）。
    """
    cfg = {
        "min_sizes": [[16, 32], [64, 128], [256, 512]],
        "steps": [8, 16, 32],
        "variance": [0.1, 0.2],
        "clip": False,
        "image_size": 840,
    }
    conf_thresh = 0.8
    top_k, nms_thresh, keep_top_k = 5000, 0.2, 750

    img = np.float32(np.asarray(rgb_image))
    im_h, im_w, _ = img.shape
    scale = np.array([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)[None]

    sess = get_face_session()
    loc, conf, landms_out = sess.run(None, {sess.get_inputs()[0].name: img})
    priors = _prior_box(cfg, (im_h, im_w))

    boxes = _decode(np.squeeze(loc, 0), priors, cfg["variance"]) * scale
    scores = np.squeeze(conf, 0)[:, 1]

    inds = np.where(scores > conf_thresh)[0]
    boxes, scores = boxes[inds], scores[inds]
    order = scores.argsort()[::-1][:top_k]
    boxes, scores = boxes[order], scores[order]
    dets = np.hstack((boxes, scores[:, None])).astype(np.float32, copy=False)
    dets = dets[_py_cpu_nms(dets, nms_thresh)[:keep_top_k], :]

    n = len(dets)
    if n != 1:
        return None
    x, y, x2, y2 = dets[0][0], dets[0][1], dets[0][2], dets[0][3]
    return (int(round(x)), int(round(y)), int(round(x2 - x + 1)), int(round(y2 - y + 1)))


def get_box_np(rgba, thresh=20):
    """人对透明 alpha 取最大连通域外接 box 的左上右下坐标 [x1,y1,x2,y2]。
    证件照主体通常整体连通；用 numpy BFS 最大连通域兜底，避免孤立噪点撑大 bbox。"""
    a = np.asarray(rgba)[:, :, 3]
    H, W = a.shape
    mask = a > thresh
    if not mask.any():
        return None
    # 逐行/逐列留白扫描（退化近似，足够证件照连通主体）
    rows = mask.any(axis=1)
    cols = mask.any(axis=0)
    ys = np.where(rows)[0]
    xs = np.where(cols)[0]
    return (int(xs[0]), int(ys[0]), int(xs[-1]), int(ys[-1]))


def idphotos_cut_np(rgba_np, k):
    """裁剪 [x1,y1,x2,y2]，越界 0 补透明。k=[x1,y1,x2,y2]。RGBA 4通道。"""
    H, W, _ = rgba_np.shape
    x1, y1, x2, y2 = k
    cw, ch = x2 - x1, y2 - y1
    bg = np.full((ch, cw, 4), 0, dtype=np.uint8)
    # clamp 到图像内
    sx1, sy1 = max(0, x1), max(0, y1)
    sx2, sy2 = min(W, x2), min(H, y2)
    dx1, dy1 = sx1 - x1, sy1 - y1
    dx2, dy2 = dx1 + (sx2 - sx1), dy1 + (sy2 - sy1)
    if sx2 > sx1 and sy2 > sy1:
        bg[dy1:dy2, dx1:dx2] = rgba_np[sy1:sy2, sx1:sx2]
    return bg


def compose_photo(rgba, face_rect, spec_w, spec_h):
    """排版：把透明人像按国标排进 spec(宽=spec_w,高=spec_h)。

    方案（2026-09-06 定稿·诚实法）：**把透明人像的主体高度填满画幅**。
    「头顶(body alpha 顶) → 身体底(body alpha 底)」按尺度映射到「画幅顶留白 → 画幅底」，
    人像自然铺满高度(底部不空)，头顶留白对齐 10%(TOP_M)。头部占画幅比例随源主体身材而定
    (这类贴脸半身照自然≈35-40%，属正常证件照长相)。水平用脸框居中。
    —— 关键修正：之前的"头高占比法"在贴脸半身照上会让 crop 超出源图(透明补边)，
    导致人像缩小、底部大片空。用"主体填满"则底部必然贴满。
    scale = 帧px/源px。rgba 透明图，face_rect=(x,y,w,h) 同空间。
    """
    rgba_np = np.array(rgba, dtype=np.uint8)
    H, W, _ = rgba_np.shape
    x, y, w, h = face_rect

    TOP_M = 0.10   # 头顶留白(画幅高比)
    SHOULDER_M = 0.02  # 肩部左右各留白(画幅宽比；模板≈0%，略收一点防裁手)
    box = get_box_np(rgba, thresh=40)
    if not box:
        return Image.fromarray(rgba_np, 'RGBA').resize((spec_w, spec_h), Image.BILINEAR)
    bxl, byb, bxr, byt = box     # (x1,y1,x2,y2)
    hair_top, body_bottom = byb, byt
    body_span = max(1, body_bottom - hair_top)

    # 垂直主导 scale：人像高度填满画幅(头顶留白→画幅底)，头部/身体比例保持源貌。
    # 水平不强制收缩(避免把远景窄照压小留白边)，而是把 body 中心对齐画幅中线 → 自动居中。
    scale = ((1 - TOP_M) * spec_h) / body_span
    scale = max(0.2, min(4.0, scale))

    # 裁剪框(源px)=填满 spec 整幅(恒 spec 比例 → 不拉伸)
    crop_h = max(1, int(round(spec_h / scale)))
    crop_w = max(1, int(round(spec_w / scale)))
    # 垂直：头顶留白对齐 TOP_M。源 p→帧 y=(p-y0)*scale；令 hair_top→TOP_M*spec_h ⇒ y0=hair_top - TOP_M*spec_h/scale
    y0 = int(hair_top - (TOP_M * spec_h) / scale)
    # 水平：body 中心(bxl+bxr)/2 对齐帧中线 → 自动居中
    x0 = int((bxl + bxr) / 2 - crop_w / 2)

    cut = idphotos_cut_np(rgba_np, [x0, y0, x0 + crop_w, y0 + crop_h])
    out = Image.fromarray(cut, 'RGBA').resize((spec_w, spec_h), Image.BILINEAR)
    return out


# =====================================================================
# BiRefNet 抠图（原逻辑，扩展返回降采样比 s 供 face_rect 对齐）
# =====================================================================
def biref_alpha_to_png(png_bytes, mask_only=False):
    img = Image.open(io.BytesIO(png_bytes)).convert('RGB')
    ow, oh = img.size
    # BiRefNet 预处理：ImageNet 均值方差归一化
    im = img.resize((SIZE, SIZE), Image.BILINEAR)
    x = (np.asarray(im) / 255.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    x = x.astype('float32').transpose(2, 0, 1)[None]
    sess = get_session()
    raw = sess.run(None, {sess.get_inputs()[0].name: x})[0][0, 0]
    # BiRefNet 输出已是 sigmoid 概率(0-1)，防御性归一化
    a = 1 / (1 + np.exp(-raw)) if raw.min() < 0 or raw.max() > 1.5 else raw
    a = (a - a.min()) / (a.max() - a.min() + 1e-9)
    amask = Image.fromarray((a * 255).astype('uint8'), 'L').resize((ow, oh), Image.BILINEAR)
    # 最小羽化收边（BiRefNet 原生 mask 已干净，不叠加反污染/深发填充）
    amask = amask.filter(ImageFilter.GaussianBlur(1.5))
    # 等比缩到 max 1024
    s = min(1.0, SIZE / max(ow, oh))
    if s < 1.0:
        ow_o, oh_o = max(1, round(ow * s)), max(1, round(oh * s))
        img_s = img.resize((ow_o, oh_o), Image.BILINEAR)
        amask_s = amask.resize((ow_o, oh_o), Image.BILINEAR)
    else:
        ow_o, oh_o, img_s, amask_s = ow, oh, img, amask
    if mask_only:
        buf = io.BytesIO(); amask_s.save(buf, 'PNG'); buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode(), mask_only, s
    out = Image.new('RGBA', (ow_o, oh_o))
    out.paste(img_s, mask=amask_s)
    buf = io.BytesIO(); out.save(buf, 'PNG'); buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode(), mask_only, s


def compose_or_fallback(png_bytes, compose, detect_only=None):
    """
    做抠图 + 可选排版。返回 (out_dict, )。
    compose={'w','h'} 时：人脸检测→排版；无脸/多脸/模型缺失→降级未排版 + detectOk:false。
    """
    t0 = time.time()
    mask_only = bool(detect_only)
    resb64, is_mask, s = biref_alpha_to_png(png_bytes, mask_only)
    if is_mask:
        return {"code": 0, "maskBase64": resb64, "source": "biref", "mask": True,
                "tookSec": round(time.time() - t0, 2)}
    if not compose:
        return {"code": 0, "pngBase64": resb64, "source": "biref",
                "tookSec": round(time.time() - t0, 2)}

    spec_w, spec_h = compose.get('w'), compose.get('h')
    if not spec_w or not spec_h:
        return {"code": 0, "pngBase64": resb64, "source": "biref", "detectOk": False,
                "tookSec": round(time.time() - t0, 2)}
    # 人脸检测：在原图坐标系得到 face_rect，再 × s 对齐到 抠图输出坐标系
    try:
        img = Image.open(io.BytesIO(png_bytes)).convert('RGB')
        fr = retinaface_get_rect(img)
        opaque = Image.open(io.BytesIO(base64.b64decode(resb64))).convert('RGBA')
        if fr is None:
            return {"code": 0, "pngBase64": resb64, "source": "biref", "detectOk": False,
                    "photoError": "no_face",
                    "tookSec": round(time.time() - t0, 2)}
        fr_s = (round(fr[0] * s), round(fr[1] * s), round(fr[2] * s), round(fr[3] * s))
        composed = compose_photo(opaque, fr_s, int(spec_w), int(spec_h))
        buf = io.BytesIO(); composed.save(buf, 'PNG'); buf.seek(0)
        return {"code": 0, "pngBase64": base64.b64encode(buf.getvalue()).decode(),
                "source": "biref", "detectOk": True, "compose": True,
                "tookSec": round(time.time() - t0, 2)}
    except Exception as e:
        sys.stderr.write(json.dumps({"compose_error": str(e)}) + "\n"); sys.stderr.flush()
        return {"code": 0, "pngBase64": resb64, "source": "biref", "detectOk": False,
                "photoError": str(e), "tookSec": round(time.time() - t0, 2)}


def main():
    # 预热抠图模型
    try:
        get_session()
    except Exception as e:
        sys.stderr.write(json.dumps({"error": str(e)}) + "\n"); sys.stderr.flush()
    # 预热人脸模型（失败不致命，排版会降级）
    try:
        get_face_session()
    except Exception as e:
        sys.stderr.write(json.dumps({"face_model_warn": str(e)}) + "\n"); sys.stderr.flush()
    # 逐行处理 stdin
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        t0 = time.time()
        try:
            req = json.loads(line)
            b64 = req.get('pngBase64') or req.get('base64') or ''
            compose = req.get('compose')
            mask_only = bool(req.get('maskOnly'))
            if not b64:
                out = {"code": "NO_BASE64", "message": "missing pngBase64"}
            else:
                png = base64.b64decode(b64)
                if mask_only:
                    out = compose_or_fallback(png, None, detect_only=True)
                else:
                    out = compose_or_fallback(png, compose)
        except Exception as e:
            out = {"code": "ERROR", "message": str(e)}
        sys.stdout.write(json.dumps(out) + "\n"); sys.stdout.flush()


if __name__ == '__main__':
    main()