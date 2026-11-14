"""合成：曝光/白平衡增益补偿 + 羽化融合 + 接缝度量。"""

from typing import Dict, Tuple

import cv2
import numpy as np

from .common import to_gray


def _odd(v) -> int:
    v = int(v)
    return v if v % 2 == 1 else v + 1


def estimate_gain_map(base: np.ndarray, aux: np.ndarray, region: np.ndarray, cfg: Dict) -> np.ndarray:
    """在两图都可见且未遮挡的区域估计空间平滑的逐通道增益，抵消曝光与白平衡差。"""
    b = base.astype(np.float32) + 1.0
    a = aux.astype(np.float32) + 1.0
    ratio = b / a

    weight = (region > 0).astype(np.float32)
    k = _odd(cfg["gain_smooth"])
    num = cv2.blur(ratio * weight[..., None], (k, k))
    den = cv2.blur(weight, (k, k))

    gm = num / (den[..., None] + 1e-6)
    valid = (den > 1e-3) & np.isfinite(gm[..., 0])
    if region.sum() > 0:
        fallback = np.median(ratio[region > 0].reshape(-1, 3), axis=0)
    else:
        fallback = np.ones(3, np.float32)
    gm[~valid] = fallback
    gm = np.clip(gm, float(cfg["gain_clip_min"]), float(cfg["gain_clip_max"]))
    return gm.astype(np.float32)


def match_color_lab(aux: np.ndarray, base: np.ndarray, mask: np.ndarray, aux_valid: np.ndarray,
                    screen: np.ndarray, cfg: Dict) -> Tuple[np.ndarray, Dict]:
    """
    把补丁的 LAB 一阶统计对齐到"洞口环带"。

    两图的差异不是乘性增益能表达的：实测 base/aux 的通道比随亮度从 0.44 单调爬到 0.98，
    属于两条色调曲线。乘法增益（哪怕分块）拟合不了，浅色卡片会被过度压红而露成蓝块。
    这里改用偏移+尺度的一阶统计对齐，只替换洞内像素，洞外保持原样。
    Lab 通道顺序为 L/a/b。默认只对齐 a/b（色度），保留 L：
    L 一起拉齐会把"卡片本来就比背景亮"这类真实内容差异也抹掉，
    实测会让填充区整体压暗并抬高边界台阶。
    """
    m8 = (mask > 0).astype(np.uint8)
    if not bool(cfg["color_match_enable"]) or int(m8.sum()) == 0:
        return aux, {"applied": False, "reason": "disabled_or_empty_mask"}

    k = _odd(cfg["color_match_ring_px"])
    ring = (cv2.dilate(m8, np.ones((k, k), np.uint8)) > 0) & (m8 == 0)
    ring &= (aux_valid > 0) & (screen > 0)

    # 只用平坦区估计统计量，避开文字/图标边缘把均值方差拉偏
    gray = to_gray(base)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    ring &= cv2.magnitude(gx, gy) < float(cfg["color_match_flat_grad_max"])

    ring_px = int(ring.sum())
    if ring_px < int(cfg["color_match_min_ring_px"]):
        return aux, {"applied": False, "reason": "ring_too_small", "ring_px": ring_px}

    lab_a = cv2.cvtColor(aux, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_b = cv2.cvtColor(base, cv2.COLOR_BGR2LAB).astype(np.float32)
    hole = m8 > 0
    out = lab_a.copy()
    lo, hi = float(cfg["color_match_std_clip"][0]), float(cfg["color_match_std_clip"][1])
    channels = range(3) if bool(cfg["color_match_align_l"]) else (1, 2)
    for c in channels:
        src = lab_a[..., c][hole]
        dst = lab_b[..., c][ring]
        scale = np.clip((float(dst.std()) + 1e-6) / (float(src.std()) + 1e-6), lo, hi)
        out[..., c][hole] = (src - float(src.mean())) * scale + float(dst.mean())

    aligned = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    result = aux.copy()
    result[hole] = aligned[hole]

    # 量化补丁整体偏离周边的程度：chroma 只看 a/b（蓝-黄、绿-红），lab 含亮度
    def _delta(image_bgr: np.ndarray) -> Tuple[float, float]:
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        diff = lab[hole].mean(axis=0) - lab[ring].mean(axis=0)
        return float(np.linalg.norm(diff[1:])), float(np.linalg.norm(diff))

    chroma_before, lab_before = _delta(aux)
    chroma_after, lab_after = _delta(result)
    info = {
        "applied": True,
        "ring_px": ring_px,
        "chroma": [round(chroma_before, 2), round(chroma_after, 2)],
        "lab": [round(lab_before, 2), round(lab_after, 2)],
    }
    return result, info


def blend(base: np.ndarray, aux_corrected: np.ndarray, mask: np.ndarray,
          aux_valid: np.ndarray, cfg: Dict) -> np.ndarray:
    """
    mask 内部采用 aux 的真实像素，边界按到边的距离线性羽化。
    mask 已经比遮挡物外扩一圈，羽化带落在干净背景上，因此不会再把人混回来，
    但能把"填充区边界"这条硬线抹掉（原来的 2px 高斯过渡在边界上仍是近乎硬替换）。
    """
    feather = max(1, int(cfg["edge_soft_px"]))
    m8 = (mask > 0).astype(np.uint8)
    dist = cv2.distanceTransform(m8, cv2.DIST_L2, 5)
    ramp = np.clip(dist / float(feather), 0.0, 1.0)
    lo = float(cfg["min_alpha"])
    alpha = (lo + (1.0 - lo) * ramp) * m8.astype(np.float32)
    alpha *= (aux_valid > 0).astype(np.float32)

    out = base.astype(np.float32) * (1.0 - alpha[..., None]) + aux_corrected.astype(np.float32) * alpha[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def seam_step(img: np.ndarray, mask: np.ndarray, band_px: int = 2) -> float:
    """mask 边界内外各 band_px 的灰度均值差，直接量化"看得见的那条线"。"""
    k = band_px * 2 + 1
    m8 = (mask > 0).astype(np.uint8)
    inner = cv2.erode(m8, np.ones((k, k), np.uint8)) > 0
    outer = (cv2.dilate(m8, np.ones((k, k), np.uint8)) > 0) & ~(m8 > 0)
    if inner.sum() == 0 or outer.sum() == 0:
        return 0.0
    gray = to_gray(img).astype(np.float32)
    return float(abs(gray[inner].mean() - gray[outer].mean()))


def seam_score(img: np.ndarray, mask: np.ndarray) -> float:
    """遮挡区边界处的梯度强度与全图平均梯度之比，越小说明接缝越不明显。"""
    if np.count_nonzero(mask) == 0:
        return 0.0
    boundary = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((9, 9), np.uint8))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    on_edge = float(mag[boundary > 0].mean()) if np.count_nonzero(boundary) else 0.0
    overall = float(mag.mean()) + 1e-6
    return on_edge / overall
