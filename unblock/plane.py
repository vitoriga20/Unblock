"""平面候选检测。

黑板/投影幕是"暗色大矩形"，白板/纸张是"亮色矩形"，两者用同一套线段交点逻辑，
只替换亮度先验；再加一条亮色轮廓策略，覆盖纸张边缘被手挡住一部分的情况。
所有候选都带分数，由上层选最优或作为交互初值。
"""

from typing import Dict, List, Optional

import cv2
import numpy as np

from .common import order_quad, resize_to, to_gray
from .geometry import _edge_support, detect_screen_quad


def inset_quad(img: np.ndarray, ratio: float = 0.04) -> np.ndarray:
    h, w = img.shape[:2]
    ix, iy = w * ratio, h * ratio
    return np.asarray([[ix, iy], [w - ix, iy], [w - ix, h - iy], [ix, h - iy]], np.float32)


def _quad_iou(a: np.ndarray, b: np.ndarray, side: int = 320) -> float:
    pts = np.vstack([a, b]).astype(np.float32)
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    s = side / max(1e-6, max(x1 - x0, y1 - y0))
    w = max(2, int(np.ceil((x1 - x0) * s)) + 2)
    h = max(2, int(np.ceil((y1 - y0) * s)) + 2)

    def _mask(q):
        m = np.zeros((h, w), np.uint8)
        p = ((q - [x0, y0]) * s + 1).astype(np.int32)
        cv2.fillConvexPoly(m, p, 255)
        return m

    ma, mb = _mask(a), _mask(b)
    inter = int(np.count_nonzero((ma > 0) & (mb > 0)))
    union = int(np.count_nonzero((ma > 0) | (mb > 0)))
    return inter / union if union else 0.0


def _contour_candidate(img: np.ndarray, screen_cfg: Dict
                       ) -> Optional[Dict]:
    """亮色轮廓策略：Otsu 取亮区，最大四边形即纸张/白板。"""
    mc = screen_cfg.get("bright", {})
    max_side = int(screen_cfg.get("detect_max_side", 1600))
    small, scale = resize_to(img, max_side)
    gray = to_gray(small)
    gray_b = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh, th = cv2.threshold(gray_b, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if float(cv2.mean(gray_b, th)[0]) < 128.0:
        th = cv2.bitwise_not(th)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    h, w = gray.shape[:2]
    grad = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
                    cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    min_area_ratio = float(mc.get("min_area_ratio", 0.04))
    max_area_ratio = float(mc.get("max_area_ratio", 0.95))
    eps_ratio = float(mc.get("contour_epsilon_ratio", 0.02))
    min_rect = float(mc.get("min_rectangularity", 0.55))
    min_aspect = float(mc.get("aspect_min", 0.45))
    max_aspect = float(mc.get("aspect_max", 3.0))
    min_contrast = float(mc.get("min_contrast", 8.0))

    best = None
    for c in contours:
        area = float(cv2.contourArea(c))
        area_ratio = area / float(w * h)
        if not (min_area_ratio <= area_ratio <= max_area_ratio):
            continue

        rect = cv2.minAreaRect(c)
        rw, rh = rect[1]
        rect_area = float(rw * rh)
        if rect_area <= 1.0:
            continue
        rectangularity = area / rect_area
        if rectangularity < min_rect:
            continue
        if min(rw, rh) < float(mc.get("min_height_ratio", 0.15)) * min(w, h):
            continue
        aspect = max(rw, rh) / max(1e-6, min(rw, rh))
        if not (min_aspect <= aspect <= max_aspect):
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, eps_ratio * peri, True)
        if len(approx) == 4:
            quad = approx.reshape(4, 2).astype(np.float32)
        else:
            quad = cv2.boxPoints(rect).astype(np.float32)
        quad = order_quad(quad)

        inner = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(inner, np.clip(quad, 0, [w - 1, h - 1]).astype(np.int32), 255)
        if int(np.count_nonzero(inner)) < 100:
            continue
        inner_gray = gray[inner > 0]
        inner_med = float(np.median(inner_gray))
        if inner_med < float(mc.get("median_luma_min", 150)):
            continue
        if inner_med > float(mc.get("median_luma_max", 252)):
            continue
        if float((inner_gray > 170).mean()) < float(mc.get("min_bright_ratio", 0.45)):
            continue

        outer = cv2.dilate(inner, np.ones((11, 11), np.uint8))
        ring = (outer > 0) & (inner == 0)
        if int(ring.sum()) >= 100:
            if inner_med - float(np.median(gray[ring])) < min_contrast:
                continue

        support = float(np.mean([
            _edge_support(grad, quad[0], quad[1]),
            _edge_support(grad, quad[1], quad[2]),
            _edge_support(grad, quad[2], quad[3]),
            _edge_support(grad, quad[3], quad[0]),
        ]))
        score = area_ratio * (0.15 + support / 255.0) * rectangularity
        if best is None or score > best["score"]:
            best = {
                "quad": order_quad(quad / scale),
                "score": float(score),
                "kind": "contour",
            }
    return best


def detect_candidates(img: np.ndarray, screen_cfg: Dict,
                      max_n: int = 4) -> List[Dict]:
    """返回按分数降序的平面候选，末尾一定带一个内缩矩形兜底。"""
    cands: List[Dict] = []
    strategies = screen_cfg.get("strategies", ["dark", "bright"])
    for mode in strategies:
        if mode not in ("dark", "bright"):
            continue
        try:
            quad, score, _ = detect_screen_quad(img, screen_cfg, mode=mode)
            if score > 0:
                cands.append({"quad": quad, "score": float(score), "kind": mode})
        except Exception:
            continue

    try:
        contour = _contour_candidate(img, screen_cfg)
        if contour is not None:
            cands.append(contour)
    except Exception:
        pass

    # 去掉几乎重合的候选，保留分数更高的
    kept: List[Dict] = []
    for cand in sorted(cands, key=lambda c: c["score"], reverse=True):
        if any(_quad_iou(cand["quad"], k["quad"]) > 0.85 for k in kept):
            continue
        kept.append(cand)
        if len(kept) >= max_n:
            break

    kept.append({"quad": inset_quad(img), "score": 0.0, "kind": "inset"})
    return kept


def detect_best(img: np.ndarray, screen_cfg: Dict) -> Dict:
    return detect_candidates(img, screen_cfg, max_n=1)[0]
