"""通用 IO 与图像工具。中文路径安全读写，全部参数由外部配置驱动。"""

import json
import os
from typing import Any, Dict, Tuple

import cv2
import numpy as np


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def imread_unicode(path: str) -> np.ndarray:
    """cv2.imread 在 Windows 中文路径下会失败，改用 fromfile + imdecode。"""
    buf = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"无法读取图片: {path}")
    return img


def imwrite_unicode(path: str, img: np.ndarray, quality: int = 95) -> None:
    ext = os.path.splitext(path)[1].lower() or ".jpg"
    params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)] if ext in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        raise IOError(f"编码失败: {path}")
    buf.tofile(path)


def resize_to(img: np.ndarray, max_side: int) -> Tuple[np.ndarray, float]:
    """只在需要缩小时缩放，返回 (图像, 缩放系数)。"""
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return img.copy(), 1.0
    scale = max_side / longest
    out = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    return out, scale


def to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def order_quad(pts: np.ndarray) -> np.ndarray:
    """统一四个角为 左上、右上、右下、左下。"""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    out = np.zeros((4, 2), dtype=np.float32)
    out[0] = pts[np.argmin(s)]
    out[2] = pts[np.argmax(s)]
    out[1] = pts[np.argmin(d)]
    out[3] = pts[np.argmax(d)]
    return out


def draw_quad(img: np.ndarray, quad: np.ndarray, color=(0, 255, 255), thickness: int = 3) -> None:
    pts = np.asarray(quad, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [pts], True, color, thickness, cv2.LINE_AA)
    for i, (x, y) in enumerate(np.asarray(quad, dtype=np.int32)):
        cv2.circle(img, (int(x), int(y)), 6, color, -1)
        cv2.putText(img, f"{i}", (int(x) + 8, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
