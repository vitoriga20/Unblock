"""人物分割：给出画面中被人体占据的像素，用于遮挡判定。

支持分块推理（tiling），把只露出后脑勺、肩线的小目标放大后再检测，提升召回。
"""

from typing import Dict, Optional

import cv2
import numpy as np

_MODEL_CACHE = {}


def _get_model(model_path: str):
    if model_path not in _MODEL_CACHE:
        from ultralytics import YOLO
        _MODEL_CACHE[model_path] = YOLO(model_path)
    return _MODEL_CACHE[model_path]


def _infer(model, patch: np.ndarray, cfg: Dict) -> np.ndarray:
    res = model(patch, classes=[0], conf=float(cfg["conf"]), imgsz=int(cfg["imgsz"]), verbose=False)[0]
    mask = np.zeros(patch.shape[:2], np.uint8)
    if res.masks is None or len(res.masks) == 0:
        return mask
    for m in res.masks.data.cpu().numpy():
        m2 = cv2.resize(m, (patch.shape[1], patch.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask[m2 > 0.5] = 255
    return mask


def person_mask(img: np.ndarray, cfg: Dict) -> Optional[np.ndarray]:
    """返回与 img 同尺寸的 0/255 人物掩码；依赖缺失时返回 None。"""
    try:
        model = _get_model(str(cfg["model"]))
    except Exception as exc:
        print(f"[unblock] 人物分割不可用: {exc}")
        return None

    tiles = max(1, int(cfg.get("tile", 1)))
    if tiles == 1:
        return _infer(model, img, cfg)

    h, w = img.shape[:2]
    overlap = float(cfg.get("tile_overlap", 0.2))
    full = np.zeros((h, w), np.uint8)
    step_y, step_x = h // tiles, w // tiles
    pad_y = int(step_y * overlap)
    pad_x = int(step_x * overlap)

    for i in range(tiles):
        for j in range(tiles):
            y0 = max(0, i * step_y - pad_y)
            y1 = min(h, (i + 1) * step_y + pad_y)
            x0 = max(0, j * step_x - pad_x)
            x1 = min(w, (j + 1) * step_x + pad_x)
            patch = img[y0:y1, x0:x1]
            if patch.size == 0:
                continue
            local = _infer(model, patch, cfg)
            # 只保留本块的核心区域，重叠部分由邻块负责，避免重复检出造成块边界伪影
            cy0 = max(0, i * step_y)
            cy1 = min(h, (i + 1) * step_y)
            cx0 = max(0, j * step_x)
            cx1 = min(w, (j + 1) * step_x)
            sub = local[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
            full[cy0:cy1, cx0:cx1] = np.maximum(full[cy0:cy1, cx0:cx1], sub)
    return full
