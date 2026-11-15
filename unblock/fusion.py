"""逐区域多源融合。

旧流程是"base 的遮挡区一律用 aux 补"，因此谁当 base 会决定补丁来源和画质。
这里把每个遮挡连通块单独评估：候选图必须有效覆盖、没有自己的遮挡物、
且补进来后与洞口环带的颜色接近；在满足条件的候选里挑最清晰的那张。
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .common import to_gray


def _bbox(mask: np.ndarray, pad: int = 2) -> Tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return 0, 0, 0, 0
    h, w = mask.shape[:2]
    x0 = max(0, int(xs.min()) - pad)
    y0 = max(0, int(ys.min()) - pad)
    x1 = min(w, int(xs.max()) + pad + 1)
    y1 = min(h, int(ys.max()) + pad + 1)
    return x0, y0, x1 - x0, y1 - y0


def _lab_mean(img: np.ndarray, mask: np.ndarray, bbox) -> np.ndarray:
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return np.zeros(3, np.float32)
    sub = img[y:y + h, x:x + w]
    m = mask[y:y + h, x:x + w] > 0
    if int(np.count_nonzero(m)) < 10:
        return np.zeros(3, np.float32)
    lab = cv2.cvtColor(sub, cv2.COLOR_BGR2LAB).astype(np.float32)
    return lab[m].mean(axis=0)


def _sharpness(img: np.ndarray, mask: np.ndarray, bbox) -> float:
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return 0.0
    sub = to_gray(img[y:y + h, x:x + w])
    m = mask[y:y + h, x:x + w] > 0
    if int(np.count_nonzero(m)) < 50:
        return 0.0
    lap = cv2.Laplacian(sub, cv2.CV_32F)
    return float(np.var(lap[m]))


def fuse(canvas: np.ndarray, sources: List[np.ndarray], valids: List[np.ndarray],
         persons: List[Optional[np.ndarray]], hole: np.ndarray, screen: np.ndarray,
         cfg: Dict) -> Tuple[np.ndarray, Dict]:
    """在 hole 的每个连通块内挑选最合适的源，并做边界羽化。"""
    info = {"regions": 0, "filled": 0, "skipped": 0, "sources": []}
    if not bool(cfg.get("enable", True)) or not sources or int(np.count_nonzero(hole)) == 0:
        return canvas, info

    feather = max(1, int(cfg.get("edge_soft_px", int(cfg.get("ring_px", 41)) // 8)))
    min_region = int(cfg.get("min_region_px", 400))
    max_overlap = float(cfg.get("max_person_overlap", 0.2))
    min_valid = float(cfg.get("min_valid_coverage", 0.85))
    max_delta = float(cfg.get("max_lab_delta", 26.0))
    ring_px = int(cfg.get("ring_px", 41))

    out = canvas.copy()
    num, labels, stats, _ = cv2.connectedComponentsWithStats((hole > 0).astype(np.uint8), 8)
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_region:
            continue
        info["regions"] += 1
        comp = (labels == i).astype(np.uint8) * 255
        bbox = _bbox(comp)
        expanded = cv2.dilate(comp, np.ones((ring_px, ring_px), np.uint8))
        ring = (expanded > 0) & ~(comp > 0) & (screen > 0) & (hole == 0)
        ring_bbox = _bbox(ring.astype(np.uint8), pad=4)
        ring_lab = _lab_mean(canvas, ring.astype(np.uint8), ring_bbox) if ring_bbox[2] > 0 else None

        best_idx, best_score, best_delta = -1, -1e9, 0.0
        fallback_idx, fallback_delta = -1, 1e9
        for idx, src in enumerate(sources):
            valid = valids[idx] if idx < len(valids) else None
            if valid is None:
                continue
            cov = float(np.count_nonzero((valid > 0) & (comp > 0))) / float(area)
            if cov < min_valid:
                continue
            person = persons[idx] if idx < len(persons) else None
            if person is not None:
                overlap = float(np.count_nonzero((person > 0) & (comp > 0))) / float(area)
                if overlap > max_overlap:
                    continue

            delta = 0.0
            if ring_lab is not None:
                src_lab = _lab_mean(src, comp, bbox)
                delta = float(np.linalg.norm(src_lab - ring_lab))

            if delta < fallback_delta:
                fallback_idx, fallback_delta = idx, delta
            if delta > max_delta:
                continue

            sharp = _sharpness(src, comp, bbox)
            score = float(np.log1p(sharp)) - delta / 20.0
            if score > best_score:
                best_idx, best_score, best_delta = idx, score, delta

        # 颜色一致性只用于排序；当只有一个可用源时不能因为色差把主遮挡物留下
        if best_idx < 0 and fallback_idx >= 0:
            best_idx, best_delta = fallback_idx, fallback_delta

        if best_idx < 0:
            info["skipped"] += 1
            continue

        src = sources[best_idx]
        valid = valids[best_idx]
        dist = cv2.distanceTransform((comp > 0).astype(np.uint8), cv2.DIST_L2, 5)
        ramp = np.clip(dist / float(feather), 0.0, 1.0).astype(np.float32)
        alpha = (float(cfg.get("min_alpha", 0.5)) +
                 (1.0 - float(cfg.get("min_alpha", 0.5))) * ramp)
        alpha *= (comp > 0).astype(np.float32)
        alpha *= (valid > 0).astype(np.float32)
        out = (out.astype(np.float32) * (1.0 - alpha[..., None]) +
               src.astype(np.float32) * alpha[..., None])
        out = np.clip(out, 0, 255).astype(np.uint8)
        info["filled"] += 1
        info["sources"].append({"region": i, "area": area, "source": best_idx,
                                "delta": round(best_delta, 2)})
    return out, info
