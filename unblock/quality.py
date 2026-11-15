"""质量评估：选基、预检与合成后的质量闸门。

稳定性的一半是"知道这次不行"。这里把原本只写进报告的数字变成可判定条件，
同时给自动选基一个统一评分，避免"谁放第一张"由用户随手决定。
"""

from typing import Dict, List, Optional

import cv2
import numpy as np

from .common import resize_to, to_gray


def sharpness(img: np.ndarray, quad: Optional[np.ndarray] = None, max_side: int = 1024) -> float:
    small, _ = resize_to(img, max_side)
    gray = to_gray(small)
    if quad is not None:
        h, w = gray.shape[:2]
        sx, sy = w / float(img.shape[1]), h / float(img.shape[0])
        q = (np.asarray(quad, np.float32) * [sx, sy]).astype(np.int32)
        mask = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(mask, q, 255)
        if int(np.count_nonzero(mask)) > 100:
            gray = cv2.bitwise_and(gray, gray, mask=mask)
            vals = gray[mask > 0]
            if vals.size > 100:
                return float(np.var(vals.astype(np.float32)))
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def clipped_ratio(img: np.ndarray) -> float:
    small, _ = resize_to(img, 640)
    gray = to_gray(small)
    clipped = int(np.count_nonzero((gray <= 3) | (gray >= 252)))
    return clipped / float(gray.size)


def frontal_score(quad: np.ndarray) -> float:
    """四边形的"正对程度"：面积 / 外接框面积，越接近 1 越正。"""
    q = np.asarray(quad, np.float32).reshape(-1, 2)
    area = 0.5 * abs(float(np.dot(q[:, 0], np.roll(q[:, 1], -1)) -
                           np.dot(q[:, 1], np.roll(q[:, 0], -1))))
    x0, y0 = q.min(axis=0)
    x1, y1 = q.max(axis=0)
    bbox = max(1e-6, float((x1 - x0) * (y1 - y0)))
    return float(np.clip(area / bbox, 0.0, 1.0))


def _normalize(values: List[float]) -> List[float]:
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [1.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def choose_base(metrics: List[Dict], cfg: Dict) -> int:
    """按遮挡、清晰度、正对程度、曝光四项打分；同分时用文件名稳定排序。"""
    if not metrics:
        return 0
    occ = _normalize([float(m["occlusion"]) for m in metrics])
    sharp = _normalize([float(m["sharpness"]) for m in metrics])
    persp = _normalize([float(m["frontal"]) for m in metrics])
    expo = _normalize([float(m["clipped"]) for m in metrics])

    w = cfg
    ranked = []
    for i, m in enumerate(metrics):
        score = (-float(w.get("occlusion_weight", 1.4)) * occ[i]
                 + float(w.get("sharpness_weight", 0.8)) * sharp[i]
                 + float(w.get("perspective_weight", 0.5)) * persp[i]
                 - float(w.get("exposure_weight", 0.3)) * expo[i])
        name = str(m.get("name", i))
        ranked.append((round(score, 6), -float(m["sharpness"]), float(m["occlusion"]), name, i))
    ranked.sort(key=lambda r: (-r[0], -r[1], r[2], r[3]))
    return int(ranked[0][4])


def qc_score(report: Dict) -> float:
    """两次候选之间比优劣用的标量分，越大越好。"""
    coverage = float(report.get("coverage", 0.0))
    blind = float(report.get("blind_ratio", 1.0))
    seam = float(report.get("seam_step", 0.0))
    return coverage - 1.5 * blind - min(seam, 100.0) / 25.0


def postcheck(report: Dict, cfg: Dict) -> Dict:
    issues: List[str] = []
    if float(report.get("coverage", 0.0)) < float(cfg.get("min_coverage", 0.45)):
        issues.append("coverage_low")
    if float(report.get("blind_ratio", 1.0)) > float(cfg.get("max_blind_ratio", 0.45)):
        issues.append("blind_high")
    if float(report.get("seam_step", 0.0)) > float(cfg.get("max_seam_step", 45.0)):
        issues.append("seam_high")
    align = report.get("align", {}) or {}
    if float(align.get("residual", 0.0)) > float(cfg.get("max_align_residual", 18.0)):
        issues.append("align_residual_high")
    if align.get("mode") == "sift" and \
            float(align.get("coverage", 0.0)) < float(cfg.get("min_inlier_coverage", 0.2)):
        issues.append("align_coverage_low")
    if float(report.get("content_change", 0.0)) > float(cfg.get("max_content_change", 0.45)):
        issues.append("content_changed")
    if report.get("align", {}).get("mode") == "failed":
        issues.append("align_failed")
    return {"pass": len(issues) == 0, "issues": issues}
