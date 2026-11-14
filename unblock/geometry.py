"""屏幕（平面）四角定位、单应对齐与正视校正。"""

from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from .common import draw_quad, order_quad, resize_to, to_gray


def _cluster_1d(values: np.ndarray, weights: np.ndarray, tol: float, top_k: int) -> list:
    """按坐标聚类并保留支持度最高的若干条，返回代表直线索引。"""
    if len(values) == 0:
        return []
    order = np.argsort(values)
    clusters = []
    for idx in order:
        v = float(values[idx])
        if clusters and abs(v - clusters[-1]["center"]) <= tol:
            clusters[-1]["ids"].append(int(idx))
            clusters[-1]["weight"] += float(weights[idx])
            clusters[-1]["center"] = float(np.average(values[clusters[-1]["ids"]],
                                                       weights=weights[clusters[-1]["ids"]]))
        else:
            clusters.append({"ids": [int(idx)], "weight": float(weights[idx]), "center": v})
    clusters.sort(key=lambda c: c["weight"], reverse=True)
    return [c["ids"] for c in clusters[:top_k]]


def _intersect(rho1: float, theta1: float, rho2: float, theta2: float) -> Optional[np.ndarray]:
    a = np.array([[np.cos(theta1), np.sin(theta1)], [np.cos(theta2), np.sin(theta2)]], dtype=np.float64)
    if abs(np.linalg.det(a)) < 1e-6:
        return None
    return np.linalg.solve(a, np.array([rho1, rho2], dtype=np.float64))


def _segments_to_lines(gray: np.ndarray, screen_cfg: Dict):
    """用 LSD 提取线段并参数化为直线 (rho, theta)，权重用线段长度。"""
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    seg = lsd.detect(gray)[0]
    if seg is None:
        return np.zeros(0), np.zeros(0), np.zeros(0)

    segs = seg.reshape(-1, 4)
    h, w = gray.shape[:2]
    min_len = max(0.10 * min(h, w), 40.0)
    rhos, thetas, lengths = [], [], []
    for x1, y1, x2, y2 in segs:
        dx, dy = x2 - x1, y2 - y1
        length = float(np.hypot(dx, dy))
        if length < min_len:
            continue
        nx, ny = -dy / length, dx / length
        theta = np.arctan2(ny, nx) % np.pi
        rho = nx * x1 + ny * y1
        if rho < 0:
            rho, theta = -rho, (theta + np.pi) % np.pi
        rhos.append(rho)
        thetas.append(theta)
        lengths.append(length)
    return (np.asarray(rhos, dtype=np.float64), np.asarray(thetas, dtype=np.float64),
            np.asarray(lengths, dtype=np.float64))


def _split_by_direction(thetas: np.ndarray, tol_rad: float) -> Tuple[list, list]:
    """按近垂直/近水平分组。"""
    vert_idx, horiz_idx = [], []
    for i, t in enumerate(thetas):
        if t <= tol_rad or t >= np.pi - tol_rad:
            vert_idx.append(i)
        elif abs(t - np.pi / 2) <= tol_rad:
            horiz_idx.append(i)
    return vert_idx, horiz_idx


def _vertical_x(rho: float, theta: float, y_ref: float) -> float:
    return float((rho - np.sin(theta) * y_ref) / np.cos(theta))


def _horizontal_y(rho: float, theta: float, x_ref: float) -> float:
    return float((rho - np.cos(theta) * x_ref) / np.sin(theta))


def _edge_support(grad: np.ndarray, p0: np.ndarray, p1: np.ndarray, samples: int = 80, radius: int = 3) -> float:
    h, w = grad.shape[:2]
    acc = []
    for t in np.linspace(0.0, 1.0, samples):
        x = p0[0] + (p1[0] - p0[0]) * t
        y = p0[1] + (p1[1] - p0[1]) * t
        if not (radius <= x < w - radius and radius <= y < h - radius):
            continue
        xi, yi = int(round(x)), int(round(y))
        acc.append(float(grad[yi - radius:yi + radius + 1, xi - radius:xi + radius + 1].max()))
    return float(np.median(acc)) if acc else 0.0


def detect_screen_quad(img: np.ndarray, screen_cfg: Dict,
                       debug_path: Optional[str] = None) -> Tuple[np.ndarray, float, np.ndarray]:
    """用线段直线交点法找屏幕四边形，返回 (原图坐标四角, 置信分, 可视化小图)。"""
    max_side = int(screen_cfg["detect_max_side"])
    small, scale = resize_to(img, max_side)
    h, w = small.shape[:2]
    gray = to_gray(small)
    gray_eq = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)

    rhos, thetas, lengths = _segments_to_lines(gray_eq, screen_cfg)
    if len(rhos) < 4:
        raise RuntimeError(f"线段太少({len(rhos)})，无法定位屏幕")

    tol_rad = np.deg2rad(float(screen_cfg["quad_angle_tolerance_deg"]))
    vert_idx, horiz_idx = _split_by_direction(thetas, tol_rad)

    vert_x = np.array([_vertical_x(rhos[i], thetas[i], h / 2.0) for i in vert_idx])
    horz_y = np.array([_horizontal_y(rhos[i], thetas[i], w / 2.0) for i in horiz_idx])

    v_groups = _cluster_1d(vert_x, lengths[vert_idx], tol=0.020 * w, top_k=8)
    h_groups = _cluster_1d(horz_y, lengths[horiz_idx], tol=0.020 * h, top_k=8)
    v_rep = [vert_idx[g[0]] for g in v_groups]
    h_rep = [horiz_idx[g[0]] for g in h_groups]
    if len(v_rep) < 2 or len(h_rep) < 2:
        raise RuntimeError("可用直线不足，无法构成四边形")

    grad = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    dark_thresh = float(np.percentile(gray, float(screen_cfg["dark_percentile"])))

    best, all_vis = None, small.copy()
    for iv in v_rep:
        if thetas[iv] <= tol_rad or thetas[iv] >= np.pi - tol_rad:
            x0 = _vertical_x(rhos[iv], thetas[iv], 0.0)
            x1 = _vertical_x(rhos[iv], thetas[iv], h - 1.0)
            cv2.line(all_vis, (int(x0), 0), (int(x1), h - 1), (0, 220, 0), 1, cv2.LINE_AA)
    for ih in h_rep:
        y0 = _horizontal_y(rhos[ih], thetas[ih], 0.0)
        y1 = _horizontal_y(rhos[ih], thetas[ih], w - 1.0)
        cv2.line(all_vis, (0, int(y0)), (w - 1, int(y1)), (255, 0, 0), 1, cv2.LINE_AA)
    for iv_l in v_rep:
        for iv_r in v_rep:
            if iv_r == iv_l:
                continue
            for ih_t in h_rep:
                for ih_b in h_rep:
                    if ih_b == ih_t:
                        continue
                    pts = []
                    ok = True
                    for iv, ih in ((iv_l, ih_t), (iv_r, ih_t), (iv_r, ih_b), (iv_l, ih_b)):
                        p = _intersect(rhos[iv], thetas[iv], rhos[ih], thetas[ih])
                        if p is None:
                            ok = False
                            break
                        pts.append(p)
                    if not ok:
                        continue
                    quad = order_quad(np.asarray(pts, dtype=np.float32))
                    xl, xr = quad[0, 0], quad[1, 0]
                    yt, yb = quad[0, 1], quad[3, 1]
                    if xr - xl < float(screen_cfg["min_width_ratio"]) * w:
                        continue
                    if yb - yt < float(screen_cfg["min_height_ratio"]) * h:
                        continue
                    if bool(screen_cfg["require_center_inside"]) and \
                            cv2.pointPolygonTest(quad.reshape(-1, 1, 2), (w / 2.0, h / 2.0), False) < 0:
                        continue

                    area_ratio = ((xr - xl) * (yb - yt)) / float(w * h)
                    if not (float(screen_cfg["min_area_ratio"]) <= area_ratio <= float(screen_cfg["max_area_ratio"])):
                        continue
                    aspect = (xr - xl) / max(1e-6, (yb - yt))
                    if not (float(screen_cfg["aspect_min"]) <= aspect <= float(screen_cfg["aspect_max"])):
                        continue

                    inner = np.zeros((h, w), np.uint8)
                    cv2.fillConvexPoly(inner, np.clip(quad, 0, [w - 1, h - 1]).astype(np.int32), 255)
                    if inner.sum() == 0:
                        continue
                    inner_gray = gray[inner > 0]
                    dark_ratio = float((inner_gray < dark_thresh).mean())
                    if dark_ratio < float(screen_cfg["min_dark_ratio"]):
                        continue
                    if screen_cfg.get("max_median_luma") is not None:
                        if float(np.median(inner_gray)) > float(screen_cfg["max_median_luma"]):
                            continue
                    bright_ratio = float((inner_gray > 170).mean())
                    if not (float(screen_cfg["min_bright_ratio"]) <= bright_ratio <= float(screen_cfg["max_bright_ratio"])):
                        continue

                    support = float(np.mean([
                        _edge_support(grad, quad[0], quad[1]),
                        _edge_support(grad, quad[1], quad[2]),
                        _edge_support(grad, quad[2], quad[3]),
                        _edge_support(grad, quad[3], quad[0]),
                    ]))
                    score = area_ratio * (0.15 + support / 255.0)
                    if best is None or score > best[0]:
                        best = (score, quad, dark_ratio, support)

    if best is None:
        if debug_path:
            cv2.imwrite(debug_path, all_vis, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        raise RuntimeError("未找到满足约束的屏幕四边形")

    score, quad, dark_ratio, support = best
    full = order_quad(quad / scale)
    vis = all_vis.copy()
    draw_quad(vis, order_quad(quad))
    cv2.putText(vis, f"score={score:.3f} dark={dark_ratio:.2f} edge={support:.0f}",
                (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    return full, float(score), vis


def homography_from_quad(aux_quad: np.ndarray, base_quad: np.ndarray) -> np.ndarray:
    """把 aux 的四角映射到 base 的四角。"""
    return cv2.getPerspectiveTransform(order_quad(aux_quad), order_quad(base_quad))


def homography_from_features(base: np.ndarray, aux: np.ndarray, align_cfg: Dict,
                             prev_h: Optional[np.ndarray] = None) -> Tuple[Optional[np.ndarray], int, np.ndarray]:
    """SIFT + RANSAC 估计 aux -> base 的单应，失败时返回 (None, 0, ...)。"""
    max_side = int(align_cfg.get("match_max_side", 2000))
    b_small, b_scale = resize_to(base, max_side)
    a_small, a_scale = resize_to(aux, max_side)
    gb, ga = to_gray(b_small), to_gray(a_small)

    sift = cv2.SIFT_create(nfeatures=int(align_cfg["sift_nfeatures"]))
    kp_b, des_b = sift.detectAndCompute(gb, None)
    kp_a, des_a = sift.detectAndCompute(ga, None)
    if des_b is None or des_a is None or len(kp_b) < 30 or len(kp_a) < 30:
        return None, 0, b_small

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    knn = matcher.knnMatch(des_a, des_b, k=2)
    ratio = float(align_cfg["ratio_test"])
    good = [m for m, n in knn if m.distance < ratio * n.distance]
    if len(good) < 12:
        return None, 0, b_small

    src = np.float32([kp_a[m.queryIdx].pt for m in good]) / a_scale
    dst = np.float32([kp_b[m.trainIdx].pt for m in good]) / b_scale
    h_mat, inliers = cv2.findHomography(src, dst, cv2.RANSAC, float(align_cfg["ransac_thresh"]))
    if h_mat is None or inliers is None:
        return None, 0, b_small

    inlier_count = int(inliers.sum())
    if inlier_count < 12:
        return None, inlier_count, b_small
    if prev_h is not None and not _is_plausible(h_mat, prev_h, aux.shape):
        return None, inlier_count, b_small
    return h_mat, inlier_count, b_small


def _is_plausible(h_mat: np.ndarray, ref_h: np.ndarray, aux_shape) -> bool:
    """检查新单应是否与四角粗对齐量级一致，防止 SIFT 在重复文字上跑飞。"""
    h, w = aux_shape[:2]
    probe = np.float32([[0, 0], [w, 0], [w, h], [0, h], [w / 2, h / 2]]).reshape(-1, 1, 2)
    p_new = cv2.perspectiveTransform(probe, h_mat).reshape(-1, 2)
    p_ref = cv2.perspectiveTransform(probe, ref_h).reshape(-1, 2)
    shift = float(np.max(np.linalg.norm(p_new - p_ref, axis=1)))
    return shift < float(max(h, w)) * 0.25


def warp_to_base(aux: np.ndarray, h_mat: np.ndarray, base_shape) -> Tuple[np.ndarray, np.ndarray]:
    h, w = base_shape[:2]
    warped = cv2.warpPerspective(aux, h_mat, (w, h), flags=cv2.INTER_LINEAR)
    valid = cv2.warpPerspective(np.full(aux.shape[:2], 255, np.uint8), h_mat, (w, h), flags=cv2.INTER_NEAREST)
    valid = cv2.erode(valid, np.ones((3, 3), np.uint8))
    return warped, valid


def refine_residual(base: np.ndarray, aux_warped: np.ndarray, aux_valid: np.ndarray,
                    cfg: Dict) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    全局单应之后仍有局部残余位移（镜头畸变、屏幕不绝对平整）。
    用网格化的特征残差估计一个平滑位移场，再做一次 remap 补偿。
    """
    grid_n = max(2, int(cfg["residual_grid"]))
    max_px = float(cfg["residual_max_px"])
    h, w = base.shape[:2]
    info = {"applied": False, "matches": 0, "max_shift": 0.0}

    max_side = int(cfg.get("match_max_side", 2000))
    b_small, b_scale = resize_to(base, max_side)
    a_small, a_scale = resize_to(aux_warped, max_side)
    gb, ga = to_gray(b_small), to_gray(a_small)

    sift = cv2.SIFT_create(nfeatures=int(cfg["sift_nfeatures"]))
    kp_b, des_b = sift.detectAndCompute(gb, None)
    kp_a, des_a = sift.detectAndCompute(ga, None)
    if des_b is None or des_a is None:
        return aux_warped, aux_valid, info

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    knn = matcher.knnMatch(des_a, des_b, k=2)
    ratio = float(cfg["ratio_test"])
    good = [m for m, n in knn if m.distance < ratio * n.distance]
    if len(good) < 40:
        return aux_warped, aux_valid, info

    src = np.float32([kp_a[m.queryIdx].pt for m in good]) / a_scale
    dst = np.float32([kp_b[m.trainIdx].pt for m in good]) / b_scale
    shift = dst - src
    keep = np.linalg.norm(shift, axis=1) < max_px
    src, shift = src[keep], shift[keep]
    if len(src) < 40:
        return aux_warped, aux_valid, info

    cell = np.zeros((grid_n, grid_n, 2), np.float32)
    count = np.zeros((grid_n, grid_n), np.float32)
    gx = np.clip((src[:, 0] / w * grid_n).astype(int), 0, grid_n - 1)
    gy = np.clip((src[:, 1] / h * grid_n).astype(int), 0, grid_n - 1)
    for i in range(len(src)):
        cell[gy[i], gx[i]] += shift[i]
        count[gy[i], gx[i]] += 1

    filled = count > 0
    cell[filled] /= count[filled][:, None]

    # 空格用邻域均值补齐，避免插值出空洞
    if (~filled).any() and filled.any():
        k = 3
        num = cv2.blur(np.where(filled[..., None], cell, 0.0), (k, k))
        den = cv2.blur(filled.astype(np.float32), (k, k))
        fallback = num / np.maximum(den, 1e-6)[..., None]
        cell = np.where(filled[..., None], cell, fallback)

    smooth_k = max(1, int(cfg["residual_smooth"]))
    if smooth_k > 1:
        cell = cv2.blur(cell, (_odd(smooth_k), _odd(smooth_k)))

    field = cv2.resize(cell, (w, h), interpolation=cv2.INTER_LINEAR)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    map_x = (xx + field[..., 0]).astype(np.float32)
    map_y = (yy + field[..., 1]).astype(np.float32)

    refined = cv2.remap(aux_warped, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    valid = cv2.remap(aux_valid, map_x, map_y, interpolation=cv2.INTER_NEAREST,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    info = {
        "applied": True,
        "matches": int(len(src)),
        "max_shift": float(np.max(np.linalg.norm(shift, axis=1))),
    }
    return refined, valid, info


def _odd(v) -> int:
    v = int(v)
    return v if v % 2 == 1 else v + 1


def rectify(img: np.ndarray, quad: np.ndarray, rectify_cfg: Dict) -> np.ndarray:
    """按四角把屏幕拉成正矩形。"""
    q = order_quad(quad)
    top = np.linalg.norm(q[1] - q[0])
    bottom = np.linalg.norm(q[2] - q[3])
    left = np.linalg.norm(q[3] - q[0])
    right = np.linalg.norm(q[2] - q[1])
    src_w = (top + bottom) / 2.0
    src_h = (left + right) / 2.0

    aspect = rectify_cfg.get("target_aspect") or (src_w / src_h)
    long_side = int(rectify_cfg["target_long_side"])
    if aspect >= 1.0:
        out_w, out_h = long_side, int(round(long_side / aspect))
    else:
        out_h, out_w = long_side, int(round(long_side * aspect))

    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
    m = cv2.getPerspectiveTransform(q, dst)
    return cv2.warpPerspective(img, m, (out_w, out_h), flags=cv2.INTER_CUBIC)
