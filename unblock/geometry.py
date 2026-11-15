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
                       debug_path: Optional[str] = None,
                       mode: str = "dark") -> Tuple[np.ndarray, float, np.ndarray]:
    """用线段直线交点法找平面四边形。

    mode="dark" 对应黑板/投影幕，"bright" 对应白板/纸张；两者共用交点逻辑，
    只替换平面内部的亮度先验，这样纸张场景不会再被"暗色大矩形"的假设挡住。
    """
    max_side = int(screen_cfg["detect_max_side"])
    mc = screen_cfg if mode == "dark" else dict(screen_cfg.get("bright", {}))
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
                    if xr - xl < float(mc.get("min_width_ratio", screen_cfg["min_width_ratio"])) * w:
                        continue
                    if yb - yt < float(mc.get("min_height_ratio", screen_cfg["min_height_ratio"])) * h:
                        continue
                    require_center = bool(mc.get("require_center_inside",
                                                 screen_cfg.get("require_center_inside", True)))
                    if require_center and \
                            cv2.pointPolygonTest(quad.reshape(-1, 1, 2), (w / 2.0, h / 2.0), False) < 0:
                        continue

                    area_ratio = ((xr - xl) * (yb - yt)) / float(w * h)
                    min_area = float(mc.get("min_area_ratio", screen_cfg["min_area_ratio"]))
                    max_area = float(mc.get("max_area_ratio", screen_cfg["max_area_ratio"]))
                    if not (min_area <= area_ratio <= max_area):
                        continue
                    aspect = (xr - xl) / max(1e-6, (yb - yt))
                    aspect_min = float(mc.get("aspect_min", screen_cfg["aspect_min"]))
                    aspect_max = float(mc.get("aspect_max", screen_cfg["aspect_max"]))
                    if not (aspect_min <= aspect <= aspect_max):
                        continue

                    inner = np.zeros((h, w), np.uint8)
                    cv2.fillConvexPoly(inner, np.clip(quad, 0, [w - 1, h - 1]).astype(np.int32), 255)
                    if inner.sum() == 0:
                        continue
                    inner_gray = gray[inner > 0]
                    support = float(np.mean([
                        _edge_support(grad, quad[0], quad[1]),
                        _edge_support(grad, quad[1], quad[2]),
                        _edge_support(grad, quad[2], quad[3]),
                        _edge_support(grad, quad[3], quad[0]),
                    ]))

                    if mode == "dark":
                        metric = float((inner_gray < dark_thresh).mean())
                        if metric < float(screen_cfg["min_dark_ratio"]):
                            continue
                        if screen_cfg.get("max_median_luma") is not None:
                            if float(np.median(inner_gray)) > float(screen_cfg["max_median_luma"]):
                                continue
                        bright_ratio = float((inner_gray > 170).mean())
                        if not (float(screen_cfg["min_bright_ratio"]) <= bright_ratio
                                <= float(screen_cfg["max_bright_ratio"])):
                            continue
                    else:
                        metric = float((inner_gray > 170).mean())
                        inner_med = float(np.median(inner_gray))
                        if inner_med < float(mc.get("median_luma_min", 150)):
                            continue
                        if inner_med > float(mc.get("median_luma_max", 252)):
                            continue
                        if metric < float(mc.get("min_bright_ratio", 0.45)):
                            continue
                        # 平面要比周围亮，避免把整面白墙/天花板当成纸
                        outer = cv2.dilate(inner, np.ones((9, 9), np.uint8))
                        ring = (outer > 0) & (inner == 0)
                        if int(ring.sum()) >= 100:
                            outer_med = float(np.median(gray[ring]))
                            if inner_med - outer_med < float(mc.get("min_contrast", 8.0)):
                                continue

                    score = area_ratio * (0.15 + support / 255.0)
                    if best is None or score > best[0]:
                        best = (score, quad, metric, support)

    if best is None:
        if debug_path:
            cv2.imwrite(debug_path, all_vis, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        raise RuntimeError("未找到满足约束的屏幕四边形")

    score, quad, metric, support = best
    full = order_quad(quad / scale)
    vis = all_vis.copy()
    draw_quad(vis, order_quad(quad))
    cv2.putText(vis, f"score={score:.3f} {mode}={metric:.2f} edge={support:.0f}",
                (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    return full, float(score), vis


def homography_from_quad(aux_quad: np.ndarray, base_quad: np.ndarray) -> np.ndarray:
    """把 aux 的四角映射到 base 的四角。"""
    return cv2.getPerspectiveTransform(order_quad(aux_quad), order_quad(base_quad))


def _normalize_h(h_mat) -> Optional[np.ndarray]:
    h = np.asarray(h_mat, dtype=np.float64)
    if h.shape != (3, 3) or abs(float(h[2, 2])) < 1e-12:
        return None
    return h / float(h[2, 2])


def _quad_area(quad: np.ndarray) -> float:
    x, y = quad[:, 0], quad[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _geometric_ok(h_mat: np.ndarray, aux_shape, base_shape) -> bool:
    """只检查单应本身是否退化，不再拿四角当语义参考。

    正确的解会把 aux 整幅画面映射成 base 坐标系里的一个凸四边形，
    它允许落在 base 画面外，但不允许翻折、退化或透视项失控。
    """
    h, w = aux_shape[:2]
    bh, bw = base_shape[:2]
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(corners, np.asarray(h_mat, np.float64)).reshape(-1, 2)
    if not np.isfinite(mapped).all():
        return False

    def _cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    signs = [_cross(mapped[i], mapped[(i + 1) % 4], mapped[(i + 2) % 4]) for i in range(4)]
    if not (all(s > 0 for s in signs) or all(s < 0 for s in signs)):
        return False

    area = _quad_area(mapped)
    base_area = float(bw * bh)
    if area < 0.03 * base_area or area > 12.0 * base_area:
        return False

    denom = float(h_mat[2, 0]) * w + float(h_mat[2, 1]) * h + float(h_mat[2, 2])
    if abs(denom) < 1e-9:
        return False
    perspective = abs(float(h_mat[2, 0]) * w / denom) + abs(float(h_mat[2, 1]) * h / denom)
    if perspective > 0.9:
        return False

    # 至少要有一块落进 base 画面，避免选到完全不重叠的解
    margin_x, margin_y = 0.25 * bw, 0.25 * bh
    if not np.any((mapped[:, 0] > -margin_x) & (mapped[:, 0] < bw + margin_x) &
                  (mapped[:, 1] > -margin_y) & (mapped[:, 1] < bh + margin_y)):
        return False
    return True


def _scaled_h(h_mat: np.ndarray, s_base: float, s_aux: float) -> np.ndarray:
    """把全分辨率单应换算到缩放图坐标系。s 为 小图/原图。"""
    sb = np.diag([s_base, s_base, 1.0])
    sa_inv = np.diag([1.0 / s_aux, 1.0 / s_aux, 1.0])
    return sb @ np.asarray(h_mat, np.float64) @ sa_inv


def _photometric_residual(base: np.ndarray, aux: np.ndarray, h_mat: np.ndarray,
                         max_side: int = 960) -> float:
    """重叠区的稳健光度残差：先拟合一阶亮度映射，再取绝对残差中位数。"""
    b_small, s_b = resize_to(base, max_side)
    a_small, s_a = resize_to(aux, max_side)
    hh, ww = b_small.shape[:2]
    hs = _scaled_h(h_mat, s_b, s_a)
    warped = cv2.warpPerspective(a_small, hs, (ww, hh), flags=cv2.INTER_LINEAR)
    valid_small = cv2.warpPerspective(np.full(a_small.shape[:2], 255, np.uint8), hs, (ww, hh),
                                      flags=cv2.INTER_NEAREST) > 128

    gb = to_gray(b_small).astype(np.float32)
    gw = to_gray(warped).astype(np.float32)
    keep = valid_small & (gb > 8) & (gw > 8)
    n = int(np.count_nonzero(keep))
    if n < 800:
        return 255.0
    idx = np.flatnonzero(keep.reshape(-1))
    if n > 20000:
        idx = np.random.default_rng(7).choice(idx, 20000, replace=False)
    x = gw.reshape(-1)[idx].astype(np.float64)
    y = gb.reshape(-1)[idx].astype(np.float64)
    try:
        k, b = np.polyfit(x, y, 1)
    except Exception:
        return 255.0
    res = np.abs(y - (k * x + b))
    return float(np.median(res))


def _inlier_coverage(pts: np.ndarray, width: int, height: int,
                     grid: int = 3, min_per_cell: int = 3) -> float:
    if len(pts) == 0:
        return 0.0
    gx = np.clip((pts[:, 0] / max(1, width) * grid).astype(int), 0, grid - 1)
    gy = np.clip((pts[:, 1] / max(1, height) * grid).astype(int), 0, grid - 1)
    counts = {}
    for x, y in zip(gx.tolist(), gy.tolist()):
        counts[(y, x)] = counts.get((y, x), 0) + 1
    filled = sum(1 for v in counts.values() if v >= min_per_cell)
    return filled / float(grid * grid)


def _sift_homography(base: np.ndarray, aux: np.ndarray, align_cfg: Dict):
    max_side = int(align_cfg.get("match_max_side", 2000))
    b_small, b_scale = resize_to(base, max_side)
    a_small, a_scale = resize_to(aux, max_side)
    gb, ga = to_gray(b_small), to_gray(a_small)

    sift = cv2.SIFT_create(nfeatures=int(align_cfg["sift_nfeatures"]))
    kp_b, des_b = sift.detectAndCompute(gb, None)
    kp_a, des_a = sift.detectAndCompute(ga, None)
    if des_b is None or des_a is None or len(kp_b) < 30 or len(kp_a) < 30:
        return None, 0, 0.0

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    knn = matcher.knnMatch(des_a, des_b, k=2)
    ratio = float(align_cfg["ratio_test"])
    good = [m for m, n in knn if m.distance < ratio * n.distance]
    if len(good) < 12:
        return None, 0, 0.0

    src = np.float32([kp_a[m.queryIdx].pt for m in good]) / a_scale
    dst = np.float32([kp_b[m.trainIdx].pt for m in good]) / b_scale
    h_mat, inliers = cv2.findHomography(src, dst, cv2.RANSAC, float(align_cfg["ransac_thresh"]))
    if h_mat is None or inliers is None:
        return None, 0, 0.0

    inlier_count = int(inliers.sum())
    if inlier_count < int(align_cfg.get("min_inliers", 12)):
        return None, inlier_count, 0.0
    h_norm = _normalize_h(h_mat)
    if h_norm is None:
        return None, inlier_count, 0.0
    sel = inliers.reshape(-1) > 0
    coverage = _inlier_coverage(dst[sel], base.shape[1], base.shape[0])
    return h_norm, inlier_count, coverage


def _ecc_homography(base: np.ndarray, aux: np.ndarray, align_cfg: Dict):
    if not bool(align_cfg.get("ecc_enable", True)):
        return None, {}
    max_side = int(align_cfg.get("ecc_max_side", 960))
    b_small, s_b = resize_to(base, max_side)
    a_small, s_a = resize_to(aux, max_side)
    hh, ww = b_small.shape[:2]
    if a_small.shape[1] != ww or a_small.shape[0] != hh:
        sx = ww / float(a_small.shape[1])
        sy = hh / float(a_small.shape[0])
        a_small = cv2.resize(a_small, (ww, hh), interpolation=cv2.INTER_AREA)
    else:
        sx, sy = 1.0, 1.0

    gb, ga = to_gray(b_small), to_gray(a_small)
    warp = np.eye(2, 3, dtype=np.float32)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            int(align_cfg.get("ecc_iterations", 60)), float(align_cfg.get("ecc_eps", 1e-5)))
    try:
        cc, warp = cv2.findTransformECC(gb, ga, warp, cv2.MOTION_AFFINE, crit, None, 5)
    except cv2.error:
        return None, {}
    if float(cc) < float(align_cfg.get("ecc_min_cc", 0.2)):
        return None, {"cc": float(cc)}

    w3 = np.vstack([warp.astype(np.float64), [0.0, 0.0, 1.0]])
    t_a = np.diag([s_a, s_a, 1.0])
    t_stretch = np.diag([sx, sy, 1.0])
    t_b_inv = np.diag([1.0 / s_b, 1.0 / s_b, 1.0])
    return _normalize_h(t_b_inv @ w3 @ t_stretch @ t_a), {"cc": float(cc)}


def _phase_homography(base: np.ndarray, aux: np.ndarray, align_cfg: Dict):
    if not bool(align_cfg.get("phase_enable", True)):
        return None, {}
    max_side = int(align_cfg.get("ecc_max_side", 960))
    b_small, s_b = resize_to(base, max_side)
    a_small, s_a = resize_to(aux, max_side)
    hh, ww = b_small.shape[:2]
    if a_small.shape[1] != ww or a_small.shape[0] != hh:
        sx = ww / float(a_small.shape[1])
        sy = hh / float(a_small.shape[0])
        a_small = cv2.resize(a_small, (ww, hh), interpolation=cv2.INTER_AREA)
    else:
        sx, sy = 1.0, 1.0

    gb = to_gray(b_small).astype(np.float32) / 255.0
    ga = to_gray(a_small).astype(np.float32) / 255.0
    win = cv2.createHanningWindow((ww, hh), cv2.CV_32F)
    try:
        (dx, dy), response = cv2.phaseCorrelate(ga, gb, win)
    except cv2.error:
        return None, {}
    if float(response) < float(align_cfg.get("phase_min_response", 0.05)):
        return None, {"response": float(response)}

    t_a = np.diag([s_a, s_a, 1.0])
    t_stretch = np.diag([sx, sy, 1.0])
    t_b_inv = np.diag([1.0 / s_b, 1.0 / s_b, 1.0])

    def _build(ox: float, oy: float) -> np.ndarray:
        w3 = np.array([[1.0, 0.0, ox], [0.0, 1.0, oy], [0.0, 0.0, 1.0]], np.float64)
        return t_b_inv @ w3 @ t_stretch @ t_a

    best_h, best_res = None, 255.0
    for ox, oy in ((dx, dy), (-dx, -dy)):
        h_cand = _normalize_h(_build(ox, oy))
        if h_cand is None or not _geometric_ok(h_cand, aux.shape, base.shape):
            continue
        res = _photometric_residual(base, aux, h_cand)
        if res < best_res:
            best_h, best_res = h_cand, res
    if best_h is None:
        return None, {"response": float(response)}
    return best_h, {"response": float(response), "residual": best_res}


def estimate_alignment(base: np.ndarray, aux: np.ndarray, align_cfg: Dict,
                       aux_quad: Optional[np.ndarray] = None,
                       base_quad: Optional[np.ndarray] = None):
    """按 SIFT -> ECC -> 相位相关 -> 四角单应 的顺序估计 aux -> base。

    与旧实现的区别：四角不再作为"正确性裁判"，只作为最后一级兜底；
    每一级都用几何合理性与重叠区光度残差独立验收，避免四角标注误差误杀正确解。
    """
    limit = float(align_cfg.get("align_max_residual_median", 25.0))
    info = {"mode": "none", "inliers": 0, "coverage": 0.0, "residual": 255.0}

    h_sift, inliers, coverage = _sift_homography(base, aux, align_cfg)
    if h_sift is not None and _geometric_ok(h_sift, aux.shape, base.shape):
        res = _photometric_residual(base, aux, h_sift)
        if res <= limit:
            info.update(mode="sift", inliers=inliers, coverage=coverage, residual=res)
            return h_sift, info
        info.update(residual=res)
    else:
        info.update(inliers=inliers, coverage=coverage)

    h_ecc, ecc_info = _ecc_homography(base, aux, align_cfg)
    if h_ecc is not None and _geometric_ok(h_ecc, aux.shape, base.shape):
        res = _photometric_residual(base, aux, h_ecc)
        if res <= limit:
            info.update(mode="ecc", residual=res, **ecc_info)
            return h_ecc, info

    h_phase, phase_info = _phase_homography(base, aux, align_cfg)
    if h_phase is not None and _geometric_ok(h_phase, aux.shape, base.shape):
        res = _photometric_residual(base, aux, h_phase)
        if res <= limit:
            info.update(mode="phase", residual=res, **phase_info)
            return h_phase, info

    # 只有两端都有真实平面四角时，四角单应才是语义正确的兜底
    if aux_quad is not None and base_quad is not None:
        h_quad = homography_from_quad(aux_quad, base_quad)
        if _geometric_ok(h_quad, aux.shape, base.shape):
            res = _photometric_residual(base, aux, h_quad)
            info.update(mode="quad", residual=res)
            return h_quad, info

    info["mode"] = "failed"
    return None, info


def homography_from_features(base: np.ndarray, aux: np.ndarray, align_cfg: Dict,
                             prev_h: Optional[np.ndarray] = None):
    """兼容旧调用：返回 (单应, 内点数, 小图)。失败时单应返回 None。"""
    h_mat, info = estimate_alignment(base, aux, align_cfg)
    if h_mat is None:
        return None, int(info.get("inliers", 0)), resize_to(base, int(align_cfg.get("match_max_side", 2000)))[0]
    if prev_h is not None:
        hh, ww = aux.shape[:2]
        probe = np.float32([[0, 0], [ww, 0], [ww, hh], [0, hh], [ww / 2, hh / 2]]).reshape(-1, 1, 2)
        p_new = cv2.perspectiveTransform(probe, h_mat).reshape(-1, 2)
        p_ref = cv2.perspectiveTransform(probe, np.asarray(prev_h, np.float64)).reshape(-1, 2)
        shift = float(np.max(np.linalg.norm(p_new - p_ref, axis=1)))
        if shift > float(max(hh, ww)) * float(align_cfg.get("guard_max_shift_ratio", 0.6)):
            return None, int(info.get("inliers", 0)), resize_to(base, int(align_cfg.get("match_max_side", 2000)))[0]
    return h_mat, int(info.get("inliers", 0)), resize_to(base, int(align_cfg.get("match_max_side", 2000)))[0]


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
