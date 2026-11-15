"""遮挡区域检测。

判据从"差异块必须连通画面边界"改成三条证据的组合：
1. 连通画面边界（黑板前站人的典型形态）；
2. 与前景语义 mask 相交（人、手持物）；
3. 画面内部的实心色块，且 base 比 aux 更偏离平面背景色（纸上的手、笔、橡皮）。

这样纸张/桌面场景里不接触边框的遮挡物也能被补掉，同时用"实心 + 厚度 + 颜色证据"
过滤掉板面内容变化造成的细笔画差异。
"""

from typing import Dict, Tuple

import cv2
import numpy as np

from .common import to_gray


def build_screen_mask(shape, quad: np.ndarray, trim_px: int) -> np.ndarray:
    h, w = shape[:2]
    mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(mask, np.asarray(quad, dtype=np.int32).reshape(-1, 2), 255)
    if trim_px > 0:
        mask = cv2.erode(mask, np.ones((trim_px * 2 + 1, trim_px * 2 + 1), np.uint8))
    return mask


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape[:2]
    ff = mask.copy()
    cv2.floodFill(ff, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 255)
    holes = cv2.bitwise_not(ff)
    return cv2.bitwise_or(mask, holes)


def _plane_background(base: np.ndarray, screen_mask: np.ndarray,
                      diff_binary: np.ndarray) -> np.ndarray:
    """平面背景色：优先取平面内未发生差异的像素，样本不足时退回整个平面。"""
    inside = screen_mask > 0
    sample = inside & (diff_binary == 0)
    if int(np.count_nonzero(sample)) < 500:
        sample = inside
    if int(np.count_nonzero(sample)) == 0:
        return np.zeros(3, np.float32)
    return np.median(base[sample].reshape(-1, base.shape[2]), axis=0).astype(np.float32)


def _keep_occluder_blobs(binary: np.ndarray, min_area: int, bridge: int,
                         screen_mask: np.ndarray, base: np.ndarray, aux: np.ndarray,
                         cfg: Dict, person: np.ndarray = None) -> np.ndarray:
    """保留遮挡物连通块。

    不再用大核膨胀去"弥合"连通性：那会把满屏文字连成一张网并误判成贴边遮挡物。
    改成直接看原始二值块，证据为 贴边 / 与语义前景相交 / 画面内部实心色块 三者之一。
    """
    h, w = binary.shape[:2]
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if num <= 1:
        return binary

    bg = _plane_background(base, screen_mask, binary)
    longest = float(max(h, w))
    delta = float(cfg.get("interior_color_delta", 18.0))
    min_thickness = max(4.0, float(cfg.get("interior_min_thickness_ratio", 0.008)) * longest)
    min_interior_area = max(min_area, int(float(cfg.get("interior_min_area_ratio", 0.006)) * h * w))
    min_solidity = float(cfg.get("interior_min_solidity", 0.22))
    interior_enable = bool(cfg.get("interior_blobs", True))
    person_near = None
    if person is not None:
        grow = max(9, int(cfg.get("person_grow_kernel", 31)))
        person_near = cv2.dilate(person, np.ones((grow, grow), np.uint8)) > 0

    keep = np.zeros((h, w), np.uint8)
    for i in range(1, num):
        x, y, bw, bh = (int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
                        int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT]))
        if bw <= 0 or bh <= 0:
            continue
        sub_bin = (labels[y:y + bh, x:x + bw] == i) & (binary[y:y + bh, x:x + bw] > 0)
        comp_px = int(np.count_nonzero(sub_bin))
        if comp_px < min_area:
            # 麦克风、笔、手这类细长物面积小，但紧贴人形，仍然要补
            if person_near is not None:
                hit = int(np.count_nonzero(person_near[y:y + bh, x:x + bw] & sub_bin))
                if hit >= max(10, int(0.3 * comp_px)):
                    keep[labels == i] = 255
            continue

        touches = x <= 1 or y <= 1 or (x + bw) >= w - 1 or (y + bh) >= h - 1
        # 距离场最大厚度：粉笔/字体边缘这类细笔画直接排除，手、杯子、书明显更厚
        dist = cv2.distanceTransform(sub_bin.astype(np.uint8), cv2.DIST_L2, 5)
        thick = float(dist.max()) >= min_thickness

        keep_it = False
        if person_near is not None:
            hit = int(np.count_nonzero(person_near[y:y + bh, x:x + bw] & sub_bin))
            keep_it = hit >= max(20, int(0.1 * comp_px))
            if keep_it:
                keep[labels == i] = 255
                continue

        if touches and thick:
            keep_it = True

        if not keep_it and interior_enable and comp_px >= min_interior_area and thick:
            solidity = comp_px / float(bw * bh)
            if solidity >= min_solidity:
                comp_full = np.zeros((h, w), bool)
                comp_full[y:y + bh, x:x + bw] = sub_bin
                mb = base[comp_full].reshape(-1, base.shape[2]).mean(axis=0)
                ma = aux[comp_full].reshape(-1, aux.shape[2]).mean(axis=0)
                db = float(np.linalg.norm(mb - bg))
                da = float(np.linalg.norm(ma - bg))
                # base 明显偏离平面背景、而 aux 更接近背景，才说明 base 是被挡的那张
                keep_it = db >= delta and (db - da) >= delta * 0.5

        if keep_it:
            keep[labels == i] = 255

    return cv2.bitwise_and(binary, keep)


def _shadow_band(gray_base: np.ndarray, gray_aux: np.ndarray, allowed: np.ndarray,
                 person: np.ndarray, cfg: Dict) -> np.ndarray:
    """找出被人物半影压暗的区域。只取 base 偏暗的方向。"""
    b = gray_base.astype(np.float32) + 1.0
    a = gray_aux.astype(np.float32) + 1.0
    ratio = b / a

    max_px = int(min(float(cfg["shadow_max_px"]),
                     max(8.0, float(max(person.shape)) * float(cfg["shadow_max_ratio"]))))
    dist = cv2.distanceTransform(1 - (person > 0).astype(np.uint8), cv2.DIST_L2, 5)
    around = dist <= max_px
    reference = allowed & ~around
    if int(reference.sum()) < 1000:
        return np.zeros(person.shape, bool)

    gain = float(np.median(ratio[reference]))
    dark = ((ratio < gain * float(cfg["shadow_ratio"])) &
            ((gain * a - b) > float(cfg["shadow_delta"])))
    shadow = dark & around & allowed
    s8 = cv2.morphologyEx(shadow.astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    s8 = cv2.morphologyEx(s8, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    return (s8 > 0) & around & allowed


def content_change_ratio(base: np.ndarray, aux_warped: np.ndarray, aux_valid: np.ndarray,
                         screen_mask: np.ndarray, cfg: Dict) -> float:
    """重叠区里"差异很大"的像素占比，用来判断两张是否还是同一份内容。"""
    if not bool(cfg.get("content_check", True)):
        return 0.0
    k = int(cfg.get("diff_blur_kernel", 9))
    gb = cv2.GaussianBlur(to_gray(base), (k, k), 0)
    ga = cv2.GaussianBlur(to_gray(aux_warped), (k, k), 0)
    diff = cv2.absdiff(gb, ga)
    overlap = (screen_mask > 0) & (aux_valid > 0)
    total = int(np.count_nonzero(overlap))
    if total < 1000:
        return 0.0
    changed = int(np.count_nonzero((diff >= int(cfg.get("content_diff_thresh", 45))) & overlap))
    return changed / float(total)


def detect_occlusion(base: np.ndarray, aux_warped: np.ndarray, aux_valid: np.ndarray,
                     screen_mask: np.ndarray, cfg: Dict,
                     person_base: np.ndarray = None, person_aux_warped: np.ndarray = None
                     ) -> Tuple[np.ndarray, Dict]:
    """返回 (base 侧遮挡 mask, 诊断信息)。

    核心规则不变：只有另一张在该处没有人/物时，才允许用它的像素填补。
    """
    k = int(cfg["diff_blur_kernel"])
    gb = cv2.GaussianBlur(to_gray(base), (k, k), 0)
    ga = cv2.GaussianBlur(to_gray(aux_warped), (k, k), 0)
    diff = cv2.absdiff(gb, ga)
    diff[aux_valid == 0] = 0
    binary = (diff >= int(cfg["diff_thresh"])).astype(np.uint8) * 255

    def _odd(v):
        v = int(v)
        return v if v % 2 == 1 else v + 1

    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((_odd(cfg["morph_open_kernel"]),) * 2, np.uint8))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((_odd(cfg["morph_close_kernel"]),) * 2, np.uint8))

    h, w = binary.shape[:2]
    min_area = int(cfg["min_blob_area_ratio"] * h * w)
    binary = _keep_occluder_blobs(binary, min_area, bridge=max(3, int(cfg["dilate_kernel"]) * 2),
                                  screen_mask=screen_mask, base=base, aux=aux_warped, cfg=cfg,
                                  person=person_base)

    if person_base is not None:
        # 麦克风/笔这类低对比细长物会被大核模糊抹掉，单独用一次小模糊、低阈值补在人物附近
        nk = int(cfg.get("near_blur_kernel", 3))
        nb = cv2.GaussianBlur(to_gray(base), (nk, nk), 0)
        na = cv2.GaussianBlur(to_gray(aux_warped), (nk, nk), 0)
        nd = cv2.absdiff(nb, na)
        nd[aux_valid == 0] = 0
        grow = max(9, int(cfg.get("person_grow_kernel", 31)))
        near = cv2.dilate(person_base, np.ones((grow, grow), np.uint8)) > 0
        extra = ((nd >= int(cfg.get("near_diff_thresh", 18))) & near).astype(np.uint8) * 255
        binary = np.maximum(binary, extra)

    screen_px = max(1, int(np.count_nonzero(screen_mask)))
    diff_ratio = float(np.count_nonzero(binary)) / float(screen_px)
    if diff_ratio > float(cfg.get("max_diff_fill_ratio", 0.15)):
        # 差异铺满平面时，多半是视角/刷新造成的整体光度差，不是遮挡。
        # 此时只保留紧贴前景的那一圈，避免把整块内容替换掉。
        if person_base is not None:
            grow = _odd(cfg.get("person_grow_kernel", 31))
            near = cv2.dilate(person_base, np.ones((grow, grow), np.uint8))
            binary = cv2.bitwise_and(binary, near)
        else:
            binary = np.zeros_like(binary)

    if person_aux_warped is not None:
        block_k = _odd(cfg.get("person_block_kernel", 81))
        aux_blocked = cv2.dilate(person_aux_warped, np.ones((block_k, block_k), np.uint8)) > 0
    else:
        aux_blocked = np.zeros((h, w), bool)

    fill = binary > 0
    if person_base is not None:
        fill |= (person_base > 0) & ~aux_blocked
    fill &= ~aux_blocked

    if person_base is not None and bool(cfg.get("shadow_enable", False)):
        allowed = (screen_mask > 0) & (aux_valid > 0) & ~aux_blocked
        fill |= _shadow_band(gb, ga, allowed, person_base, cfg)

    fill = fill.astype(np.uint8) * 255
    fill = cv2.morphologyEx(fill, cv2.MORPH_CLOSE, np.ones((_odd(cfg["morph_close_kernel"]),) * 2, np.uint8))
    fill = cv2.dilate(fill, np.ones((_odd(cfg["dilate_kernel"]),) * 2, np.uint8))

    screen = screen_mask > 0
    scoped = np.zeros_like(fill)
    scoped[screen] = fill[screen]
    scoped = _fill_holes(scoped)
    scoped[screen == False] = 0

    covered = int(np.count_nonzero(scoped & (aux_valid > 0)))
    total = int(np.count_nonzero(scoped))
    info = {
        "mask_pixels": total,
        "fillable_pixels": covered,
        "coverage": (covered / total) if total else 0.0,
        "blind_ratio": (1.0 - covered / total) if total else 0.0,
        "diff_mean_on_mask": float(diff[scoped > 0].mean()) if total else 0.0,
        "content_change": content_change_ratio(base, aux_warped, aux_valid, screen_mask, cfg),
    }
    return scoped, info
