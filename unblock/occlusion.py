"""遮挡区域检测。基于对齐后帧差，并用"连通到画面边界"来区分遮挡物与画面本身内容。"""

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


def _keep_border_blobs(binary: np.ndarray, min_area: int, bridge: int) -> np.ndarray:
    """
    只保留"最终能延伸到画面边界"的块。
    先膨胀弥合头/身分离，判定连通到边界后再回取原二值，从而排除完全位于屏幕内的图形元素。
    """
    h, w = binary.shape[:2]
    merged = cv2.dilate(binary, np.ones((bridge, bridge), np.uint8)) if bridge > 1 else binary

    num, labels, stats, _ = cv2.connectedComponentsWithStats(merged, 8)
    keep = np.zeros((h, w), np.uint8)
    for i in range(1, num):
        x, y, bw, bh, area = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
            stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT], stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        touches = x <= 1 or y <= 1 or (x + bw) >= w - 1 or (y + bh) >= h - 1
        if not touches:
            continue
        keep[labels == i] = 255

    kept = cv2.bitwise_and(binary, keep)
    return kept


def _shadow_band(gray_base: np.ndarray, gray_aux: np.ndarray, allowed: np.ndarray,
                 person: np.ndarray, cfg: Dict) -> np.ndarray:
    """
    找出被人物半影压暗的区域。投影幕/黑板前站人时，人身上会投出一圈阴影，
    只按人物轮廓填充会留下这圈暗影，视觉上就是一圈"描边"。
    判据：相对全局增益明显偏暗，且只取 base 偏暗的方向（aux 偏暗说明另一张才是被挡的）。
    """
    b = gray_base.astype(np.float32) + 1.0
    a = gray_aux.astype(np.float32) + 1.0
    ratio = b / a

    # 阴影可能离人物轮廓很远（投影灯在幕前，人影偏出去大半个人），用距离场而不是膨胀核来限制范围
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


def detect_occlusion(base: np.ndarray, aux_warped: np.ndarray, aux_valid: np.ndarray,
                     screen_mask: np.ndarray, cfg: Dict,
                     person_base: np.ndarray = None, person_aux_warped: np.ndarray = None) -> Tuple[np.ndarray, Dict]:
    """
    返回 (base 侧遮挡 mask, 诊断信息)。
    核心判据：只有当"另一张图在该处没有人"时，才允许用它的像素填补；
    这样能排除"base 可见、aux 被挡"的反向区域。
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
    binary = _keep_border_blobs(binary, min_area, bridge=max(3, int(cfg["dilate_kernel"]) * 2))

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
    }
    return scoped, info
