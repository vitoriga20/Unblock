"""Unblock 技术验证主流程。

用法：
    python -m unblock.pipeline --config config.json --stage corners
    python -m unblock.pipeline --config config.json --stage all
"""

import argparse
import json
import os
import time
from typing import Dict, Optional

import cv2
import numpy as np

from .common import (draw_quad, ensure_dir, imread_unicode, imwrite_unicode, load_config,
                     order_quad, resize_to)
from .compose import blend, estimate_gain_map, match_color_lab, seam_score, seam_step
from .geometry import (detect_screen_quad, homography_from_features, homography_from_quad,
                       rectify, refine_residual, warp_to_base)
from .occlusion import build_screen_mask, detect_occlusion
from .segment import person_mask


def log(msg: str) -> None:
    print(f"[unblock] {msg}", flush=True)


def _save(path: str, img: np.ndarray, quality: int) -> str:
    imwrite_unicode(path, img, quality)
    return path


def resolve_quad(img: np.ndarray, override: Optional[list], screen_cfg: Dict,
                 tag: str, debug_dir: str, quality: int):
    """返回 (原图坐标四角, 置信分, 来源)。同时输出四角诊断图。"""
    max_side = int(screen_cfg["detect_max_side"])
    if override is not None:
        quad = order_quad(np.asarray(override, dtype=np.float32))
        small, scale = resize_to(img, max_side)
        vis = small.copy()
        draw_quad(vis, quad * scale)
        src = "override"
        score = 1.0
    else:
        quad, score, vis = detect_screen_quad(
            img, screen_cfg, os.path.join(debug_dir, f"dbg_{tag}_lines.jpg"))
        src = "auto"

    _save(os.path.join(debug_dir, f"dbg_{tag}_corners.jpg"), vis, quality)
    log(f"{tag} 四角来源={src} 置信分={score:.4f} 四角={quad.round(1).tolist()}")
    return quad, score, src


def process_group(group: Dict, cfg: Dict) -> Dict:
    ws, out_cfg = cfg["workspace"], cfg["output"]
    quality = int(out_cfg["jpeg_quality"])
    debug_dir = ws["debug_dir"]
    gid = group["id"]

    base_path = os.path.join(ws["input_dir"], group["base"])
    aux_path = os.path.join(ws["input_dir"], group["aux"])
    base = imread_unicode(base_path)
    aux = imread_unicode(aux_path)
    log(f"[{group['id']}] {group['base']} {base.shape[1]}x{base.shape[0]}  {group['aux']} {aux.shape[1]}x{aux.shape[0]}")

    t0 = time.time()
    base_quad, base_score, base_src = resolve_quad(
        base, group["corners"]["base"], cfg["screen"], f"{gid}_base", debug_dir, quality)
    aux_quad, aux_score, aux_src = resolve_quad(
        aux, group["corners"]["aux"], cfg["screen"], f"{gid}_aux", debug_dir, quality)

    h_quad = homography_from_quad(aux_quad, base_quad)
    h_used, inliers, mode = h_quad, 0, "quad"
    if bool(cfg["align"]["refine"]):
        h_ref, inl, _ = homography_from_features(base, aux, cfg["align"], h_quad)
        if h_ref is not None:
            h_used, inliers, mode = h_ref, inl, "sift+quad-guard"
        else:
            log(f"[{gid}] SIFT 精修未通过校验，回退到四角单应")
    log(f"[{gid}] 对齐方式={mode} 内点={inliers}")

    aux_warped, aux_valid = warp_to_base(aux, h_used, base.shape)
    residual_info = {"applied": False, "matches": 0, "max_shift": 0.0}
    if bool(cfg["align"]["residual_refine"]):
        aux_warped, aux_valid, residual_info = refine_residual(base, aux_warped, aux_valid, cfg["align"])
        log(f"[{gid}] 局部残余精修 applied={residual_info['applied']} "
            f"matches={residual_info['matches']} max_shift={residual_info['max_shift']:.1f}px")
    overlay = cv2.addWeighted(base, 0.5, aux_warped, 0.5, 0)
    _save(os.path.join(debug_dir, f"dbg_{gid}_warp_overlay.jpg"), overlay, quality)

    person_base = person_mask(base, cfg["segment"])
    person_aux = person_mask(aux, cfg["segment"])
    person_aux_w = None
    if person_aux is not None:
        person_aux_w = cv2.warpPerspective(person_aux, h_used, (base.shape[1], base.shape[0]),
                                           flags=cv2.INTER_NEAREST)
    if person_base is not None:
        pv = base.copy()
        pv[person_base > 0] = (pv[person_base > 0] * 0.4 + np.array([0, 0, 255]) * 0.6).astype(np.uint8)
        _save(os.path.join(debug_dir, f"dbg_{gid}_person.jpg"), pv, quality)

    screen_mask = build_screen_mask(base.shape, base_quad, int(cfg["occlusion"]["border_trim_px"]))
    mask, occ_info = detect_occlusion(base, aux_warped, aux_valid, screen_mask, cfg["occlusion"],
                                      person_base, person_aux_w)

    region = ((screen_mask > 0) & (mask == 0) & (aux_valid > 0)).astype(np.uint8)
    pad = int(cfg["compose"]["gain_exclude_px"])
    if pad > 0:
        # 接缝外侧一圈像素可能还带着人物残留/半影，参与增益估计会把整张图拉偏
        ring = cv2.dilate(mask, np.ones((pad * 2 + 1, pad * 2 + 1), np.uint8))
        region[ring > 0] = 0
    gain_map = estimate_gain_map(base, aux_warped, region, cfg["compose"])
    aux_corrected = np.clip(aux_warped.astype(np.float32) * gain_map, 0, 255).astype(np.uint8)
    aux_corrected, color_info = match_color_lab(aux_corrected, base, mask, aux_valid,
                                                screen_mask, cfg["compose"])
    if color_info.get("applied"):
        log(f"[{gid}] 环带色彩对齐 ring={color_info['ring_px']}px "
            f"色度偏差 {color_info['chroma'][0]} -> {color_info['chroma'][1]} "
            f"LAB 偏差 {color_info['lab'][0]} -> {color_info['lab'][1]}")

    composed = blend(base, aux_corrected, mask, aux_valid, cfg["compose"])
    seam = seam_score(composed, mask)
    step = seam_step(composed, mask, int(cfg["compose"]["seam_band_px"]))

    mask_vis = base.copy()
    mask_vis[mask > 0] = (mask_vis[mask > 0] * 0.35 + np.array([0, 0, 255]) * 0.65).astype(np.uint8)
    _save(os.path.join(debug_dir, f"dbg_{gid}_mask.jpg"), mask_vis, quality)
    cv2.imwrite(os.path.join(debug_dir, f"dbg_{gid}_mask_raw.png"), mask)
    _save(os.path.join(debug_dir, f"dbg_{gid}_composed_screen.jpg"), composed, quality)

    restored = rectify(composed, base_quad, cfg["rectify"])
    out_path = os.path.join(ws["output_dir"], f"{out_cfg['restored_prefix']}_{gid}.jpg")
    _save(out_path, restored, quality)

    report = {
        "group": gid,
        "base": group["base"],
        "aux": group["aux"],
        "resolution": [base.shape[1], base.shape[0]],
        "quad_source": {"base": base_src, "aux": aux_src},
        "quad_score": {"base": round(float(base_score), 4), "aux": round(float(aux_score), 4)},
        "align": {"mode": mode, "inliers": inliers, "residual": residual_info},
        "gain": [round(float(v), 3) for v in np.median(gain_map.reshape(-1, 3), axis=0)],
        "color_match": color_info,
        "mask": {k: round(float(v), 4) for k, v in occ_info.items()},
        "seam_ratio": round(float(seam), 3),
        "seam_step": round(float(step), 2),
        "output": os.path.basename(out_path),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    log(f"[{gid}] 结果 {json.dumps(report, ensure_ascii=False)}")
    return report


def run(config_path: str, stage: str, only: Optional[str]) -> None:
    cfg = load_config(config_path)
    ws = cfg["workspace"]
    ensure_dir(ws["output_dir"])
    ensure_dir(ws["debug_dir"])

    reports = []
    for group in cfg["groups"]:
        if only and group["id"] != only:
            continue
        if stage == "corners":
            base = imread_unicode(os.path.join(ws["input_dir"], group["base"]))
            aux = imread_unicode(os.path.join(ws["input_dir"], group["aux"]))
            log(f"[{group['id']}] {group['base']} {base.shape[1]}x{base.shape[0]}  {group['aux']} {aux.shape[1]}x{aux.shape[0]}")
            qb, sb, mb = resolve_quad(base, group["corners"]["base"], cfg["screen"],
                                      f"{group['id']}_base", ws["debug_dir"], int(cfg["output"]["jpeg_quality"]))
            qa, sa, ma = resolve_quad(aux, group["corners"]["aux"], cfg["screen"],
                                      f"{group['id']}_aux", ws["debug_dir"], int(cfg["output"]["jpeg_quality"]))
            reports.append({"group": group["id"], "base": {"quad": qb.round(1).tolist(), "src": mb},
                            "aux": {"quad": qa.round(1).tolist(), "src": ma}})
        else:
            reports.append(process_group(group, cfg))

    os.makedirs(os.path.dirname(config_path) or ".", exist_ok=True)
    report_path = os.path.join(ws["output_dir"], "report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(reports, f, ensure_ascii=False, indent=2)
    log(f"报告已写入 {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="遮不住 Unblock 技术验证")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--stage", default="all", choices=["corners", "all"])
    parser.add_argument("--group", default=None)
    args = parser.parse_args()
    run(args.config, args.stage, args.group)


if __name__ == "__main__":
    main()
