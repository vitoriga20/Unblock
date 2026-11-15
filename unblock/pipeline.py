"""Unblock 主流程。

相对于旧版的三处结构变化：
1. 平面检测改成多策略候选（暗色 / 亮色 / 轮廓 / 内缩），不再假定只有黑板；
2. 先评估每张图再决定谁是 base，必要时对另一个 base 重跑一次择优；
3. 遮挡补丁从"base 单向吸入一张 aux"改成"每个遮挡块单独挑最合适的源"。
"""

import argparse
import json
import os
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .common import (draw_quad, ensure_dir, imread_unicode, imwrite_unicode, load_config,
                     order_quad, resize_to)
from .compose import (blend, estimate_gain_map, estimate_gain_map_robust, match_color_lab,
                      seam_score, seam_step)
from .fusion import fuse
from .geometry import estimate_alignment, rectify, refine_residual, warp_to_base
from .occlusion import build_screen_mask, detect_occlusion
from .plane import detect_candidates
from .quality import choose_base, clipped_ratio, frontal_score, postcheck, qc_score, sharpness
from .segment import person_mask


def log(msg: str) -> None:
    print(f"[unblock] {msg}", flush=True)


def _save(path: str, img: np.ndarray, quality: int) -> str:
    imwrite_unicode(path, img, quality)
    return path


def resolve_quad(img: np.ndarray, override: Optional[list], screen_cfg: Dict,
                 tag: str, debug_dir: str, quality: int):
    """返回 (原图坐标四角, 置信分, 来源, 候选列表)。"""
    if override is not None:
        quad = order_quad(np.asarray(override, dtype=np.float32))
        src, score, kind = "override", 1.0, "override"
        candidates = []
    else:
        candidates = detect_candidates(img, screen_cfg)
        best = candidates[0]
        quad = best["quad"]
        score = float(best["score"])
        kind = str(best["kind"])
        src = "auto"

    if debug_dir:
        max_side = int(screen_cfg["detect_max_side"])
        small, scale = resize_to(img, max_side)
        vis = small.copy()
        draw_quad(vis, quad * scale)
        cv2.putText(vis, f"{kind} score={score:.3f}", (12, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        _save(os.path.join(debug_dir, f"dbg_{tag}_corners.jpg"), vis, quality)

    log(f"{tag} 四角来源={src} 类型={kind} 置信分={score:.4f} 四角={quad.round(1).tolist()}")
    return quad, score, kind, candidates


def _prepare_source(base: np.ndarray, base_quad: np.ndarray, aux: np.ndarray,
                    aux_quad: Optional[np.ndarray], person_base: Optional[np.ndarray],
                    person_aux: Optional[np.ndarray], cfg: Dict) -> Dict:
    """把一张 aux 对齐、去遮挡、配色，准备成可供融合的源。"""
    h_mat, align_info = estimate_alignment(base, aux, cfg["align"], aux_quad, base_quad)
    if h_mat is None:
        raise RuntimeError("两张照片无法对齐：特征、ECC、相位相关都未通过验收")

    aux_warped, aux_valid = warp_to_base(aux, h_mat, base.shape)
    residual_info = {"applied": False, "matches": 0, "max_shift": 0.0}
    if bool(cfg["align"]["residual_refine"]):
        aux_warped, aux_valid, residual_info = refine_residual(base, aux_warped, aux_valid, cfg["align"])
        log(f"局部残余精修 applied={residual_info['applied']} "
            f"matches={residual_info['matches']} max_shift={residual_info['max_shift']:.1f}px")

    person_aux_w = None
    if person_aux is not None:
        person_aux_w = cv2.warpPerspective(person_aux, h_mat, (base.shape[1], base.shape[0]),
                                           flags=cv2.INTER_NEAREST)

    screen_mask = build_screen_mask(base.shape, base_quad, int(cfg["occlusion"]["border_trim_px"]))
    # 先做一次对人物不敏感的块中位数增益校正，否则视角造成的幕面亮度差会被当成遮挡
    pre_region = ((screen_mask > 0) & (aux_valid > 0)).astype(np.uint8)
    pre_gain = estimate_gain_map_robust(base, aux_warped, pre_region, cfg["compose"],
                                        grid=int(cfg["compose"].get("gain_grid", 8)))
    aux_pre = np.clip(aux_warped.astype(np.float32) * pre_gain, 0, 255).astype(np.uint8)

    mask, occ_info = detect_occlusion(base, aux_pre, aux_valid, screen_mask, cfg["occlusion"],
                                      person_base, person_aux_w)

    region = ((screen_mask > 0) & (mask == 0) & (aux_valid > 0)).astype(np.uint8)
    pad = int(cfg["compose"]["gain_exclude_px"])
    if pad > 0:
        ring = cv2.dilate(mask, np.ones((pad * 2 + 1, pad * 2 + 1), np.uint8))
        region[ring > 0] = 0
    gain_map = estimate_gain_map(base, aux_pre, region, cfg["compose"])
    aux_corrected = np.clip(aux_pre.astype(np.float32) * gain_map, 0, 255).astype(np.uint8)
    aux_corrected, color_info = match_color_lab(aux_corrected, base, mask, aux_valid,
                                                screen_mask, cfg["compose"])

    return {
        "h": h_mat,
        "align": align_info,
        "residual": residual_info,
        "mask": mask,
        "occ": occ_info,
        "valid": aux_valid,
        "warped": aux_pre,
        "person": person_aux_w,
        "corrected": aux_corrected,
        "gain": [round(float(v), 3) for v in
                 np.median((pre_gain * gain_map).reshape(-1, 3), axis=0)],
        "color": color_info,
        "screen_mask": screen_mask,
    }


def _run_once(images: List[np.ndarray], quads: List[np.ndarray], infos: List[Dict],
              persons: List[Optional[np.ndarray]], base_idx: int, cfg: Dict) -> Tuple[Dict, Dict]:
    """以 base_idx 为画布跑一遍完整合成，返回 (报告, 中间产物)。"""
    base = images[base_idx]
    base_quad = quads[base_idx]
    order = [base_idx] + [i for i in range(len(images)) if i != base_idx]

    sources, masks = [], []
    for idx in order[1:]:
        src = _prepare_source(base, base_quad, images[idx], quads[idx],
                              persons[base_idx], persons[idx], cfg)
        src["index"] = idx
        sources.append(src)
        masks.append(src["mask"])

    if not sources:
        raise RuntimeError("至少需要一张补图")

    hole = np.zeros(base.shape[:2], np.uint8)
    for m in masks:
        hole = np.maximum(hole, m)
    screen_mask = sources[0]["screen_mask"]

    if bool(cfg["fusion"].get("enable", True)):
        fusion_cfg = dict(cfg["fusion"])
        fusion_cfg["min_alpha"] = cfg["compose"]["min_alpha"]
        fusion_cfg["edge_soft_px"] = cfg["compose"]["edge_soft_px"]
        composed, fusion_info = fuse(base, [s["corrected"] for s in sources],
                                     [s["valid"] for s in sources],
                                     [s["person"] for s in sources],
                                     hole, screen_mask, fusion_cfg)
    else:
        composed = base
        for s in sources:
            composed = blend(composed, s["corrected"], s["mask"], s["valid"], cfg["compose"])
        fusion_info = {"regions": 0, "filled": 0, "skipped": 0, "sources": []}

    valid_union = np.zeros(base.shape[:2], np.uint8)
    for s in sources:
        valid_union = np.maximum(valid_union, s["valid"])
    covered = int(np.count_nonzero((hole > 0) & (valid_union > 0)))
    total = int(np.count_nonzero(hole))
    coverage = covered / total if total else 1.0
    blind = 1.0 - coverage if total else 0.0

    seam = seam_score(composed, hole)
    step = seam_step(composed, hole, int(cfg["compose"]["seam_band_px"]))
    restored = rectify(composed, base_quad, cfg["rectify"])

    first = sources[0]
    align_summary = {
        "mode": first["align"].get("mode", "none"),
        "inliers": int(first["align"].get("inliers", 0)),
        "coverage": round(float(first["align"].get("coverage", 0.0)), 3),
        "residual": round(float(first["align"].get("residual", 0.0)), 2),
        "residual_refine": first["residual"],
    }
    content_change = max(float(s["occ"].get("content_change", 0.0)) for s in sources)

    report = {
        "base": os.path.basename(str(infos[base_idx]["name"])),
        "base_index": base_idx,
        "resolution": [int(base.shape[1]), int(base.shape[0])],
        "scene": {"kind": infos[base_idx]["kind"], "score": round(float(infos[base_idx]["plane_score"]), 4)},
        "align": align_summary,
        "coverage": round(float(coverage), 4),
        "blind_ratio": round(float(blind), 4),
        "content_change": round(float(content_change), 4),
        "fusion": {k: v for k, v in fusion_info.items() if k != "sources"},
        "fusion_sources": fusion_info.get("sources", []),
        "sources": [{
            "index": int(s["index"]),
            "align_mode": s["align"].get("mode", "none"),
            "residual": round(float(s["align"].get("residual", 0.0)), 2),
            "gain": s["gain"],
            "color": s["color"],
            "mask": {k: round(float(v), 4) for k, v in s["occ"].items()},
        } for s in sources],
        "seam_ratio": round(float(seam), 3),
        "seam_step": round(float(step), 2),
    }
    report["quality"] = postcheck(report, cfg["quality"])

    artifacts = {
        "base": base,
        "composed": composed,
        "restored": restored,
        "hole": hole,
        "screen_mask": screen_mask,
        "warp_overlay": cv2.addWeighted(base, 0.5, sources[0]["warped"], 0.5, 0),
        "person_base": persons[base_idx],
    }
    return report, artifacts


def _metrics(images: List[np.ndarray], quads: List[np.ndarray], infos: List[Dict],
             persons: List[Optional[np.ndarray]]) -> List[Dict]:
    out = []
    for i, img in enumerate(images):
        quad = quads[i]
        # 选基指标刻意不依赖四角：四角本身可能不可靠，用整图前景占比和整图清晰度更稳定
        occ = 0.0
        if persons[i] is not None:
            occ = float(np.count_nonzero(persons[i] > 0) / float(img.shape[0] * img.shape[1]))
        out.append({
            "index": i,
            "name": infos[i]["name"],
            "occlusion": occ,
            "sharpness": sharpness(img),
            "frontal": frontal_score(quad),
            "clipped": clipped_ratio(img),
            "plane_score": float(infos[i]["plane_score"]),
            "kind": infos[i]["kind"],
        })
    return out


def process_group(group: Dict, cfg: Dict, force_base: Optional[int] = None,
                  return_artifacts: bool = False):
    ws, out_cfg = cfg["workspace"], cfg["output"]
    quality = int(out_cfg["jpeg_quality"])
    debug_dir = ws["debug_dir"]
    gid = group["id"]

    names = [group["base"], group["aux"]]
    paths = [os.path.join(ws["input_dir"], n) for n in names]
    images = [imread_unicode(p) for p in paths]
    for n, img in zip(names, images):
        log(f"[{gid}] {n} {img.shape[1]}x{img.shape[0]}")

    t0 = time.time()
    quads, infos = [], []
    for view, img, name in zip(("base", "aux"), images, names):
        override = (group.get("corners") or {}).get(view)
        quad, score, kind, candidates = resolve_quad(
            img, override, cfg["screen"], f"{gid}_{view}", debug_dir, quality)
        quads.append(quad)
        infos.append({"name": name, "plane_score": score, "kind": kind,
                      "candidates": [{"kind": c["kind"], "score": round(float(c["score"]), 4)}
                                     for c in candidates]})

    persons = [person_mask(img, cfg["segment"]) for img in images]
    metrics = _metrics(images, quads, infos, persons)

    if force_base is not None:
        base_idx = int(force_base)
        select_info = {"enabled": False, "forced": base_idx}
    elif bool(cfg["base_select"].get("enable", True)):
        base_idx = choose_base(metrics, cfg["base_select"])
        select_info = {"enabled": True, "chosen": base_idx,
                       "scores": [{k: (round(v, 4) if isinstance(v, float) else v)
                                   for k, v in m.items()} for m in metrics]}
    else:
        base_idx = 0
        select_info = {"enabled": False, "chosen": 0}

    attempts = []
    report, artifacts = _run_once(images, quads, infos, persons, base_idx, cfg)
    attempts.append({"base_index": base_idx, "pass": report["quality"]["pass"],
                     "issues": report["quality"]["issues"], "score": round(qc_score(report), 4)})

    retry = (force_base is None and bool(cfg["quality"].get("retry_alt_base", True))
             and len(images) > 1 and not report["quality"]["pass"])
    if retry:
        alt_idx = (base_idx + 1) % len(images)
        try:
            alt_report, alt_artifacts = _run_once(images, quads, infos, persons, alt_idx, cfg)
            attempts.append({"base_index": alt_idx, "pass": alt_report["quality"]["pass"],
                             "issues": alt_report["quality"]["issues"],
                             "score": round(qc_score(alt_report), 4)})
            if (alt_report["quality"]["pass"] and not report["quality"]["pass"]) or \
                    qc_score(alt_report) > qc_score(report):
                report, artifacts, base_idx = alt_report, alt_artifacts, alt_idx
        except Exception as exc:
            attempts.append({"base_index": alt_idx, "error": str(exc)})

    report["group"] = gid
    report["base_selection"] = select_info
    report["attempts"] = attempts
    report["elapsed_sec"] = round(time.time() - t0, 1)

    composed, hole = artifacts["composed"], artifacts["hole"]
    mask_vis = artifacts["base"].copy()
    mask_vis[hole > 0] = (mask_vis[hole > 0] * 0.35 + np.array([0, 0, 255]) * 0.65).astype(np.uint8)
    _save(os.path.join(debug_dir, f"dbg_{gid}_mask.jpg"), mask_vis, quality)
    cv2.imwrite(os.path.join(debug_dir, f"dbg_{gid}_mask_raw.png"), hole)
    _save(os.path.join(debug_dir, f"dbg_{gid}_composed_screen.jpg"), composed, quality)
    _save(os.path.join(debug_dir, f"dbg_{gid}_warp_overlay.jpg"), artifacts["warp_overlay"], quality)
    if artifacts.get("person_base") is not None:
        pv = artifacts["base"].copy()
        pm = artifacts["person_base"]
        pv[pm > 0] = (pv[pm > 0] * 0.4 + np.array([0, 0, 255]) * 0.6).astype(np.uint8)
        _save(os.path.join(debug_dir, f"dbg_{gid}_person.jpg"), pv, quality)

    out_path = os.path.join(ws["output_dir"], f"{out_cfg['restored_prefix']}_{gid}.jpg")
    _save(out_path, artifacts["restored"], quality)
    report["output"] = os.path.basename(out_path)

    log(f"[{gid}] 结果 {json.dumps(report, ensure_ascii=False)}")
    if return_artifacts:
        return report, artifacts
    return report


def run(config_path: str, stage: str, only: Optional[str], force_base: Optional[int] = None) -> None:
    cfg = load_config(config_path)
    ws = cfg["workspace"]
    ensure_dir(ws["output_dir"])
    ensure_dir(ws["debug_dir"])

    reports = []
    for group in cfg["groups"]:
        if only and group["id"] != only:
            continue
        if stage == "corners":
            for view in ("base", "aux"):
                img = imread_unicode(os.path.join(ws["input_dir"], group[view]))
                quad, score, kind, cands = resolve_quad(
                    img, (group.get("corners") or {}).get(view), cfg["screen"],
                    f"{group['id']}_{view}", ws["debug_dir"], int(cfg["output"]["jpeg_quality"]))
                reports.append({"group": group["id"], "view": view, "kind": kind,
                                "score": round(float(score), 4),
                                "quad": quad.round(1).tolist(),
                                "candidates": [{"kind": c["kind"], "score": round(float(c["score"]), 4)}
                                               for c in cands]})
        else:
            reports.append(process_group(group, cfg, force_base=force_base))

    report_path = os.path.join(ws["output_dir"], "report.json")
    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(reports, f, ensure_ascii=False, indent=2)
    log(f"报告已写入 {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="遮不住 Unblock")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--stage", default="all", choices=["corners", "all"])
    parser.add_argument("--group", default=None)
    parser.add_argument("--base", type=int, default=None, help="强制基准图序号（0 或 1），用于回归对比")
    args = parser.parse_args()
    run(args.config, args.stage, args.group, args.base)


if __name__ == "__main__":
    main()
