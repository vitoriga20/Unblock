"""稳定性回归。

三个指标：
- 顺序不变性：强制 0 当 base 与强制 1 当 base，两张结果在统一画布上的差异；
- 角点敏感度：四角按长边比例抖动后，遮挡 mask 的 IoU 与输出差异；
- 内容一致性：沿用主流程里算出的 content_change。

用法：
    python -m unblock.regress --config config.json
    python -m unblock.regress --config config.json --group group0 --jitters 3
"""

import argparse
import copy
import json
import os
from typing import Dict, List

import cv2
import numpy as np

from .common import ensure_dir, load_config
from .pipeline import process_group


def _resize_for_compare(img: np.ndarray, size=(960, 540)) -> np.ndarray:
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def _image_diff(a: np.ndarray, b: np.ndarray) -> float:
    ga = cv2.cvtColor(_resize_for_compare(a), cv2.COLOR_BGR2GRAY).astype(np.float32)
    gb = cv2.cvtColor(_resize_for_compare(b), cv2.COLOR_BGR2GRAY).astype(np.float32)
    return float(np.mean(np.abs(ga - gb)) / 255.0)


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    ma, mb = a > 0, b > 0
    union = int(np.count_nonzero(ma | mb))
    if union == 0:
        return 1.0
    return float(np.count_nonzero(ma & mb) / union)


def _jitter(quad, ratio: float, step: int, diag: float):
    angles = np.deg2rad([45, 135, 225, 315])
    out = np.asarray(quad, np.float32).copy()
    for i in range(4):
        ang = angles[(i + step) % 4]
        out[i, 0] += float(np.cos(ang)) * ratio * diag
        out[i, 1] += float(np.sin(ang)) * ratio * diag
    return out


def run_group(group: Dict, cfg: Dict, jitters: int, ratio: float) -> Dict:
    # 产品路径：自动选基。交换输入顺序后应选到同一张物理图，结果应基本一致。
    auto_report, auto_art = process_group(group, cfg, return_artifacts=True)
    swapped = copy.deepcopy(group)
    swapped["base"], swapped["aux"] = group["aux"], group["base"]
    corners = group.get("corners") or {}
    if corners.get("base") is not None and corners.get("aux") is not None:
        swapped["corners"] = {"base": corners["aux"], "aux": corners["base"]}
    swap_report, swap_art = process_group(swapped, cfg, return_artifacts=True)
    auto_diff = _image_diff(auto_art["restored"], swap_art["restored"])

    base_report, base_art = process_group(group, cfg, force_base=0, return_artifacts=True)
    alt_report, alt_art = process_group(group, cfg, force_base=1, return_artifacts=True)
    order_diff = _image_diff(base_art["restored"], alt_art["restored"])

    result = {
        "group": group["id"],
        "auto_order": {
            "diff": round(auto_diff, 4),
            "base": [auto_report.get("base"), swap_report.get("base")],
            "pass": [auto_report["quality"]["pass"], swap_report["quality"]["pass"]],
        },
        "order": {
            "diff": round(order_diff, 4),
            "base0": {"pass": base_report["quality"]["pass"], "issues": base_report["quality"]["issues"]},
            "base1": {"pass": alt_report["quality"]["pass"], "issues": alt_report["quality"]["issues"]},
        },
        "content_change": max(float(base_report.get("content_change", 0.0)),
                              float(alt_report.get("content_change", 0.0))),
        "corner": {"runs": []},
    }

    corners = (group.get("corners") or {}).get("base")
    if jitters <= 0 or corners is None:
        return result

    diag = float(np.hypot(*np.ptp(np.asarray(corners, np.float32), axis=0)))
    ious: List[float] = []
    diffs: List[float] = []
    for step in range(jitters):
        g2 = copy.deepcopy(group)
        g2.setdefault("corners", {})["base"] = _jitter(np.asarray(corners, np.float32),
                                                       ratio, step, diag).tolist()
        rep, art = process_group(g2, cfg, force_base=0, return_artifacts=True)
        iou = _mask_iou(base_art["hole"], art["hole"])
        d = _image_diff(base_art["restored"], art["restored"])
        ious.append(iou)
        diffs.append(d)
        result["corner"]["runs"].append({"step": step, "mask_iou": round(iou, 4),
                                         "out_diff": round(d, 4),
                                         "pass": rep["quality"]["pass"]})

    result["corner"].update({
        "mask_iou_min": round(float(min(ious)), 4),
        "mask_iou_mean": round(float(np.mean(ious)), 4),
        "out_diff_mean": round(float(np.mean(diffs)), 4),
        "out_diff_max": round(float(max(diffs)), 4),
    })
    return result


def run(config_path: str, only: str, jitters: int, ratio: float) -> None:
    cfg = load_config(config_path)
    ensure_dir(cfg["workspace"]["output_dir"])
    results = []
    for group in cfg["groups"]:
        if only and group["id"] != only:
            continue
        print(f"[regress] {group['id']} ...", flush=True)
        results.append(run_group(group, cfg, jitters, ratio))

    out_path = os.path.join(cfg["workspace"]["output_dir"], "regress.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    for r in results:
        c = r["corner"]
        print(f"[regress] {r['group']} 顺序差异={r['order']['diff']:.3f} "
              f"角点 IoU={(c.get('mask_iou_min', 0)):.3f}~{(c.get('mask_iou_mean', 0)):.3f} "
              f"输出差异均值={c.get('out_diff_mean', 0):.3f}", flush=True)
    print(f"[regress] 写入 {out_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="遮不住 稳定性回归")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--group", default=None)
    parser.add_argument("--jitters", type=int, default=3)
    parser.add_argument("--jitter-ratio", type=float, default=0.02)
    args = parser.parse_args()
    run(args.config, args.group, args.jitters, args.jitter_ratio)


if __name__ == "__main__":
    main()
