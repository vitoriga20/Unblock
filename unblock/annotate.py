"""四角交互标注工具。

用法：
    python -m unblock.annotate --config config.json            # 标注所有缺失四角的图
    python -m unblock.annotate --config config.json --force    # 全部重标
    python -m unblock.annotate --config config.json --render    # 不交互，只渲染现有四角做核对

操作：鼠标左键依次点四个角；s 保存并写回 config.json；r 重来；q 放弃本张。
"""

import argparse
import json
import os

import cv2
import numpy as np

from .common import draw_quad, imread_unicode, load_config, order_quad, resize_to


def _pick(img: np.ndarray, existing, max_side: int, title: str):
    small, scale = resize_to(img, max_side)
    points = []
    win = "unblock-annotate"

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((int(x), int(y)))

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, small.shape[1], small.shape[0])
    cv2.setMouseCallback(win, on_mouse)

    while True:
        vis = small.copy()
        if existing is not None and not points:
            draw_quad(vis, np.asarray(existing, float) * scale, color=(120, 120, 120), thickness=2)
        for i, p in enumerate(points):
            cv2.circle(vis, p, 6, (0, 255, 255), -1)
            cv2.putText(vis, str(i + 1), (p[0] + 8, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            if i > 0:
                cv2.line(vis, points[i - 1], p, (0, 255, 255), 2)
        cv2.putText(vis, f"{title}  click 4 corners | s=save  r=reset  q=skip",
                    (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow(win, vis)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("s") and len(points) == 4:
            break
        if key == ord("r"):
            points.clear()
        if key == ord("q"):
            points.clear()
            break

    cv2.destroyWindow(win)
    if len(points) != 4:
        return None
    return (order_quad(np.asarray(points, dtype=np.float32)) / scale).tolist()


def _save(config_path: str, group_id: str, view: str, corners) -> None:
    cfg = load_config(config_path)
    for g in cfg["groups"]:
        if g["id"] == group_id:
            g.setdefault("corners", {})[view] = [[round(float(x), 1), round(float(y), 1)] for x, y in corners]
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def run(config_path: str, force: bool, render: bool, max_side: int) -> None:
    cfg = load_config(config_path)
    ws = cfg["workspace"]
    for group in cfg["groups"]:
        for view in ("base", "aux"):
            corners = (group.get("corners") or {}).get(view)
            if not force and corners and not render:
                print(f"[skip] {group['id']}/{view} 已有四角")
                continue
            img = imread_unicode(os.path.join(ws["input_dir"], group[view]))
            print(f"[open] {group['id']}/{view} {img.shape[1]}x{img.shape[0]}")
            if render:
                vis, scale = resize_to(img, max_side)
                if corners:
                    draw_quad(vis, np.asarray(corners, float) * scale)
                out = os.path.join(ws["debug_dir"], f"corners_{group['id']}_{view}.jpg")
                cv2.imwrite(out, vis)
                print(f"[render] {out}")
                continue
            picked = _pick(img, corners, max_side, f"{group['id']}/{view}")
            if picked is None:
                print(f"[skip] {group['id']}/{view} 未标注")
                continue
            _save(config_path, group["id"], view, picked)
            print(f"[saved] {group['id']}/{view} -> {picked}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Unblock 四角标注")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--max-side", type=int, default=1280)
    args = parser.parse_args()
    run(args.config, args.force, args.render, args.max_side)


if __name__ == "__main__":
    main()
