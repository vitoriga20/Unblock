"""图标单一源：一份几何数据，同时产出设计母版 SVG 与 Android vector。

之前 design/cover/icon-*.svg 和 res/drawable/ic_launcher_*.xml 是同一图形的手工两份，
改一处忘一处就会让桌面图标和设计稿慢慢对不上。这个脚本把几何收进 GEOMETRY，
其余全部派生，改形只改这里。

用法：python design/cover/build_icon.py
产出：
  design/cover/icon-dark.svg
  design/cover/icon-paper.svg
  design/cover/icon-monochrome.svg
  android/app/src/main/res/drawable/ic_launcher_foreground.xml
  android/app/src/main/res/drawable/ic_launcher_monochrome.xml
"""

from __future__ import annotations

from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent.parent
RES_DRAWABLE = ROOT / "android" / "app" / "src" / "main" / "res" / "drawable"

CANVAS = 108.0
INK = "#111114"
PAPER = "#FFFFFF"

# 唯一的一份几何：108 画布，图形全部落在直径 72 的可见圆内
GEOMETRY = {
    "head": {"cx": 36.0, "cy": 38.0, "r": 9.5},
    "body": "M27 78C27 60 31 52 36 52C41 52 45 60 45 78Z",
    "bars": [
        {"x": 56.0, "y": 33.0, "w": 29.0, "h": 9.0},
        {"x": 56.0, "y": 49.0, "w": 29.0, "h": 9.0},
        {"x": 56.0, "y": 65.0, "w": 20.0, "h": 9.0},
    ],
}


def num(value: float) -> str:
    """数字统一格式化，避免 36.0 和 36 两种写法在不同产物里出现。"""
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return text or "0"


def head_path() -> str:
    head = GEOMETRY["head"]
    r = num(head["r"])
    left = num(head["cx"] - head["r"])
    right = num(head["r"] * 2)
    top = num(head["cy"])
    return f"M{left},{top}a{r},{r} 0 1,0 {right},0a{r},{r} 0 1,0 -{right},0z"


def bar_path(bar: dict) -> str:
    x, y, w, h = bar["x"], bar["y"], bar["w"], bar["h"]
    r = h / 2
    return (
        f"M{num(x)},{num(y + r)}"
        f"A{num(r)},{num(r)} 0 0 1 {num(x + r)},{num(y)}"
        f"H{num(x + w - r)}"
        f"A{num(r)},{num(r)} 0 0 1 {num(x + w)},{num(y + r)}"
        f"A{num(r)},{num(r)} 0 0 1 {num(x + w - r)},{num(y + h)}"
        f"H{num(x + r)}"
        f"A{num(r)},{num(r)} 0 0 1 {num(x)},{num(y + r)}Z"
    )


def shapes() -> list[str]:
    return [head_path(), GEOMETRY["body"]] + [bar_path(bar) for bar in GEOMETRY["bars"]]


def write_svg(path: Path, fill: str, background: str | None, label: str) -> None:
    paths = "\n".join(f'    <path fill="{fill}" d="{d}"/>' for d in shapes())
    bg = f'  <rect width="{num(CANVAS)}" height="{num(CANVAS)}" fill="{background}"/>\n' if background else ""
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {num(CANVAS)} {num(CANVAS)}" '
        f'width="{num(CANVAS)}" height="{num(CANVAS)}" role="img" aria-label="{label}">\n'
        f"{bg}"
        f"  <g>\n{paths}\n  </g>\n"
        f"</svg>\n"
    )
    path.write_text(svg, encoding="utf-8")


def write_vector(path: Path, comment: str) -> None:
    paths = "\n\n".join(
        f'    <path\n        android:fillColor="#FFFFFF"\n        android:pathData="{d}" />' for d in shapes()
    )
    xml = (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        f"<!-- {comment} -->\n"
        '<vector xmlns:android="http://schemas.android.com/apk/res/android"\n'
        '    android:width="108dp"\n'
        '    android:height="108dp"\n'
        f'    android:viewportWidth="{num(CANVAS)}"\n'
        f'    android:viewportHeight="{num(CANVAS)}">\n\n'
        f"{paths}\n"
        "</vector>\n"
    )
    path.write_text(xml, encoding="utf-8")


def main() -> None:
    write_svg(BASE / "icon-dark.svg", PAPER, INK, "遮不住 图标 黑底")
    write_svg(BASE / "icon-paper.svg", INK, PAPER, "遮不住 图标 纸白")
    write_svg(BASE / "icon-monochrome.svg", PAPER, None, "遮不住 单色层")

    write_vector(
        RES_DRAWABLE / "ic_launcher_foreground.xml",
        "自适应图标前景层：左边一个人，右边三行板书。由 design/cover/build_icon.py 生成，不要手改。",
    )
    write_vector(
        RES_DRAWABLE / "ic_launcher_monochrome.xml",
        "Android 13 主题图标层：只能有形状、不能有底色，系统会整体染色。由 build_icon.py 生成。",
    )
    print("icons written:", BASE, RES_DRAWABLE)


if __name__ == "__main__":
    main()
