"""生成「遮不住」字标。

设计说明：
- 字形取等线 Bold，中性几何无衬线，和纯黑白界面同气质；
- 三字不齐距：字重收紧、末两字降到 92%，让视线从「遮」起笔；
- 底下一道板书行，从「遮」左下起笔、略微越过「住」右缘，
  既把三字收成一个整体，也是产品语义（被挡住的那行字）的最小图形。

输出：
- design/cover/wordmark-ink.svg   深色字，浅底用
- design/cover/wordmark-paper.svg 反白字，深底用
- design/cover/wordmark-ink.png / wordmark-paper.png

用法：python design/cover/build_wordmark.py
"""

from __future__ import annotations

from pathlib import Path

import cairosvg
from fontTools.misc.transform import Transform
from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont

BASE = Path(__file__).resolve().parent
FONT_PATH = r"C:\Windows\Fonts\Dengb.ttf"
TEXT = "遮不住"

EM = 1000.0
TRACKING = -34.0
SMALL_SCALE = 0.92          # 后两字相对「遮」的缩放
RULE_THICKNESS = 18.0
RULE_GAP = 150.0            # 板书行与基线的距离
RULE_OVERHANG = 74.0        # 板书行右侧超出末字的部分
RULE_LEFT_EXTEND = 26.0
PAD = 40.0


def build_paths() -> tuple[str, float, float, float]:
    font = TTFont(FONT_PATH)
    glyphs = font.getGlyphSet()
    cmap = font.getBestCmap()
    upem = font["head"].unitsPerEm
    base_scale = EM / upem

    # 第一遍：按基线排版，量出真实墨迹范围（ascent 覆盖不到全部字形，不能拿它当画布）
    boxes: list[tuple[float, float]] = []
    cursor = 0.0
    for index, ch in enumerate(TEXT):
        glyph = glyphs[cmap[ord(ch)]]
        scale = base_scale if index == 0 else base_scale * SMALL_SCALE
        bounds_pen = BoundsPen(glyphs)
        glyph.draw(TransformPen(bounds_pen, Transform(scale, 0, 0, -scale, cursor, 0)))
        if bounds_pen.bounds is None:
            raise RuntimeError(f"字形没有轮廓：{ch}")
        boxes.append(bounds_pen.bounds)
        cursor += glyph.width * scale + TRACKING

    left = min(box[0] for box in boxes)
    top = min(box[1] for box in boxes)
    right = max(box[2] for box in boxes)
    bottom = max(box[3] for box in boxes)

    # 第二遍：整体平移到留白之后的位置
    dx = PAD - left
    dy = PAD - top

    parts: list[str] = []
    cursor = 0.0
    for index, ch in enumerate(TEXT):
        glyph = glyphs[cmap[ord(ch)]]
        scale = base_scale if index == 0 else base_scale * SMALL_SCALE
        pen = SVGPathPen(glyphs)
        glyph.draw(TransformPen(pen, Transform(scale, 0, 0, -scale, cursor + dx, dy)))
        parts.append(pen.getCommands())
        cursor += glyph.width * scale + TRACKING

    text_width = right - left
    text_height = bottom - top
    rule_y = PAD + text_height + RULE_GAP
    rule_x0 = PAD - RULE_LEFT_EXTEND
    rule_x1 = PAD + text_width + RULE_OVERHANG

    rule = (
        f"M{rule_x0:.2f} {rule_y:.2f} "
        f"H{rule_x1:.2f} "
        f"V{rule_y + RULE_THICKNESS:.2f} "
        f"H{rule_x0:.2f} Z"
    )

    width = text_width + PAD * 2 + RULE_OVERHANG
    height = rule_y + RULE_THICKNESS + PAD
    return " ".join(parts) + " " + rule, width, height, PAD + text_height


def write_svg(path: Path, fill: str, d: str, width: float, height: float) -> None:
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.2f} {height:.2f}" '
        f'width="{width:.0f}" height="{height:.0f}" role="img" aria-label="遮不住 字标">\n'
        f"  <!-- 字形由 Dengb.ttf 转轮廓，已去掉字体依赖；板书行与三字同色同层 -->\n"
        f'  <path fill="{fill}" fill-rule="nonzero" d="{d}"/>\n'
        f"</svg>\n"
    )
    path.write_text(svg, encoding="utf-8")


def main() -> None:
    d, width, height, _ = build_paths()
    for name, fill in (("ink", "#111114"), ("paper", "#FFFFFF")):
        svg_path = BASE / f"wordmark-{name}.svg"
        write_svg(svg_path, fill, d, width, height)
        png_path = BASE / f"wordmark-{name}.png"
        cairosvg.svg2png(url=str(svg_path), write_to=str(png_path), output_width=1200)
        print(svg_path, png_path)


if __name__ == "__main__":
    main()
