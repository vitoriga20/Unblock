"""把母题按多种尺寸渲染成对照表，用于肉眼检查小尺寸可读性。

用法：python design/cover/build_contact_sheet.py
输出：design/cover/contact-<组名>.png
"""

from __future__ import annotations

import io
from pathlib import Path

import cairosvg
from PIL import Image, ImageDraw

BASE = Path(__file__).resolve().parent

# 组名 -> [(相对路径, 标签)]
GROUPS: dict[str, list[tuple[str, str]]] = {
    "colorways": [
        ("icon-dark.svg", "黑底"),
        ("icon-paper.svg", "白底"),
        ("icon-monochrome.svg", "单色层"),
    ],
}

SIZES = [72, 48, 32, 24]
ZOOM = 200
ROW_H = 118
COL_W = 250
PAD = 28
HEADER_H = 46


def render(rel: str, size: int) -> Image.Image:
    png = cairosvg.svg2png(url=str(BASE / rel), output_width=size, output_height=size)
    return Image.open(io.BytesIO(png)).convert("RGBA")


def build_group(name: str, icons: list[tuple[str, str]]) -> None:
    width = PAD * 2 + COL_W * len(icons)
    height = HEADER_H + ROW_H * 2 + PAD
    sheet = Image.new("RGB", (width, height), "#FFFFFF")
    draw = ImageDraw.Draw(sheet)
    draw.text((PAD, 16), f"遮不住 · {name} · 上浅下深", fill="#111114")

    for row, backdrop in enumerate(("#F2F2F7", "#0A0A0B")):
        top = HEADER_H + row * ROW_H
        draw.rectangle([0, top, width, top + ROW_H], fill=backdrop)
        for col, (rel, label) in enumerate(icons):
            x = PAD + col * COL_W
            draw.text((x, top + 10), label, fill="#111114" if row == 0 else "#FFFFFF")
            cursor = x
            for size in SIZES:
                img = render(rel, size)
                sheet.paste(img, (cursor, top + 34), img)
                draw.text((cursor, top + 34 + size + 4), str(size), fill="#6C6C72" if row == 0 else "#8E8E93")
                cursor += size + 16

    out = BASE / f"contact-{name}.png"
    sheet.save(out)
    print(out, sheet.size)

    zoom = Image.new("RGB", (PAD * 2 + (ZOOM + 20) * len(icons), ZOOM + PAD * 2), "#FFFFFF")
    zdraw = ImageDraw.Draw(zoom)
    for col, (rel, label) in enumerate(icons):
        x = PAD + col * (ZOOM + 20)
        img = render(rel, ZOOM)
        zoom.paste(img, (x, PAD), img)
        zdraw.text((x, ZOOM + PAD + 6), label, fill="#6C6C72")
    zout = BASE / f"zoom-{name}.png"
    zoom.save(zout)
    print(zout, zoom.size)


if __name__ == "__main__":
    for group, icons in GROUPS.items():
        build_group(group, icons)
