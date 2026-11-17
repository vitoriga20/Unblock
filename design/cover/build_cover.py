"""合成 1024×500 横版封面。

左侧是生成素材（板书前站着一个人），转黑白以贴合产品的纯黑白基调；
右侧黑场放图标、字标与副标。

用法：python design/cover/build_cover.py
输出：design/cover/cover-1024x500.png
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import cairosvg
from PIL import Image, ImageDraw, ImageEnhance, ImageFont, ImageOps

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

from build_wordmark import main as build_wordmark  # noqa: E402

W, H = 1024, 500
PHOTO_W = 620
PANEL_X = PHOTO_W

INK = (17, 17, 20)
MUTED = (150, 150, 158)

SRC_PHOTO = BASE / "gen" / "board-with-person.jpg"
PHOTO_CONTRAST = 1.12

ICON_SIZE = 52
TEXT_LEFT = PANEL_X + 44
WORDMARK_W = 268
FONT_CN = r"C:\Windows\Fonts\Deng.ttf"


def load_photo(size: tuple[int, int]) -> Image.Image:
    image = Image.open(SRC_PHOTO).convert("RGB")
    width, height = image.size
    target_ar = size[0] / size[1]
    crop_h = min(height, width / target_ar)
    crop_w = crop_h * target_ar
    # 人站在左下，靠左裁；纵向往下压一点，避开板顶那道过亮的天花板
    x0, y0 = 0, (height - crop_h) / 2 + height * 0.05
    image = image.crop((round(x0), round(y0), round(x0 + crop_w), round(y0 + crop_h)))
    image = image.resize(size, Image.LANCZOS)

    image = ImageOps.grayscale(image)
    image = ImageEnhance.Contrast(image).enhance(PHOTO_CONTRAST)
    return image.convert("RGB")


def render_icon(size: int) -> Image.Image:
    png = cairosvg.svg2png(url=str(BASE / "icon-dark.svg"), output_width=size, output_height=size)
    return Image.open(io.BytesIO(png)).convert("RGBA")


def main() -> None:
    build_wordmark()

    cover = Image.new("RGB", (W, H), INK)
    cover.paste(load_photo((PHOTO_W, H)), (0, 0))

    wordmark = Image.open(BASE / "wordmark-paper.png").convert("RGBA")
    wordmark = wordmark.resize((WORDMARK_W, round(wordmark.height * WORDMARK_W / wordmark.width)), Image.LANCZOS)
    cover.paste(wordmark, (TEXT_LEFT, 176), wordmark)

    # 字标自带留白，按真实墨迹左缘对齐图标和副标，不然三行左边缘会各差几像素
    ink_left = wordmark.getbbox()[0] if wordmark.getbbox() else 0
    text_x = TEXT_LEFT + ink_left

    icon = render_icon(ICON_SIZE)
    cover.paste(icon, (text_x, 96), icon)

    draw = ImageDraw.Draw(cover)
    tag_font = ImageFont.truetype(FONT_CN, 17)
    draw.text((text_x, 344), "用另一张照片，", font=tag_font, fill=MUTED)
    draw.text((text_x, 372), "救回被挡住的那行字", font=tag_font, fill=MUTED)

    out = BASE / "cover-1024x500.png"
    cover.save(out, quality=95)
    print(out, cover.size)


if __name__ == "__main__":
    main()
