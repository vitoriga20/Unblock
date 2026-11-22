# -*- coding: utf-8 -*-
"""
生成教程示例素材：同一块板书、两个机位、遮挡位置互补的两张图。

设计约束：
- 板书由代码渲染，两张图共用同一份，保证内容绝对一致
- 两个机位通过透视变换模拟，差异控制在可对齐范围内
- 人物以背光剪影形式叠加，避免肖像问题与干扰板书阅读
- 输出像素四角坐标，格式与 assets/config.json 的 groups 段一致

用法：
    python make_tutorial.py
产物：
    demo_a.jpg / demo_b.jpg / demo_corners.json
"""

import json
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------- 全局参数（集中在此，避免散落字面量） ----------
OUT_W, OUT_H = 2400, 1600          # 输出画布，长边 2400 与端上缓存一致
BOARD_W, BOARD_H = 2000, 1300      # 板书渲染尺寸
FONT_PATH = r"C:\Windows\Fonts\Deng.ttf"   # 等线，笔画细，接近粉笔

BOARD_BG_BGR = (33, 47, 42)        # 深墨绿黑板
CHALK_RGB = (238, 241, 235)        # 粉笔白
FRAME_BGR = (146, 148, 146)        # 板框浅灰
WALL_BGR = (74, 76, 74)            # 板外墙面
PERSON_BGR = (26, 28, 30)          # 背光人影

FONT_TITLE = 80
FONT_SECTION = 54
FONT_BODY = 46
LINE_GAP = 94

# 两个机位的板书四角（画布像素坐标，顺序 左上 右上 右下 左下）
QUAD_A = [(185, 205), (2195, 150), (2245, 1478), (140, 1520)]
QUAD_B = [(265, 125), (2282, 215), (2222, 1502), (205, 1442)]

# 人物站位（画布像素）：A 挡左侧，B 挡右侧
PERSON_A = dict(cx=720, visible_h=1340)
PERSON_B = dict(cx=1745, visible_h=1360)


def render_board() -> np.ndarray:
    """渲染一块带粉笔字与抛物线的黑板。"""
    base = np.full((BOARD_H, BOARD_W, 3), BOARD_BG_BGR, np.float32)

    # 低频不均匀，模拟擦痕与反光（幅度压低，避免像迷彩）
    low = np.random.normal(0, 1, (BOARD_H // 70, BOARD_W // 70)).astype(np.float32)
    low = cv2.resize(low, (BOARD_W, BOARD_H), interpolation=cv2.INTER_CUBIC) * 3.0
    base += low[:, :, None]
    # 细颗粒
    base += np.random.normal(0, 2.2, (BOARD_H, BOARD_W, 1)).astype(np.float32)
    board = np.clip(base, 0, 255)

    # 文字与图形单独绘制到灰度层
    layer = Image.new("L", (BOARD_W, BOARD_H), 0)
    d = ImageDraw.Draw(layer)
    f_title = ImageFont.truetype(FONT_PATH, FONT_TITLE)
    f_section = ImageFont.truetype(FONT_PATH, FONT_SECTION)
    f_body = ImageFont.truetype(FONT_PATH, FONT_BODY)

    d.text((BOARD_W // 2, 86), "第三章  二次函数", font=f_title, fill=255, anchor="mm")
    d.line([(110, 158), (BOARD_W - 110, 158)], fill=165, width=3)

    # 左列
    left_x = 110
    y = 240
    d.text((left_x, y), "一、标准式", font=f_section, fill=255)
    y += LINE_GAP
    for line in ["y = ax\u00b2 + bx + c", "a > 0 开口向上", "对称轴 x = -b/2a"]:
        d.text((left_x + 16, y), line, font=f_body, fill=232)
        y += LINE_GAP

    # 中间抛物线
    cx, cy = BOARD_W // 2, 760
    pts = []
    for x in range(-400, 401, 6):
        pts.append((cx + x, cy - (x * x) / 900.0 + 250))
    d.line(pts, fill=212, width=5)
    d.line([(cx, cy - 170), (cx, cy + 300)], fill=140, width=3)
    d.text((cx + 28, cy + 285), "图象", font=f_body, fill=215)

    # 右列
    right_x = 1180
    y = 240
    d.text((right_x, y), "二、顶点与判别式", font=f_section, fill=255)
    y += LINE_GAP
    for line in ["顶点 (-b/2a, (4ac-b\u00b2)/4a)", "\u0394 = b\u00b2 - 4ac", "\u0394 > 0 有两个交点"]:
        d.text((right_x + 16, y), line, font=f_body, fill=232)
        y += LINE_GAP

    # 练习分两行，避免超出板面
    y += 24
    d.text((right_x, y), "练习：求 y = x\u00b2 - 4x + 3", font=f_body, fill=225)
    y += LINE_GAP
    d.text((right_x + 16, y), "的顶点坐标", font=f_body, fill=225)

    # 粉笔颗粒：文本边缘轻微模糊，再乘颗粒衰减
    mask = np.array(layer, np.float32)
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    grain = np.random.uniform(0.55, 1.0, (BOARD_H, BOARD_W)).astype(np.float32)
    mask = np.clip(mask * grain, 0, 255)

    alpha = (mask / 255.0)[:, :, None]
    chalk = np.array(CHALK_RGB[::-1], np.float32)  # RGB -> BGR
    out = board * (1 - alpha) + chalk * alpha

    # 板框：四周浅灰边 + 内侧一道暗线，跟着透视一起变形
    fw = 14
    out[:fw, :] = FRAME_BGR
    out[-fw:, :] = FRAME_BGR
    out[:, :fw] = FRAME_BGR
    out[:, -fw:] = FRAME_BGR
    out[fw + 2:fw + 5, fw:-fw] = np.array(FRAME_BGR, np.float32) * 0.55
    out[-fw - 5:-fw - 2, fw:-fw] = np.array(FRAME_BGR, np.float32) * 0.55
    out[fw:-fw, fw + 2:fw + 5] = np.array(FRAME_BGR, np.float32) * 0.55
    out[fw:-fw, -fw - 5:-fw - 2] = np.array(FRAME_BGR, np.float32) * 0.55

    return np.clip(out, 0, 255).astype(np.uint8)


def perspective_to(canvas: np.ndarray, board: np.ndarray, quad) -> None:
    """把板书按四角透视贴到画布上。"""
    src = np.float32([[0, 0], [BOARD_W, 0], [BOARD_W, BOARD_H], [0, BOARD_H]])
    dst = np.float32(quad)
    m = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(board, m, (canvas.shape[1], canvas.shape[0]),
                                 flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    mask = np.zeros((canvas.shape[0], canvas.shape[1]), np.uint8)
    cv2.fillConvexPoly(mask, dst.astype(np.int32), 255)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)[:, :, None].astype(np.float32) / 255.0
    canvas[:] = (canvas * (1 - mask) + warped * mask).astype(np.uint8)


def draw_person(canvas: np.ndarray, cx: int, visible_h: int) -> None:
    """在画布上叠加一个背光人影：头 + 肩 + 收窄的躯干，底部延伸出画面。"""
    h, w = canvas.shape[:2]
    bottom_y = h + 60
    top_y = bottom_y - visible_h

    head_rx = int(visible_h * 0.064)
    head_ry = int(visible_h * 0.076)
    head_cy = top_y + head_ry + int(visible_h * 0.010)

    neck_w = int(visible_h * 0.062)
    neck_top = head_cy + head_ry - int(visible_h * 0.016)
    neck_bot = neck_top + int(visible_h * 0.050)

    shoulder_w = int(visible_h * 0.330)
    shoulder_cy = neck_bot + int(visible_h * 0.020)
    shoulder_ry = int(visible_h * 0.052)

    torso_top = shoulder_cy
    torso_top_w = int(shoulder_w * 0.86)
    torso_bot_w = int(shoulder_w * 0.80)

    mask = np.zeros((h, w), np.uint8)
    # 头
    cv2.ellipse(mask, (cx, head_cy), (head_rx, head_ry), 0, 0, 360, 255, -1)
    # 脖子
    cv2.rectangle(mask, (cx - neck_w // 2, neck_top), (cx + neck_w // 2, neck_bot), 255, -1)
    # 肩（宽扁椭圆，制造斜肩）
    cv2.ellipse(mask, (cx, shoulder_cy), (shoulder_w // 2, shoulder_ry), 0, 0, 360, 255, -1)
    # 躯干（上宽下窄的梯形）
    body = np.array([
        [cx - torso_top_w // 2, torso_top],
        [cx + torso_top_w // 2, torso_top],
        [cx + torso_bot_w // 2, bottom_y],
        [cx - torso_bot_w // 2, bottom_y],
    ], np.int32)
    cv2.fillPoly(mask, [body], 255)

    mask = cv2.GaussianBlur(mask, (9, 9), 2.0).astype(np.float32) / 255.0

    # 纵向渐变：头顶略亮，向下加深
    grad = np.linspace(1.32, 0.84, h, dtype=np.float32)[:, None, None]
    person = np.clip(np.array(PERSON_BGR, np.float32) * grad, 0, 255)

    alpha = mask[:, :, None]
    out = canvas.astype(np.float32) * (1 - alpha) + person * alpha

    # 左缘轮廓光，避免剪影变成死黑一块
    mask_u8 = (mask * 255).astype(np.uint8)
    ring = cv2.subtract(mask_u8, cv2.erode(mask_u8, np.ones((7, 7), np.uint8)))
    xs = np.arange(w, dtype=np.float32)[None, :]
    left_gate = np.clip((cx + 20 - xs) / 80.0, 0, 1)
    rim = (ring.astype(np.float32) / 255.0) * left_gate * 0.5
    rim_bgr = np.array((172, 178, 180), np.float32)
    out = out * (1 - rim[:, :, None]) + rim_bgr * rim[:, :, None]

    canvas[:] = np.clip(out, 0, 255).astype(np.uint8)


def add_camera_effects(canvas: np.ndarray, seed: int) -> np.ndarray:
    """光照渐变、暗角、轻微模糊与噪声，模拟手机拍摄。"""
    rng = np.random.default_rng(seed)
    h, w = canvas.shape[:2]
    img = canvas.astype(np.float32)

    gx = np.linspace(1.07, 0.95, w, dtype=np.float32)[None, :, None]
    gy = np.linspace(1.06, 0.96, h, dtype=np.float32)[:, None, None]
    img *= gx * gy

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xx / w - 0.5) * 2
    ny = (yy / h - 0.5) * 2
    vig = 1 - 0.20 * np.clip(nx * nx + ny * ny, 0, 1)
    img *= vig[:, :, None]

    img = cv2.GaussianBlur(img, (3, 3), 0.5)
    img += rng.normal(0, 2.0, img.shape).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


def build(quad, person_cx, person_h, seed):
    canvas = np.full((OUT_H, OUT_W, 3), WALL_BGR, np.uint8)
    wall_noise = np.random.default_rng(seed + 7).normal(0, 2.5, (OUT_H, OUT_W, 3))
    canvas = np.clip(canvas.astype(np.float32) + wall_noise, 0, 255).astype(np.uint8)

    board = render_board()
    perspective_to(canvas, board, quad)
    draw_person(canvas, person_cx, person_h)
    return add_camera_effects(canvas, seed)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(here, exist_ok=True)
    np.random.seed(20261122)

    img_a = build(QUAD_A, PERSON_A["cx"], PERSON_A["visible_h"], seed=11)
    img_b = build(QUAD_B, PERSON_B["cx"], PERSON_B["visible_h"], seed=23)

    # 示例页第三屏：合成结果。用同一份板书，只加轻微成像痕迹，保持"照片感"
    result = render_board()
    result = cv2.GaussianBlur(result, (3, 3), 0.4)
    result = np.clip(
        result.astype(np.float32) + np.random.default_rng(31).normal(0, 1.8, result.shape),
        0, 255
    ).astype(np.uint8)

    path_a = os.path.join(here, "demo_a.jpg")
    path_b = os.path.join(here, "demo_b.jpg")
    path_r = os.path.join(here, "demo_result.jpg")
    cv2.imwrite(path_a, img_a, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
    cv2.imwrite(path_b, img_b, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
    cv2.imwrite(path_r, result, [int(cv2.IMWRITE_JPEG_QUALITY), 94])

    corners = {
        "demo_a": {"file": "demo_a.jpg", "quad": QUAD_A},
        "demo_b": {"file": "demo_b.jpg", "quad": QUAD_B},
    }
    with open(os.path.join(here, "demo_corners.json"), "w", encoding="utf-8") as f:
        json.dump(corners, f, ensure_ascii=False, indent=2)

    print("已生成：")
    for p in (path_a, path_b):
        print(" ", p, os.path.getsize(p) // 1024, "KB")


if __name__ == "__main__":
    main()
