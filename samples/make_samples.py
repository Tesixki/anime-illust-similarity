"""デモ用サンプル画像を生成する（本物のイラストの代替品）。

PILで簡易的な「キャラ顔」を描画し、以下を samples/ に出力する:
- char_a_1.png / char_a_2.png : 同じキャラ（配色同じ・表情と背景違い）
- char_b_1.png                : 別キャラ（配色が大きく違う）
- landscape.png               : キャラなしの風景（全く違う画像の例）

使い方:
    python samples/make_samples.py
"""

from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path(__file__).parent
SIZE = (512, 512)


def draw_character(bg, hair, eye, mouth="smile") -> Image.Image:
    """簡易キャラ顔: 髪・顔・目・口だけのプロシージャル画像"""
    img = Image.new("RGB", SIZE, bg)
    d = ImageDraw.Draw(img)
    w, h = SIZE
    cx, cy = w // 2, h // 2 + 20

    # 髪（頭を覆う大きな円）
    d.ellipse([cx - 170, cy - 200, cx + 170, cy + 140], fill=hair)
    # 顔（肌色の楕円で髪の下半分を隠す）
    d.ellipse([cx - 140, cy - 130, cx + 140, cy + 160], fill=(255, 224, 196))
    # 前髪
    d.pieslice([cx - 145, cy - 190, cx + 145, cy - 30], 180, 360, fill=hair)
    # 目
    for dx in (-60, 60):
        d.ellipse([cx + dx - 24, cy + 10, cx + dx + 24, cy + 70], fill=eye)
        d.ellipse([cx + dx - 8, cy + 22, cx + dx + 6, cy + 40], fill="white")
    # 口
    if mouth == "smile":
        d.arc([cx - 30, cy + 90, cx + 30, cy + 130], 10, 170, fill=(200, 90, 90), width=6)
    else:  # open
        d.ellipse([cx - 25, cy + 95, cx + 25, cy + 135], fill=(200, 90, 90))
    return img


def draw_landscape() -> Image.Image:
    img = Image.new("RGB", SIZE, (135, 206, 235))
    d = ImageDraw.Draw(img)
    d.ellipse([380, 40, 470, 130], fill=(255, 220, 60))  # 太陽
    d.polygon([(0, 512), (200, 220), (400, 512)], fill=(90, 140, 90))  # 山
    d.polygon([(150, 512), (380, 280), (512, 512)], fill=(70, 110, 70))
    d.rectangle([0, 420, 512, 512], fill=(60, 160, 80))  # 地面
    return img


def main():
    OUT.mkdir(exist_ok=True)
    # キャラA: 青髪・青目。表情と背景だけ変えた2枚
    draw_character((220, 230, 255), (60, 90, 200), (40, 60, 160), "smile").save(
        OUT / "char_a_1.png"
    )
    draw_character((255, 240, 220), (60, 90, 200), (40, 60, 160), "open").save(
        OUT / "char_a_2.png"
    )
    # キャラB: 赤髪・緑目（配色が違う別キャラ）
    draw_character((240, 255, 230), (200, 60, 60), (40, 140, 60), "smile").save(
        OUT / "char_b_1.png"
    )
    draw_landscape().save(OUT / "landscape.png")
    print(f"generated -> {OUT}")


if __name__ == "__main__":
    main()
