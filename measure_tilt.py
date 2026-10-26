"""
手を前に倒したときの認識を計測する(一時的なツール。計測が終わったら消してよい)

使い方: python measure_tilt.py
画面の指示どおりに右手で形を作り、SPACE を押すと 2 秒記録する。結果は measure_out/ に保存される。
"""
import json
import math
import os
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import hand_control as hc

FONT_PATH = "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "measure_out")
REC_SEC = 2.0

# (形の名前, 形の説明, 倒す角度)
STEPS = [
    ("パー", "指を5本ぜんぶ開く", 0),
    ("パー", "指を5本ぜんぶ開く", 30),
    ("パー", "指を5本ぜんぶ開く", 45),
    ("パー", "指を5本ぜんぶ開く", 60),
    ("人差し指だけ", "人差し指だけ立てて、ほかは握る", 0),
    ("人差し指だけ", "人差し指だけ立てて、ほかは握る", 45),
    ("チョキ", "人差し指と中指を立てる", 45),
    ("グー", "ぜんぶ握る", 45),
]

_fonts = {}


def font(size):
    if size not in _fonts:
        _fonts[size] = ImageFont.truetype(FONT_PATH, size)
    return _fonts[size]


def draw_tilt_icon(draw, x, y, angle):
    """横から見た手の角度の図。左がカメラ、手は下(手首)から上(指先)へ伸びる"""
    size = 150
    draw.rectangle([x, y, x + size, y + size], fill=(40, 40, 40))
    draw.text((x + 8, y + 6), "横から見た図", font=font(16), fill=(200, 200, 200))
    draw.text((x + 8, y + size - 50), "カメラ", font=font(14), fill=(200, 200, 200))
    bx, by = x + 100, y + size - 20     # 手首
    length = 95
    # まっすぐ(点線の代わりに灰色)
    draw.line([bx, by, bx, by - length], fill=(110, 110, 110), width=3)
    # 倒した手(カメラ側=左へ倒す)
    rad = math.radians(angle)
    tx, ty = bx - length * math.sin(rad), by - length * math.cos(rad)
    draw.line([bx, by, tx, ty], fill=(0, 220, 255), width=8)
    draw.ellipse([tx - 6, ty - 6, tx + 6, ty + 6], fill=(0, 220, 255))
    draw.text((bx - 20, y + 26), f"{angle}°", font=font(22), fill=(0, 220, 255))


def overlay(frame, step_no, step, status, status_color, hand):
    """指示パネルを画面上部に描く"""
    name, desc, angle = step
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img, "RGBA")
    w = img.width
    draw.rectangle([0, 0, w, 200], fill=(0, 0, 0, 200))
    draw.text((20, 12), f"{step_no} / {len(STEPS)}   右手で", font=font(24), fill=(200, 200, 200))
    draw.text((20, 46), f"{name}", font=font(56), fill=(255, 255, 255))
    draw.text((20 + 60 * len(name) + 20, 66), f"({desc})", font=font(28), fill=(255, 255, 255))
    if angle == 0:
        tilt = "手のひらをカメラに向けて、指先はまっすぐ上"
    else:
        tilt = f"手のひらをカメラに向けたまま、指先をカメラの方へ約{angle}°倒す"
    draw.text((20, 120), tilt, font=font(28), fill=(0, 220, 255))
    draw.text((20, 160), status, font=font(28), fill=status_color)
    draw_tilt_icon(draw, w - 170, 20, angle)
    # いまの認識結果
    if hand is None:
        info = "右手: 見えていない"
    else:
        cos = " ".join(f"{c:+.1f}" for c in hc.finger_cos(hand))
        info = f"右手: {hand.gesture} {hand.gesture_score:.2f}   指 I M R P: {cos}"
    draw.rectangle([0, img.height - 50, w, img.height], fill=(0, 0, 0, 200))
    draw.text((20, img.height - 42), info, font=font(26), fill=(255, 255, 0))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        print("カメラを開けません")
        sys.exit(1)

    log = []
    for i, step in enumerate(STEPS, 1):
        recording_since = None
        saved = False
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            hands = hc.detect(frame)
            hand = hc.right_hand(hands)
            hc.draw_hands(frame, hands)

            if recording_since is None:
                status, color = "形ができたら SPACE で記録(2秒)   ESC で終了", (120, 255, 120)
            else:
                left = REC_SEC - (time.time() - recording_since)
                if left <= 0:
                    break
                status, color = f"記録中… そのまま {left:.1f} 秒", (255, 80, 80)
                log.append({
                    "step": i, "shape": step[0], "angle": step[2],
                    "right": hand is not None,
                    "gesture": hand.gesture if hand else None,
                    "gscore": round(hand.gesture_score, 2) if hand else None,
                    "cos": [round(c, 2) for c in hc.finger_cos(hand)] if hand else None,
                    "hands": [(h.gesture, round(h.gesture_score, 2), h.is_right, round(h.handedness_score, 2))
                              for h in hands],
                })
                if not saved and time.time() - recording_since > REC_SEC / 2:
                    cv2.imwrite(os.path.join(OUT_DIR, f"{i}_{step[0]}_{step[2]}.jpg"), frame)
                    saved = True

            cv2.imshow("measure", overlay(frame, i, step, status, color, hand))
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key == 32 and recording_since is None:
                recording_since = time.time()
        if key == 27:
            break

    with open(os.path.join(OUT_DIR, "log.json"), "w") as f:
        json.dump(log, f, ensure_ascii=False)
    cap.release()
    cv2.destroyAllWindows()
    hc.close()
    print(f"完了: {len(log)} フレームを {OUT_DIR} に保存しました")


if __name__ == "__main__":
    main()
