"""
録画した映像で、手の検出方法を比べる(試験用。本体のプログラムは変えない)

比べる方法:
  MediaPipe        今の本体と同じ Gesture Recognizer(動画モード、最大2手)
  MediaPipe+CLAHE  明るさの局所補正(CLAHE)をかけてから MediaPipe。逆光で暗い手の対策候補
  YOLO             手検出用の学習済み YOLOv8n(枠だけ。指の形はわからない)
  YOLO+切り出し     MediaPipe が見失ったフレームで、YOLO の枠を切り出して拡大し MediaPipe にかけ直す

録画中は手をずっと映しておく前提で、「手を見つけたフレームの割合」を比べる。

準備(最初に一度だけ。PyTorch が入るので約1.5GB。本体の .venv とは別にする):
  uv venv experiments/.venv --python 3.11
  uv pip install -p experiments/.venv/bin/python -r experiments/requirements.txt
  mkdir -p experiments/models
  curl -L -o experiments/models/hand_yolov8n.pt https://huggingface.co/Bingsu/adetailer/resolve/main/hand_yolov8n.pt

使い方(go2-hand-follow フォルダで):
  experiments/.venv/bin/python experiments/compare_detectors.py recordings/*.mp4
"""
import argparse
import os
import sys
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import hand_control as hc  # noqa: E402  本体と同じモデル・しきい値を使う

YOLO_PATH = os.path.join(HERE, "models", "hand_yolov8n.pt")
YOLO_CONF = 0.3
CROP_SCALE = 1.6        # YOLO の枠をこの倍率で広げて切り出す(指先や手首が切れないように)
CROP_SIZE = 384         # 切り出した画像をこの大きさに拡大してから MediaPipe にかける
SAMPLES = 6             # 「MediaPipe は見失い、YOLO は見つけた」フレームを保存する枚数


def make_recognizer(video_mode):
    options = vision.GestureRecognizerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=hc.MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO if video_mode else vision.RunningMode.IMAGE,
        num_hands=hc.MAX_NUM_HANDS,
        min_hand_detection_confidence=hc.HAND_DETECTION_CONFIDENCE,
        min_hand_presence_confidence=hc.HAND_DETECTION_CONFIDENCE,
        min_tracking_confidence=hc.HAND_TRACKING_CONFIDENCE,
    )
    return vision.GestureRecognizer.create_from_options(options)


def to_mp_image(bgr):
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def apply_clahe(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = _clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def crop_box(frame, box):
    """YOLO の枠を正方形に広げて切り出し、CROP_SIZE に拡大する"""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = max(x2 - x1, y2 - y1) * CROP_SCALE / 2
    a, b = int(max(cx - half, 0)), int(max(cy - half, 0))
    c, d = int(min(cx + half, w)), int(min(cy + half, h))
    if c - a < 8 or d - b < 8:
        return None
    return cv2.resize(frame[b:d, a:c], (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_LINEAR)


def timed(fn, times, key):
    t = time.perf_counter()
    result = fn()
    times[key] += time.perf_counter() - t
    return result


def compare(path, yolo, out_dir, max_frames):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"開けません: {path}")
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 14.0
    mp_plain, mp_clahe, mp_crop = make_recognizer(True), make_recognizer(True), make_recognizer(False)
    found = {"mp": 0, "clahe": 0, "yolo": 0, "miss_yolo": 0, "crop": 0}
    times = {"mp": 0.0, "clahe": 0.0, "yolo": 0.0, "crop": 0.0}
    crop_runs = 0
    n = 0
    saved = 0
    name = os.path.splitext(os.path.basename(path))[0]
    while max_frames is None or n < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        ts = int(n * 1000 / fps) + 1      # 動画モードは時刻が増えている必要がある
        n += 1

        r = timed(lambda: mp_plain.recognize_for_video(to_mp_image(frame), ts), times, "mp")
        mp_ok = len(r.hand_landmarks) > 0
        r = timed(lambda: mp_clahe.recognize_for_video(to_mp_image(apply_clahe(frame)), ts), times, "clahe")
        clahe_ok = len(r.hand_landmarks) > 0
        boxes = timed(lambda: yolo.predict(frame, conf=YOLO_CONF, verbose=False)[0].boxes, times, "yolo")
        yolo_ok = len(boxes) > 0

        found["mp"] += mp_ok
        found["clahe"] += clahe_ok
        found["yolo"] += yolo_ok
        if yolo_ok and not mp_ok:
            found["miss_yolo"] += 1
            box = boxes.xyxy[int(boxes.conf.argmax())].tolist()
            crop = crop_box(frame, box)
            recovered = False
            if crop is not None:
                crop_runs += 1
                rc = timed(lambda: mp_crop.recognize(to_mp_image(crop)), times, "crop")
                recovered = len(rc.hand_landmarks) > 0
            found["crop"] += recovered
            if saved < SAMPLES:
                img = frame.copy()
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0) if recovered else (0, 0, 255), 3)
                cv2.putText(img, "crop: OK" if recovered else "crop: NG", (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if recovered else (0, 0, 255), 2)
                cv2.imwrite(os.path.join(out_dir, f"{name}_{n:05d}.jpg"), img)
                saved += 1

    cap.release()
    for r in (mp_plain, mp_clahe, mp_crop):
        r.close()
    if n == 0:
        return None
    return {
        "name": name, "frames": n, "found": found,
        "ms": {k: times[k] * 1000 / (crop_runs if k == "crop" else n) if (crop_runs if k == "crop" else n) else 0.0
               for k in times},
    }


def pct(a, n):
    return f"{a / n * 100:5.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--out", default=os.path.join(HERE, "compare_out"),
                    help="MediaPipe が見失い YOLO が見つけたフレームの画像を保存する場所")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    yolo = YOLO(YOLO_PATH)
    yolo.predict(np.zeros((720, 1280, 3), np.uint8), verbose=False)    # 初回の準備時間を測定に入れない

    rows = [res for res in (compare(v, yolo, args.out, args.max_frames) for v in args.videos) if res]
    print()
    print("手を見つけたフレームの割合(録画中は手をずっと映している前提)")
    print(f"{'映像':<24}{'枚数':>6}  {'MediaPipe':>10}{'+CLAHE':>9}{'YOLO':>9}  "
          f"{'MP見失い→YOLO発見':>12}{'→切り出しで21点':>10}")
    for r in rows:
        f, n = r["found"], r["frames"]
        print(f"{r['name']:<24}{n:>6}  {pct(f['mp'], n):>10}{pct(f['clahe'], n):>9}{pct(f['yolo'], n):>9}  "
              f"{f['miss_yolo']:>12}{f['crop']:>14}")
    print()
    print("1フレームの処理時間 [ms](切り出しは、かけ直したフレームだけの平均)")
    for r in rows:
        m = r["ms"]
        print(f"{r['name']:<24}MediaPipe {m['mp']:5.1f}  +CLAHE {m['clahe']:5.1f}  "
              f"YOLO {m['yolo']:5.1f}  切り出し {m['crop']:5.1f}")
    print(f"\n見失ったのに YOLO が見つけたフレームの例: {args.out}")


if __name__ == "__main__":
    main()
