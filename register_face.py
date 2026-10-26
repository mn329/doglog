"""
飼い主の顔を登録する(seek.py --owner で、登録した人にだけ近寄るようにするため)

カメラに顔を映して SPACE で 1 枚ずつ撮る。正面・少し左右・少し上下など、向きを変えて 5 枚ほど撮るとよい。
ESC で保存して終了。顔の特徴(128 個の数)を models/owner_face.npy に保存する。写真そのものは保存しない。
models/ は .gitignore に入っているので、リポジトリには入らない。

使い方(go2-hand-follow フォルダで、.venv を有効にして):
  python register_face.py                 # Mac のカメラで撮る
  python register_face.py --go2 --add     # Go2 のカメラで撮って、今の登録に足す(いちばんおすすめ)
  python register_face.py --images a.jpg  # 写真から登録する(1 枚に顔が 1 つだけ写っているもの)

顔が小さく写ると照合しにくい(証明写真だけだと、1m 以上離れると別の人と判定されやすい)。
Go2 のカメラで、1m・2m・3m と距離を変え、正面と少し横向きを撮っておくと見つけやすくなる。
--add を付けないと、今の登録は消えて新しく撮った分だけになる。

準備(最初に一度だけ): 顔の検出と照合のモデルをダウンロードする
  curl -L -o models/face_detection_yunet_2023mar.onnx https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
  curl -L -o models/face_recognition_sface_2021dec.onnx https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx
"""
import argparse
import asyncio
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FACE_DET_PATH = os.path.join(HERE, "models", "face_detection_yunet_2023mar.onnx")
FACE_REC_PATH = os.path.join(HERE, "models", "face_recognition_sface_2021dec.onnx")
OWNER_PATH = os.path.join(HERE, "models", "owner_face.npy")
FACE_SCORE = 0.8        # 顔とみなす確からしさ
OWNER_MATCH = 0.363     # 登録した顔との似ている度合い(cos)がこれ以上なら同じ人(SFace の推奨値)

_det = None
_rec = None


def _models():
    global _det, _rec
    if _det is None:
        for path in (FACE_DET_PATH, FACE_REC_PATH):
            if not os.path.exists(path):
                raise FileNotFoundError(f"モデルがありません: {path}(register_face.py の先頭の準備を参照)")
        _det = cv2.FaceDetectorYN.create(FACE_DET_PATH, "", (320, 320), FACE_SCORE)
        _rec = cv2.FaceRecognizerSF.create(FACE_REC_PATH, "")
    return _det, _rec


def find_faces(frame):
    """フレーム(BGR)の顔を探す。YuNet の結果(1 行 = 1 つの顔。先頭 4 つが 左, 上, 幅, 高さ [px])"""
    det, _ = _models()
    h, w = frame.shape[:2]
    det.setInputSize((w, h))
    _, faces = det.detect(frame)
    return [] if faces is None else list(faces)


def face_feature(frame, face):
    """顔の特徴(128 個の数)"""
    _, rec = _models()
    return rec.feature(rec.alignCrop(frame, face))


def similarity(feature, owner):
    """登録した顔(複数枚)のうち、いちばん似ているものとの cos"""
    _, rec = _models()
    return max(rec.match(feature, o.reshape(1, -1), cv2.FaceRecognizerSF_FR_COSINE) for o in owner)


def load_owner():
    if not os.path.exists(OWNER_PATH):
        raise FileNotFoundError(f"飼い主の顔が登録されていません: {OWNER_PATH}(python register_face.py で登録する)")
    return np.load(OWNER_PATH)


def save(features):
    np.save(OWNER_PATH, np.vstack(features))
    print(f"{len(features)} 枚分の顔の特徴を保存しました: {OWNER_PATH}")


def from_images(paths):
    features = []
    for path in paths:
        frame = cv2.imread(path)
        faces = find_faces(frame) if frame is not None else []
        if len(faces) != 1:
            print(f"{path}: 顔が {len(faces)} 個見つかったので使いません(1 個だけ写っている写真にする)")
            continue
        features.append(face_feature(frame, faces[0]))
        print(f"{path}: 登録")
    return features


def from_camera(index):
    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        raise RuntimeError(f"カメラ {index} を開けません")

    async def frames():
        while True:
            ok, frame = await asyncio.to_thread(cap.read)
            if ok:
                yield frame
    try:
        return asyncio.run(shoot(frames()))
    finally:
        cap.release()


def from_go2(ip, aes_key):
    """Go2 のカメラで撮る(ロボットは動かさない)"""
    from go2_follow import Go2HandFollower

    async def run():
        robot = Go2HandFollower(ip, aes_key, dry_run=True)
        await robot.connect()

        async def frames():
            last_id = 0
            while True:
                if robot.frame_id == last_id:
                    await asyncio.sleep(0.005)
                    continue
                last_id = robot.frame_id
                yield await asyncio.to_thread(robot.latest_frame.to_ndarray, format="bgr24")
        try:
            return await shoot(frames())
        finally:
            await robot.conn.disconnect()
    return asyncio.run(run())


async def shoot(frames):
    """映像を表示し、SPACE で顔を 1 枚ずつ撮る。ESC で終わって、撮った顔の特徴を返す"""
    owner = np.load(OWNER_PATH) if os.path.exists(OWNER_PATH) else None
    features = []
    print("顔を映して SPACE で撮る(距離と向きを変えて 5 枚ほど)。ESC で保存して終了")
    try:
        async for frame in frames:
            faces = find_faces(frame)
            view = frame.copy()
            for f in faces:
                x, y, w, h = (int(v) for v in f[:4])
                cv2.rectangle(view, (x, y), (x + w, y + h), (0, 255, 0) if len(faces) == 1 else (0, 0, 255), 3)
                # 今の登録との似ている度合い(登録済みのとき)。低い距離・向きを撮り足すとよい
                label = f"{int(w)}px"
                if owner is not None:
                    sim = similarity(face_feature(frame, f), owner)
                    label += f"  now {sim:.2f} {'OWNER' if sim >= OWNER_MATCH else '-'}"
                cv2.putText(view, label, (x, max(30, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            msg = f"{len(features)} shots  SPACE=shoot  ESC=save"
            if len(faces) != 1:
                msg += f"  ({len(faces)} faces - need exactly 1)"
            cv2.putText(view, msg, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.imshow("Register Face", view)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key == ord(' ') and len(faces) == 1:
                features.append(face_feature(frame, faces[0]))
                print(f"{len(features)} 枚目を撮りました(顔の幅 {int(faces[0][2])}px)")
            await asyncio.sleep(0)
    finally:
        cv2.destroyAllWindows()
    return features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--webcam", type=int, default=0, help="使うカメラの番号")
    ap.add_argument("--images", nargs="+", help="カメラの代わりに写真から登録する")
    ap.add_argument("--go2", action="store_true", help="Go2 のカメラで撮る(ロボットは動かさない)")
    ap.add_argument("--ip", help="--go2 のときの Go2 の IP。省略時は .env の GO2_SN で探し、だめなら GO2_IP")
    ap.add_argument("--add", action="store_true", help="今の登録を消さずに、撮った分を足す")
    args = ap.parse_args()
    if args.images:
        features = from_images(args.images)
    elif args.go2:
        from config import GO2_AES_KEY, find_ip
        features = from_go2(args.ip or find_ip(), GO2_AES_KEY)
    else:
        features = from_camera(args.webcam)
    if not features:
        print("1 枚も撮れなかったので保存しません")
        return
    if args.add and os.path.exists(OWNER_PATH):
        features = list(np.load(OWNER_PATH)) + [f.reshape(-1) for f in features]
    save([np.asarray(f).reshape(1, -1) for f in features])


if __name__ == '__main__':
    main()
