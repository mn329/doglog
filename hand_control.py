import os
import time
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

# パラメータ
HAND_DETECTION_CONFIDENCE = 0.5   # 手を見失った後の再検出のしきい値。高いと速い動き(ブレ)で再検出できない
HAND_TRACKING_CONFIDENCE = 0.5
MAX_NUM_HANDS = 2       # 両手を検出し、操作には右手だけを使う(左手を先に見つけても右手を選べるように)

# ジェスチャー認識モデル(MediaPipe Gesture Recognizer)。手の検出・21点・左右判定・ジェスチャー分類をまとめて行う
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "gesture_recognizer.task")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
             "gesture_recognizer/float16/1/gesture_recognizer.task")
GESTURE_MIN_SCORE = 0.5     # ジェスチャー分類の確からしさがこれより低いときは使わない(指の角度のルールで判定する)

# 右手の判定
# Gesture Recognizer は、反転していない映像(Go2 も Mac のカメラ(OpenCV)もそう)で実際の左右をそのまま返す。
# 旧 API(mp.solutions.hands)は自撮りの反転画像が前提で左右が逆('Left' が実際の右手)なので注意
MP_LABEL_RIGHT = 'Right'
HANDEDNESS_MIN_SCORE = 0.8  # 左右判定の確からしさがこれより低い手は右手とみなさない(手の甲を向けた時など)

# 制御パラメータ
TARGET_SIZE = 0.22      # 手(手首〜中指付け根)の画面高さに対する目標の大きさ。大きいほど近い
SIZE_DEADZONE = 0.03    # この範囲内なら前後に動かない
X_DEADZONE = 0.08       # 画面中心からこの範囲内なら旋回しない
K_YAW = 1.6             # 旋回ゲイン
K_FWD = 2.5             # 前後ゲイン
MAX_VX = 0.35           # 最大前後速度 [m/s]
MAX_VYAW = 0.8          # 最大旋回速度 [rad/s]

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# 各指の (先端, 第二関節) のランドマーク番号（親指以外）
FINGER_TIP_PIP = [(8, 6), (12, 10), (16, 14), (20, 18)]


@dataclass
class Hand:
    """検出した手 1 つ分"""
    landmarks: landmark_pb2.NormalizedLandmarkList  # 画面上の 21 点(x, y は 0〜1)。描画や位置に使う
    world: np.ndarray       # 21 点の 3D 座標 [m](手の中心が原点)。指の曲がり具合に使う
    is_right: bool          # 操作に使う右手か(左右判定の確からしさが低いときは False)
    handedness_score: float
    gesture: str            # 'Open_Palm', 'Closed_Fist', 'Victory', 'Pointing_Up', 'None' など
    gesture_score: float


_recognizer = None
_last_ts = 0


def _get_recognizer():
    """Gesture Recognizer を初回だけ生成して使い回す"""
    global _recognizer
    if _recognizer is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"モデルがありません: {MODEL_PATH}\n"
                f"次のコマンドでダウンロードしてください:\n"
                f"  mkdir -p models && curl -L -o models/gesture_recognizer.task {MODEL_URL}")
        options = vision.GestureRecognizerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=MAX_NUM_HANDS,
            min_hand_detection_confidence=HAND_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=HAND_DETECTION_CONFIDENCE,
            min_tracking_confidence=HAND_TRACKING_CONFIDENCE,
        )
        _recognizer = vision.GestureRecognizer.create_from_options(options)
    return _recognizer


def close():
    """Gesture Recognizer を解放"""
    global _recognizer
    if _recognizer is not None:
        _recognizer.close()
        _recognizer = None


_gamma_lut = None       # 白飛び対策の明るさ補正表(None なら補正しない)


def set_gamma(gamma):
    """
    白飛び対策の明るさ補正(ガンマ補正)を設定する

    gamma > 1 で全体を暗くし、明るい部分の明暗差を広げる(白っぽい手の輪郭が出やすくなる)。
    完全に真っ白(255)になった部分は情報が残っていないので戻せない。1.0 で補正なし。
    """
    global _gamma_lut
    if gamma == 1.0:
        _gamma_lut = None
    else:
        _gamma_lut = np.array([(i / 255.0) ** gamma * 255.0 for i in range(256)]).round().astype(np.uint8)


def adjust_exposure(frame):
    """set_gamma で設定した明るさ補正をかける"""
    return frame if _gamma_lut is None else cv2.LUT(frame, _gamma_lut)


def overexposed_ratio(frame):
    """白飛び(ほぼ真っ白)している画素の割合(0〜1)。縮小して計算するので軽い"""
    small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
    return float(np.mean(small.min(axis=2) >= 250))


def detect(frame):
    """
    フレームから手とジェスチャーを検出

    Args:
        frame: OpenCVフレーム (BGR)

    Returns:
        hands: 検出した手(Hand)のリスト。見つからなければ空
    """
    global _last_ts
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    # 動画モードは時刻(ミリ秒)が毎回増えている必要がある
    _last_ts = max(int(time.monotonic() * 1000), _last_ts + 1)
    result = _get_recognizer().recognize_for_video(image, _last_ts)

    hands = []
    for i, points in enumerate(result.hand_landmarks):
        landmarks = landmark_pb2.NormalizedLandmarkList()
        for q in points:
            landmarks.landmark.add(x=q.x, y=q.y, z=q.z)
        handedness = result.handedness[i][0]
        gesture = result.gestures[i][0] if result.gestures[i] else None
        hands.append(Hand(
            landmarks=landmarks,
            world=np.array([[q.x, q.y, q.z] for q in result.hand_world_landmarks[i]]),
            is_right=(handedness.category_name == MP_LABEL_RIGHT
                      and handedness.score >= HANDEDNESS_MIN_SCORE),
            handedness_score=handedness.score,
            gesture=gesture.category_name if gesture else 'None',
            gesture_score=gesture.score if gesture else 0.0,
        ))
    return hands


def right_hand(hands):
    """
    操作に使う右手(Hand)を返す。右手が見つからなければ None

    右手らしい手が2つあるとき(左右の判定違い)は、より確からしい方を選ぶ。
    """
    candidates = [hand for hand in hands if hand.is_right]
    return max(candidates, key=lambda hand: hand.handedness_score, default=None)


def gesture_of(hand):
    """ジェスチャー分類の結果。確からしさが低いときは 'None'"""
    return hand.gesture if hand.gesture_score >= GESTURE_MIN_SCORE else 'None'


_OTHER_HAND_STYLE = mp_drawing.DrawingSpec(color=(128, 128, 128), thickness=2, circle_radius=2)


def draw_hands(frame, hands, label=None):
    """
    検出した手を描く。操作に使う右手は通常の色で R の印を付け、それ以外の手は灰色で描く

    Args:
        label: 右手の印の横に出す最終的な判定(例: 'PAW (fingers)')。None なら印だけ
    """
    selected = right_hand(hands)
    h, w = frame.shape[:2]
    for hand in hands:
        if hand is selected:
            mp_drawing.draw_landmarks(frame, hand.landmarks, mp_hands.HAND_CONNECTIONS)
            wrist = hand.landmarks.landmark[0]
            text = "R" if label is None else f"R -> {label}"
            (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            x = int(clamp(wrist.x * w, 0, w - tw))     # 画面の右端からはみ出さないように
            y = int(clamp(wrist.y * h + 30, 20, h - 10))
            cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            mp_drawing.draw_landmarks(frame, hand.landmarks, mp_hands.HAND_CONNECTIONS,
                                      _OTHER_HAND_STYLE, _OTHER_HAND_STYLE)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def deadzone(v, dz):
    if abs(v) < dz:
        return 0.0
    return v - dz if v > 0 else v + dz


def count_fingers(pts):
    """伸びている指の本数を数える（親指以外の4本）"""
    wrist = pts[0]
    count = 0
    for tip, pip in FINGER_TIP_PIP:
        # 先端が第二関節より手首から遠ければ伸びているとみなす
        if np.linalg.norm(pts[tip] - wrist) > np.linalg.norm(pts[pip] - wrist):
            count += 1
    return count


# 各指の (付け根, 第二関節, 先端) のランドマーク番号（親指以外）
FINGER_MCP_PIP_TIP = [(5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)]
EXTENDED_COS = 0.6      # 第二関節の曲がりがこれより小さい(cosが大きい)と伸びている(約53°以内)
FOLDED_COS = 0.2        # これより曲がっている(cosが小さい)と曲がっている(約78°以上)


def finger_cos(hand):
    """
    人差し指・中指・薬指・小指の第二関節の曲がり具合を返す

    MediaPipe が推定した手の 3D 座標(メートル単位)で第二関節の曲がり角を見るので、
    指をカメラに向けても(画面上で短く見えても)伸びていると判定できる。
    画面座標の z(奥行き)よりぶれが小さい。

    Returns:
        [人差し指, 中指, 薬指, 小指] の cos。1 に近いほどまっすぐ、0 以下は大きく曲がっている
    """
    p = hand.world
    result = []
    for mcp, pip, tip in FINGER_MCP_PIP_TIP:
        v1 = p[pip] - p[mcp]
        v2 = p[tip] - p[pip]
        result.append(float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)))
    return result


def finger_states(hand):
    """
    人差し指・中指・薬指・小指がそれぞれ伸びているかを判定する

    Returns:
        [人差し指, 中指, 薬指, 小指] の各要素が True(伸び) / False(曲げ) / None(どちらとも言えない)
    """
    states = []
    for cos in finger_cos(hand):
        if cos > EXTENDED_COS:
            states.append(True)
        elif cos < FOLDED_COS:
            states.append(False)
        else:
            states.append(None)
    return states


def compute_cmd(hands, frame_shape):
    """
    検出結果からロボット制御コマンドを計算

    右手が画面の右にあれば右旋回、手が小さい(遠い)なら前進、大きい(近い)なら後退。
    グーの場合は停止する。止める側は安全のため左右どちらの手のグーでもよく、
    ジェスチャー分類(Closed_Fist)と指の判定のどちらか一方でもグーなら止める。

    Args:
        hands: detect() の結果
        frame_shape: フレームのサイズ (height, width, channels)

    Returns:
        cmd: ロボット制御コマンド
            forward: 前後速度 [m/s]（前進が正）
            turn: 旋回速度 [rad/s]（左回りが正）
            gesture: 'open' / 'fist' / None(右手が見つからない)
    """
    cmd = {
        'forward': 0.0,
        'turn': 0.0,
        'gesture': None
    }

    h, w = frame_shape[:2]

    def to_pts(hand):
        # x, y は 0〜1 の正規化座標。縦横比を補正して大きさを画面高さ基準にする
        return np.array([[p.x * w / h, p.y] for p in hand.landmarks.landmark])

    if any(gesture_of(hand) == 'Closed_Fist' or count_fingers(to_pts(hand)) == 0 for hand in hands):
        cmd['gesture'] = 'fist'
        return cmd

    hand = right_hand(hands)
    if hand is None:
        return cmd
    pts = to_pts(hand)
    cmd['gesture'] = 'open'

    size = float(np.linalg.norm(pts[0] - pts[9]))   # 手首〜中指付け根の長さ
    cx = float(np.mean([hand.landmarks.landmark[i].x for i in (0, 5, 9, 13, 17)]))  # 手のひら中心

    # Go2のyawは左回りが正なので、手が右(cx>0.5)ならマイナス
    turn = -K_YAW * deadzone(cx - 0.5, X_DEADZONE)
    forward = K_FWD * deadzone(TARGET_SIZE - size, SIZE_DEADZONE)

    # + 0.0 で -0.0 を 0.0 にそろえる
    cmd['forward'] = clamp(forward, -MAX_VX, MAX_VX) + 0.0
    cmd['turn'] = clamp(turn, -MAX_VYAW, MAX_VYAW) + 0.0
    return cmd
