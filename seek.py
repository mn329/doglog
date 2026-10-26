"""
その場で見回して、人を見つけたら近寄って止まり、手の形で芸をする(go2_trick.py と同じジェスチャー)

歩き回らないので、本体の障害物回避も LiDAR も使わない。近寄るのは人に向かってだけで、
人が画面の高さの STOP_HEIGHT 以上に大きく見えたら(=近くまで来たら)止まる。

  見回す(その場で旋回) → 人を見つけた → 近寄る(人を画面の真ん中に保つ) → 着いた
  → 手を待つ(指先を下に向けたパー=お手 人差し指=おすわり グー=立つ チョキ=おまけ。3秒続けると芸)
  → WAIT_SEC 秒だれも手を見せなければ、座っていれば立って、また見回す
  近寄る途中で人を見失ったら、止まって見回すに戻る

いつでも使える合図:
  両手のパー      止まれ。歩くのも芸もすぐやめる。両手を下ろすと続きを再開する
  パーを左右に傾ける(手を待っているとき)  体を同じ側に傾ける(鏡のまね)。まっすぐに戻すと体も戻る。
                  上に向けたパーは傾けるだけで、お手にはならない(お手は指先を下に向けたパーを 3 秒)

--owner を付けると、register_face.py で登録した人(飼い主)にだけ近寄る。見回している間に顔を照合し、
ほかの人は無視する。近寄り始めたら、顔が画面から外れても同じ人を追い続ける。

使い方(go2-hand-follow フォルダで、.venv を有効にして):
  python seek.py --webcam 0     # Mac のカメラで、人の認識・手の判定・状態の切り替わりを試す(ロボットなし)
  python seek.py --owner        # 飼い主にだけ近寄る(先に python register_face.py で顔を登録する)
  python seek.py --dry-run      # Go2 の映像で試す(ロボットは動かない)
  python seek.py                # 動かす。スペースで止めて一時停止(もう一度で再開)、ESC で終了

準備(最初に一度だけ): 人物検出のモデルをダウンロードする
  curl -L -o models/efficientdet_lite0.tflite https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite

まわりを空け、人のほうへ歩いてくるので、近寄られる人はすぐ止められる位置に立つ。
"""
import argparse
import asyncio
import math
import os
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

from config import GO2_AES_KEY, find_ip
from go2_follow import _convert_and_detect
from go2_trick import (Go2Trick, pick_hand, classify, trick_from_fingers, TRICK_LABEL, GRACE_SEC,
                       ACTION_SEC)
from hand_control import adjust_exposure, draw_hands, gesture_of, finger_cos, hand_size, hand_up_cos, EXTENDED_COS
from register_face import find_faces, face_feature, similarity, load_owner, OWNER_MATCH
from unitree_webrtc_connect import SPORT_CMD

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "efficientdet_lite0.tflite")
CMD_HZ = 10             # 指令送信頻度
DETECT_WIDTH = 640      # この幅に縮小してから人を探す(処理を軽くする)
MIN_SCORE = 0.6         # 人とみなす確からしさ(0.5 ではソファーを人と見間違えた)
MIN_TALL = 1.0          # 枠の 縦 / 横 がこれ以上のものだけ人とみなす(立っている人は縦長。ソファーなど横長の物を除く)
FOUND_FRAMES = 5        # この回数見えたら「見つけた」
HOLD_SEC = 3.0          # 同じ合図をこの秒数続けたら芸をする(1 秒では見間違いで芸をしてしまった。--hold で変えられる)
FOUND_GAP = 0.5         # 見回し中、この秒数以内の見えない間は続けて見えているとみなす(顔の照合がちらついても数え直さない)
PEEK_SEC = 1.0          # 見回し中に相手が見えたら、この秒数は回るのを止めて確かめる
LOOK_YAW = 0.5          # 見回すときの旋回の速さ [rad/s]
APPROACH_SPEED = 0.2    # 近寄る速さ [m/s]
K_TURN = 1.2            # 人が画面の端にいるほど強く曲がる
MAX_TURN = 0.6          # 近寄るときの旋回の上限 [rad/s]
STOP_HEIGHT = 0.7       # 人の枠が画面の高さのこの割合以上になったら「着いた」
LOST_SEC = 1.0          # 近寄る途中でこの秒数見えなければ「見失った」
WAIT_SEC = 20.0         # 着いてから(または芸のあと)この秒数、手が見えなければ見回しに戻る
IGNORE_SEC = 3.0        # 見回し始めてこの秒数は、人がいても無視する(同じ人に続けて行かない)
VIDEO_TIMEOUT = 0.5     # 映像がこの秒数届かなければ止まる
FACE_WIDTH = 960        # --owner: この幅に縮小してから顔を探す(遠くの小さい顔も見つけられるよう大きめ)

STOP_FRAMES = 3         # 両手のパーがこの回数続けて見えたら止まれ(1 秒待たずにすぐ止める)
RESUME_SEC = 1.0        # 両手のパーがこの秒数見えなければ再開
TILT_START = math.radians(15)   # パーの傾きがこれ以上なら体を傾ける(これより小さければまっすぐ扱い)
TILT_FULL = math.radians(60)    # パーの傾きがこれで体の傾きが最大(--tilt-max)になる。その間は手の傾きに比例
TILT_MAX_DEG = 30       # 体の傾きの最大 [度](--tilt-max で変えられる。Go2 の上限は約 43°)
TILT_HOLD = 0.5         # パーが一瞬見えなくなっても、この秒数は傾きを保つ(ちらつき対策)
PAW_DOWN_COS = -0.5     # 指先が真下から 60° 以内を向いたパーをお手の合図にする(真上との cos がこれ以下)
TILT_PAW_BLOCK = 1.5    # 傾けるのをやめてからこの秒数は、お手にしない(傾けて戻す途中のまっすぐで、お手にならないように)
TILT_HZ = 5             # 体の傾きの指令を送る頻度
ROLL_SIGN = 1           # 体を傾ける向き。実機で手と逆に傾いたら -1 にする

_detector = None
_last_ts = 0


def detect_people(frame):
    """フレーム(BGR)から人を探し、(左, 上, 幅, 高さ, 確からしさ) を画面に対する割合(0〜1)で返す"""
    global _detector, _last_ts
    if _detector is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"モデルがありません: {MODEL_PATH}(このファイルの先頭の準備を参照)")
        options = vision.ObjectDetectorOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO,
            score_threshold=MIN_SCORE,
            category_allowlist=["person"],
        )
        _detector = vision.ObjectDetector.create_from_options(options)
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (DETECT_WIDTH, int(h * DETECT_WIDTH / w)))
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
    _last_ts = max(int(time.monotonic() * 1000), _last_ts + 1)
    result = _detector.detect_for_video(image, _last_ts)
    sh, sw = small.shape[:2]
    return [(d.bounding_box.origin_x / sw, d.bounding_box.origin_y / sh,
             d.bounding_box.width / sw, d.bounding_box.height / sh, d.categories[0].score)
            for d in result.detections if d.bounding_box.height >= MIN_TALL * d.bounding_box.width]


def _detect(av_frame, find_people, owner=None, use_hands=True):
    """
    手を探す(合図はいつでも使うので毎回)。find_people なら人も探す。owner(登録した顔)があれば顔も照合する

    Returns:
        frame, hands, people, faces。faces は (中心x, 中心y, 幅, 高さ, 似ている度合い) で、位置は画面に対する割合
    """
    if use_hands:
        frame, hands, _ = _convert_and_detect(av_frame)
    else:
        frame, hands = adjust_exposure(av_frame.to_ndarray(format="bgr24")), []
    if not find_people:
        return frame, hands, [], []
    people = detect_people(frame)
    faces = []
    if owner is not None and people:
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (FACE_WIDTH, int(h * FACE_WIDTH / w)))
        sh, sw = small.shape[:2]
        for f in find_faces(small):
            x, y, fw, fh = f[:4]
            faces.append(((x + fw / 2) / sw, (y + fh / 2) / sh, fw / sw, fh / sh,
                          similarity(face_feature(small, f), owner)))
    return frame, hands, people, faces


def _contains(person, face):
    x, y, w, h = person[:4]
    return x <= face[0] <= x + w and y <= face[1] <= y + h


def is_open(hand):
    """パー(指を全部開いた手)か。go2_trick.py のお手と同じ判定"""
    gesture = gesture_of(hand)
    if gesture == 'None':
        return trick_from_fingers(finger_cos(hand)) == 'paw'
    return gesture == 'Open_Palm'


def palm_angle(hand, frame_shape):
    """
    手(手首→中指付け根)の傾き [rad]。まっすぐ上で 0、画面の右へ傾くと正

    Go2 のカメラ画面の右は Go2 から見て右。人が自分の左へ手を倒すと画面では右へ傾く(鏡のまね)。
    """
    h, w = frame_shape[:2]
    a, b = hand.landmarks.landmark[0], hand.landmarks.landmark[9]
    return math.atan2((b.x - a.x) * w / h, -(b.y - a.y))


def is_tilt_palm(hand):
    """
    体を傾ける合図のパーか。指が 4 本とも伸びていればよい

    傾けた手はジェスチャー分類でパー(Open_Palm)と判定されにくいので、指の曲がり具合だけで見る
    (指の曲がり具合は手の向きに関係なく測れる)。
    """
    return gesture_of(hand) == 'Open_Palm' or all(c > EXTENDED_COS for c in finger_cos(hand))


def pick_down_palm(hands, frame_shape, min_size):
    """
    お手の合図: 指先を下に向けたパー。min_size 以上に大きく映った手のうち、いちばん大きいもの。無ければ None

    下を向いた手は左右の判定があてにならないので、右手に限らない。
    """
    candidates = [hand for hand in hands if hand_size(hand, frame_shape) >= min_size
                  and hand_up_cos(hand, frame_shape) <= PAW_DOWN_COS
                  and all(c > EXTENDED_COS for c in finger_cos(hand))]
    return max(candidates, key=lambda hand: hand_size(hand, frame_shape), default=None)


def roll_for(angle, max_roll):
    """手の傾きから体の傾き [rad] を決める。TILT_START 未満は 0、そこから手の傾きに比例し、TILT_FULL で max_roll"""
    over = abs(angle) - TILT_START
    if over <= 0:
        return 0.0
    return math.copysign(max_roll * min(1.0, over / (TILT_FULL - TILT_START)), angle) * ROLL_SIGN


def _ts():
    return f"[{time.strftime('%H:%M:%S')}.{int(time.time() * 1000) % 1000:03d}]"


class Seeker(Go2Trick):
    """接続と芸(お手・おすわり・立つ・おまけ)は go2_trick.py のものを使い、見回して近寄る部分を足す"""

    def __init__(self, robot_ip, aes_key=None, dry_run=False, webcam=None, owner=False, use_hands=True,
                 tilt_max=TILT_MAX_DEG, hold_sec=HOLD_SEC):
        super().__init__(robot_ip, aes_key, dry_run, webcam, hold_sec=hold_sec)
        self.max_roll = math.radians(min(tilt_max, 43))   # 体の傾きの最大 [rad]
        self.use_hands = use_hands  # False なら手を見ない(合図も芸もなし。近くまで来たら相手のほうを向き続ける)
        self.owner = load_owner() if owner else None   # 登録した飼い主の顔(None なら誰にでも近寄る)
        self.state = "look"
        self.since = time.time()
        self.before_stop = None     # 止まれの前の状態(再開するときに戻る)
        self.roll = 0.0             # 今送っている体の傾き
        self._euler_task = None

    def send_roll(self, roll):
        """体の傾きを応答を待たずに送る(前の指令の応答待ち中か、傾きがほぼ変わらないなら送らない)"""
        if self._euler_task is not None and not self._euler_task.done():
            return
        if abs(roll - self.roll) < math.radians(1) and (roll != 0.0 or not self._tilted):
            return
        self.roll = roll
        if self.dry_run:
            return

        async def send():
            if not self._tilted:
                await self._request(SPORT_CMD["BalanceStand"])     # 体の傾きは止まって立っているときに効く
            self._tilted = roll != 0.0
            await self._request(SPORT_CMD["Euler"], {"x": roll, "y": 0.0, "z": 0.0})
        self._euler_task = asyncio.create_task(send())

    def pick_target(self, people, faces, last):
        """
        近寄る相手を選ぶ。見つかっていなければ None

        last(近寄っている途中の相手)があれば、その位置にいちばん近い人を追う(途中でほかの人に乗り換えない)。
        飼い主モードで見回し中は、登録した顔が枠の中にある人だけを選ぶ。
        """
        if self.owner is not None:
            owners = [p for p in people if any(f[4] >= OWNER_MATCH and _contains(p, f) for f in faces)]
            if owners:
                return max(owners, key=lambda p: p[3])
            if last is None:
                return None
        if not people:
            return None
        if last is None:
            return max(people, key=lambda p: p[3])
        center = lambda p: (p[0] + p[2] / 2, p[1] + p[3] / 2)
        lx, ly = center(last)
        return min(people, key=lambda p: (center(p)[0] - lx) ** 2 + (center(p)[1] - ly) ** 2)

    async def stop_now(self):
        """止まれ: 歩くのも芸もすぐやめて、体の傾きも戻す"""
        if self._trick_task is not None and not self._trick_task.done():
            self._trick_task.cancel()
        await self._stop()
        if self.roll != 0.0 or self._tilted:
            self.roll = 0.0
            if not self.dry_run:
                await self._untilt()

    def go(self, state, message):
        print(f"{_ts()} {message}")
        self.state, self.since = state, time.time()

    async def leave(self):
        """手を待つのをやめて見回しに戻る。座っていれば先に立つ(座ったまま歩かせない)"""
        if self._trick_task is not None and not self._trick_task.done():
            await self._trick_task
        if self.sitting:
            print(f"{_ts()} 座っているので立つ")
            await self.do_trick("stand")
            await asyncio.sleep(2)
        self.go("look", "だれも手を見せない → 見回す")

    async def run(self):
        await self.connect()
        print("接続OK。スペース=止めて一時停止(もう一度で再開)  ESC=終了"
              + ("  [DRY-RUN: ロボットは動かない]" if self.dry_run else ""))
        seen, last_seen, target = 0, 0.0, None
        stop_seen, stop_last, next_tilt = 0, 0.0, 0.0
        tilt_t = 0.0                # 最後に傾けたパーが見えた時刻
        candidate, candidate_t, other_t, fired = None, 0.0, None, False
        busy_until, next_cmd = 0.0, 0.0
        last_id, frame_t = 0, time.time()

        try:
            while True:
                if self.frame_id == last_id:
                    if self.state in ("look", "approach") and time.time() - frame_t > VIDEO_TIMEOUT:
                        await self._stop()
                        self.go("look", "映像が途切れたので止まる")
                        frame_t = time.time()
                    await asyncio.sleep(0.005)
                    continue
                last_id, frame_t = self.frame_id, time.time()
                now = time.time()

                frame, hands, people, faces = await asyncio.to_thread(
                    _detect, self.latest_frame, self.state in ("look", "approach"), self.owner, self.use_hands)

                # 止まれ(両手のパー)はどの状態でも先に見る
                if sum(is_open(x) for x in hands) >= 2:
                    stop_seen, stop_last = stop_seen + 1, now
                else:
                    stop_seen = 0
                if self.state != "stopped" and stop_seen >= STOP_FRAMES and not self.paused:
                    await self.stop_now()
                    self.before_stop = self.state
                    self.go("stopped", "両手のパー → 止まれ")
                elif self.state == "stopped" and now - stop_last >= RESUME_SEC:
                    # 近寄る途中だったなら見回しからやり直す(止まっている間に人が動いたかもしれない)
                    back = "wait" if self.before_stop == "wait" else "look"
                    self.go(back, "両手を下ろした → 再開")
                    if back == "look":
                        self.since -= IGNORE_SEC
                    candidate = None

                if self.state == "stopped":
                    draw_hands(frame, hands)
                elif self.state == "wait":
                    # 着いたあとは手を見る(おすわり・立つ・チョキは go2_trick.py と同じ)
                    # seek では、上に向けたパーは体を傾けるだけ。お手は指先を下に向けたパー
                    hand = pick_hand(hands, frame.shape, self.min_size)
                    trick, source = classify(hand)
                    roll = 0.0
                    if hand is not None and trick in ('paw', None) and is_tilt_palm(hand):
                        angle = palm_angle(hand, frame.shape)
                        roll = roll_for(angle, self.max_roll)
                        trick, source = None, f"palm {math.degrees(angle):+.0f}deg"
                        if roll != 0.0:
                            tilt_t = now
                    down = pick_down_palm(hands, frame.shape, self.min_size)
                    if down is not None and trick is None and roll == 0.0:
                        hand, trick, source = down, 'paw', "palm down"
                    if roll == 0.0 and self.roll != 0.0 and now - tilt_t < TILT_HOLD and trick is None:
                        roll = self.roll    # 一瞬見えなくなっただけなら傾きを保つ
                    if now - tilt_t < TILT_PAW_BLOCK:
                        if trick == 'paw':
                            trick, source = None, "tilt"
                        if candidate == 'paw':
                            candidate = None    # 傾ける前にたまっていたお手の待ち時間も消す
                    if self.paused:
                        trick, roll = None, 0.0
                    trick_running = self._trick_task is not None and not self._trick_task.done()
                    # 芸の動きの途中(芸を始めてから ACTION_SEC)は傾けない。お手などを上書きして止めてしまうため
                    if now < busy_until:
                        roll = 0.0
                    if (now >= next_tilt and not self.sitting and not trick_running and now >= busy_until
                            and (roll != 0.0 or self.roll != 0.0)):
                        next_tilt = now + 1.0 / TILT_HZ
                        self.send_roll(roll)
                    if hand is not None:
                        self.since = max(self.since, now - WAIT_SEC / 2)    # 手が見えている間は待ち続ける
                    if trick == candidate:
                        other_t = None
                    elif candidate is None or (other_t is not None and now - other_t >= GRACE_SEC):
                        candidate, candidate_t, other_t, fired = trick, now, None, False
                    elif other_t is None:
                        other_t = now
                    hold = min((now - candidate_t) / self.hold_sec, 1.0) if candidate else 0.0
                    if candidate and hold >= 1.0 and now >= busy_until and not fired:
                        fired = True
                        if self.start_trick(candidate):
                            busy_until = now + ACTION_SEC
                            self.since = now
                    draw_hands(frame, hands, f"{trick.upper() if trick else '-'} ({source})" if source else None, hand)
                    if candidate and not fired:
                        cv2.putText(frame, f"{TRICK_LABEL[candidate]} {hold * 100:.0f}%", (10, 90),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
                    if now - self.since >= WAIT_SEC and now >= busy_until and not self.paused:
                        if self.roll != 0.0:
                            await self.stop_now()
                        await self.leave()
                        candidate = None
                else:
                    draw_hands(frame, hands)
                    picked = self.pick_target(people, faces, target if self.state == "approach" else None)
                    if picked is not None:
                        seen, last_seen, target = seen + 1, now, picked
                    elif self.state != "look" or now - last_seen > FOUND_GAP:
                        seen = 0
                    for x, y, bw, bh, score in people:
                        h, w = frame.shape[:2]
                        cv2.rectangle(frame, (int(x * w), int(y * h)), (int((x + bw) * w), int((y + bh) * h)),
                                      (0, 200, 255), 3)
                        cv2.putText(frame, f"person {score:.2f}  h {bh * 100:.0f}%", (int(x * w), max(30, int(y * h) - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
                    for cx, cy, fw, fh, sim in faces:
                        h, w = frame.shape[:2]
                        ok = sim >= OWNER_MATCH
                        x0, y0 = int((cx - fw / 2) * w), int((cy - fh / 2) * h)
                        cv2.rectangle(frame, (x0, y0), (int((cx + fw / 2) * w), int((cy + fh / 2) * h)),
                                      (0, 255, 0) if ok else (128, 128, 128), 2)
                        cv2.putText(frame, f"{'OWNER' if ok else 'face'} {sim:.2f}", (x0, max(30, y0 - 8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if ok else (128, 128, 128), 2)

                    if self.paused:
                        pass
                    elif self.state == "look" and seen >= FOUND_FRAMES and now - self.since >= IGNORE_SEC:
                        self.go("approach", ("飼い主" if self.owner is not None else "人") + "を見つけた → 近寄る")
                    elif self.state == "approach" and now - last_seen > LOST_SEC:
                        await self._stop()
                        self.go("look", "見失った → 止まって見回す")
                        self.since -= IGNORE_SEC        # 見失ったときはすぐ探し直してよい
                    elif self.state == "approach" and target[3] >= STOP_HEIGHT and self.use_hands:
                        await self._stop()
                        self.go("wait", f"着いた(人の高さ {target[3] * 100:.0f}%)→ 手を待つ")
                        candidate = None

                    if now >= next_cmd and not self.paused:
                        next_cmd = now + 1.0 / CMD_HZ
                        if self.state == "look":
                            # 相手がちらっと見えたら、回るのを止めて確かめる(回りながらだと通り過ぎてしまう)
                            peek = seen > 0 and now - last_seen < PEEK_SEC and now - self.since >= IGNORE_SEC
                            self._move(0.0, 0.0 if peek else LOOK_YAW)
                        elif self.state == "approach":
                            # 手を見ないときは、近くまで来たら前には出ず、相手のほうを向き続ける(離れたらまた近寄る)
                            cx = target[0] + target[2] / 2
                            speed = 0.0 if target[3] >= STOP_HEIGHT else APPROACH_SPEED
                            self._move(speed, max(-MAX_TURN, min(MAX_TURN, -K_TURN * (cx - 0.5))))

                close = self.state == "approach" and target is not None and target[3] >= STOP_HEIGHT
                label = {"look": "LOOK", "approach": "NEAR - WATCHING" if close else "APPROACH",
                         "wait": "SHOW ME YOUR HAND",
                         "stopped": "STOP (both palms)"}[self.state]
                if self.state == "wait":
                    label += f"  {max(0.0, WAIT_SEC - (now - self.since)):.0f}s"
                    label += f"  tilt {math.degrees(self.roll):+.0f}deg" if self.roll else ""
                    label += "  SITTING" if self.sitting else ""
                if self.paused:
                    label = "PAUSED - SPACE to resume"
                cv2.putText(frame, label + ("  [DRY-RUN]" if self.dry_run else ""), (10, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255) if self.paused else (0, 255, 0), 3)
                cv2.imshow("Go2 Seek", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break
                if key == ord(' '):
                    if self.paused:
                        self.paused = False
                        print("再開")
                    else:
                        await self.emergency_stop()
                        busy_until = 0.0
                await asyncio.sleep(0)
        finally:
            await self.disconnect()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", help="Go2のIPアドレス (STAモード)。省略時は .env の GO2_SN で探し、だめなら GO2_IP")
    ap.add_argument("--aes-key", default=GO2_AES_KEY,
                    help="新ファームウェア用 AES-128 キー(32桁hex)。省略時は .env の GO2_AES_KEY")
    ap.add_argument("--dry-run", action="store_true", help="Go2 の映像で試すが、ロボットは動かさない")
    ap.add_argument("--webcam", type=int, help="Go2 の代わりに Mac のカメラ(番号)を使う。ロボットは動かない")
    ap.add_argument("--owner", action="store_true", help="register_face.py で登録した人にだけ近寄る")
    ap.add_argument("--hold", type=float, default=HOLD_SEC,
                    help=f"同じ合図をこの秒数続けたら芸をする(既定 {HOLD_SEC:g})")
    ap.add_argument("--tilt-max", type=float, default=TILT_MAX_DEG,
                    help=f"パーを傾けたときの体の傾きの最大 [度](既定 {TILT_MAX_DEG}、上限 43)")
    ap.add_argument("--no-hands", action="store_true",
                    help="手を見ない(合図も芸もなし)。近くまで来たら止まって相手のほうを向き続け、離れたらまた近寄る。"
                         "止めるのはスペースか ESC")
    args = ap.parse_args()
    if args.webcam is None and args.ip is None:
        args.ip = find_ip()
        if args.ip is None:
            ap.error("Go2 の IP がありません。.env に GO2_IP を書くか --ip で指定してください")

    try:
        asyncio.run(Seeker(args.ip, args.aes_key, args.dry_run, args.webcam, args.owner, not args.no_hands,
                                    args.tilt_max, args.hold).run())
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
