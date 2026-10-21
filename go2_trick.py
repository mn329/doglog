import argparse
import asyncio
import time

import cv2
import av
from hand_control import finger_cos, set_gamma, right_hand, gesture_of, draw_hands, EXTENDED_COS, FOLDED_COS
from go2_follow import Go2HandFollower, _convert_and_detect
from unitree_webrtc_connect import SPORT_CMD

HOLD_SEC = 1.0          # 同じジェスチャーをこの秒数続けたら指示として受け付ける
SIT_MIDDLE_COS = 0.45   # おすわりで中指がこれより曲がっていればよい(約63°以上)。ハート(中指 0.6 超)との間は判定しない
GRACE_SEC = 0.3         # 判定がこの秒数以内だけ揺れても、続けているとみなす
ACTION_SEC = 4.0        # 動作中はこの秒数、次の指示を受け付けない

# ジェスチャー → 指示
#   パー(4本とも伸び)              : お手
#   人差し指だけ伸び(中指は曲げ)    : おすわり
#   グー(4本とも伸びていない)      : 立つ(おすわりを解除)
#   チョキ(人差し指と中指だけ伸び)  : ハート
# ジェスチャー分類(Gesture Recognizer)の結果 → 指示。'None'(該当なし)のときだけ指の角度のルールで判定する
GESTURE_TRICK = {'Open_Palm': 'paw', 'Pointing_Up': 'sit', 'Closed_Fist': 'stand', 'Victory': 'heart'}
KEY_TRICK = {ord('h'): 'paw', ord('s'): 'sit', ord('u'): 'stand', ord('v'): 'heart'}
TRICK_LABEL = {'paw': 'PAW (Hello)', 'sit': 'SIT', 'stand': 'STAND', 'heart': 'HEART'}


def classify(hands):
    """
    右手の検出結果から指示を判定する(左手は使わない)

    ジェスチャー分類の結果を優先し、分類できなかった('None')ときだけ指の角度のルールで判定する。
    いいね(Thumb_Up)など、指示に割り当てていない形と分類されたときは指示なし。

    Returns:
        (trick, source): trick は該当なしなら None。source は画面表示用の判定の出どころ
            ('model 0.79' = ジェスチャー分類、'fingers' = 指の角度のルール)。右手が無ければ None
    """
    hand = right_hand(hands)
    if hand is None:
        return None, None
    gesture = gesture_of(hand)
    if gesture == 'None':
        return trick_from_fingers(finger_cos(hand)), 'fingers'
    return GESTURE_TRICK.get(gesture), f"model {hand.gesture_score:.2f}"


def trick_from_fingers(cos):
    """
    指の曲がり具合 [人差し指, 中指, 薬指, 小指] の cos から指示を判定。該当なしなら None

    人差し指を立てたとき中指は軽く丸まるだけのことが多いので、おすわりの中指は
    「はっきり曲がっている」までは求めない(SIT_MIDDLE_COS)。
    中指が伸び切らないチョキと間違えないよう、ハートとの間に判定しない幅を残している。
    """
    ext = [c > EXTENDED_COS for c in cos]
    fold = [c < FOLDED_COS for c in cos]
    index, middle, ring, pinky = cos
    if all(ext):
        return 'paw'
    if ext[0] and ext[1] and not ext[2] and not ext[3]:
        return 'heart'
    if ext[0] and middle < SIT_MIDDLE_COS and not ext[2] and not ext[3]:
        return 'sit'
    if not any(ext) and sum(fold) >= 3:
        return 'stand'
    return None


def fingers_label(cos):
    """画面表示用: 各指の cos(1 に近いほどまっすぐ)"""
    return " ".join(f"{c:+.1f}" for c in cos)


class Go2Trick(Go2HandFollower):
    """ジェスチャーで お手 / おすわり をさせるデモ"""

    def __init__(self, robot_ip, aes_key=None, dry_run=False, webcam=None):
        super().__init__(robot_ip, aes_key, dry_run or webcam is not None)
        self.webcam = webcam    # カメラ番号。指定時は Go2 に接続せず Mac のカメラで試す
        self.sitting = False
        self._trick_task = None     # 実行中の動作

    async def connect(self):
        if self.webcam is None:
            await super().connect()
            return
        cap = cv2.VideoCapture(self.webcam, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            raise RuntimeError(f"カメラ {self.webcam} を開けません")
        self.cap = cap
        asyncio.create_task(self._read_webcam())

    async def _read_webcam(self):
        while True:
            ret, frame = await asyncio.to_thread(self.cap.read)
            if ret:
                self.latest_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
                self.frame_id += 1

    async def disconnect(self):
        if self._trick_task is not None:
            self._trick_task.cancel()
        if self.webcam is None:
            await super().disconnect()
            return
        self.cap.release()
        cv2.destroyAllWindows()

    async def _sport(self, name):
        if self.dry_run:
            return
        await self._request(SPORT_CMD[name])

    def start_trick(self, trick):
        """
        指示に応じた動作を裏で始める(画面の処理を止めないため)

        Returns:
            動作する場合 True。すでにその姿勢で何もしない場合 False
        """
        print(f"指示: {TRICK_LABEL[trick]}")
        if (trick == 'sit' and self.sitting) or (trick == 'stand' and not self.sitting):
            return False
        self._trick_task = asyncio.create_task(self.do_trick(trick))
        return True

    async def do_trick(self, trick):
        """指示に応じた動作をさせる"""
        if trick == 'sit':
            await self._sport("Sit")
            self.sitting = True
        elif trick == 'stand':
            await self._sport("RiseSit")
            self.sitting = False
        elif trick in ('paw', 'heart'):
            # お手とハートは立った状態から行う
            if self.sitting:
                await self._sport("RiseSit")
                self.sitting = False
                await asyncio.sleep(2)
            await self._sport("Hello" if trick == 'paw' else "FingerHeart")

    async def run(self):
        await self.connect()
        print("接続OK。映像を待っています... (ESCで終了)")
        print("パー=お手  人差し指だけ=おすわり  グー=立つ  チョキ=ハート  / キー h=お手 s=おすわり u=立つ v=ハート")

        last_id = 0
        candidate = None        # 判定中のジェスチャー
        candidate_t = 0.0       # そのジェスチャーが始まった時刻
        other_t = None          # 判定中と違う結果が出始めた時刻(揺れの判定用)
        fired = False           # 判定中のジェスチャーで指示を出し済み(手を下ろすか形を変えるまで再発火しない)
        busy_until = 0.0        # 動作中は次の指示を受け付けない
        last_trick = None

        try:
            while True:
                if self.frame_id == last_id:
                    await asyncio.sleep(0.005)
                    continue
                last_id = self.frame_id
                frame, hands, white = await asyncio.to_thread(_convert_and_detect, self.latest_frame)
                now = time.time()

                # 同じジェスチャーを HOLD_SEC 続けたら確定
                trick, source = classify(hands)
                # 判定が GRACE_SEC 以内だけ揺れても、同じジェスチャーを続けているとみなす
                if trick == candidate:
                    other_t = None
                elif candidate is None or (other_t is not None and now - other_t >= GRACE_SEC):
                    candidate, candidate_t, other_t, fired = trick, now, None, False
                elif other_t is None:
                    other_t = now
                hold = min((now - candidate_t) / HOLD_SEC, 1.0) if candidate else 0.0

                draw_hands(frame, hands, f"{trick.upper() if trick else '-'} ({source})" if source else None)

                busy = now < busy_until

                # お手はその場で行う(手に近づいたり向き直ったりしない。追いかけるのは go2_follow.py)
                if candidate and hold >= 1.0 and not busy and not fired:
                    fired = True
                    last_trick = candidate
                    if self.start_trick(last_trick):
                        busy_until = now + ACTION_SEC

                # 表示
                state = "SITTING" if self.sitting else "STANDING"
                label = f"{state}  last: {TRICK_LABEL.get(last_trick, '-')}"
                if self.dry_run:
                    label += "  [DRY-RUN]"
                cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 255) if busy else (0, 255, 0), 2)
                if busy:
                    cv2.putText(frame, f"BUSY {busy_until - now:.1f}s", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                elif candidate and fired:
                    cv2.putText(frame, f"{TRICK_LABEL[candidate]} done - lower hand to repeat", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                elif candidate:
                    cv2.putText(frame, f"{TRICK_LABEL[candidate]} {hold * 100:.0f}%", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                hand = right_hand(hands)
                if hand is not None:
                    cos = finger_cos(hand)
                    cv2.putText(frame, f"I M R P: {fingers_label(cos)}", (10, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(frame, f"white {white * 100:.0f}%", (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.imshow('Go2 Trick', frame)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESCキーで終了
                    break
                if key in KEY_TRICK and not busy:
                    last_trick = KEY_TRICK[key]
                    if self.start_trick(last_trick):
                        busy_until = now + ACTION_SEC

                await asyncio.sleep(0)   # 映像受信に処理を譲る

        finally:
            await self.disconnect()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", help="Go2のIPアドレス (STAモード)")
    ap.add_argument("--aes-key", default=None, help="新ファームウェア用 AES-128 キー(32桁hex)")
    ap.add_argument("--dry-run", action="store_true", help="ロボットを動かさず判定結果の表示のみ")
    ap.add_argument("--webcam", type=int, default=None, metavar="N",
                    help="Go2 の代わりに Mac のカメラ N で試す(ロボットは動かさない)")
    ap.add_argument("--gamma", type=float, default=1.0,
                    help="白飛び対策の明るさ補正。1.0 で補正なし、1.5〜2.5 で明るい部分を暗くする")
    args = ap.parse_args()
    set_gamma(args.gamma)
    if args.ip is None and args.webcam is None:
        ap.error("--ip か --webcam のどちらかを指定してください")

    trick = Go2Trick(args.ip, args.aes_key, args.dry_run, args.webcam)
    try:
        asyncio.run(trick.run())
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
