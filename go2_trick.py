import argparse
import asyncio
import time

import cv2
import av
from config import GO2_AES_KEY, find_ip
from hand_control import (finger_cos, set_gamma, gesture_of, draw_hands, hand_size, palm_center, hand_up_cos,
                          EXTENDED_COS, FOLDED_COS)
from go2_follow import Go2HandFollower, _convert_and_detect
from unitree_webrtc_connect import SPORT_CMD

HOLD_SEC = 1.0          # 同じジェスチャーをこの秒数続けたら指示として受け付ける(--hold で変えられる)
MIN_HAND_SIZE = 0.0     # これより小さく映る手(遠くの人の手)は無視する(--min-size で変えられる。0 で無視しない)
MIN_UP_COS = 0.0        # 指の向きの真上との cos がこれより小さい手(指が真横より下を向いた手)は無視する。
                        # 下げた手や、通りがかりの人のぶら下げた手で動かないように
JUMP_DIST = 0.15        # 操作中の手が 1 フレームでこれ以上(画面幅に対する割合)動いたら別の手とみなし、判定をやり直す
SIT_MIDDLE_COS = 0.45   # おすわりで中指がこれより曲がっていればよい(約63°以上)。チョキ(中指 0.6 超)との間は判定しない
GRACE_SEC = 0.3         # 判定がこの秒数以内だけ揺れても、続けているとみなす
ACTION_SEC = 4.0        # 動作中はこの秒数、次の指示を受け付けない

# ジェスチャー → 指示
#   パー(4本とも伸び)              : お手
#   人差し指だけ伸び(中指は曲げ)    : おすわり
#   グー(4本とも伸びていない)      : 立つ(おすわりを解除)
#   チョキ(人差し指と中指だけ伸び)  : おまけの動き(--victory で選ぶ。既定は首かしげ)
# ジェスチャー分類(Gesture Recognizer)の結果 → 指示。'None'(該当なし)のときだけ指の角度のルールで判定する
GESTURE_TRICK = {'Open_Palm': 'paw', 'Pointing_Up': 'sit', 'Closed_Fist': 'stand', 'Victory': 'victory'}
KEY_TRICK = {ord('h'): 'paw', ord('s'): 'sit', ord('u'): 'stand', ord('v'): 'victory'}
TRICK_LABEL = {'paw': 'PAW (Hello)', 'sit': 'SIT', 'stand': 'STAND', 'victory': 'VICTORY'}

# チョキ(と v キー)でさせる動き。展示会では写真のピースでも動くので、急に動いても危なくない小さな動きにする。
# 'tilt' 以外は本体の決まった動きの再生。どれも Go2 Air で動くかは実機で未確認(try_action.py で確かめる)
VICTORY_ACTIONS = {
    'tilt': 'TILT',             # 首かしげ: 体を横に少し傾けて戻す(Euler)。その場で、足は動かない
    'wiggle': 'WiggleHips',     # おしりふり
    'stretch': 'Stretch',       # 伸び
    'content': 'Content',       # うれしい
    'heart': 'FingerHeart',     # ハート。後ろ足で立ち上がる大きな動きなので展示会では使わない
    'none': None,               # 何もしない(チョキを無視する)
}
TILT_ROLL = 0.52        # 首かしげで傾ける角度 [rad](約30°)
SETTLE_SEC = {'Hello': 3.0}   # 動きが終わるまで待ってから、その場に立たせる秒数
STANDUP_SEC = 2.0       # 座った状態から立つとき、StandUp のあと BalanceStand までに待つ秒数
RISE_SEC = 1.0          # 終了時、立ったあと(StandUp → BalanceStand)落ち着くまで待つ秒数
TILT_SEC = 1.2          # 首かしげで傾けたままにする秒数


def pick_hand(hands, frame_shape, min_size):
    """
    指示に使う手を選ぶ。右手のうち min_size 以上に大きく映り、指が上(真横より上)を向いている手で、
    いちばん大きい(近い)もの。無ければ None

    人混みでは後ろを通る人の手も映るので、確からしさではなく大きさで選び、小さい(遠い)手は無視する。
    下を向いた手(下げた手)も無視する。
    """
    candidates = [hand for hand in hands if hand.is_right and hand_size(hand, frame_shape) >= min_size
                  and hand_up_cos(hand, frame_shape) >= MIN_UP_COS]
    return max(candidates, key=lambda hand: hand_size(hand, frame_shape), default=None)


def classify(hand):
    """
    指示に使う手(pick_hand の結果)から指示を判定する

    ジェスチャー分類の結果を優先し、分類できなかった('None')ときだけ指の角度のルールで判定する。
    いいね(Thumb_Up)など、指示に割り当てていない形と分類されたときは指示なし。

    Returns:
        (trick, source): trick は該当なしなら None。source は画面表示用の判定の出どころ
            ('model 0.79' = ジェスチャー分類、'fingers' = 指の角度のルール)。手が無ければ None
    """
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
    中指が伸び切らないチョキと間違えないよう、チョキとの間に判定しない幅を残している。
    """
    ext = [c > EXTENDED_COS for c in cos]
    fold = [c < FOLDED_COS for c in cos]
    index, middle, ring, pinky = cos
    if all(ext):
        return 'paw'
    if ext[0] and ext[1] and not ext[2] and not ext[3]:
        return 'victory'
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

    def __init__(self, robot_ip, aes_key=None, dry_run=False, webcam=None, record=False,
                 hold_sec=HOLD_SEC, min_size=MIN_HAND_SIZE, victory='tilt', ap=False):
        super().__init__(robot_ip, aes_key, dry_run or webcam is not None, record, ap)
        self.webcam = webcam    # カメラ番号。指定時は Go2 に接続せず Mac のカメラで試す
        self.hold_sec = hold_sec
        self.min_size = min_size
        self.victory = VICTORY_ACTIONS[victory]     # チョキでさせる動き(None なら何もしない)
        self.sitting = False
        self.paused = False         # 一時停止中はジェスチャーを受け付けない(スペースキーで切り替え)
        self._trick_task = None     # 実行中の動作
        self._tilted = False        # 首かしげで体を傾けている途中(中断されたら元に戻すため)

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
            if self.conn is not None and not self.dry_run:
                await self._untilt()
                if self.sitting:
                    # 座ったまま終わらせない。立ち上がりきる前に止める指令(StopMove)を送らないよう待つ
                    print("座っているので、立ってから終わります(数秒かかります)")
                    await self._stand_up()
                    self.sitting = False
                    await asyncio.sleep(RISE_SEC)
            await super().disconnect()
            return
        self.cap.release()
        if self.recorder is not None:
            self.recorder.close()
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
        # チョキは写真のピースでも出るので、座っているときは立ち上がらせず何もしない
        if trick == 'victory' and (self.victory is None or self.sitting):
            return False
        if self._trick_task is not None and not self._trick_task.done():
            self._trick_task.cancel()   # 前の動きの「その場に立たせる」待ちが残っていれば取り消す(新しい芸を上書きしないため)
        self._trick_task = asyncio.create_task(self.do_trick(trick))
        return True

    async def do_trick(self, trick):
        """指示に応じた動作をさせる"""
        if trick == 'sit':
            self.sitting = True     # 送る前に記録する(座る途中で終了しても、終了時に立たせるため)
            await self._sport("Sit")
        elif trick == 'stand':
            await self._stand_up()
            self.sitting = False
        elif trick == 'victory':
            if self.victory == 'TILT':
                await self._tilt()
            else:
                await self._sport(self.victory)
                await self._settle(self.victory)
        elif trick == 'paw':
            # お手は立った状態から行う
            if self.sitting:
                await self._stand_up()
                self.sitting = False
                await asyncio.sleep(1)
            await self._sport("Hello")
            await self._settle("Hello")

    async def _stand_up(self):
        """
        座った状態から、その場で立つ

        RiseSit(おすわりをやめる)は前に足を出しながら立つので 0.15m ほど前に出る。
        StandUp で立ってから BalanceStand(足を動かさずに立つ)にすると 0.05m ほどで済む(実機で比べた)。
        """
        if self.dry_run:
            return
        await self._request(SPORT_CMD["StandUp"])
        await asyncio.sleep(STANDUP_SEC)
        await self._request(SPORT_CMD["BalanceStand"])

    async def _settle(self, name):
        """
        動きが終わったら、止めてその場に立たせる

        立つ・お手のあと、Go2 は足踏みを続けて少しずつ前に動いてしまう(実機で確認)。
        StopMove で速さを 0 にし、BalanceStand(足を動かさずに立つ)にする。
        """
        if self.dry_run:
            return
        await asyncio.sleep(SETTLE_SEC.get(name, 2.0))
        await self._request(SPORT_CMD["StopMove"])
        await self._request(SPORT_CMD["BalanceStand"])

    async def _tilt(self):
        """首かしげ: その場で体を横に少し傾けて、元に戻す"""
        if self.dry_run:
            return
        await self._request(SPORT_CMD["BalanceStand"])     # 体の傾き(Euler)は止まって立っているときに効く
        self._tilted = True
        await self._request(SPORT_CMD["Euler"], {"x": TILT_ROLL, "y": 0.0, "z": 0.0})
        await asyncio.sleep(TILT_SEC)
        await self._untilt()

    async def _untilt(self):
        """首かしげの傾きを元に戻す(傾けていなければ何もしない)"""
        if self._tilted:
            self._tilted = False
            await self._request(SPORT_CMD["Euler"], {"x": 0.0, "y": 0.0, "z": 0.0})

    async def emergency_stop(self):
        """スペースキー: 動作を中断して止め、ジェスチャーの受け付けを止める"""
        if self._trick_task is not None:
            self._trick_task.cancel()
        self.paused = True
        print("一時停止(スペースキーでもう一度押すと再開)")
        if not self.dry_run:
            await self._untilt()
            await self._request(SPORT_CMD["StopMove"])

    async def run(self):
        await self.connect()
        print("接続OK。映像を待っています... (ESCで終了)")
        print("パー=お手  人差し指だけ=おすわり  グー=立つ  チョキ=おまけ  / キー h=お手 s=おすわり u=立つ v=おまけ")
        print("スペース=止めて一時停止(もう一度で再開)  ESC=終了")

        last_id = 0
        candidate = None        # 判定中のジェスチャー
        candidate_t = 0.0       # そのジェスチャーが始まった時刻
        other_t = None          # 判定中と違う結果が出始めた時刻(揺れの判定用)
        fired = False           # 判定中のジェスチャーで指示を出し済み(手を下ろすか形を変えるまで再発火しない)
        busy_until = 0.0        # 動作中は次の指示を受け付けない
        last_trick = None
        last_center = None      # 前のフレームで指示に使った手の位置(別の手に入れ替わったかの判定用)
        fps = 0.0
        last_t = time.time()

        try:
            while True:
                if self.frame_id == last_id:
                    await asyncio.sleep(0.005)
                    continue
                last_id = self.frame_id
                frame, hands, white = await asyncio.to_thread(_convert_and_detect, self.latest_frame, self.recorder)
                now = time.time()
                fps += 0.1 * (1.0 / max(now - last_t, 1e-3) - fps)   # 処理fps(移動平均)
                last_t = now

                # 近くに大きく映った右手だけを使う
                hand = pick_hand(hands, frame.shape, self.min_size)
                trick, source = classify(hand)
                if self.paused:
                    trick = None
                # 手の位置が大きく飛んだら別の人の手とみなし、続けた時間を数え直す
                if hand is not None:
                    center = palm_center(hand)
                    if last_center is not None and abs(center[0] - last_center[0]) + abs(center[1] - last_center[1]) > JUMP_DIST:
                        candidate = None
                    last_center = center
                # 同じジェスチャーを hold_sec 続けたら確定
                # 判定が GRACE_SEC 以内だけ揺れても、同じジェスチャーを続けているとみなす
                if trick == candidate:
                    other_t = None
                elif candidate is None or (other_t is not None and now - other_t >= GRACE_SEC):
                    candidate, candidate_t, other_t, fired = trick, now, None, False
                elif other_t is None:
                    other_t = now
                hold = min((now - candidate_t) / self.hold_sec, 1.0) if candidate else 0.0

                draw_hands(frame, hands, f"{trick.upper() if trick else '-'} ({source})" if source else None, hand)

                busy = now < busy_until

                # お手はその場で行う(手に近づいたり向き直ったりしない。追いかけるのは go2_follow.py)
                if candidate and hold >= 1.0 and not busy and not fired:
                    fired = True
                    last_trick = candidate
                    if self.start_trick(last_trick):
                        busy_until = now + ACTION_SEC

                # 表示
                if self.paused:
                    h, w = frame.shape[:2]
                    cv2.putText(frame, "PAUSED - SPACE to resume", (10, h // 2), cv2.FONT_HERSHEY_SIMPLEX,
                                1.2, (0, 0, 255), 3)
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
                if hand is not None:
                    cos = finger_cos(hand)
                    cv2.putText(frame, f"I M R P: {fingers_label(cos)}", (10, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                # 手の大きさ(--min-size の調整用)。右手ごとに出す。min 未満の手と、下向き(DOWN)の手は無視される
                sizes = " ".join(f"{hand_size(x, frame.shape):.2f}"
                                 + ("" if hand_up_cos(x, frame.shape) >= MIN_UP_COS else " DOWN")
                                 for x in hands if x.is_right)
                cv2.putText(frame, f"size {sizes or '-'} (min {self.min_size:.2f})", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(frame, f"{fps:.0f} fps  white {white * 100:.0f}%", (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                if self.recorder is not None:
                    cv2.putText(frame, "REC", (frame.shape[1] - 80, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.imshow('Go2 Trick', frame)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESCキーで終了
                    break
                if key == ord(' '):     # スペースキーで止めて一時停止、もう一度で再開
                    if self.paused:
                        self.paused = False
                        print("再開")
                    else:
                        await self.emergency_stop()
                        busy_until = 0.0
                if key in KEY_TRICK and not busy:
                    last_trick = KEY_TRICK[key]
                    if self.start_trick(last_trick):
                        busy_until = now + ACTION_SEC

                await asyncio.sleep(0)   # 映像受信に処理を譲る

        finally:
            await self.disconnect()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default=None,
                    help="Go2のIPアドレス (STAモード)。省略時は .env の GO2_SN で探し、見つからなければ GO2_IP")
    ap.add_argument("--aes-key", default=GO2_AES_KEY,
                    help="新ファームウェア用 AES-128 キー(32桁hex)。省略時は .env の GO2_AES_KEY")
    ap.add_argument("--dry-run", action="store_true", help="ロボットを動かさず判定結果の表示のみ")
    ap.add_argument("--webcam", type=int, default=None, metavar="N",
                    help="Go2 の代わりに Mac のカメラ N で試す(ロボットは動かさない)")
    ap.add_argument("--gamma", type=float, default=1.0,
                    help="白飛び対策の明るさ補正。1.0 で補正なし、1.5〜2.5 で明るい部分を暗くする")
    ap.add_argument("--record", action="store_true",
                    help="受信した映像を recordings/ に動画で保存する(明るさ補正の前の映像)")
    ap.add_argument("--hold", type=float, default=HOLD_SEC,
                    help=f"ジェスチャーを何秒続けたら指示にするか(既定 {HOLD_SEC})。人が多い場所では長めにする")
    ap.add_argument("--min-size", type=float, default=MIN_HAND_SIZE,
                    help="これより小さく映る手(遠くの人の手)を無視する。画面の size の値を見て決める(0 で無視しない)")
    ap.add_argument("--victory", choices=VICTORY_ACTIONS, default='tilt',
                    help="チョキでさせる動き(既定 tilt=首かしげ)。none で何もしない。heart は大きく危ないので展示会では使わない")
    ap.add_argument("--ap", action="store_true",
                    help="Go2 自身の Wi-Fi(AP モード)に Mac を直接つないで使う。ルーターも IP も不要")
    args = ap.parse_args()
    set_gamma(args.gamma)
    if args.ip is None and args.webcam is None and not args.ap:
        args.ip = find_ip()
    if args.ip is None and args.webcam is None and not args.ap:
        ap.error("Go2 の IP がありません。.env に GO2_IP を書くか --ip で指定してください(実機なしなら --webcam 0)")

    trick = Go2Trick(args.ip, args.aes_key, args.dry_run, args.webcam, args.record,
                     args.hold, args.min_size, args.victory, args.ap)
    try:
        asyncio.run(trick.run())
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
