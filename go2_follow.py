import argparse
import asyncio
import json
import os
import time

import cv2
from config import GO2_IP, GO2_AES_KEY
from hand_control import detect, compute_cmd, close, set_gamma, adjust_exposure, overexposed_ratio, draw_hands
from unitree_webrtc_connect import (
    UnitreeWebRTCConnection,
    WebRTCConnectionMethod,
    RTC_TOPIC,
    SPORT_CMD,
)

CMD_HZ = 10             # 指令送信頻度
SMOOTH = 0.4            # 0〜1。小さいほどなめらか(遅延は増える)
LOST_TIMEOUT = 0.5      # 手を見失ってから停止するまでの秒数
REPLY_TIMEOUT = 0.5     # ロボットからの応答をこの秒数以上は待たない
SPORT_NAME = {v: k for k, v in SPORT_CMD.items()}   # ログ表示用
RECORD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recordings")
RECORD_FPS = 14         # Go2 の映像は約 14 fps


class Recorder:
    """受信した映像を、明るさ補正をかける前の状態で動画ファイルに保存する(手の検出方法の比較用)"""

    def __init__(self):
        os.makedirs(RECORD_DIR, exist_ok=True)
        self.path = os.path.join(RECORD_DIR, time.strftime("%Y%m%d_%H%M%S") + ".mp4")
        self.writer = None

    def write(self, frame):
        if self.writer is None:     # 映像の大きさは最初のフレームが来るまでわからない
            h, w = frame.shape[:2]
            self.writer = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*"mp4v"), RECORD_FPS, (w, h))
        self.writer.write(frame)

    def close(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            print(f"録画を保存しました: {self.path}")


def _convert_and_detect(av_frame, recorder=None):
    """受信フレームを BGR 画像に変換し、明るさ補正をかけてハンドジェスチャーを検出"""
    raw = av_frame.to_ndarray(format="bgr24")
    if recorder is not None:
        recorder.write(raw)
    frame = adjust_exposure(raw)
    return frame, detect(frame), overexposed_ratio(raw)


class Go2HandFollower:
    """Go2ロボット用ハンドジェスチャー追従コントローラー"""

    def __init__(self, robot_ip, aes_key=None, dry_run=False, record=False, ap=False):
        """
        Args:
            robot_ip: Go2ロボットのIPアドレス(ap=True のときは使わない)
            aes_key: 新しめのファームウェアで必要な AES-128 キー(32桁hex)
            dry_run: True ならロボットを動かさず表示のみ
            record: True なら受信した映像を recordings/ に保存する
            ap: True なら Go2 自身の Wi-Fi(AP モード)に直接つなぐ。Go2 の IP は 192.168.12.1 に決まっている
        """
        self.robot_ip = robot_ip
        self.ap = ap
        self.aes_key = aes_key
        self.dry_run = dry_run
        self.recorder = Recorder() if record else None
        self.conn = None
        self.latest_frame = None    # 最新フレームのみ保持(古いフレームは捨てて遅延を防ぐ)
        self.frame_id = 0           # 受信したフレームの通し番号(同じフレームを二度処理しないため)
        self.forward = 0.0          # 平滑化後の指令
        self.turn = 0.0
        self.moving = False
        self._move_task = None      # 応答待ち中の移動指令
        self._last_logged_move = None

    async def connect(self):
        """ロボットに接続して映像受信を開始"""
        kwargs = {} if self.ap else {"ip": self.robot_ip}
        if self.aes_key:
            kwargs["aes_128_key"] = self.aes_key
        method = WebRTCConnectionMethod.LocalAP if self.ap else WebRTCConnectionMethod.LocalSTA
        self.conn = UnitreeWebRTCConnection(method, **kwargs)
        await self.conn.connect()

        if not self.dry_run:
            await self._set_motion_mode("normal")

        self.conn.video.switchVideoChannel(True)
        self.conn.video.add_track_callback(self._on_track)

    async def _set_motion_mode(self, name):
        """歩行モードを切り替える(すでにそのモードなら何もしない)"""
        pub_sub = self.conn.datachannel.pub_sub
        resp = await pub_sub.publish_request_new(
            RTC_TOPIC["MOTION_SWITCHER"], {"api_id": 1001}
        )
        current = json.loads(resp["data"]["data"])["name"]
        if current != name:
            await pub_sub.publish_request_new(
                RTC_TOPIC["MOTION_SWITCHER"],
                {"api_id": 1002, "parameter": {"name": name}}
            )
            await asyncio.sleep(5)

    async def _on_track(self, track):
        while True:
            # 画像への変換は使うフレームだけ行う(受信処理を軽くして映像の乱れを防ぐ)
            self.latest_frame = await track.recv()
            self.frame_id += 1

    async def run(self):
        """ハンドジェスチャー追従を開始"""
        await self.connect()
        print("接続OK。映像を待っています... (ESCで終了)")

        period = 1.0 / CMD_HZ
        next_t = time.time()
        last_seen = 0.0
        cmd = None
        last_id = 0
        fps = 0.0
        last_t = time.time()

        try:
            while True:
                # 新しいフレームが来るまで待つ(同じフレームを繰り返し検出しない)
                if self.frame_id == last_id:
                    # 映像が途切れたら停止する
                    if self.moving and time.time() - last_t >= LOST_TIMEOUT:
                        cmd = None
                        await self._stop()
                    await asyncio.sleep(0.005)
                    continue
                last_id = self.frame_id
                # 画像変換とハンドジェスチャー検出は別スレッドで行う。
                # 同じスレッドで行うと、その間 WebRTC の受信が止まり映像が乱れ遅れる
                frame, hands, white = await asyncio.to_thread(_convert_and_detect, self.latest_frame, self.recorder)
                now = time.time()
                fps += 0.1 * (1.0 / max(now - last_t, 1e-3) - fps)   # 処理fps(移動平均)
                last_t = now

                # ロボット制御コマンド計算(右手が見えない、または左手だけのときは見失った扱い)
                new_cmd = compute_cmd(hands, frame.shape)
                if new_cmd['gesture'] is not None:
                    cmd = new_cmd
                    last_seen = now
                elif now - last_seen >= LOST_TIMEOUT:
                    cmd = None

                # ロボットに制御コマンドを送信
                if now >= next_t:
                    next_t = now + period
                    await self._send_command(cmd)

                # フレーム表示（デバッグ用）。操作に使う右手は R の印、ほかの手は灰色
                draw_hands(frame, hands, (new_cmd['gesture'] or '-').upper())

                label = f"Forward: {self.forward:+.2f}, Turn: {self.turn:+.2f}"
                if self.dry_run:
                    label += "  [DRY-RUN]"
                cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0) if self.moving else (0, 0, 255), 2)
                cv2.putText(frame, f"{fps:.0f} fps  white {white * 100:.0f}%", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                if self.recorder is not None:
                    cv2.putText(frame, "REC", (frame.shape[1] - 80, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.imshow('Go2 Hand Follow', frame)

                if cv2.waitKey(1) & 0xFF == 27:  # ESCキーで終了
                    break

                await asyncio.sleep(0)   # 映像受信に処理を譲る

        finally:
            await self.disconnect()

    async def _send_command(self, cmd):
        """
        ロボットにコマンドを送信

        Args:
            cmd: 制御コマンド。None(手を見失った)またはグーなら停止する
        """
        if cmd is None or cmd['gesture'] != 'open':
            if self.moving:
                await self._stop()
            return

        self.forward += SMOOTH * (cmd['forward'] - self.forward)
        self.turn += SMOOTH * (cmd['turn'] - self.turn)
        self.moving = True

        self._move(self.forward, self.turn)

    async def _request(self, api_id, parameter=None):
        """スポーツモードの指令を送り、応答を待つ(REPLY_TIMEOUT まで)"""
        options = {"api_id": api_id}
        if parameter is not None:
            options["parameter"] = parameter
        name = SPORT_NAME.get(api_id, api_id)
        # 送った指令をログに出す(Move は値が変わったときだけ)
        log = name != "Move" or parameter != self._last_logged_move
        if name == "Move":
            self._last_logged_move = parameter
        if log:
            args = "" if parameter is None else " " + " ".join(f"{k}={v:+.2f}" for k, v in parameter.items())
            print(f"[{time.strftime('%H:%M:%S')}.{int(time.time() * 1000) % 1000:03d}] 送信 {name}{args}")
        try:
            resp = await asyncio.wait_for(
                self.conn.datachannel.pub_sub.publish_request_new(RTC_TOPIC["SPORT_MOD"], options),
                REPLY_TIMEOUT,
            )
            try:
                code = resp["data"]["header"]["status"]["code"]
            except (TypeError, KeyError):
                code = None     # 応答の形式が想定と違うときはコードを表示しない
            if code not in (None, 0):
                print(f"    -> {name} の応答コード {code}(0以外は失敗の可能性)")
        except asyncio.TimeoutError:
            if log:
                print(f"    -> {name} の応答なし({REPLY_TIMEOUT}秒)")

    def _move(self, x, z):
        """
        移動指令を応答を待たずに送る(x: 前後 [m/s]、z: 旋回 [rad/s])

        応答を待つと、その間(Wi-Fi の往復分)画面の処理が止まって重くなるため。
        前の指令の応答待ち中なら今回は送らない(指令は 0.1 秒ごとに来るので次で追いつく)。
        """
        if self.dry_run:
            return
        if self._move_task is not None and not self._move_task.done():
            return
        self._move_task = asyncio.create_task(
            self._request(SPORT_CMD["Move"], {"x": x, "y": 0.0, "z": z})
        )

    async def _stop(self):
        """ロボットを停止"""
        self.forward = self.turn = 0.0
        self.moving = False
        if self.dry_run:
            return
        await self._request(SPORT_CMD["StopMove"])

    async def disconnect(self):
        """ロボットを停止して接続を切断"""
        if self.conn is not None:
            await self._stop()
            await self.conn.disconnect()
        if self.recorder is not None:
            self.recorder.close()
        close()
        cv2.destroyAllWindows()


def main():
    """メイン処理"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default=GO2_IP, help="Go2のIPアドレス (STAモード)。省略時は .env の GO2_IP")
    ap.add_argument("--aes-key", default=GO2_AES_KEY,
                    help="新ファームウェア用 AES-128 キー(32桁hex)。省略時は .env の GO2_AES_KEY")
    ap.add_argument("--dry-run", action="store_true", help="ロボットを動かさず検出と指令値の表示のみ")
    ap.add_argument("--gamma", type=float, default=1.0,
                    help="白飛び対策の明るさ補正。1.0 で補正なし、1.5〜2.5 で明るい部分を暗くする")
    ap.add_argument("--record", action="store_true",
                    help="受信した映像を recordings/ に動画で保存する(明るさ補正の前の映像)")
    ap.add_argument("--ap", action="store_true",
                    help="Go2 自身の Wi-Fi(AP モード)に Mac を直接つないで使う。--ip は不要")
    args = ap.parse_args()
    if args.ip is None and not args.ap:
        ap.error("Go2 の IP がありません。.env に GO2_IP を書くか --ip で指定してください")
    set_gamma(args.gamma)

    follower = Go2HandFollower(args.ip, args.aes_key, args.dry_run, args.record, args.ap)
    try:
        asyncio.run(follower.run())
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
