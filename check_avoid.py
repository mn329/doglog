"""
Go2 Air で本体の障害物回避と LiDAR が WebRTC から使えるかを確かめる(試験用)

確かめること:
  1. 状態の取得(動かない)  歩行モード、障害物回避の ON/OFF、LiDAR・位置情報が届くか
  2. 前進テスト(動く)     障害物回避を ON にして、回避用の移動指令でゆっくり前進する。
                          前に箱などを置いておき、ぶつかる前に止まる(またはよける)かを見る

使い方(go2-hand-follow フォルダで、.venv を有効にして):
  python check_avoid.py --dry-run     # 1 だけ(ロボットは動かない)
  python check_avoid.py               # 1 のあと Enter で 2 を行う

箱の前で止まらなかったときは、移動指令の送り方を変えて試す(--method):
  req       既定。ライブラリの指令と同じ形(type=req)
  msg       go2_ros2_sdk と同じ形(type=msg)。中身は req と同じ
  joystick  リモコンのふりをする(rt/wirelesscontroller)。アプリの操作で止まったのと同じ経路のはず。
            スティックの倒し具合と速さの関係がわからないので、遅めに換算している(進んだ距離で確かめる)

前進テストは、Go2 を立たせた状態で、前方 1.5〜2m に箱などを置き、まわりを広く空けて行う。
Ctrl+C でいつでも止まる。終わると障害物回避の設定は元に戻す。
"""
import argparse
import asyncio
import json
import sys
import time

from config import GO2_AES_KEY, find_ip
from unitree_webrtc_connect import (
    UnitreeWebRTCConnection,
    WebRTCConnectionMethod,
    RTC_TOPIC,
    SPORT_CMD,
    OBSTACLES_AVOID_API,
    DATA_CHANNEL_TYPE,
)

LISTEN_SEC = 5.0        # LiDAR・位置情報が届くかを見る秒数
CMD_HZ = 10             # 前進テストの指令送信頻度
JOY_FULL_SPEED = 2.5    # joystick: スティックを倒しきったときの速さ [m/s] と仮定(多めに見積もって遅く動かす)
JOY_FULL_YAW = 3.0      # joystick: 倒しきったときの旋回の速さ [rad/s] と仮定
METHODS = ("req", "msg", "joystick")
REPLY_TIMEOUT = 2.0     # 応答をこの秒数以上は待たない


def _ts():
    return f"[{time.strftime('%H:%M:%S')}.{int(time.time() * 1000) % 1000:03d}]"


def _parse_data(resp):
    """応答の data.data(JSON 文字列のことが多い)を取り出す。取れなければ None"""
    try:
        data = resp["data"]["data"]
    except (TypeError, KeyError):
        return None
    if isinstance(data, str):
        try:
            return json.loads(data)
        except ValueError:
            return data
    return data


async def wait_enter():
    """Enter を待つ(input() を別スレッドで待つと Ctrl+C のあと終了できなくなるため)"""
    loop = asyncio.get_running_loop()
    done = loop.create_future()
    loop.add_reader(sys.stdin, lambda: done.done() or done.set_result(sys.stdin.readline()))
    try:
        await done
    finally:
        loop.remove_reader(sys.stdin)


def _code(resp):
    try:
        return resp["data"]["header"]["status"]["code"]
    except (TypeError, KeyError):
        return None


class AvoidChecker:
    def __init__(self, robot_ip, aes_key=None, method="req"):
        self.method = method        # 回避用の移動指令の送り方(METHODS のどれか)
        self.robot_ip = robot_ip
        self.aes_key = aes_key
        self.conn = None
        self.counts = {}            # トピックごとの受信数
        self.samples = {}           # トピックごとの最初のメッセージ(形を表示する用)
        self.position = None        # sportmodestate の位置 [x, y, z]
        self.avoid_original = None  # テスト前の障害物回避の ON/OFF(終了時に戻す)
        self.avoid_changed = False

    @property
    def pub_sub(self):
        return self.conn.datachannel.pub_sub

    async def connect(self):
        kwargs = {"ip": self.robot_ip}
        if self.aes_key:
            kwargs["aes_128_key"] = self.aes_key
        self.conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, **kwargs)
        await self.conn.connect()

    async def request(self, topic, api_id, parameter=None):
        """指令を送って応答を返す。応答が来なければ None"""
        options = {"api_id": api_id}
        if parameter is not None:
            options["parameter"] = parameter
        try:
            return await asyncio.wait_for(self.pub_sub.publish_request_new(topic, options), REPLY_TIMEOUT)
        except asyncio.TimeoutError:
            return None

    def request_no_reply(self, topic, api_id, parameter, msg_type=DATA_CHANNEL_TYPE["REQUEST"]):
        """応答を返さない指令(回避用の移動指令)を送る"""
        payload = {
            "header": {"identity": {"id": int(time.time() * 1000) % 2147483648, "api_id": api_id}},
            "parameter": json.dumps(parameter),
        }
        self.pub_sub.publish_without_callback(topic, payload, msg_type)

    def avoid_move(self, x, yaw):
        """回避機能を通る移動指令を、self.method の送り方で送る(x: 前後 [m/s]、yaw: 旋回 [rad/s]、左が正)"""
        if self.method == "joystick":
            # ly: 前に倒すと前進、rx: 右に倒すと右旋回(yaw は左が正なので符号を反転)
            stick = {"lx": 0.0, "ly": max(-1.0, min(1.0, x / JOY_FULL_SPEED)),
                     "rx": max(-1.0, min(1.0, -yaw / JOY_FULL_YAW)), "ry": 0.0, "keys": 0}
            self.pub_sub.publish_without_callback(RTC_TOPIC["WIRELESS_CONTROLLER"], stick, DATA_CHANNEL_TYPE["MSG"])
            return
        msg_type = DATA_CHANNEL_TYPE["MSG"] if self.method == "msg" else DATA_CHANNEL_TYPE["REQUEST"]
        self.request_no_reply(RTC_TOPIC["OBSTACLES_AVOID"], OBSTACLES_AVOID_API["MOVE"],
                              {"x": x, "y": 0.0, "yaw": yaw, "mode": 0}, msg_type)

    def _listen(self, name):
        def callback(message):
            self.counts[name] = self.counts.get(name, 0) + 1
            if name not in self.samples:
                self.samples[name] = message.get("data")
            if name == "LF_SPORT_MOD_STATE":
                data = message.get("data")
                if isinstance(data, dict) and "position" in data:
                    self.position = data["position"]
        return callback

    # ---- 1. 状態の取得 ----

    async def check_status(self):
        print("\n=== 1. 状態の取得(動きません) ===")

        resp = await self.request(RTC_TOPIC["MOTION_SWITCHER"], 1001)
        mode = _parse_data(resp)
        print(f"歩行モード: {mode.get('name') if isinstance(mode, dict) else mode}")

        enable = await self.get_avoid()
        if enable is None:
            print("障害物回避の状態: 取得できませんでした(Air では使えない可能性)")
        else:
            print(f"障害物回避の状態: {'ON' if enable else 'OFF'}")
            self.avoid_original = enable

        # LiDAR は通信量節約を切ってからスイッチを入れないと届かない
        try:
            await asyncio.wait_for(self.conn.datachannel.disableTrafficSaving(True), REPLY_TIMEOUT)
        except asyncio.TimeoutError:
            print("通信量節約の解除: 応答なし")
        self.pub_sub.publish_without_callback(RTC_TOPIC["ULIDAR_SWITCH"], "on")
        for name in ("ULIDAR_ARRAY", "ULIDAR_STATE", "ROBOTODOM", "LF_SPORT_MOD_STATE"):
            self.pub_sub.subscribe(RTC_TOPIC[name], self._listen(name))

        print(f"LiDAR・位置情報を {LISTEN_SEC:g} 秒待っています...")
        await asyncio.sleep(LISTEN_SEC)
        for name in ("ULIDAR_ARRAY", "ULIDAR_STATE", "ROBOTODOM", "LF_SPORT_MOD_STATE"):
            n = self.counts.get(name, 0)
            print(f"  {name:16s} {n:4d} 件 ({n / LISTEN_SEC:.1f} Hz)  {self._describe(name)}")

    def _describe(self, name):
        """受信したメッセージの中身を短く表す"""
        sample = self.samples.get(name)
        if sample is None:
            return "← 届いていない"
        if name == "ULIDAR_ARRAY" and isinstance(sample, dict):
            points = sample.get("data", {}).get("positions") if isinstance(sample.get("data"), dict) else None
            if points is not None:
                return f"点の数(座標の要素数) {len(points)}"
        if name == "LF_SPORT_MOD_STATE" and self.position is not None:
            return "位置 " + ", ".join(f"{v:+.2f}" for v in self.position[:3])
        if isinstance(sample, dict):
            return "キー: " + ", ".join(list(sample.keys())[:8])
        return str(sample)[:80]

    async def get_avoid(self):
        resp = await self.request(RTC_TOPIC["OBSTACLES_AVOID"], OBSTACLES_AVOID_API["SWITCH_GET"])
        data = _parse_data(resp)
        if isinstance(data, dict) and "enable" in data:
            return bool(data["enable"])
        if resp is not None:
            print(f"    障害物回避の応答(想定外の形): code={_code(resp)} data={data}")
        return None

    # ---- 2. 前進テスト ----

    async def forward_test(self, speed, duration):
        print("\n=== 2. 前進テスト(動きます) ===")

        resp = await self.request(RTC_TOPIC["OBSTACLES_AVOID"], OBSTACLES_AVOID_API["SWITCH_SET"], {"enable": True})
        self.avoid_changed = True
        print(f"障害物回避を ON: 応答コード {_code(resp) if resp else '応答なし'}")
        enable = await self.get_avoid()
        print(f"障害物回避の状態(確認): {'取得できず' if enable is None else ('ON' if enable else 'OFF')}")
        if not enable:
            print("ON にできなかったので前進テストはしません")
            return

        await self.use_api_commands()

        start = list(self.position) if self.position else None
        print(f"{_ts()} 前進 {speed:.2f} m/s を {duration:.0f} 秒(Ctrl+C で止まる)")
        period = 1.0 / CMD_HZ
        end_t = time.time() + duration
        last_print = 0.0
        while time.time() < end_t:
            self.avoid_move(speed, 0.0)
            if start and self.position and time.time() - last_print >= 0.5:
                last_print = time.time()
                print(f"{_ts()}   進んだ距離 {self._distance(start):.2f} m")
            await asyncio.sleep(period)
        self.stop_avoid_move()

        if start and self.position:
            await asyncio.sleep(0.5)
            print(f"合計で進んだ距離: {self._distance(start):.2f} m(指令どおりなら {speed * duration:.2f} m)")
        else:
            print("位置情報が届いていないので進んだ距離はわかりません。見た目で判断してください")

    async def use_api_commands(self):
        """回避用の移動指令を API から受け付けるようにする(joystick はリモコンの経路なので API 側は切っておく)"""
        from_api = self.method != "joystick"
        resp = await self.request(RTC_TOPIC["OBSTACLES_AVOID"], OBSTACLES_AVOID_API["USE_REMOTE_COMMAND_FROM_API"],
                                  {"is_remote_commands_from_api": from_api})
        print(f"移動指令の送り方: {self.method}(API からの移動指令 {'有効' if from_api else '無効'}: "
              f"応答コード {_code(resp) if resp else '応答なし'})")

    def _distance(self, start):
        return ((self.position[0] - start[0]) ** 2 + (self.position[1] - start[1]) ** 2) ** 0.5

    def stop_avoid_move(self):
        self.avoid_move(0.0, 0.0)

    async def cleanup(self):
        """止めて、障害物回避の設定を元に戻して切断する"""
        if self.conn is None:
            return
        if self.avoid_changed:
            self.stop_avoid_move()
            await self.request(RTC_TOPIC["OBSTACLES_AVOID"], OBSTACLES_AVOID_API["USE_REMOTE_COMMAND_FROM_API"],
                               {"is_remote_commands_from_api": False})
            # 最初の状態が取れなかったときは OFF に戻す(ON のまま残すと、ほかのプログラムで思わぬ止まり方をする)
            original = bool(self.avoid_original)
            await self.request(RTC_TOPIC["OBSTACLES_AVOID"], OBSTACLES_AVOID_API["SWITCH_SET"], {"enable": original})
            print(f"障害物回避を元に戻しました({'ON' if original else 'OFF'})")
            await self.request(RTC_TOPIC["SPORT_MOD"], SPORT_CMD["StopMove"])
        await self.conn.disconnect()

    async def run(self, dry_run, speed, duration):
        try:
            await self.connect()
            await self.check_status()
            if dry_run:
                return
            print("\n前方 1.5〜2m に箱などを置き、まわりを空けてから Enter(やめるなら Ctrl+C)")
            await wait_enter()
            await self.forward_test(speed, duration)
        finally:
            await self.cleanup()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", help="Go2のIPアドレス (STAモード)。省略時は .env の GO2_SN で探し、だめなら GO2_IP")
    ap.add_argument("--aes-key", default=GO2_AES_KEY,
                    help="新ファームウェア用 AES-128 キー(32桁hex)。省略時は .env の GO2_AES_KEY")
    ap.add_argument("--dry-run", action="store_true", help="状態の取得だけ行い、ロボットを動かさない")
    ap.add_argument("--speed", type=float, default=0.3, help="前進テストの速さ [m/s]")
    ap.add_argument("--duration", type=float, default=6.0, help="前進テストの秒数")
    ap.add_argument("--method", choices=METHODS, default="req",
                    help="回避用の移動指令の送り方。箱の前で止まらなかったら msg → joystick の順に試す")
    args = ap.parse_args()
    if args.ip is None:
        args.ip = find_ip()
    if args.ip is None:
        ap.error("Go2 の IP がありません。.env に GO2_IP を書くか --ip で指定してください")
    if not 0 < args.speed <= 0.5:
        ap.error("--speed は 0〜0.5 m/s にしてください(テストなのでゆっくり)")

    try:
        asyncio.run(AvoidChecker(args.ip, args.aes_key, args.method).run(args.dry_run, args.speed, args.duration))
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
