"""
Go2 に部屋の中をうろうろ歩かせる(本体の障害物回避で止まり、こちらで向きを変える)

Go2 本体の障害物回避は「ぶつかる前に止まる」までで、よけはしない。そこで:
  前進 → 進めていない(回避で止まった)と気づいたら → その場で向きを変える → また前進
をくり返す。同じ向きにばかり進まないよう、ときどき気まぐれにも曲がる。

本体の回避が Wi-Fi 越しの命令で効かなかったときは --lidar を付ける。LiDAR の点群を自分で見て、
体の前に物があれば止まって向きを変える(本体の回避は使わず、ふつうの Move で歩く)。

使い方(go2-hand-follow フォルダで、.venv を有効にして):
  python wander.py --dry-run          # 命令を送らずに、状態の切り替わりだけ表示する
  python wander.py                    # 歩かせる(既定で 120 秒。Enter か Ctrl+C で止まる)
  python wander.py --lidar --dry-run  # 動かさずに、前方の物までの距離を表示する(箱を置いて確かめる)
  python wander.py --lidar            # LiDAR で止まりながら歩かせる

先に check_avoid.py で、障害物回避が ON にでき、箱の前で止まることを確かめておく。
まわりを広く空け、段差や階段のない場所で、すぐ止められる状態で行う。
"""
import argparse
import asyncio
import math
import random
import time

import numpy as np

from check_avoid import AvoidChecker, METHODS, wait_enter, _code, _ts
from config import GO2_AES_KEY, find_ip
from unitree_webrtc_connect import RTC_TOPIC, OBSTACLES_AVOID_API, SPORT_CMD

CMD_HZ = 10             # 指令送信頻度
SPEED = 0.3             # 前進の速さ [m/s]
TURN_RATE = 0.8         # 向きを変える速さ [rad/s]
TURN_SEC = (1.5, 3.0)   # 向きを変える秒数の範囲(約 70〜140 度)
WHIM_SEC = (6.0, 12.0)  # この秒数前進したら、気まぐれに曲がる
STUCK_WINDOW = 1.5      # この秒数のあいだに
STUCK_DIST = 0.10       # これしか進んでいなければ「止められた」とみなす [m]
START_GRACE = 1.0       # 前進を始めてからこの秒数は、加速中なので止められた判定をしない
STATE_TIMEOUT = 1.0     # ロボットの状態がこの秒数届かなければ、通信が切れたとみなして止まる

# --lidar のときの「前に物がある」の判定(ロボットの中心から見た範囲)
BODY_FRONT = 0.35       # 胴体の中心から前端まで [m]
AHEAD = 0.5             # 前端からこの距離 [m] 以内に点があれば止まる
HALF_WIDTH = 0.25       # 左右この幅 [m] の中だけを見る(胴体の幅 31cm + 余裕)
HEIGHT = (-0.2, 0.3)    # 胴体の高さから見たこの範囲 [m] の点だけを見る(床を拾わないため)
MIN_POINTS = 3          # この数以上の点があれば物があるとみなす(ノイズ対策)
LIDAR_TIMEOUT = 1.0     # 点群がこの秒数届かなければ止まる


def front_distance(message, pose):
    """
    点群の中で、ロボットの前の範囲にある点までの距離 [m](前端から)を返す。なければ None

    点の位置 = positions × resolution + origin(地図の座標)。go2_ros2_sdk と同じ変換。
    pose は robot_pose の (x, y, z, 向き[rad])。
    """
    data = message.get("data") or {}
    inner = data.get("data") if isinstance(data.get("data"), dict) else {}
    positions = inner.get("positions")
    if positions is None or len(positions) == 0:
        return None
    pts = np.asarray(positions, dtype=np.float32).reshape(-1, 3) * float(data.get("resolution", 0.0))
    pts += np.asarray(data.get("origin", [0.0, 0.0, 0.0]), dtype=np.float32)
    x, y, z, yaw = pose
    dx, dy = pts[:, 0] - x, pts[:, 1] - y
    fwd = dx * math.cos(yaw) + dy * math.sin(yaw)       # ロボットから見て前方向
    side = -dx * math.sin(yaw) + dy * math.cos(yaw)     # ロボットから見て左方向
    h = pts[:, 2] - z
    mask = (fwd > BODY_FRONT) & (np.abs(side) < HALF_WIDTH) & (h > HEIGHT[0]) & (h < HEIGHT[1])
    if mask.sum() < MIN_POINTS:
        return None
    return float(np.sort(fwd[mask])[MIN_POINTS - 1] - BODY_FRONT)


def yaw_of(q):
    """クォータニオン {x, y, z, w} から向き(z 軸まわりの角度)[rad]"""
    return math.atan2(2 * (q["w"] * q["z"] + q["x"] * q["y"]), 1 - 2 * (q["y"] ** 2 + q["z"] ** 2))


class Wanderer(AvoidChecker):
    def __init__(self, robot_ip, aes_key=None, dry_run=False, lidar=False, method="req"):
        super().__init__(robot_ip, aes_key, method)
        self.dry_run = dry_run
        self.lidar = lidar
        self.pose = None            # robot_pose の (x, y, z, 向き)
        self.ahead = None           # 前の物までの距離 [m]。なければ None
        self.lidar_t = 0.0          # 点群を最後に受け取った時刻
        self.state_t = 0.0          # ロボットの状態を最後に受け取った時刻
        self.trail = []             # 前進中の (時刻, x, y)。止められた判定に使う

    def _on_state(self, message):
        data = message.get("data")
        if isinstance(data, dict) and "position" in data:
            self.position = data["position"]
            self.state_t = time.time()

    def _on_pose(self, message):
        try:
            pose = message["data"]["pose"]
            p = pose["position"]
            self.pose = (p["x"], p["y"], p["z"], yaw_of(pose["orientation"]))
        except (TypeError, KeyError):
            pass

    def _on_lidar(self, message):
        if self.pose is None:
            return
        try:
            self.ahead = front_distance(message, self.pose)
            self.lidar_t = time.time()
        except (TypeError, ValueError) as e:
            print(f"{_ts()} 点群を読めませんでした: {e}")

    async def setup(self):
        await self.connect()
        self.pub_sub.subscribe(RTC_TOPIC["LF_SPORT_MOD_STATE"], self._on_state)
        if self.lidar:
            return await self.setup_lidar()
        enable = await self.get_avoid()
        self.avoid_original = enable
        print(f"障害物回避の状態: {'取得できず' if enable is None else ('ON' if enable else 'OFF')}")
        if self.dry_run:
            return True

        resp = await self.request(RTC_TOPIC["OBSTACLES_AVOID"], OBSTACLES_AVOID_API["SWITCH_SET"], {"enable": True})
        self.avoid_changed = True
        if not await self.get_avoid():
            print(f"障害物回避を ON にできませんでした(応答コード {_code(resp) if resp else '応答なし'})。歩かせません")
            return False
        await self.use_api_commands()
        print("障害物回避を ON にしました")
        return True

    async def setup_lidar(self):
        """LiDAR の点群とロボットの位置を受け取り始める。届かなければ歩かせない"""
        try:
            await asyncio.wait_for(self.conn.datachannel.disableTrafficSaving(True), 2.0)
        except asyncio.TimeoutError:
            pass
        self.pub_sub.publish_without_callback(RTC_TOPIC["ULIDAR_SWITCH"], "on")
        self.pub_sub.subscribe(RTC_TOPIC["ROBOTODOM"], self._on_pose)
        self.pub_sub.subscribe(RTC_TOPIC["ULIDAR_ARRAY"], self._on_lidar)
        await asyncio.sleep(2.0)
        if not self.lidar_t:
            print(f"LiDAR の点群{'' if self.pose else 'とロボットの位置'}が届きません。歩かせません")
            return False
        print("LiDAR モード: 本体の回避は使わず、点群を見て止まります")
        return True

    def send(self, x, yaw):
        """移動指令を送る(dry-run では送らない)。LiDAR モードはふつうの Move、それ以外は回避機能を通す"""
        if self.dry_run:
            return
        if self.lidar:
            self.request_no_reply(RTC_TOPIC["SPORT_MOD"], SPORT_CMD["Move"], {"x": x, "y": 0.0, "z": yaw})
        else:
            self.avoid_move(x, yaw)

    def blocked(self, now, started):
        """前進しているのに、最近ほとんど進めていなければ True(dry-run では動かないので判定しない)"""
        if self.dry_run or self.position is None:
            return False
        self.trail.append((now, self.position[0], self.position[1]))
        self.trail = [p for p in self.trail if now - p[0] <= STUCK_WINDOW]
        if now - started < max(START_GRACE, STUCK_WINDOW):
            return False
        _, x0, y0 = self.trail[0]
        return math.hypot(self.position[0] - x0, self.position[1] - y0) < STUCK_DIST

    async def wander(self, duration):
        """前進と方向転換をくり返す。duration 秒たつか、Enter か、通信が切れたら終わる"""
        stop_key = asyncio.ensure_future(wait_enter())
        end_t = time.time() + duration
        period = 1.0 / CMD_HZ
        state, started, until = None, 0.0, 0.0
        last_print = 0.0
        turns = {"前に物": 0, "止められた": 0, "気まぐれ": 0}
        print(f"{_ts()} うろうろ開始({duration:.0f} 秒。Enter か Ctrl+C で止まる)")
        try:
            while time.time() < end_t and not stop_key.done():
                now = time.time()
                if not self.dry_run and now - self.state_t > STATE_TIMEOUT:
                    print(f"{_ts()} ロボットの状態が {STATE_TIMEOUT} 秒届きません。通信が切れたとみなして止まります")
                    break
                if self.lidar and now - self.lidar_t > LIDAR_TIMEOUT:
                    print(f"{_ts()} LiDAR の点群が {LIDAR_TIMEOUT} 秒届きません。止まります")
                    break
                if self.lidar and now - last_print >= 1.0:
                    last_print = now
                    print(f"{_ts()}   前の物まで {'なし' if self.ahead is None else f'{self.ahead:.2f} m'}")

                if state is None or (state == "turn" and now >= until):
                    state, started = "forward", now
                    until = now + random.uniform(*WHIM_SEC)
                    self.trail = []
                    print(f"{_ts()} 前進")
                elif state == "forward":
                    if self.lidar and self.ahead is not None and self.ahead < AHEAD:
                        reason = "前に物"
                    elif self.blocked(now, started):
                        reason = "止められた"
                    else:
                        reason = "気まぐれ" if now >= until else None
                    if reason:
                        turns[reason] += 1
                        sec = random.uniform(*TURN_SEC)
                        yaw = TURN_RATE * random.choice((-1, 1))
                        state, started, until = "turn", now, now + sec
                        print(f"{_ts()} {reason} → {'左' if yaw > 0 else '右'}に {math.degrees(TURN_RATE * sec):.0f}° 向きを変える")

                if state == "forward":
                    self.send(SPEED, 0.0)
                else:
                    self.send(0.0, yaw)
                await asyncio.sleep(period)
        finally:
            stop_key.cancel()
            self.stop_avoid_move()
            if self.lidar and not self.dry_run:
                self.send(0.0, 0.0)
                await self.request(RTC_TOPIC["SPORT_MOD"], SPORT_CMD["StopMove"])
            print(f"{_ts()} 停止。向きを変えた回数: " + "、".join(f"{k} {v} 回" for k, v in turns.items()))

    async def run(self, duration):
        try:
            if await self.setup():
                await self.wander(duration)
        finally:
            await self.cleanup()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", help="Go2のIPアドレス (STAモード)。省略時は .env の GO2_SN で探し、だめなら GO2_IP")
    ap.add_argument("--aes-key", default=GO2_AES_KEY,
                    help="新ファームウェア用 AES-128 キー(32桁hex)。省略時は .env の GO2_AES_KEY")
    ap.add_argument("--dry-run", action="store_true", help="命令を送らず、状態の切り替わりだけ表示する")
    ap.add_argument("--duration", type=float, default=120.0, help="歩かせる秒数")
    ap.add_argument("--lidar", action="store_true",
                    help="本体の回避の代わりに、LiDAR の点群を見て止まる(回避が効かなかったとき用)")
    ap.add_argument("--method", choices=METHODS, default="req",
                    help="回避用の移動指令の送り方。check_avoid.py で箱の前で止まった送り方を指定する")
    args = ap.parse_args()
    if args.ip is None:
        args.ip = find_ip()
    if args.ip is None:
        ap.error("Go2 の IP がありません。.env に GO2_IP を書くか --ip で指定してください")

    try:
        asyncio.run(Wanderer(args.ip, args.aes_key, args.dry_run, args.lidar, args.method).run(args.duration))
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
