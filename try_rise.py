"""
座った状態から、前に出ずに立てる方法を探す試験(go2-hand-follow フォルダで実行する)

  python try_rise.py 方法

方法:
  risesit    今のやり方(RiseSit のみ)
  cut        RiseSit を送り、0.8 秒後に StopMove + BalanceStand で足踏みを止める
  balance    RiseSit の代わりに BalanceStand(足を動かさずに立つ)
  standup    RiseSit の代わりに StandUp → BalanceStand
  recovery   RiseSit の代わりに RecoveryStand(起き上がり)
"""
import asyncio
import os
import sys

sys.path.insert(0, os.getcwd())
from config import GO2_AES_KEY, find_ip
from go2_trick import Go2Trick
from unitree_webrtc_connect import SPORT_CMD
from unitree_webrtc_connect.constants import RTC_TOPIC

METHODS = ("risesit", "cut", "balance", "standup", "recovery")


async def main(method):
    robot = Go2Trick(find_ip(), GO2_AES_KEY)
    await robot.connect()
    req = robot._request
    robot._pos = None

    def on_state(message):
        data = message.get("data")
        if isinstance(data, dict) and "position" in data:
            robot._pos = data["position"]
    robot.conn.datachannel.pub_sub.subscribe(RTC_TOPIC["LF_SPORT_MOD_STATE"], on_state)
    await asyncio.sleep(1)
    try:
        print("座ります")
        await req(SPORT_CMD["Sit"])
        await asyncio.sleep(4)
        start = robot_pos(robot)
        print(f"立ちます: {method}")
        if method == "risesit":
            await req(SPORT_CMD["RiseSit"])
        elif method == "cut":
            await req(SPORT_CMD["RiseSit"])
            await asyncio.sleep(0.8)
            await req(SPORT_CMD["StopMove"])
            await req(SPORT_CMD["BalanceStand"])
        elif method == "balance":
            await req(SPORT_CMD["BalanceStand"])
        elif method == "standup":
            await req(SPORT_CMD["StandUp"])
            await asyncio.sleep(1.5)
            await req(SPORT_CMD["BalanceStand"])
        elif method == "recovery":
            await req(SPORT_CMD["RecoveryStand"])
        await asyncio.sleep(5)
        end = robot_pos(robot)
        if start and end:
            d = ((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2) ** 0.5
            print(f"立つ間に動いた距離: {d:.2f} m")
        print("立てていなければ、アプリかリモコンで立たせてください")
    finally:
        await robot.conn.disconnect()


def robot_pos(robot):
    return getattr(robot, "_pos", None)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in METHODS:
        print(__doc__)
        sys.exit(1)
    asyncio.run(main(sys.argv[1]))
