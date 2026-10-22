"""
ハートの代わりの動きの候補を、Go2 実機で 1 つずつ試す(試験用)

go2_trick.py の --victory で選べる動きが Go2 Air で動くか、どのくらいの大きさの動きかを確かめる。
ロボットの応答コードを表示する(0 以外は失敗の可能性。Air では使えない動きのこともある)。

使い方(go2-hand-follow フォルダで、.venv を有効にして):
  python try_action.py tilt       # 首かしげ
  python try_action.py wiggle     # おしりふり
  python try_action.py stretch    # 伸び
  python try_action.py content    # うれしい

Go2 を立たせた状態で、まわり 1.5m 以内に人や物がない場所で行う。
ハート(heart)や、宙返り・ジャンプなどの大きな動きはここからは試せない。
"""
import argparse
import asyncio

from config import GO2_AES_KEY, find_ip
from go2_trick import Go2Trick, VICTORY_ACTIONS

CANDIDATES = [name for name in VICTORY_ACTIONS if name not in ('heart', 'none')]
WATCH_SEC = 5.0         # 動作を送ったあと、終わるまで待つ秒数


async def try_action(trick, name):
    await trick.connect()
    print(f"接続OK。{name}({VICTORY_ACTIONS[name]})を送ります")
    try:
        await trick.do_trick('victory')
        await asyncio.sleep(WATCH_SEC)
    finally:
        await trick.disconnect()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=CANDIDATES, help="試す動き")
    ap.add_argument("--ip", default=None, help="Go2のIPアドレス。省略時は .env の GO2_SN で探し、見つからなければ GO2_IP")
    ap.add_argument("--aes-key", default=GO2_AES_KEY, help="AES-128 キー。省略時は .env の GO2_AES_KEY")
    ap.add_argument("--ap", action="store_true", help="Go2 自身の Wi-Fi(AP モード)に直接つないで使う")
    args = ap.parse_args()
    ip = None if args.ap else (args.ip or find_ip())
    if ip is None and not args.ap:
        ap.error("Go2 の IP がありません。.env に GO2_IP を書くか --ip で指定してください")
    trick = Go2Trick(ip, args.aes_key, victory=args.action, ap=args.ap)
    try:
        asyncio.run(try_action(trick, args.action))
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
