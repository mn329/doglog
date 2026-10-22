"""
Go2 の接続情報を .env から読む

IP や AES キーはリポジトリに入れないよう、このフォルダの .env に書く(.env.example をコピーして作る)。
すでに環境変数に設定されている値のほうを優先する。
GO2_SN を書いておくと、find_ip() でシリアル番号から今の IP を探せる(DHCP で IP が変わっても .env を直さなくてよい)。
"""
import os

ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")


def load_env(path=ENV_PATH):
    """.env の「名前=値」を環境変数に読み込む。ファイルが無ければ何もしない"""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_env()
GO2_IP = os.environ.get("GO2_IP") or None
GO2_AES_KEY = os.environ.get("GO2_AES_KEY") or None
GO2_SN = os.environ.get("GO2_SN") or None


def find_ip(sn=GO2_SN, fallback=GO2_IP):
    """
    シリアル番号 sn の Go2 を LAN 上で探して IP を返す。見つからなければ fallback を返す

    同じ LAN にほかの Unitree ロボットがいても、シリアル番号で自分の機体だけを選ぶ。
    """
    if not sn:
        return fallback
    from unitree_webrtc_connect import discover_ip_sn
    found = discover_ip_sn(sn=sn).get(sn)
    if found:
        print(f"Go2 ({sn}) を {found} で見つけました")
        return found
    print(f"Go2 ({sn}) が見つかりませんでした。.env の GO2_IP ({fallback}) を使います")
    return fallback
