"""
Go2 の接続情報を .env から読む

IP や AES キーはリポジトリに入れないよう、このフォルダの .env に書く(.env.example をコピーして作る)。
すでに環境変数に設定されている値のほうを優先する。
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
