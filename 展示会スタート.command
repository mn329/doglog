#!/bin/bash
# 展示会用: ダブルクリックで お手・おすわりデモ を起動する
# Go2 と Mac を同じルーター(またはスマホのテザリング)につないで使う。Go2 の Wi-Fi に直接つなぐときは「展示会スタート(Go2直結).command」
# 値の意味は doc/展示会マニュアル.md を参照。会場で調整したらここを書き換える(2つのファイルとも)
cd "$(dirname "$0")" || exit 1
if [ ! -x .venv/bin/python ] || [ ! -f .env ]; then
  echo "まだ準備ができていません。先に「セットアップ.command」をダブルクリックしてください。"
  read -r -p "Enter キーでこの画面を閉じます"
  exit 1
fi
.venv/bin/python go2_trick.py --hold 1.5 --min-size 0.12 --victory tilt "$@"
echo
read -r -p "Enter キーでこの画面を閉じます"
