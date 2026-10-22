#!/bin/bash
# 展示会用: Mac を Go2 自身の Wi-Fi(AP モード)に直接つないで、お手・おすわりデモを起動する
# ルーターやテザリングを使うときは「展示会スタート.command」のほうを使う
# 値の意味は doc/展示会マニュアル.md を参照。会場で調整したらここを書き換える(2つのファイルとも)
cd "$(dirname "$0")" || exit 1
source .venv/bin/activate
python go2_trick.py --ap --hold 1.5 --min-size 0.12 --victory tilt "$@"
echo
read -r -p "Enter キーでこの画面を閉じます"
