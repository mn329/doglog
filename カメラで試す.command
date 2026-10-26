#!/bin/bash
# 実機なしで、Mac のカメラで手の形の判定を試す(Go2 にはつながない・動かさない)
# セットアップがうまくいったかの確認と、手の形の練習に使う。ESC で終了
cd "$(dirname "$0")" || exit 1
if [ ! -x .venv/bin/python ]; then
  echo "まだ準備ができていません。先に「セットアップ.command」をダブルクリックしてください。"
  read -r -p "Enter キーでこの画面を閉じます"
  exit 1
fi
echo "Mac のカメラで試します(Go2 は動きません)。映像のウィンドウで右手を見せてください。ESC で終了。"
# Mac のカメラは Go2 と映り方が違うので、手の大きさでは絞らない(--min-size 0)
.venv/bin/python go2_trick.py --webcam 0 --hold 1.5 --min-size 0 --victory tilt "$@"
echo
read -r -p "Enter キーでこの画面を閉じます"
