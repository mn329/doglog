#!/bin/bash
# 展示会用のリポジトリ(dotlog-inc/doglog)に、展示会デモに要るファイルだけをコピーする
# 使い方: ./sync_expo.sh [展示会リポジトリのフォルダ](省略時は ../doglog-expo)
# コピーしたあと、展示会リポジトリで git diff を見てからコミット・push する
# README.md・.gitignore・.env.example は展示会リポジトリ側のものを使うのでコピーしない
set -e
cd "$(dirname "$0")"
DEST="${1:-../doglog-expo}"
[ -d "$DEST/.git" ] || { echo "$DEST が git のフォルダではありません"; exit 1; }

FILES=(
  go2_trick.py go2_follow.py hand_control.py config.py try_action.py
  requirements-lock.txt
  セットアップ.command IPを変える.command カメラで試す.command 展示会スタート.command "展示会スタート(Go2直結).command"
  doc/展示会マニュアル.md doc/展示会マニュアル.html doc/実機接続手順.md
)
mkdir -p "$DEST/doc"
for f in "${FILES[@]}"; do
  cp -p "$f" "$DEST/$f"
done
echo "${#FILES[@]} 個のファイルを $DEST にコピーしました"
git -C "$DEST" status --short
