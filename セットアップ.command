#!/bin/bash
# 初めて使う Mac で一度だけダブルクリックする(インターネットにつながった場所で)
# Python や VS Code は入っていなくてよい。uv が Python 3.11 ごと用意する
# 何度ダブルクリックしてもよい(できているところは飛ばす)
cd "$(dirname "$0")" || exit 1

UV="$HOME/.local/bin/uv"
MODEL_URL="https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"

fail() {
  echo
  echo "×  $1"
  echo "   この画面を写真に撮って、開発担当に送ってください。"
  echo
  read -r -p "Enter キーでこの画面を閉じます"
  exit 1
}

echo "=== Go2 デモの準備を始めます(5〜10 分かかります) ==="
echo

# 1. uv(Python と部品を入れる道具)
if [ ! -x "$UV" ]; then
  echo "[1/5] uv を入れています..."
  curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh >/dev/null || fail "uv を入れられませんでした。インターネットにつながっているか確かめてください。"
fi
echo "[1/5] uv: OK"

# 2. Python 3.11 と仮想環境(mediapipe 0.10.14 は 3.11 までしか動かない)
if [ ! -x .venv/bin/python ] || ! .venv/bin/python -c 'import sys; sys.exit(sys.version_info[:2] != (3, 11))' 2>/dev/null; then
  echo "[2/5] Python 3.11 を用意しています..."
  rm -rf .venv
  "$UV" venv --quiet --python 3.11 .venv || fail "Python を用意できませんでした。"
fi
echo "[2/5] Python: OK"

# 3. 部品(requirements-lock.txt)。ソースからのビルドは Xcode が要るので、ビルド済みのものだけ使う
echo "[3/5] 部品を入れています..."
"$UV" pip install --quiet --python .venv/bin/python --only-binary :all: --no-deps -r requirements-lock.txt || fail "部品を入れられませんでした。"
echo "[3/5] 部品: OK"

# 4. ジェスチャー認識モデル
if [ ! -s models/gesture_recognizer.task ]; then
  echo "[4/5] ジェスチャー認識モデルをダウンロードしています..."
  mkdir -p models
  curl -fL --progress-bar -o models/gesture_recognizer.task "$MODEL_URL" || { rm -f models/gesture_recognizer.task; fail "モデルをダウンロードできませんでした。"; }
fi
echo "[4/5] モデル: OK"

# 5. Go2 の接続情報(.env)。AES キーは画面に出さない
if [ ! -f .env ]; then
  echo
  echo "[5/5] Go2 の接続情報を入れます。開発担当から受け取った値を貼り付けて Enter。"
  read -r -p "  シリアル番号(GO2_SN、わからなければ空のまま Enter): " sn
  read -r -p "  Go2 の IP(例 172.20.10.4): " ip
  read -r -s -p "  AES キー(32 桁。貼り付けても画面には出ません): " key
  echo
  umask 077
  printf 'GO2_SN=%s\nGO2_IP=%s\nGO2_AES_KEY=%s\n' "$sn" "$ip" "$key" > .env
  unset key
fi
echo "[5/5] 接続情報: OK(変えるときは「IPを変える.command」)"

# AirDrop などで実行の許可が外れていることがあるので付け直す
chmod +x ./*.command 2>/dev/null

echo
echo "動作確認をしています..."
.venv/bin/python -c "import mediapipe, cv2, unitree_webrtc_connect, go2_trick" 2>/dev/null || fail "動作確認に失敗しました。"

echo
echo "○  準備OK。次からは「展示会スタート.command」をダブルクリックします。"
echo
read -r -p "Enter キーでこの画面を閉じます"
