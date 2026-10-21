# go2-hand-follow

GO2 ロボット用のハンドジェスチャー認識・追従システム

## セットアップ

### 必要なもの
- Python 3.8以上
- Go2 ロボット

### インストール

```bash
pip install -r requirements.txt
# ジェスチャー認識モデル(MediaPipe Gesture Recognizer、約8MB)
mkdir -p models && curl -L -o models/gesture_recognizer.task https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task
```

## 使用方法

### ウェブカメラデモ

```bash
python webcam_demo.py
```

パソコンのウェブカメラを使用してハンドジェスチャーを認識します。

### Go2ロボット実機での実行

```bash
# 最初に一度だけ: Go2 の IP と AES キーを .env に書く(.env はリポジトリに入らない)
cp .env.example .env
# まず --dry-run で、ロボットを動かさずに検出と指令値の表示だけ確認する
python go2_follow.py --dry-run
# 問題なければ(広い場所で、すぐ止められる状態で)
python go2_follow.py
# 手の検出方法を比べるために、受信した映像を recordings/ に保存する
python go2_follow.py --dry-run --record
```

`.env` の値の代わりに `--ip` / `--aes-key` で指定することもできます。
IP の調べ方、AES キーの取得、エラーへの対処は [doc/実機接続手順.md](doc/実機接続手順.md) を参照してください。
今後の予定は [doc/今後の展望.md](doc/今後の展望.md) にまとめています。
操作は **右手** で行います(左手は灰色で表示され、指示には使いません。画面で `R` の印が付いた手が操作に使われている手です)。
右手を開くと追従、グー(左右どちらの手でも)または右手を見失うと停止、ESC で停止して終了します。

### お手・おすわりデモ

```bash
# 実機なしで Mac のカメラで判定を試す(ロボットは動かない)
python go2_trick.py --webcam 0
# 実機(まず --dry-run で確認)
python go2_trick.py --dry-run
python go2_trick.py
```

右手のジェスチャーを1秒続けると指示になります(左手は使いません)。
形は MediaPipe の学習済みモデル(Gesture Recognizer)で判定し、分類できなかったときだけ指の角度で判定します。画面の `R` の横に分類結果と確からしさが出ます。動作中の4秒間は次の指示を受け付けません。

| ジェスチャー | キー | 動作 |
|---|---|---|
| パー | `h` | その場でお手(Hello、右前足)。Go2 は動かないので手の位置は自分で合わせる(高い位置の手には足が届かない)。座っていれば先に立つ |
| 人差し指だけ(中指は軽く曲げればよい) | `s` | おすわり(Sit) |
| グー | `u` | 立つ(RiseSit) |
| チョキ | `v` | ハート(FingerHeart)。座っていれば先に立つ |

## ファイル説明

- `config.py` - `.env` から Go2 の IP と AES キーを読む
- `hand_control.py` - ハンドジェスチャー認識とコマンド計算の共通処理
- `models/gesture_recognizer.task` - ジェスチャー認識モデル(上記のコマンドでダウンロード)
- `webcam_demo.py` - ウェブカメラ版のデモンストレーション
- `go2_follow.py` - Go2ロボット実機版
- `go2_trick.py` - お手・おすわりデモ
