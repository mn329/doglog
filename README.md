# go2-hand-follow

GO2 ロボット用のハンドジェスチャー認識・追従システム

## セットアップ

### いちばん簡単な方法(エンジニアでなくても)

展示会でデモを動かす人は、展示会に要るファイルだけを入れた [dotlog-inc/doglog](https://github.com/dotlog-inc/doglog)（Private）を使う。`sync_expo.sh` でこちらからコピーする。どちらも `セットアップ.command` をダブルクリックすれば準備できる。Python がなくても、uv が Python 3.11・部品(`requirements-lock.txt`)・ジェスチャー認識モデルを入れ、接続情報(`.env`)を聞いてくる。手順は [doc/展示会マニュアル.md](doc/展示会マニュアル.md) の「事前準備」。

### 必要なもの(手で入れる場合)
- Python 3.11(mediapipe 0.10.14 のため)
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
| チョキ | `v` | 首かしげ(体を少し横に傾ける。`--victory` で変えられる)。座っているときは何もしない(写真のピースで立ち上がらないように) |
| — | スペース | 止めて一時停止(ジェスチャーに反応しなくなる)。もう一度押すと再開 |

展示会では `展示会スタート.command` をダブルクリックして起動する。人混みで誤作動しないよう、小さく映る(遠い)手を無視し(`--min-size`)、同じ手で 1.5 秒続けたときだけ動く(`--hold`)。手順は [doc/展示会マニュアル.md](doc/展示会マニュアル.md)。

## ファイル説明

- `config.py` - `.env` から Go2 の IP と AES キーを読む
- `hand_control.py` - ハンドジェスチャー認識とコマンド計算の共通処理
- `models/gesture_recognizer.task` - ジェスチャー認識モデル(上記のコマンドでダウンロード)
- `webcam_demo.py` - ウェブカメラ版のデモンストレーション
- `go2_follow.py` - Go2ロボット実機版
- `go2_trick.py` - お手・おすわりデモ
- `wander.py` - 部屋の中をうろうろ歩かせる。本体の障害物回避で止まったら、向きを変えてまた前進する(先に `check_avoid.py` で回避が使えることを確かめる)。回避が効かないときは `--lidar` で、LiDAR の点群を見て止まる。`--lidar --dry-run` なら動かさずに前方の物までの距離を表示する
- `try_action.py` - チョキでさせる動き(ハートの代わり)の候補を実機で 1 つずつ試す
- `カメラで試す.command` - 実機なしで、Mac のカメラで手の形の判定を試す(前日までの確認・練習)
- `展示会スタート.command` - 展示会用の起動(ダブルクリック)。ルーターやスマホのテザリングで Go2 と Mac をつなぐとき
- `展示会スタート(Go2直結).command` - 同じく、Mac を Go2 自身の Wi-Fi(AP モード)に直接つなぐとき(`--ap`)
- `seek.py` - その場で見回して、人を見つけたら近寄って止まり、手の形で芸をする(ジェスチャーと芸は `go2_trick.py` と同じ。両手のパーで止まれ、パーを左右に傾けると体も傾く。歩き回らないので回避も LiDAR も使わない)。`--webcam 0` で Mac のカメラで試せる。人物検出モデルの準備はファイル先頭を参照
- `check_avoid.py` - 本体の障害物回避と LiDAR が WebRTC から使えるかを確かめる(試験用。`--dry-run` で動かさずに状態だけ見る)
