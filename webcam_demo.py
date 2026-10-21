import sys
import time

import cv2
from hand_control import detect, compute_cmd, close, draw_hands


def main():
    """ウェブカメラからのハンドジェスチャー認識デモ"""
    # カメラ番号は引数で指定（例: python webcam_demo.py 1）。省略時は 0
    camera_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)

    if not cap.isOpened():
        print(f"カメラ {camera_index} を開けません")
        return

    fps = 0.0
    last_t = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ハンドジェスチャー検出
        hands = detect(frame)
        now = time.time()
        fps += 0.1 * (1.0 / max(now - last_t, 1e-3) - fps)   # 処理fps(移動平均)
        last_t = now

        # ロボット制御コマンド計算
        cmd = compute_cmd(hands, frame.shape)

        # 検出結果を描画。操作に使う右手は R の印と判定、ほかの手は灰色
        draw_hands(frame, hands, (cmd['gesture'] or '-').upper())

        # コマンド表示
        cv2.putText(
            frame,
            f"Forward: {cmd['forward']:.2f}, Turn: {cmd['turn']:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        if cmd['gesture']:
            cv2.putText(
                frame,
                f"Gesture: {cmd['gesture']}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        cv2.putText(frame, f"{fps:.0f} fps", (10, 110), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 0), 2)

        # フレーム表示
        cv2.imshow('Hand Gesture Demo', frame)

        # ESCキーで終了
        if cv2.waitKey(1) & 0xFF == 27:
            break

    close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
