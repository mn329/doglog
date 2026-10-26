#!/bin/bash
# Go2 の IP が変わったときにダブルクリックする。.env の GO2_IP だけを書き換える
cd "$(dirname "$0")" || exit 1

if [ ! -f .env ]; then
  echo "まだ準備ができていません。先に「セットアップ.command」をダブルクリックしてください。"
  read -r -p "Enter キーでこの画面を閉じます"
  exit 1
fi

now=$(grep '^GO2_IP=' .env | cut -d= -f2)
echo "今の Go2 の IP: ${now:-(なし)}"
echo "Unitree Go アプリで Go2 の IP を調べて入れてください(テザリングなら 172.20.10.x)。"
read -r -p "新しい IP: " ip

if ! [[ "$ip" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "IP の形になっていません(例 172.20.10.4)。変えずに終わります。"
  read -r -p "Enter キーでこの画面を閉じます"
  exit 1
fi

grep -v '^GO2_IP=' .env > .env.tmp && echo "GO2_IP=$ip" >> .env.tmp && chmod 600 .env.tmp && mv .env.tmp .env
echo "GO2_IP を $ip にしました。"
read -r -p "Enter キーでこの画面を閉じます"
