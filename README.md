# Real-World Traffic Signal Optimization Using Deep Reinforcement Learning

強化學習期末專題。目標是在真實單一路口，比較 Fixed-Time、DQN 與 PPO 面對時變實測車流的表現。第一階段用合成十字路口驗證環境。每個進口一條直行車道，採固定時制；真實地圖、行人與模型訓練排在後續階段。

## 安裝

Ubuntu 24.04，Python 3.11，使用系統的 SUMO 與虛擬環境中的 Python 套件。先安裝 python3.11 及對應 venv，或用既有 Python 3.11 環境。圖形示範需要桌面顯示。

```bash
sudo apt-get update
sudo apt-get install sumo sumo-tools
export SUMO_HOME=/usr/share/sumo
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/check_environment.py
```

## 執行與驗收

從專案根目錄執行。每個指令都會使用 netconvert 從版本控制內的 XML 建立路網。

```bash
python scripts/demo.py
python scripts/demo.py --gui --output results/demo-gui.csv
python scripts/smoke_rl.py
```

固定時制 Demo 每秒寫出號誌相位、燈色、各進口停止車輛數及累積出發／抵達數。南北方向各每 6 秒一車，東西方向各每 8 秒一車，注入 240 秒，再觀察 120 秒。驗收條件為 140 輛車全數抵達、期末待處理數為零，且曾出現排隊與相位變換。GUI 與無圖形版本使用相同種子，應產生一致的 CSV。

SUMO-RL 測試每 5 秒送出一次動作，檢查 observation、reward、實際綠燈切換，以及第 12 次決策的 episode 結束訊號。此處的固定動作序列僅用於介面測試。真實路口後續須另建轉向衝突、全紅及行人清空限制，並驗證 RL 控制下的每次轉相。

## 實驗順序

1. 完成 [路口及資料來源查核](docs/data_sources.md)，保存分時段、分方向觀測與資料缺口。
2. 建立實際路網、轉向需求及固定時制，以獨立時段觀測檢查流量與排隊。
3. 固定合法相位、需求及模擬步數，再訓練 DQN 與 PPO。
4. 比較離峰、均衡尖峰、方向尖峰和未見需求。配對使用測試種子，記錄等待、排隊、通行量與未完成車次。行人 reward 消融視觀測完整度進行。

GitHub Actions 的 Phase 1 SUMO smoke test 可重跑上述驗收並保留兩份逐秒 CSV。
