# Real-World Traffic Signal Optimization Using Deep Reinforcement Learning

強化學習期末專題。正式研究路口為臺中市南區興大路 × 學府路（國立中興大學校門口）。目標是在此路口，比較 Fixed-Time、DQN 與 PPO 面對時變實測車流的表現。第一階段用合成十字路口驗證環境。每個進口一條直行車道，採固定時制；真實地圖、行人與模型訓練排在後續階段。

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

1. 已定位[中興大學校門口與 OSM 路網](docs/data_sources.md)。確認各進口可通行車種、時段與號誌時相。
2. [校門口 Phase 3／4](docs/nchu_simulation.md) 已提供 15 分鐘觀測匯入、三組合成需求、OSM 道路幾何、行穿線與可執行的 SUMO-GUI 場景。本地實測需求、現行時制及校準仍待補。
3. 已完成 [Phase 5 Gymnasium 環境與 Baseline](docs/phase5.md)，可執行固定時制與排隊感應控制。[Phase 6 訓練與模型重播](docs/phase6.md)使用相同環境，比較 DQN、PPO 及兩項單因素試驗。
4. [Phase 7 配對評估及 Phase 8 繳交成果](docs/final.md)涵蓋 1,000 次模擬、兩項 PPO 消融、英文報告、簡報與展示影片。

GitHub Actions 的 Phase 1 SUMO smoke test 可重跑上述驗收並保留兩份逐秒 CSV。

## 既有資料處理範例

臺北 EI035 的[調查來源](docs/ei035_sources.md)、[需求轉檔及核對結果](docs/demand.md)保留供重現。現有 `scripts/process_demand.py` 與 `data/processed/ei035_*` 專屬臺北案例。
