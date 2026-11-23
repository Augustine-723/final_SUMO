# Phase 5｜校門口強化學習環境與 Baseline

場景沿用 Phase 4 的興大路 × 學府路、校門中興路、4 條行穿線及 3 份合成需求。Gymnasium 環境以 TraCI 直接控制既有號誌組合，保留行人清空與全紅規則。

| 項目 | 定義 |
|---|---|
| 觀測 | 19 維 float32。依 N/E/S/W 排列 4 個進口的停止車數與 4 個進口在途車數（各除以 40 並截於 1）；8 維目前號誌階段 one-hot、階段已用秒數除以 45（截於 1）、episode 已用時間比例、是否有行人位於行穿線。進口車數是車輛數的代理量，未依車道容量換算密度。 |
| 動作 | Discrete(2)。0 請求東西向，1 請求南北向。每 5 秒決策一次，請求受共用控制器的最短綠燈、最長綠燈及轉相規則約束。 |
| Reward | 每個決策區間取 5 個逐秒「四進口停止車數合計」之負和，單位為負車輛秒。越長的排隊等待產生越大成本。此版採車輛排隊成本以便逐步核算，行人等待另作評估指標。 |
| Episode | 現有情境注入需求 1,800 秒，再清空 900 秒，合計 540 次決策。到時回傳 truncated=True，讀取未完成車輛與行人數。 |
| 重現性 | `reset(seed=...)` 共同設定逐程出發時刻與 SUMO 種子。每個方法記錄 route XML 的 SHA-256，配對比較須相符。每個環境有獨立暫存目錄及 TraCI 連線。 |

## 控制與驗收

車輛與行人共同放行階段最短 10 秒、最長 45 秒；之後依序執行車輛續綠但行人停止進入 5 秒、黃燈 3 秒及全紅至少 2 秒。如果仍有人在行穿線上，全紅繼續到行穿線清空。兩個 Baseline 與未來 RL 共用同一套轉相邏輯；兩組綠燈含 SUMO 路網指定的讓行轉向。秒數為模擬設計假設。

- **Fixed-Time**：共同放行達 20 秒後，於決策點請求另一組綠燈。正常循環為 60 秒；發生行人清空延長時，後續請求對齊 5 秒決策點。
- **Actuated**：若對向進口合計排隊多於當前方向，請求切換；平手時續綠。持續續綠仍受最長綠燈限制。

平均車輛等待、平均旅行時間及平均行人等待分別取已完成車程與步行階段的 SUMO 輸出。通行量計算到達數；未完成數以需求總數減去到達數，包含尚未進網者。平均排隊以全 2,700 秒為分母，含清空時間。每次執行保留最大排隊、總 reward、碰撞與 teleport 計數。缺少行人 teleport 欄位的 SUMO 版本會留下空值。

在專案根目錄與已安裝 requirements.txt 的 Python 環境執行：

```bash
python -m unittest discover -s tests
python scripts/nchu_baselines.py --seeds 42 43
```

[本地 SUMO 1.24.0 的 12 筆逐回合結果](../experiments/phase5/sumo_1.24_seed42_43.csv)。GitHub Actions 也以專案環境另跑測試並保存 `nchu-phase5-baselines` 成績檔。這些是合成需求下的基準測試，沒有現地校準或訓練模型。

## Phase 6 的 Python 入口

從專案根目錄將 scripts 加入 Python 模組搜尋路徑（或將呼叫腳本放在 scripts）。使用 `from nchu_env import GateEnv`，以 `GateEnv("data/nchu/morning_synthetic.csv")` 建立環境。按照 Gymnasium 的 reset、step、close 介面操作。終局 info 會包含逐回合通行、等待、排隊、未完成數及需求雜湊。

## 使用 Antigravity 同步 host

在 host 上的 Antigravity 開啟專案目錄，貼上以下任務。可於專案完成後重用，會取得當時的 main。

> 請在這台 Ubuntu host 同步 Augustine-723/final_SUMO。先找出目前開啟的 Git 工作目錄，檢查 origin 指向正確專案，回報分支與 git status。若沒有本地副本，clone https://github.com/Augustine-723/final_SUMO.git 到空的新目錄。若已有乾淨的 main，執行 git pull --ff-only origin main。遇到未提交修改、不同分支或分歧時，保留現況並回報。同步後執行 git rev-parse HEAD 與 git rev-parse origin/main，回報兩者是否相同。啟用此專案的 Python 環境，檢查 SUMO_HOME、sumo 與 netconvert，再執行上方兩條驗收命令，回報測試結果與 results/phase5/baselines.csv 的完整路徑。
