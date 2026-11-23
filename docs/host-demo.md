# Ubuntu host 排錯與校門口模擬展示

本次 host 紀錄：Python 3.11、Conda 環境 sumo311。兩個 unittest 均通過（9.439 秒），並已產生 results/phase5/baselines.csv。以下展示可直接沿用這個環境。

| 現象 | 原因與處理 | 目前狀態 |
|---|---|---|
| No module named traci | 在專案環境執行 python -m pip install -r requirements.txt | 已可匯入 |
| ml-dtypes 要求 NumPy 2；專案指定 1.26.4 | 建立獨立的 sumo311，保留原 py311 做其他作業 | 專案測試已在獨立環境通過 |
| FileNotFoundError: netconvert | Ubuntu 的 sumo 套件提供路網轉換工具 | 已能建網 |
| SUMO_HOME 警告 | 模擬採用內建型別，略過 XML 驗證。若 /usr/share/sumo/data/xsd 存在，可設定下列環境變數 | 日誌仍有警告，模擬已完成 |
| Retrying in 1 seconds | TraCI 啟動模擬時等待連線，本次隨後成功 | baseline 已輸出 |

## 開啟互動展示

在登入 Ubuntu 圖形桌面的終端機執行。第一次建立場景時會跑一次無視窗模擬，產生可重播的網路與路線檔。第二個指令以每步 100 毫秒播放，可在視窗中暫停或調整 Delay。

```bash
cd ~/Desktop/SUMO/final_SUMO
conda activate sumo311
if [ -d /usr/share/sumo/data/xsd ]; then export SUMO_HOME=/usr/share/sumo; fi
python scripts/nchu_scenario.py data/nchu/morning_synthetic.csv --seed 42 --out results/nchu/morning && \
  sumo-gui -c results/nchu/morning/gate.sumocfg --seed 42 --end 2700 --delay 100 --start
```

此展示使用興大路 × 學府路的地圖幾何、合成早峰需求與固定時制。往路口中央放大即可觀察排隊、轉向與行穿線。若終端機無法連上 display，請從有圖形桌面的 host 終端機啟動。觀看已訓練 DQN 控制器的實際軌跡，可直接播放 `submission/Demo.mp4`。
