# Phase 6｜在中興大學校門口訓練 DQN 與 PPO

場景沿用興大路 × 學府路的四臂路網及 Phase 5 的行人清空控制。核心實驗使用相同的 19 維觀測、兩個合法綠燈請求與車輛排隊獎勵。需求均為合成。研究問題是策略能否從不同出發時刻的互動經驗中，學會兼顧各進口的排隊。

## 訓練設計

| 項目 | 設定 |
|---|---|
| 預算 | 每模型 30 個完整 episode，每回合 540 次決策，總計 16,200 個 5 秒決策。每個 episode 含 1,800 秒需求與 900 秒清空。 |
| 交通情境 | 每三回合把離峰、早峰、晚峰各跑一次，三者順序由訓練種子產生。每個情境出現 10 次。 |
| 配對 | DQN 與 PPO 各用 11、22、33 三個訓練種子。同種子的情境序列與每回合路線檔 SHA-256 逐一核對。 |
| 網路 | 使用 Stable-Baselines3 的 MlpPolicy，兩層各 64 個單元。DQN 使用預設 ReLU；PPO 各自建立 actor、critic，使用預設 Tanh。 |
| DQN | 學習率 0.0005，折扣 0.98，回放容量 20,000，前 540 步收集經驗。每 4 步做 1 次 batch 64 更新，目標網路每 540 步更新。 |
| PPO | 學習率 0.0003，折扣 0.98，每批 rollout 180 步，batch 60，每批更新 5 輪，GAE 0.95，clip 0.2。 |
| 單因素試驗 | PPO 固定種子 11、同預算與同一條訓練需求序列。low_lr 僅把學習率改成 0.0001；pedestrian 僅在原 reward 加入本決策區間新增行人停止人秒乘以 −0.25。 |
| 模型選擇 | 保存中點和最終模型。使用事先指定的最終模型進行確定性驗證；未按驗證分數挑 checkpoint。 |
| 驗證 | 每個最終模型各跑三種需求 × 兩個新交通種子（101、102）。Fixed-Time 與 Actuated 在相同路線檔重跑。這是相同需求輪廓換出發時刻的驗證；新需求輪廓與正式測試留給 Phase 7。 |

每個訓練目錄包含 `run.json`（種子、超參數、版本、輸入及模型雜湊）、`training.csv`（逐回合目標、排隊、等待與安全指標）、`midpoint.zip`、`final.zip`、`validation.csv`。模型為 SB3 推論 checkpoint；精確中斷續訓還需要當時的模擬器、隨機狀態及 DQN replay buffer。各驗證回合從磁碟載入 final.zip；重新載入後逐張量核對 policy 參數。

主曲線將每 3 個回合的原始排隊 reward 平均，每點皆涵蓋三種情境，淡線是各訓練種子，主線是跨種子平均。分數越接近零表示進口排隊車輛秒越少。不同 reward 的單因素比較請以共同的實體指標（車輛等待、行人停止、排隊等）解讀。

## 在 Ubuntu host 重現

先依 README 安裝 SUMO 及啟用 Python 3.11 虛擬環境，再安裝 CPU 訓練套件。在專案根目錄執行：

```bash
python -m pip install 'torch==2.5.1' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-training.txt
python scripts/nchu_train.py DQN --seed 11 --out results/phase6/policy-DQN-core-11
python scripts/nchu_train.py PPO --seed 11 --out results/phase6/policy-PPO-core-11
python scripts/nchu_replay.py results/phase6/policy-PPO-core-11 --seed 101
```

訓練指令也接受 `--pilot`（3 回合）及 PPO 的 `--variant low_lr` 或 `--variant pedestrian`。每次選新的輸出目錄，成功完成時才會寫入正式路徑。所有種子與單因素比較可從 GitHub Actions 的 **NCHU Phase 6 training** 重跑。完整產物解壓後即可直接重播；可用 `gh run download` 取回訓練檔案。一般 git pull 會同步程式與報告，模型下載為另一步。

## 本次執行結果

[訓練與驗證記錄](https://github.com/Augustine-723/final_SUMO/actions/runs/45134925800) · [模型、逐回合紀錄與學習曲線](https://github.com/Augustine-723/final_SUMO/actions/runs/45134925800/artifacts/14800105135) · [逐情境摘要](../experiments/phase6/validation_summary.csv)

8 個模型均完成訓練，包含 6 個核心模型與 2 個單種子試驗。驗證共 48 個模型回合與 12 個基準回合。全部車輛及行人抵達，記錄到的碰撞與車輛 teleport 均為零。以車輛平均等待排序，三種情境皆為感應控制、DQN、固定時制、核心 PPO。晚峰 DQN 的行人平均停止時間為 11.06 秒，感應控制為 12.43 秒。此處使用同一需求輪廓的兩個驗證交通種子；正式泛化比較待 Phase 7。

單一種子加入行人停止成本的 PPO 在三種情境得到與 DQN 相同的彙總指標。相同成績本身無法辨別模型是否學會相同策略；若需解釋控制行為，應比較逐步動作、實際放行階段與切換次數。

日後在 host 同步程式後，可用已登入的 GitHub CLI 取得本次模型，並重播相同驗證路線：

```bash
gh run download 45134925800 -R Augustine-723/final_SUMO -n nchu-phase6-models-and-results -D results/phase6-run45134925800
python scripts/nchu_replay.py results/phase6-run45134925800/policy-DQN-core-11 --seed 101
```
