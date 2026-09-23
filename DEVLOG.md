# DEVLOG: TinyRouter

實驗進度與觀察。最新條目在最上面。

---

## 2026-09-23（夜）：PR #7 審查修正

### 本次工作 / 執行摘要
- 新增 `pr-text.yml`（job `pr-text-hygiene`）：squash merge 的 commit 取自 PR 標題與內文，原本的 commit-hygiene 看不到。標題與內文只經 env 傳入。兩個 job 共用 `.github/scripts/check-disallowed-text.sh` 與同一份 pattern 檔。
- pilot 重用 AC2 改為 validation-only：archive 整檔照算 SHA-256，但只讀 metadata 與 validation 陣列；結果 JSON 只取 `run_name`、`config`、`training`、`logits`、`environment`，不含 `metrics`。
- 權重刪除前先確認結果 JSON、manifest、archive 三處一致；曲線索引對非重用點也驗 archive。
- 曲線索引逐點比對「訓練時記下的抽樣指紋」與「現在重抽的 `curve_sample(k, seed)`」，numpy 升版造成樣本改變時會報錯；AC2 重用點補上現算的指紋並標註來源。測試裡另釘了 5 組抽樣指紋。
- 等價判斷新增：torch 與 transformers 版本必須與目前安裝的相同（忽略 `+cpu` 這類 build 標籤）；schedule 層補獨立測試。
- TF-IDF 的 100 倍：修正 docstring，PLAN §4.1 註明 RQ3 對 TF-IDF 只報溫度校準後的訊號與 margin，多數類排除在 RQ3 外。
- scipy 列入 dev group。

### 核心發現 / 數據
- 審查實測（seed 42 val，k=100）：TF-IDF 未校準 MSP 的 AURC 在純餘弦為 0.0854、100 倍為 0.0533；entropy 為 0.1418 與 0.0539。倍數確實改變未校準訊號，所以 RQ3 不用它們。
- 關於 test 洩漏：Drew 在 PR #6（AC2 結果）看過 BERT lr 5e-5 的 test 數字。lr pilot 的選擇規則是機械式的（只比 validation 的正確筆數，再比 validation OOS 正確筆數，再取較小 lr），看過 test 數字不會影響它選出什麼，所以不構成洩漏。

### Blockers / 遇到的問題
- (無)

### Next
- [ ] PR #7 合併後，由協調者把 `pr-text-hygiene` 加進 main 的 required checks
- [ ] 其餘同上一則

### Files / Budget
- `.github/workflows/pr-text.yml`、`.github/scripts/check-disallowed-text.sh`、`tests/test_text_hygiene.py`
- API 花費：US$0

---

## 2026-09-23（晚）：步驟 3 準備，抽樣、步數協定、pilot、執行器與 ModernBERT 相容性試跑

### 本次工作 / 執行摘要
- 超參數協定寫進 `docs/PLAN.md` §4.1：步數 = max(S_min, 5 epoch 步數)，S_min 與各 encoder 的 learning rate 都只用 validation pilot 選，選定值手動填進 `configs/curve.yaml`。
- 新模組：`sampling.py`（k-shot 抽樣，OOS 用寫死對照表）、`steps.py`（步數規劃，訓練後核對實際步數）、`runs.py`（續跑規則與「等價 run」判斷）、`protocol.py`、`pilots.py`、`curves.py`、`baselines.py`；`metrics.wilson_interval`。
- logits archive 升到 format 2：加三個 split parquet 的 SHA-256；讀 format 1（AC2）時只在 dataset revision 等於鎖定值才從 `SPLIT_FILES` 補上並標註來源，不符就報錯。
- AC2 重用：BERT 若選中 lr 5e-5，曲線的 k=100 三個點與 lr pilot 的 5e-5 點都重用 AC2，前提是逐欄設定、訓練列數、步數與 seed 都對得上，且 archive 的 SHA-256 三處一致；任何一項不符就重訓。對照已提交的三個 AC2 結果 JSON 測過，S_min 取 100、200、400 或不設都判定等價。
- 基準：多數類與 TF-IDF centroid 在每個 (k, seed) 的同一份抽樣上計算，存成與 encoder 相同的 archive 與結果 JSON。
- ModernBERT 相容性試跑（`scripts/compat_trial.py`、`scripts/export_onnx.py`），結果在 `results/compat/`。只跑 50 步與一次 validation，不是正式 pilot 或曲線。

### 核心發現 / 數據
**ModernBERT 在 M4 MPS 上可以訓練。** 兩個模型同設定：batch 32、max_length 64、50 步、lr 5e-5（試跑用佔位值）、seed 42。

| | ModernBERT-base | bert-base-uncased |
|---|---:|---:|
| 參數量 | 149.7M | 109.6M |
| 每步時間（中位數，去掉前 5 步） | 0.234 s | 0.139 s |
| 第一步 | 1.13 s | 0.51 s |
| 峰值記憶體（Metal driver，採樣） | 4.80 GiB | 3.18 GiB |
| 峰值記憶體（存活 tensor，採樣） | 2.30 GiB | 1.66 GiB |
| loss（前 5 步 → 後 5 步） | 5.22 → 4.22 | 5.08 → 5.00 |
| NaN | 無 | 無 |

- attention 走 `sdpa`（MPS 上沒有 Flash Attention 與 unpadding），沒有報錯或退回警告。
- 每步時間 ModernBERT 約為 BERT 的 1.69 倍。
- **全量 5 epoch（2,385 步）預估：** 直接乘是 559 s，但同法估 BERT 得 332 s，而 AC2 實測是 822 到 886 s（每步約 0.36 s），短試跑低估約 2.6 倍（原因未查：可能是長時間執行的降頻、每個 epoch 存檔與每步記憶體採樣）。用 AC2 實測乘上 1.69 倍，**ModernBERT 全量每個 seed 約 24 分鐘**，以這個數字排時程。
- 粗估整條曲線（S_min 以 400 計，每個 seed 約 5,400 步）：ModernBERT 18 點約 2.7 小時，OOS 消融 3 次約 1.2 小時；BERT 若重用 AC2，剩 15 點約 1 小時。
- **ONNX 匯出成功，兩種 exporter 都可以。** `torch.onnx.export`（dynamo 與 TorchScript）加 ONNX Runtime 1.30 CPU，10 筆 validation（padding 到 14）最大絕對 logit 誤差 2.1e-5，argmax 10/10 一致；另換一批 3 筆、長度 25 的輸入也只差 1.1e-5，動態 batch 與長度可用。模型檔約 600 MB。TorchScript exporter 有 TracerWarning（masking 相關），實測沒出錯，但 RQ7 以 dynamo 為主。int8 量化還沒試，留給步驟 6。**RQ7 可以維持用 ModernBERT。**
- TF-IDF centroid 若直接把餘弦（0 到 1）當 logit，validation 的溫度擬合在 k ≥ 5 撞到搜尋下界 T = 0.05；改用 CLIP 慣例的 100 倍餘弦，T 落在 2.9 到 7.2 之間。這是 seed 42、43 在 scratch 目錄的檢查，不是正式基準結果。

### Blockers / 遇到的問題
- 磁碟剩約 11 GB。試跑權重與兩個 ONNX 檔（共 1.7 GB）已刪除。
- 短試跑的每步時間與長時間訓練差很多（見上），時程估計以 AC2 實測為準。
- pilot 重用 AC2 時，判斷等價要讀 AC2 結果 JSON 的 `config` 與 `training` 欄位；那個檔案裡也有 test 數字，但程式只取這兩欄，validation 數字從 archive 只讀 validation 陣列。

### Next
- [ ] `make pilot-lr`（ModernBERT 3 次 + BERT 2 次，約 1.7 小時），把選定值填進 `configs/curve.yaml`
- [ ] `make pilot-steps`（6 次 k=5 短訓練），填 S_min
- [ ] `make curve MODEL=bert`、`make curve MODEL=modernbert`、`make oos-ablation`
- [ ] logits 檔附到 GitHub Release

### Files / Budget
- 新增：`src/tinyrouter/{sampling,steps,runs,protocol,pilots,curves,baselines}.py`、`configs/{modernbert-base,curve}.yaml`、`scripts/{compat_trial,export_onnx}.py`、`results/compat/*.json`
- 相依：optional group `onnx`（onnx 1.23.0、onnxruntime 1.30.0、onnxscript 0.7.2），CI 不安裝
- API 花費：US$0

---

## 2026-09-23 (晚)：learning rate pilot（validation only）

### 本次工作 / 執行摘要
- `make pilot-lr`：k=100、seed 42，兩個 encoder 各掃 {1e-5, 2e-5, 5e-5}，只看 validation。BERT 5e-5 重用 AC2 seed 42（判定為等價 run），其餘 5 次新訓練。
- Drew 確認兩者都用 5e-5，寫入 `configs/curve.yaml`。

### 核心發現 / 數據
| 模型 | 1e-5 | 2e-5 | 5e-5 |
|---|---:|---:|---:|
| BERT（val in-scope / OOS recall） | 90.17% / 45% | 95.73% / 66% | **96.73% / 68%** |
| ModernBERT | 94.77% / 67% | 95.93% / 72% | **97.13% / 75%** |

- **兩個模型的最佳值都在掃描範圍上限。** 只能說「在 {1e-5, 2e-5, 5e-5} 中 5e-5 最好」，真正的最佳 LR 可能更高。Drew 決定照凍結的協定接受 5e-5，不在看到結果後擴充範圍；兩者停在同一個邊界，比較仍對稱。README 需寫明此限制。
- LR 太小會學不完：BERT 1e-5 在第 5 個 epoch 的 loss 仍有 0.40（5e-5 為 0.03），val in-scope 低 6.5pp。
- 單一 seed、val 只有 100 筆 OOS，ModernBERT 與 BERT 的 OOS recall 差 7 筆，不下結論。
- 每次 run 時間（M4）：BERT 約 15 分鐘，ModernBERT 約 21 到 23 分鐘。

### Blockers / 遇到的問題
- (無)

### Next
- [ ] `make pilot-steps`（k=5、seed 42，S_min ∈ {100, 200, 400}）→ Drew 確認 → 寫入 `configs/curve.yaml`
- [ ] `make curve MODEL=bert`、`make curve MODEL=modernbert`、`make oos-ablation`

### Files / Budget
- `results/pilots/lr.json`、`configs/curve.yaml`
- API 花費：US$0

---

## 2026-09-23：AC2 通過（BERT 流程正確性檢查）

### 本次工作 / 執行摘要
- `make ac2`：`bert-base-uncased` 在完整 OOS+ 訓練集（15,000 in-scope + 250 OOS）訓練 151 類，seeds 42 / 43 / 44，commit `b3a59e2`，工作目錄乾淨（`git_dirty: false`）。
- 超參數是骨架的起始值，未調參：lr 5e-5、5 epochs、batch 32、max_length 64、warmup 10%、weight decay 0.01。
- 開跑前 PR #5 經兩輪審查：續跑比對 config、三處 sha 一致才算完成、判定驗 seed 身分、過期的 `ac2.json` 開跑即刪。

### 核心發現 / 數據
| seed | val in-scope | val OOS recall | test in-scope | test OOS recall | 訓練時間 |
|---:|---:|---:|---:|---:|---:|
| 42 | 96.73% | 68.0% | 96.51% | 58.3% | 886 s |
| 43 | 96.73% | 69.0% | 96.40% | 55.8% | 847 s |
| 44 | 96.73% | 73.0% | 96.22% | 56.7% | 822 s |

- AC2 門檻 95.7%（test in-scope，三個 seed 都要過）：**PASS**。test in-scope 平均 96.38%；Larson et al. 2019（BERT、OOS+、oos-train）為 96.7%，OOS recall 59.2%，本次 56.9%。這是工程驗收，不宣稱重現原論文。
- 三個 seed 的 val in-scope 都是 2,902 / 3,000。已從 logits 檔直接重算確認不是同一份結果：三個檔 SHA-256 不同，val 上約 2.5% 的預測彼此不同，test 數字也不同；判定為巧合。
- **OOS 是弱點，符合預期**：in-scope 96%，但 test 上約 43% 的 OOS 被分進某個 intent。這正是 RQ2 到 RQ4 要處理的問題。
- 效率（M4、MPS）：參數 109.6M；每個 seed 約 14 分鐘（2,385 步）；峰值記憶體 driver 約 4.6 GB、存活 tensor 約 1.8 GB（MPS 沒有原生峰值 API，採樣值，量法見結果檔）。

### Blockers / 遇到的問題
- 磁碟可用空間從 15 GB 降到 11 GB。主要佔用是 uv 全域快取（33 GB，全機共用，非本專案）與 HF 模型快取（2.3 GB）；建議 Drew 執行 `uv cache prune`，本專案不代為刪除。
- 背景執行時包了一層 `echo exit`，所以背景任務的離開碼永遠是 0，不能拿來判斷成功與否；要看 log 裡 `make` 的輸出與 `results/ac2.json`。

### Next
- [ ] logits 檔附到 GitHub Release（manifest 已記 SHA-256）
- [ ] ModernBERT 相容性試跑（MPS 訓練、Optimum ONNX 匯出）
- [ ] k-shot 抽樣模組（每個 intent k 筆、OOS ⌈2.5k⌉ 筆）與兩條學習曲線

### Files / Budget
- `results/ac2.json`、`results/runs/bert-base-uncased-full-seed{42,43,44}.json`、`results/logits-manifest.json`
- logits 檔（不進 git）：`results/logits/bert-base-uncased-full-seed{42,43,44}.npz`，各約 4.8 MB
- API 花費：US$0
