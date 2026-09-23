# DEVLOG: TinyRouter

實驗進度與觀察。最新條目在最上面。

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
