# DEVLOG: TinyRouter

實驗進度與觀察。最新條目在最上面。

---

## 2026-09-29（深夜）：步驟 4 後半，RQ2 到 RQ4 離線分析

### 本次工作 / 執行摘要
- 新增 `make analysis`（`analysis_run.py`、`analysis.py`、`selective.py`、`haiku.py`）：只讀 75 個 logits archive 與 Haiku 8,600 筆逐筆預測，不訓練、不打 API。輸出 `results/analysis/summary.json`、`curves.json`、`haiku.json`，兩次執行逐位元組相同（沒有任何亂數）。
- 開跑前比照 completeness.py 驗證：四個 curve index 重跑 `verify_index`（BERT、ModernBERT 各 18、消融 3、基準 36；BERT k=100 即 AC2 三個 run），archive 總數 75 與 Haiku 列數 8,600 都是字面值；每個 split 內所有 archive 的 labels 必須相同，Haiku 的 `gold_intent` 必須與之逐列相等。輸出先寫暫存檔、讀回檢查（25 組、每組 3 seeds、兩個 Haiku split）才換上，最後印 `completed analysis (75/75 archives, 8600/8600 llm rows, 25 groups)`。
- 只用 validation 決定、程式層級守住（傳 test 就 `LeakageError`，有測試）：溫度、8 類聚合（validation 8 類準確率高者，平手取 argmax）、每個訊號在每個目標 risk 的門檻、hybrid 用哪個訊號（validation coverage 大者，再比 validation AURC）、操作曲線上的門檻。另有一條測試把 test labels 打亂，確認所有選擇都不變。
- 門檻：候選是 validation 上所有不同的分數，取「selective risk 的單側 95% Wilson 上界（z = 1.645）≤ 目標」中 coverage 最大者；同分的列永遠一起收或一起退。沒有門檻可行時，router 全部交給 LLM（coverage 0），並記 `feasible: false`。目標 2% 與 5% 兩檔。
- 訊號（越大越有信心）：`msp`、`entropy`（負熵）在 T = 1；`msp_t` 在擬合的 T；`margin` 是擬合 T 下前兩名 log 機率的差，argmax 聚合時等於 (z1 − z2)/T，排序與原始分數 margin 相同。argmax 聚合的訊號取自 151 類分布，summed 取自 8 類分布。TF-IDF 只報 `msp_t` 與 `margin`、只報校準後 ECE；多數類不擬合 T、不報訊號（PLAN §4.1）。
- oracle：只把小模型 8 類判錯的查詢交給 Haiku。錯誤回收率 = 被交出去且 Haiku 救回的小模型錯誤 ÷ 小模型全部錯誤；「抓到的可回收錯誤」= 同一個分子 ÷ Haiku 會救回的小模型錯誤（oracle 的分子）。
- 並列：LLM-only、small-only、hybrid（每個目標 risk）、oracle。std 是 3 seeds 的樣本標準差（ddof=1）。
- 舊專案解析規則（子字串、取集合迭代到的第一個）重做成 `legacy_parse`，順序改用 `AGENTS` 固定，並另計「含兩個以上標籤、在舊程式裡答案不固定」的列數。

### 核心發現 / 數據
（全部取自 `results/analysis/*.json`，test，百分比，mean ± std）
- **Haiku 4.5 zero-shot**：8 類準確率 82.1，OOS recall 56.8（Wilson 95% 53.7 到 59.8），每 1K 查詢 US$0.369。8,600 筆回覆全是 8 個標籤之一，parse_failed 0 列，新舊解析規則不一致 0 列，順序相依 0 列。
- **ModernBERT k=100（主模型，validation 選 argmax）**：small-only 91.9 ± 0.1，OOS recall 61.1 ± 0.4；oracle 95.1 ± 0.1、LLM 呼叫 8.1 ± 0.1。
- **validation 選的門檻在 test 上守不住目標 risk**。目標 2%：ModernBERT k=100 test selective risk 7.3 ± 0.9（coverage 98.7）、k=25 為 5.3 ± 0.5；目標 5% 在 k ≥ 25 時 coverage 幾乎 100%，test selective risk 8.0 到 13.5。原因是 CLINC150 的 validation 只有 100/3,100（3.2%）OOS，test 是 1,000/5,500（18.2%）；小模型的錯誤大多是 OOS，validation 上錯誤率本來就低，Wilson 上界再保守也看不到 test 的 OOS 比例。這不是程式錯，是協定（只用 validation）在這個資料集上的已知代價，需要 Drew 決定要不要處理（見 Blockers）。
- 小資料量時 hybrid 的價值最清楚：ModernBERT k=10 在目標 2% 下 88.0 ± 0.4，LLM 呼叫 23.9 ± 3.8，高於 Haiku 單獨的 82.1 與 small-only 的 81.5 ± 0.7。
- **OOS 0 筆消融**：small-only OOS recall 0.0（從不預測 oos），但只靠不確定性當 OOS 分數時 AUROC 反而更高：msp_t 98.0 ± 0.1、AUPRC 92.2 ± 0.4，對照同點 OOS 250 筆的 91.9 ± 0.5、72.0 ± 0.7。注意這個比較對有 OOS 訓練的模型不利：它有信心地預測成 oos 的查詢，在「低信心 = OOS」的分數下反而被當成最不像 OOS。以 router 實際行為比較：消融在目標 2% 下 hybrid OOS recall 40.6 ± 1.0、高信心 OOS 誤派 37.7 ± 2.9；目標 5% 時門檻放到全收，高信心 OOS 誤派 99.3 ± 0.2。
- 聚合方式：ModernBERT k ≤ 10 三個 seed 都選 summed，k ≥ 50 都選 argmax；BERT 在 k=10、50、100 三個 seed 選得不一致，兩者差距都在 1pp 內。兩種的 test 數字都在 JSON。
- 多數類：argmax 聚合永遠猜 oos（k-shot 樣本裡 oos 是單一最大 intent），summed 永遠猜 finance；validation 選 summed，test 8 類 20.7。

### Blockers / 遇到的問題
- **待 Drew 決定**：validation 與 test 的 OOS 比例差 5.7 倍，使「只用 validation 選門檻」在 test 上系統性地超出目標 risk。可選：(a) 照實報，當成部署時要重新校準門檻的證據；(b) 另報一個敏感度版本，validation 的 OOS 列依「假設的部署 OOS 比例」加權（比例是事先宣告的參數，不從 test 估），這會動到 PLAN 的協定，所以這次沒做。
- 「不確定性當 OOS 分數」對有 OOS 訓練的模型不公平（上面的消融段）；若要公平比較，可另加「預測為 oos 或低信心」的組合分數。這次照任務只報四個訊號。

### Next
- [ ] Drew 決定上面兩點
- [ ] 步驟 5：README 與圖（risk-coverage、操作曲線、reliability），成本表（RQ5），數字由 `make report` 從 `results/analysis/*.json` 產生

### Files / Budget
- 新增：`src/tinyrouter/analysis.py`、`analysis_run.py`、`selective.py`、`haiku.py`；`tests/test_analysis.py`、`test_analysis_run.py`、`test_selective.py`、`test_haiku.py`；`results/analysis/summary.json`、`curves.json`、`haiku.json`
- 修改：`Makefile`（`make analysis`）、`README.md`（指令表與目錄）、`DEVLOG.md`
- API 花費：US$0

---

## 2026-09-29（夜）：llm-smoke 第一次實跑失敗，temperature 改走 extra_body

### 本次工作 / 執行摘要
- 事故：PR #14 合併後第一次 `make llm-smoke`，每筆呼叫都在送出前就失敗，錯誤是 `TypeError: Messages.create() got an unexpected keyword argument 'temperature'`。沒有請求到達 API，花費 US$0，輸出裡沒有 key。
- 原因：anthropic 1.x 把 `temperature`、`top_p`、`top_k` 從 `messages.create()` 的簽名拿掉了，但 API 本身沒有移除，Haiku 4.5 仍然接受。測試用的假 client 什麼參數都收，所以測不出和真 SDK 簽名不符。
- 修法：`request_params` 改成 `extra_body={"temperature": 0.0}`，SDK 會把它原樣併進 request JSON（claude-api skill，`python/claude-api/sdk-upgrade.md` Step 6 的建議：模型仍接受、程式又依賴這個設定時，移到 extra_body，不要刪）。identity 仍記 temperature 0，語意沒變，所以身分雜湊不變，smoke journal 裡的 failed 紀錄下次會照常重試。
- `count_tokens` 只送 model、system、messages，沒有同樣的問題；它的參數也抽成 `count_tokens_params()`，跟 `request_params()` 一樣是唯一產生參數的地方。
- 疤痕變腳本：新增 `test_the_arguments_we_send_fit_the_installed_sdk_signatures`，把實際送出的 kwargs `bind` 到真 SDK 的 `Messages.create` 與 `Messages.count_tokens` 簽名上，不需要網路或 key；另一條測試確認假 client 收到的參數正是這兩個函式產生的，簽名測試因此涵蓋實際送出的內容。

### 核心發現 / 數據
- (無實跑數據)。定價仍是 Haiku 4.5 輸入 US$1、輸出 US$5 per MTok。

### Blockers / 遇到的問題
- (無)

### Next
- [ ] 合併後重跑 `make llm-smoke`

### Files / Budget
- `src/tinyrouter/llm.py`、`tests/test_llm.py`、`tests/llm_fakes.py`
- API 花費：US$0

---

## 2026-09-29（晚）：PR #14 審查修正（4 medium、6 low）

### 本次工作 / 執行摘要
- 單一執行鎖：`results/llm/<name>.lock` 用 `flock(LOCK_EX|LOCK_NB)`，run 與 finalize 都要拿鎖；拿不到就 exit 2。審查實測兩個行程並行時各自花到上限，合計到 144%。
- parser：`oos` 以單字邊界也算候選，候選數 ≥ 2 就判 oos 並標 `parse_failed`。`"oos (not travel_agent)"` 原本會被判成 travel_agent，是 RQ2 最危險的靜默誤派。
- journal 尾行：先修尾再載入。完整但缺換行的尾行保留並補換行（那筆已付費），只有無法解析的半行才截掉。原本是先載入再截尾，會把記憶體裡算成功、磁碟上已刪掉的那筆再付一次費。
- `count_tokens` 走同一套重試與遮罩（529 會重試、錯誤字串遮 key）。
- `settle` 把非預期例外也記成 failed 並停跑；Ctrl-C 時先等在途呼叫回來並記帳再往外拋；回應的 input 或 output 超過上界就停跑（`bound violated`，exit 1）。
- summary 新增 `parser_sha256`（重新解析用的是當下的 parser，而 parser 不在 identity 裡）與 `cap_scope`。
- 補五個守門的測試：非 ASCII 的上界（bytes 不是字元數）、journal 混入別的身分、gold 被改、只改 manifest 的 SHA、同一列成功兩次。

### 核心發現 / 數據
- **上限的範圍是「每個身分 × 每個 target」**：smoke 與改 prompt 之前的花費不算在內。AC6 的總花費要手動把 `results/llm/haiku-8way.json` 與 smoke 的 summary 加總。
- 上限看不到的部分：client 端逾時但伺服器已計費的請求，重試時沿用同一份預留；每次這種逾時最多多出一個單筆上界，並行時一波約 workers 個上界。
- (無實跑數據)

### Blockers / 遇到的問題
- (無)

### Next
- [ ] 下一個分析 PR 同時報新舊兩種解析規則（舊：子字串比對取第一個命中），以及兩者判定不一致的列數

### Files / Budget
- `src/tinyrouter/llm.py`、`src/tinyrouter/llm_run.py`、`tests/test_llm.py`、`tests/test_llm_run.py`、`tests/llm_fakes.py`、`.gitignore`
- API 花費：US$0

---

## 2026-09-29：步驟 4 之一，Haiku 8 類執行器（尚未實跑）

### 本次工作 / 執行摘要
- 新增 `src/tinyrouter/llm_run.py` 與 `make llm`、`make llm-smoke`、`make verify-llm`：validation 3,100 + test 5,500 逐筆呼叫 Haiku，逐筆寫入 journal（split、index、query 的 SHA-256、gold intent 與 agent、原始回覆、解析後標籤、`parse_failed`、tokens、花費、延遲、嘗試次數、request id）。
- query 原文不存，只存 SHA-256：資料集公開且鎖定 revision，雜湊足以證明紀錄對應哪一列，續跑時也拿它比對資料有沒有變。
- 續跑：journal 檔名含身分雜湊（模型、system prompt 的 SHA-256、temperature、max_tokens、user 內容格式）。身分一變就換新檔，舊回覆不會被拿來用；同身分重跑只補沒有成功紀錄的列。最後一行寫到一半（當機）會被截掉重做。
- 成本上限：開跑前用 token counting 取 prompt 的基礎 token 數，印出上界估計；每筆開打前預留「基礎 + 每個 byte 算一個 token 的輸入、max_tokens 的輸出」的上界，累計花費（含之前幾次）加上在途預留會超過 `--max-usd` 就不開新呼叫。實際花費依回傳的 usage 計算。
- 重試改由程式自己做（SDK 的 `max_retries=0`）：429、5xx、408、409、連線錯誤指數退避並尊重 retry-after，5 次用盡記為失敗、整體 exit 1；400、401、403、404 不重試，而且停止開新呼叫。
- 完成判定比照 `completeness.py`：預測檔從磁碟讀回，(split, index) 恰為預期集合且各一次、身分一致，SHA-256 在檔案、summary、`results/llm-manifest.json` 三處相同，才印 `completed 8600/8600 llm predictions`。預期列數寫成字面值。
- CI 的 test job 改裝 `--group llm`，讓重試與 key 遮蔽的測試用 SDK 真正的例外類別。仍不設 key、不打 API。
- `classify` 原本把任何例外都當可重試，改為依錯誤類型判斷。

### 核心發現 / 數據
- system prompt 與 cost-aware-hybrid-router `src/routers/llm_router.py` 逐位元組相同（SHA-256 `560d22c5...5df574`，測試釘住）。模型、temperature 0、max_tokens 20、query 原樣當唯一 user 訊息，都與舊專案相同。
- 解析規則與舊專案**不同**：舊版對回覆做子字串比對、取集合迭代到的第一個命中（順序不固定）；這裡只接受完整標籤，或恰好命中一個 in-scope agent，其餘判為 oos 並標 `parse_failed`。原始回覆都有存，要用舊規則重算不必再打 API。
- 定價：Haiku 4.5 每百萬 tokens 輸入 US$1、輸出 US$5（claude-api skill 的模型表，快取日期 2026-06-24）。粗估全量約 US$2.5，上界約 US$3.5，低於 AC6 的 US$5。
- (無實跑數據)

### Blockers / 遇到的問題
- (無)

### Next
- [ ] Drew：`.env` 放 key 後 `make llm-smoke`，看 20 筆的實際 token 與推估全量花費
- [ ] `make llm`，把 `results/llm/haiku-8way.jsonl` 附到 Release
- [ ] 下一個 PR：不確定性、risk-coverage、fallback、oracle（RQ3、RQ4、AC6）

### Files / Budget
- 新增：`src/tinyrouter/llm_run.py`、`tests/test_llm_run.py`、`tests/test_llm_deps.py`、`tests/llm_fakes.py`
- 修改：`src/tinyrouter/llm.py`、`tests/test_llm.py`、`tests/test_makefile.py`、`Makefile`、`.github/workflows/ci.yml`、`.gitignore`、`pyproject.toml`（只改註解）、`README.md`、`docs/OPERATIONS.md`
- API 花費：US$0

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

## 2026-09-29：Haiku 4.5 zero-shot 全量完成（validation 3,100 + test 5,500）

### 本次工作 / 執行摘要
- PR #14（Haiku 執行器：journal、續跑、預留上界的成本護欄、執行鎖）與 PR #15（temperature 改走 `extra_body`，並用真 SDK 簽名檢查送出的參數）合併後執行。
- `make llm-smoke`（validation 前 20 筆）：US$0.0074，20/20，0 筆解析失敗，實際 input 338 到 342 tokens 都在上界內；推估全量 US$3.18。前 20 筆全是同一個 intent（資料依 intent 排序），所以 smoke 只驗證格式與成本，沒有驗證回覆的多樣性。
- Drew 確認花費後執行 `make llm`：`completed 8600/8600 llm predictions`，0 筆失敗，輸出無 key，`make verify-llm` 通過。

### 核心發現 / 數據
- **花費**：正式 US$3.1787（2,925,596 input + 50,630 output tokens）+ smoke US$0.0074 = **US$3.19**，AC6 上限 US$5。成本上限是每個身分 × 每個 target，所以總額需手動加總（見 `cap_scope`）。
- Haiku 8 類：validation 87.52%、OOS recall 68.0%；**test 82.07%、OOS recall 56.8%**；0 筆解析失敗（新規則）。
- 初步對照（同一份 test、8 類）：ModernBERT k=10 82.0%、BERT k=10 82.2%，約每個 intent 10 筆標註即追上 Haiku；ModernBERT k=100 91.9%。統計檢定與正式表格在分析 PR。
- 舊專案的 82.9% 是 1,200 筆分層抽樣（OOS 佔 1/8），與這裡的 test（OOS 佔 18.2%）不是同一批查詢，不直接比較。

### Blockers / 遇到的問題
- 第一次 smoke 全部失敗（`TypeError: ... 'temperature'`，anthropic 1.8.0 從簽名移除取樣參數），花費 US$0。假 client 什麼參數都收，所以測試沒抓到；PR #15 補上真 SDK 簽名檢查。

### Next
- [ ] `haiku-8way.jsonl` 附到 GitHub Release
- [ ] 分析 PR：RQ2 到 RQ4（不確定性訊號、risk-coverage、fallback、oracle、兩個 OOS 誤派指標、Wilson 信賴區間），同時報新舊兩種解析規則與不一致列數，並逐列交叉比對 Haiku 的 gold 與 logits archive 的 labels

### Files / Budget
- `results/llm/haiku-8way.json`、`results/llm-manifest.json`（jsonl 不進 git）
- API 花費：US$3.19（累計）

---

## 2026-09-23 (夜)：兩條學習曲線與 OOS 消融完成

### 本次工作 / 執行摘要
- 協定凍結後（lr 5e-5、S_min 400、5 epochs）依序單獨執行，每條以程式印出的完成行為證據（不看離開碼）：
  - `make curve MODEL=bert`：`completed 18/18 encoder points (bert)`，18:26 到 19:24。k=1 到 50 的 15 點新訓練；k=100 的 3 點重用 AC2（判定為等價 run，archive 完整）。
  - `make curve MODEL=modernbert`：`completed 18/18 encoder points (modernbert)`，19:25 到 21:47，18 點全部新訓練。
  - `make oos-ablation`：`completed 3/3 ablation points`，21:48 到 22:47（ModernBERT、k=100、OOS 訓練 0 筆 × 3 seeds）。
- `make verify-logits`：75 個 archive 全部與 manifest SHA-256 一致（AC2 3 + 基準 36 + BERT 新 15 + ModernBERT 18 + 消融 3）。
- 曲線權重全部刪除，只保留 logits。

### 核心發現 / 數據
- (本條只記錄執行完成；數字分析在步驟 4 由分析程式從 logits 產生，不在此手抄)

### Blockers / 遇到的問題
- 第一次開跑（17:22）的 shell 迴圈在 zsh 下沒拆 `make $t`，兩條曲線沒跑卻 exit 0。PR #11 加上完成性檢查與 Makefile 無目標防護（`make "curve MODEL=bert"` 現在 exit 2），並有測試對應這次事故。

### Next
- [ ] logits 附到 GitHub Release
- [ ] 步驟 4：重跑 Haiku（需 `ANTHROPIC_API_KEY`，預算 ≤ US$5）→ 不確定性分析、fallback、oracle

### Files / Budget
- `results/curves/{bert,modernbert,oos-ablation}.json`、`results/runs/*.json`、`results/logits-manifest.json`
- API 花費：US$0

---

## 2026-09-23 (晚 2)：S_min pilot（validation only），協定凍結

### 本次工作 / 執行摘要
- `make pilot-steps`：k=5（每個 intent 5 筆、OOS 13 筆）、seed 42、lr 5e-5，兩個 encoder 各試 S_min ∈ {100, 200, 400}，只看 validation。
- 規則：兩個模型 val in-scope 答對數加總最高者，平手取較小 S_min。選出 400，Drew 確認，寫入 `configs/curve.yaml`。
- **超參數協定至此全部凍結**：lr 5e-5（兩個模型）、S_min 400、5 epochs、batch 32、max_length 64。正式曲線開跑後不再依中途結果修改任何一項。

### 核心發現 / 數據
| S_min（實際步數） | BERT val in-scope / OOS | ModernBERT val in-scope / OOS |
|---|---:|---:|
| 100（120） | 9.73% / 10% | 56.20% / 16% |
| 200（200） | 36.73% / 45% | 63.43% / 21% |
| 400（400） | **75.87% / 47%** | **67.20% / 27%** |

- **又落在範圍上限，而且小 k 尚未訓練飽和**：BERT 從 200 到 400 步仍大幅上升（37% → 76%）。所以 k=1、5、10 的曲線點量的是「400 步預算下」的表現，不是模型極限；README 必須寫明。k ≥ 25 時 5 個 epoch 已超過 400 步，S_min 不生效。
- **學習速度 vs 最終表現**：步數很少時 ModernBERT 學得快很多（120 步 56% vs 10%），到 400 步 BERT 反超。「誰比較有效率」取決於訓練預算。單一 seed，不下結論，等正式曲線。
- `S_min=100` 在 k=5 實際是 120 步，因為 5 個 epoch = 120 步 > 100，符合 max(S_min, epoch 步數) 的定義。

### Blockers / 遇到的問題
- (無)

### Next
- [ ] `make curve MODEL=bert`、`make curve MODEL=modernbert`、`make oos-ablation`（約 5 小時）

### Files / Budget
- `results/pilots/steps.json`、`configs/curve.yaml`
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
