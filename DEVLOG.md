# DEVLOG: TinyRouter

實驗進度與觀察。最新條目在最上面。

---

## 2026-09-29（深夜，五）：PR #18 審查修正（R1 到 R6）

### 本次工作 / 執行摘要
- 狀態措辭：Tier 1 研究問題（RQ1 到 RQ5）完成；AC1 尚未達成（待決定拆成 AC1a 從 Release artifacts 重現、AC1b 完整重訓）。AC1 的實作不在 PR #18。
- **R1**：原本「README 與重算一致」只守一致性，欄位接錯後重生 README，CI 仍綠（審查突變 X3、X4 存活）。新增表格驅動測試：首屏 12 個數字各自對應 `summary.json` 的明確路徑，測試自己格式化，不經 report.py，斷言渲染字串等於該欄位的值。
- **R2**：k=10 hybrid 補上三個 seed 的絕對呼叫次數：1,235、1,157、1,548（test 共 5,500 筆），並納入 R1 的測試。
- **R3**：routers 圖 (a) 原本是長條圖且 y 軸截斷在 70，會放大 91.9 對 92.1 的差距。改成點圖加誤差棒，y 軸從 75% 起，圖說寫明「看點的距離，不看長條長度」；(b) 仍是從 0 開始的長條圖。
- R4：「約 87%」改寫為「把 test 加權到 validation 的 OOS 比例後，validation 與 test 的 risk 差距縮小約 87%」，屬描述性寫法。R5：效率表註腳改為「準確率與 OOS recall 用 validation 選的 8 類聚合，ECE 是 151 類」。R6：Haiku 延遲註明不是同條件比較（不同機器，中間有網路）。
- 延遲的同架構比對：計時模型改用訓練時同一個 loader（`train.load_model_and_tokenizer`）建立；model revision、max_length、torch 與 transformers 版本必須等於訓練 run 的記錄，否則停止。另記錄 `attention_implementation`（兩個模型都是 sdpa；訓練 run 沒有記錄這個欄位，因為 loader 與 transformers 版本相同，選法一致）。
- 因為延遲改了程式，重跑 `make bench-cpu`，接著重生 cost、report、figures。

### 核心發現 / 數據
- 重量後 CPU p50 / p95：BERT 15.5 / 18.4 ms，ModernBERT 20.9 / 28.5 ms（上一則是 15.6 / 17.0 與 20.2 / 23.2）。p50 差不到 1 ms，p95 對同機其他負載較敏感，重量時差了約 5 ms；這是同一台機器兩次量測的差異，不是程式造成的。
- 損益兩平只有 ModernBERT k=100 hybrid 在 US$2/h 時從 1,830 變成 1,831，其他格不變。

### Blockers / 遇到的問題
- (無)

### Next
- [ ] 決定 AC1 是否拆成 AC1a（從 Release artifacts 重現）與 AC1b（完整重訓）
- [ ] Tier 2（步驟 6）

### Files / Budget
- `src/tinyrouter/report.py`、`figures.py`、`latency.py`；`tests/test_report.py`、`test_latency.py`；`results/efficiency/cpu_latency.json`、`results/cost/cost.json`、`results/figures/*.png`、`results/report.md`；`README.md`；`DEVLOG.md`
- API 花費：US$0

---

## 2026-09-29（深夜，四）：步驟 5，效率表、成本、圖與 README 首屏自動產生

### 本次工作 / 執行摘要
- **CPU 延遲（AC5，`make bench-cpu`）**：曲線權重已刪，改用同架構量：鎖定 revision 的預訓練骨架 + 151 類分類頭（seed 42 隨機初始化）+ 同 tokenizer 與 max_length。延遲只取決於架構與輸入形狀，不取決於權重數值；程式比對參數量必須等於 k=100 訓練 run 記錄的值（BERT 109,598,359、ModernBERT 149,720,983），不等就停。量法：CPU、batch 1、`torch.inference_mode()`、4 個 intra-op 執行緒、interop 1、暖機 50 筆，再依序量 validation 前 500 筆；報 tokenization + forward + argmax 與 forward-only 兩種。
- **Haiku 延遲（`make llm-latency`）**：journal 每筆的 `latency_ms` 是用戶端量的最後一次嘗試時間，含網路往返與 API 排隊，執行時最多 6 個呼叫同時進行；8,600 筆沒有任何重試。
- **成本（RQ5，`make cost`）**：`results/cost/cost.json` 把實測（M4 訓練 wall-clock、CPU 延遲、Haiku token 與花費、各 router 在 test 的實際 Haiku 花費）與假設（accelerator 每小時 US$0.5、1、2；標註每筆 US$0.05、0.2、1；本機推論每 vCPU-hour US$0.05、滿載）分開存。不把 Mac 購買價算進 run 的成本。損益兩平 = 一次性成本 ÷（LLM-only 每筆 − router 每筆），對 ModernBERT k=10、k=100 的 small-only 與 hybrid、每個價格情境都算，標為情境敏感度。
- **`make report`**：從 commit 的 JSON 產生 `results/report.md` 與 README 標記之間的區塊（首屏雙欄、router 表、效率表、成本表、聚合方式、圖、Limitations）。`tests/test_report.py` 在 CI 重算並比對，不一致就紅。
- **`make figures`**：四張 PNG（學習曲線、risk-coverage、router 比較、threshold transfer），Okabe-Ito 配色加線型與標記，300 dpi，拿掉 PNG 的 Software 欄位，兩次輸出逐位元組相同（有測試）。matplotlib 3.11.2 放在新的 `figures` group，CI 的 test job 一併安裝。
- README 另加隱私說明：Release 存的是公開 CLINC 查詢的 hash 與 LLM 回覆，這個 journal 設計不應原封不動套到含私人查詢的產品。

### 核心發現 / 數據
（取自 `results/efficiency/*.json`、`results/cost/cost.json`、`results/analysis/summary.json`）
- CPU p50 / p95（Apple M4，4 執行緒，端到端）：BERT 15.6 / 17.0 ms，ModernBERT 20.2 / 23.2 ms；Haiku test 669 / 916 ms（含網路）。
- 訓練（k=100，M4 MPS）：BERT 852 ± 33 s、峰值 4.31 GiB；ModernBERT 1,198 ± 1 s、峰值 6.24 GiB（MPS driver 記憶體取樣）。ECE（151 類，test）溫度校準前後：BERT 3.90 → 3.80，ModernBERT 3.95 → 2.45。
- Haiku 每 1K test 查詢 US$0.369（約 340K 輸入、5.9K 輸出 token）。
- 損益兩平（只算訓練，US$1/h）：ModernBERT k=10 hybrid 198 筆、k=100 hybrid 915 筆。訓練算力只值幾分錢；一旦要付標註費，標註主導（k=100、每筆 US$0.2 時約 839 萬筆）。
- k=100 的 hybrid 在三個 seed 各呼叫 Haiku 170、9、31 次（共 5,500 筆 test），8 類準確率 91.9 → 92.1，是安全與診斷槓桿，不是準確率主要來源；k=10 時 81.5 → 88.0（Haiku 呼叫 23.9%）才是 fallback 價值的主要證據。

### Blockers / 遇到的問題
- (無)

### Next
- [ ] Tier 2（步驟 6）：ONNX int8 + FastAPI + Docker + 壓測，完成後把 RQ5 的本機延遲換成 ONNX 實測

### Files / Budget
- 新增：`src/tinyrouter/latency.py`、`cost.py`、`figures.py`；`tests/test_latency.py`、`test_cost.py`、`test_report.py`、`test_figures.py`；`results/efficiency/*.json`、`results/cost/cost.json`、`results/figures/*.png`、`results/report.md`
- 修改：`src/tinyrouter/report.py`（改寫）、`Makefile`、`README.md`、`pyproject.toml`、`uv.lock`、`.github/workflows/ci.yml`、`.gitignore`、`docs/OPERATIONS.md`、`DEVLOG.md`
- API 花費：US$0

---

## 2026-09-29（深夜，三）：PR #17 複查小修（R1 到 R4）

### 本次工作 / 執行摘要
- 複查確認主結果（(a)、消融、歸因）與洩漏實驗無誤，獨立重算到第 6 位一致；剩下三個 MEDIUM 與一個 LOW。
- **R1**：新增測試，coverage 平手時 AURC 較好的訊號（`msp_t`）在 `SIGNALS` 裡排在較差的（`msp`）後面，仍要選到它。原本的測試裡較好的剛好排第一，拿掉 AURC 判定也會過。
- **R2**：新增 summed 聚合的 `oos_score` 測試：oos agent 拿走大部分機率時，分數要接近 1，且等於 1 減最大 in-scope agent 機率。
- **R3**：`select_threshold` 的加權分支（只有 (b) 用到）改用 Kish 有效樣本數：每個切點 p = 加權誤判率、n_eff = (Σw)²/Σw²（只算被收下的列）、k = p·n_eff 代入 Wilson。原本用名目加權計數，100 筆 OOS 權重約 5.64 時偏樂觀。權重全為 1 時與未加權結果相同（有測試）；權重不均時上界比名目計數寬（有測試）。
- **R4**：三個 seed 選到的訊號不同時，τ 不再算 mean 與 std，只保留逐 seed 的值與訊號名（final hybrid、各聚合的 hybrid、diagnostics 與 (b) 的 τ）。`by_signal` 底下同一個訊號的 τ 照常平均。

### 核心發現 / 數據
- (b) 敏感度（ModernBERT k=100，目標 2%）：test risk 2.60 ± 0.48（2.17、3.12、2.50），coverage 87.33 ± 1.85；原本 2.93 ± 0.63、88.55 ± 1.76。結論不變：依部署 OOS 比例加權後仍略高於 2%。
- `summary.json` 逐欄比對：變動只在 (b) 的欄位（`diagnostics.*.sensitivity_reweighted_validation`）與 R4 的 τ 表示方式；其他數字全部不變，`curves.json` 逐位元組相同。

### Blockers / 遇到的問題
- (無)

### Next
- [ ] 同上一則

### Files / Budget
- `src/tinyrouter/selective.py`、`analysis_run.py`；`tests/test_selective.py`、`test_analysis.py`、`test_analysis_run.py`；`results/analysis/summary.json`；`DEVLOG.md`
- API 花費：US$0

---

## 2026-09-29（深夜，二）：PR #17 審查修正（守門缺口與兩個會誤導的結論）

### 本次工作 / 執行摘要
- 審查結論：計算正確（無洩漏、可逐位元重現、獨立重算吻合），退回原因是 9 條突變存活與兩個報告結論。這次一次修完 F1 到 F11。
- **守門不再只看 split 名稱（F2）**：`select_threshold`、`coverage_thresholds` 改成只收 `Scored`，由 `Routed.scored(signal)` 從同一個 routed split 取出名稱、分數與錯誤，不能再「傳 validation 名稱配 test 陣列」。打亂 labels 的測試之外，新增「test logits 換成亂數」的測試，並把操作曲線的 τ、hybrid 的訊號、(b) 敏感度的 τ 都納入比對。
- **RQ2 OOS 偵測改用專用分數（F3）**：`1 − max in-scope 機率`（擬合的 T 下；argmax 聚合用 151 類、summed 用 8 類），寫在每個聚合的 `oos_detection`。四個信心訊號只留給 RQ3。`summary.json` 新增 `ablation_comparison`：偵測固定用 argmax 聚合，router 行為用 final router，全部由程式產生。
- **門檻失效的診斷（F4）**：`diagnostics.py`，對每個 encoder 點的 final hybrid 報 val 與 test 的 selective risk、只看 in-scope 的 risk、被收下 OOS 的誤判率、test 依 validation OOS 比例重新加權後的 risk、比例解釋的缺口比例，另報選項 (b) 的敏感度（validation 依 18.2% OOS 加權後選 τ 再套 test，註明這個比例是看過 test 才知道的，只作診斷）。加權的 Wilson 上界在複查時改用 Kish 有效樣本數（見「深夜，三」那一則）。
- **決定（協調者依 Drew 的專案原則拍板）**：主結果採 (a) 照實報，validation-only 協定不變；(b) 只當診斷。
- 其他：F1 OOS AUROC／AUPRC 方向的呼叫端測試（對 sklearn）；F5 `select_signal` 用手算 choices 測；F6 entropy 方向；F7 summed 用溫度校準機率（T 會改變 argmax 的案例）；F8 分數恰好等於 τ 的列由小模型保留（`deferred_below`，有測試）；F9 可行 seed 少於 2 個時 std 記 null 並保留 n（例：bert/k1 目標 2% 只有 1 個 seed 可行）；F10 兩個 index 指向同一個 archive 會紅；F11 正式程式碼的 `assert isinstance` 改成明確 raise。

### 核心發現 / 數據
（全部取自 `results/analysis/summary.json`，test，百分比，mean ± std）
- **消融改正後**（偵測固定 argmax 聚合、分數 `1 − max in-scope p`）：OOS 250 筆 AUROC 98.33 ± 0.07、AUPRC 94.34 ± 0.25；OOS 0 筆 97.80 ± 0.04、91.52 ± 0.26。有 OOS 訓練資料的模型較好，與前一則的方向相反。
- **消融的主比較是 router 行為**（目標 2%，OOS 250 對 0）：hybrid OOS recall 62.43 ± 1.33 對 40.60 ± 1.04；高信心 OOS 誤派 34.53 ± 5.53 對 37.67 ± 2.90；LLM 呼叫率 1.27 ± 1.59 對 12.07 ± 0.73。沒有 OOS 訓練資料時，要多交 LLM 近 10 倍的查詢，OOS 仍少攔約 22pp。
- **門檻失效的歸因**（ModernBERT k=100，目標 2%）：val 1.50 ± 0.13，test 7.34 ± 0.86，test 依 validation OOS 比例加權後 2.29 ± 0.11，比例解釋缺口的 86.66 ± 1.39%。其餘來自 test 的 OOS 較難：被收下 OOS 的誤判率 val 19.91 ± 2.93 對 test 36.48 ± 3.04；in-scope risk 0.90 ± 0.23 對 1.20 ± 0.07。目標 2% 下其他 encoder 點比例解釋約 78% 到 97%（BERT k=5、k=10 超過 100%，加權後反而低於 val）；目標 5% 的 ModernBERT k=1 只有 27%，那一點 coverage 很低、數字不穩。
- **(b) 敏感度**（同點，已依複查改用 Kish 有效樣本數重算）：test risk 2.60 ± 0.48（三個 seed 2.17、3.12、2.50），coverage 87.33 ± 1.85（85.45、89.15、87.38）。依部署 OOS 比例加權後仍略高於 2%。
- **結論**：validation 選的門檻在 test 上守不住目標 risk，主要來自 OOS 比例差異（約 86%），其餘是 test OOS 較難；依部署 OOS 比例加權仍不足，部署時應在接近真實流量的資料上重新校準門檻。
- 重構後 `curves.json` 與改前逐位元組相同（所有 τ 與曲線不變），`summary.json` 除新增欄位外只有 48 處 std 由 0 改成 null（F9）；`make analysis` 連跑兩次輸出相同。

### Blockers / 遇到的問題
- 重新下載 Release 時連線中斷，5 個 archive 下載不完整；`make verify-logits` 以 SHA-256 不符擋下，重下後通過。

### Next
- [ ] 步驟 5：README 與圖、成本表（RQ5），數字由 `make report` 從 `results/analysis/*.json` 產生；README 寫明門檻要在接近真實流量的資料上重新校準

### Files / Budget
- 新增：`src/tinyrouter/diagnostics.py`、`tests/test_diagnostics.py`
- 修改：`src/tinyrouter/selective.py`、`analysis.py`、`analysis_run.py`；`tests/test_selective.py`、`test_analysis.py`、`test_analysis_run.py`；`results/analysis/summary.json`；`README.md`；`DEVLOG.md`
- API 花費：US$0

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
- **validation 選的門檻在 test 上守不住目標 risk**。目標 2%：ModernBERT k=100 test selective risk 7.3 ± 0.9（coverage 98.7）、k=25 為 5.3 ± 0.5；目標 5% 在 k ≥ 25 時 coverage 幾乎 100%，test selective risk 8.0 到 13.5。歸因見下一則（審查修正）的診斷：主要來自 OOS 比例差異，其餘是 test 的 OOS 較難。
- 小資料量時 hybrid 的價值最清楚：ModernBERT k=10 在目標 2% 下 88.0 ± 0.4，LLM 呼叫 23.9 ± 3.8，高於 Haiku 單獨的 82.1 與 small-only 的 81.5 ± 0.7。
- **OOS 0 筆消融**：small-only OOS recall 0.0（從不預測 oos）。本則原先寫「只靠不確定性當 OOS 分數時 AUROC 反而更高（msp_t 98.0 對 91.9）」，那是分數選擇造成的假象，已在下一則改用 `1 − max in-scope p` 並固定聚合重算，方向反過來。以 router 實際行為比較：消融在目標 2% 下 hybrid OOS recall 40.6 ± 1.0、高信心 OOS 誤派 37.7 ± 2.9；目標 5% 時門檻放到全收，高信心 OOS 誤派 99.3 ± 0.2。
- 聚合方式：ModernBERT k ≤ 10 三個 seed 都選 summed，k ≥ 50 都選 argmax；BERT 在 k=10、50、100 三個 seed 選得不一致，兩者差距都在 1pp 內。兩種的 test 數字都在 JSON。
- 多數類：argmax 聚合永遠猜 oos（k-shot 樣本裡 oos 是單一最大 intent），summed 永遠猜 finance；validation 選 summed，test 8 類 20.7。

### Blockers / 遇到的問題
- 門檻失效要不要處理：已決定（見下一則），主結果照實報，(b) 只當診斷。
- 「低信心當 OOS 分數」對有 OOS 訓練的模型不公平：已在下一則改用專用的 OOS 偵測分數。

### Next
- [x] 上面兩點已在下一則處理
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
