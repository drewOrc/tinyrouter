# TinyRouter：計畫與驗收條件（v0.1）

> **Learning When Not to Ask the LLM.** Pipeline 13 階段 1–2 產出。2026-09-23 初版，同日改 v0.1。
> 範圍：微調、評估並校準一個小編碼器與一個小型生成模型（LoRA / QLoRA），再把它們接進有成本約束、可實際服務請求的 routing 系統。重點是工程流程完整，不追求研究新穎性。

## 0. v0 → v0.1 改了什麼

採納外部審查的「分析層」建議（全部從已存的逐筆 logits 離線計算，不多花訓練與 API 成本）：
更細的資料量刻度、OOS 訓練量對齊、多種不確定性訊號、risk-coverage 評估、oracle 上限、兩個層級分開報。
**同日再擴充**：把生成式 LLM 微調（LoRA、QLoRA，RQ6）與推論優化及服務化（RQ7）拉回 v0.1，因為這兩項是「訓練完之後怎麼用」與「生成模型微調」的證據，encoder 分類單獨撐不起來。
**不採納進 v0**：自建 RoutingBench（含 multi-intent，需要多標籤模型、且自造資料的可信度問題）。第二個 benchmark 排 v0.2，見 §8。

## 1. 問題陳述

`cost-aware-hybrid-router` 的 LLM-only（Claude Haiku 4.5 zero-shot）在 8 類（7 agent + OOS）上 82.9%（分層抽樣 1,200 筆）。
TinyRouter 問三件事：**小模型要多少標註資料才夠？它知不知道自己什麼時候不確定？不確定時交給 LLM 能救回多少，特別是 OOS？**

已知前提（Larson et al. 2019, Table 2, OOS+ / oos-train）：BERT in-scope 96.7%、OOS recall 59.2%；Small 設定（每類 50 筆）in-scope 仍有 96.4%。所以全量資料的 in-scope 答案大致已知，**低資料量、OOS、deferral 才是主要觀察點**。

**比較規則**：150 類 intent 準確率與 8 類 routing 準確率是不同任務，**不互相比較**。與 Haiku 的比較一律在 8 類、同一批查詢上做。

## 2. 分級

**L1**（只有 Drew 依賴）。公開 showcase，額外加 CI 與可重現性。

## 3. 研究問題

| | 問題 | 產出 |
|---|---|---|
| RQ1 資料效率 | 每個 intent 1 / 5 / 10 / 25 / 50 / 100 / 全部 筆時，兩個層級的準確率與 OOS 指標怎麼變？8 類在哪一點追上 Haiku？ | 學習曲線（3 seeds mean ± std） |
| RQ2 OOS | OOS 偵測隨資料量怎麼變？**完全沒有 OOS 訓練資料時**（實務常態），只靠不確定性能攔下多少？ | OOS recall / precision / F1 / AUROC / AUPRC |
| RQ3 不確定性 | MSP、entropy、margin、temperature-scaled MSP，哪個最能預測錯誤？ | risk-coverage 曲線、AURC、reliability diagram |
| RQ4 LLM fallback | 在目標錯誤率下，小模型能自己處理多少？交給 Haiku 救回多少錯誤？離 oracle 上限差多少？ | coverage @ 目標 risk、LLM 呼叫率、錯誤回收率 |
| RQ5 成本 | 標註 + 訓練的一次性成本 vs 每 1K 查詢的 API 成本，損益兩平點 | 一張表 + 一句結論 |
| RQ6 生成式微調 | 同一個 routing 任務，LoRA 微調的小型生成模型 vs 全微調 encoder：準確率、校準、延遲、記憶體各差多少？QLoRA 相對 LoRA 省多少記憶體、掉多少準確率？ | encoder vs LoRA vs QLoRA 對照表 |
| RQ7 推論與服務 | encoder 匯出 ONNX 並 int8 量化後，準確率掉多少、延遲與模型大小省多少？包成服務後單機能撐多少 QPS？ | 延遲 p50/p95、吞吐量、大小、準確率差 |

**四個 router**：LLM-only、small-only、uncertainty-aware hybrid、oracle（知道小模型哪些會錯，理論上限；方法沿用 τ-bench 專案的 oracle ceiling）。

**不做（v0.1）**：keyword 前置層、自建 RoutingBench、SetFit 每個刻度重跑（引用舊 repo 16-shot 70.2%）。

## 4. 設計決策

| 決策 | 選擇 | 為什麼不選另外的 |
|---|---|---|
| 標籤粒度 | 訓練 151 類（150 intent + oos），推論時對應到 8 類；兩層都報 | 8 類直接訓練：丟掉細粒度訊號，也無法對照 Larson |
| 資料量抽樣 | in-scope 每個 intent 抽 k 筆；**OOS 同樣抽 k 筆**（上限為可用的 250 筆）；每個 seed 抽樣不同、固定可重現 | OOS 永遠給全量：學習曲線會把「資料量」與「OOS 比例」混在一起 |
| OOS 消融 | in-scope 全量下，OOS 訓練 0 筆 vs 250 筆 | 只有 k 對齊一條線：分不出 OOS 能力來自 OOS 資料還是不確定性 |
| 骨架模型 | `bert-base-uncased` 只跑全量（對照原論文，過 AC2）；學習曲線用一個小模型（階段 2 以 5 分鐘試跑在 `microsoft/deberta-v3-small` 與 MiniLM 類之間決定） | 所有刻度都用 bert-base：磁碟與時間成本高，對 showcase 沒有多的資訊 |
| 8 類聚合 | 預設 argmax intent 再對應；summed 版一併報，用 validation 選 | summed 在結構上壓低 OOS（finance 38 個 intent vs oos 1 個） |
| 校準 | temperature scaling，只在 validation 上 fit | Platt 已在舊 repo 做過 |
| 不確定性訊號與門檻 | 全部由**已存檔的逐筆 logits** 離線計算；門檻只由 validation 決定 | 每種訊號各訓練一次：浪費，也不必要 |
| 便宜基準 | TF-IDF centroid 在每個刻度重算（秒級） | 沒有它，學習曲線缺「不微調」的對照 |
| LLM 對照 | 重跑 Haiku 4.5（temperature 0）於 val 3,100 + test 5,500，**逐筆預測與 token 用量存檔** | 舊 repo 沒存逐筆預測，無法組 cascade |
| 執行環境 | M4 MPS 訓練；CI 只跑 CPU 單元測試 + 極小模型煙霧測試 | CI 上訓練完整模型：慢、貴、不可重現 |
| 生成模型 | `Qwen/Qwen3-0.6B`（Apache-2.0）為首選，`Qwen/Qwen2.5-0.5B-Instruct`（Apache-2.0）為備案；以 30 分鐘試跑在 MPS 上確認 bf16 訓練可行後決定 | 1B 以上：16 GB 統一記憶體加上訓練開銷太緊，而且對「會不會做 LoRA」沒有多的證據 |
| 生成式微調形式 | SFT：用 chat template 讓模型輸出 agent 標籤；信心度 = 對 8 個候選標籤的序列 log-prob 做 softmax，存成與 encoder 相同格式，RQ3/RQ4 分析直接重用 | 接分類頭（SequenceClassification）：更簡單，但那不是業界說「LLM 微調」時指的技能；只生成不算機率：無法做校準與 fallback |
| 生成模型任務層級 | 只做 8 類 routing（不做 151 類 intent） | 151 個候選標籤逐一算 log-prob 太慢；與 Haiku 的比較本來就在 8 類 |
| LoRA 函式庫 | PEFT + TRL（`SFTTrainer`） | MLX-LM：Mac 上很快，但生態與業界主流不同，面試官較少見 |
| QLoRA 執行位置 | 先在 MPS 試 bitsandbytes 4-bit（官方有 macOS arm64 wheel，但 MPS 上的 4-bit 訓練是否可用**尚未確認**）；不行就在 Kaggle 免費 CUDA GPU 上跑，腳本與環境鎖版本進 repo | 放棄 QLoRA：少一個常見 JD 關鍵字，而且 RQ6 的記憶體比較會缺一半 |
| 推論優化 | encoder 用 Optimum 匯出 ONNX，ONNX Runtime 動態 int8 量化；CPU、batch 1 量延遲 | TensorRT：需要 NVIDIA GPU；只量 PyTorch：沒有優化可講 |
| 服務化 | FastAPI `/route` 端點（encoder ONNX + 門檻 + 可關閉的 Haiku fallback）、`/healthz`、Dockerfile；本機壓測 | 只交 notebook：沒有「上線」證據；上雲長期運行：L1 專案不值得維運成本 |

**訓練次數估計**：bert-base 全量 × 3 seeds + 小模型 7 刻度 × 3 seeds + OOS 消融 1 × 3 seeds = 27 次 encoder；另加 LoRA（全量 + 10-shot）× 3 seeds = 6 次、QLoRA 全量 × 1 seed = 1 次。encoder 小刻度每次數十秒、全量數分鐘到二十分鐘；LoRA 全量預估每次一小時上下，試跑後更新。

## 5. 驗收條件

| # | 條件 | 怎麼驗 |
|---|---|---|
| AC1 | 乾淨 clone 後 `make setup && make reproduce` 跑完整流程（下載資料含 checksum → 訓練 → 評估 → 產表） | 在新目錄實跑一次 |
| AC2 | **流程正確性**：`bert-base-uncased` 全量訓練，150 類 in-scope 準確率 ≥ 95.7%（原論文 96.7% 減 1pp），3 seeds。第一次沒過先用 validation 調參，不下結論 | `results/` JSON + report-check |
| AC3 | 每一次訓練都把 val 與 test 的**逐筆 logits** 存檔（含 split、seed、k、模型 revision），之後所有 RQ2–RQ5 分析只讀存檔，不重跑模型 | 測試：分析函式只接受存檔格式 |
| AC4 | **test 不參與任何調整**：T、門檻、聚合方式只由 validation 決定；有測試守住 | pytest（已有 `LeakageError`） |
| AC5 | RQ1 七個刻度 × 3 seeds 全跑完；k 刻度的抽樣有測試（每個 intent 正好 k 筆、OOS k 筆、不同 seed 抽樣不同、同 seed 相同） | pytest + results JSON |
| AC6 | RQ4 報 test 上 coverage、selective risk、整體準確率、OOS recall、LLM 呼叫率、每 1K 成本，並列 oracle 上限；Haiku 花費 ≤ US$5 | results JSON 含實際 token 用量與花費 |
| AC7 | CI 綠燈：ruff、pytest、em dash 守門、commit hygiene | GitHub Actions |
| AC8 | README 首屏數字全部由 `make report` 從 JSON 產生，不手打；結論依結果寫；兩個層級不混比 | report-check 比對 |
| AC9 | 模型上 Hugging Face Hub，model card 寫明限制（英文、CLINC 領域、OOS 數字） | **上架前需 Drew 同意** |
| AC10 | LoRA：生成模型全量與 10-shot 各 3 seeds，報 8 類準確率、OOS recall、ECE、推論延遲、訓練峰值記憶體；逐筆候選標籤機率以 AC3 同格式存檔，RQ3/RQ4 分析不改程式就能跑 | results JSON + 同一支分析指令 |
| AC11 | QLoRA：至少 1 次全量，與 LoRA 同超參數，報峰值記憶體與準確率差；執行環境（MPS 或 Kaggle CUDA）與相依版本寫進結果檔 | results JSON；Kaggle 路徑附可重跑的腳本 |
| AC12 | ONNX int8：8 類準確率相對 PyTorch 下降 ≤ 0.5pp；報 CPU batch 1 的 p50/p95 延遲、模型大小，並記錄硬體型號 | `make bench`，結果含硬體資訊 |
| AC13 | 服務：`make serve` 起 FastAPI；Docker image 在 CI 建置並用極小模型打 `/healthz` 與一筆 `/route`；報本機壓測的 QPS 與 p95；Haiku fallback 預設關閉，沒有 API key 也能跑 | CI job + 壓測結果檔 |

## 6. 風險

| 風險 | 處理 |
|---|---|
| 磁碟只剩約 19 GB | `save_total_limit=1`；學習曲線的權重不保留，只存 logits；只保留要上 HF 的那一份 |
| workspace 在 iCloud 同步範圍 | `checkpoints.nosync/`、`.venv.nosync/` + symlink |
| MPS 非確定性 | 3 seeds 報變異；README 註明 |
| 1-shot 全微調 151 類可能完全不收斂 | 照實報；這本身是資料效率曲線的一個點 |
| 撞題（arXiv 2608.20371） | 定位是工程 showcase，README 引用並說明差異 |
| API key | Drew 自行設定環境變數，不進 repo |
| 16 GB 統一記憶體跑 LoRA | 0.6B 模型、bf16、gradient checkpointing、短序列；試跑量峰值記憶體，不夠就降到 0.5B |
| QLoRA 在 MPS 不可用 | 改 Kaggle 免費 GPU；結果檔記錄環境，README 寫明兩個環境不同，記憶體數字不跨環境比較 |
| 公開服務被濫用燒 API 費 | 公開 demo（若有）一律關閉 fallback；本機才開 |
| 磁碟（再加生成模型與 Docker image） | Qwen 權重約 1 到 1.5 GB、adapter 數十 MB；Docker 用 slim base 並只放 ONNX 模型；每步結束清 build cache |

## 7. 時程（約 14–18 天）

1. 骨架 + CI + 資料載入（階段 3–5）✅ 2026-09-23
2. bert-base 全量，過 AC2
3. k-shot 抽樣 + 小模型學習曲線 + OOS 消融（RQ1、RQ2）
4. Haiku 重跑 + 不確定性分析 + cascade + oracle（RQ3、RQ4）
5. 生成模型試跑（MPS bf16、QLoRA 可行性）→ LoRA 全量與 10-shot → QLoRA（RQ6）
6. ONNX 匯出與 int8 量化 + FastAPI 服務 + Docker + 壓測（RQ7）
7. 成本表（RQ5，把 LoRA 與服務的實測延遲一起算進去）+ README + model card

## 8. 第二個 benchmark（v0.2，Drew 2026-09-23 決定）

候選：Zhang et al. 2022 的 **BANKING77-OOS / CLINC-Single-Domain-OOS**。它區分 in-domain OOS（同領域但不支援的請求）與 out-of-domain OOS，比 CLINC 標準 OOS 更接近 agent routing 的真實難點，而且格式與本專案流程相同（單標籤分類 + OOS），不需要新的模型結構。
排在 v0.1 做完、AC2 通過之後（v0.2），用同一條流程重跑。
