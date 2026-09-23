# TinyRouter：計畫與驗收條件（v0.1，凍結版）

> **Learning When Not to Ask the LLM.** Pipeline 13 階段 1–2 產出。2026-09-23 初版，同日改 v0.1。
> 範圍：微調、評估並校準一個小編碼器與一個小型生成模型（LoRA / QLoRA），再把它們接進有成本約束、可實際服務請求的 routing 系統。重點是工程流程完整，不追求研究新穎性。

## 0. v0 → v0.1 改了什麼

採納外部審查的「分析層」建議（全部從已存的逐筆 logits 離線計算，不多花訓練與 API 成本）：
更細的資料量刻度、OOS 訓練量對齊、多種不確定性訊號、risk-coverage 評估、oracle 上限、兩個層級分開報。
**同日再擴充**：把生成式 LLM 微調（LoRA、QLoRA，RQ6）與推論優化及服務化（RQ7）拉回 v0.1，因為這兩項是「訓練完之後怎麼用」與「生成模型微調」的證據，encoder 分類單獨撐不起來。
**凍結前最後一輪（外部審查第三、四輪）**：主模型改 ModernBERT-base，BERT 降為歷史基準與流程檢查；AC2 措辭改為工程驗收、不宣稱重現；RQ 分三層優先序（§3.1），RQ6 不得阻塞 RQ1 到 RQ5；新增 OOS 靜默誤派指標與多數類基準；RQ5 不宣稱真實標註成本；資料量刻度修正（CLINC150 每個 intent 正好 100 筆，所以 k=100 就是全量）。**此後 v0.1 範圍凍結**，新想法進 §9 待辦，不改驗收條件。
**凍結後釐清（第五輪審查，不改範圍）**：OOS 誤派拆成兩個定義明確的指標（§3）；BERT 與 ModernBERT 的比較加上參數量、訓練時間、峰值記憶體、延遲（AC5）；資料來源已逐筆比對原作者檔案（`docs/DATA.md`）。
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
| RQ1 資料效率 | 每個 intent k = 1 / 5 / 10 / 25 / 50 / 100 筆（k=100 即全量，CLINC150 每個 intent 正好 100 筆）時，ModernBERT 與 BERT 在兩個層級的準確率與 OOS 指標怎麼變？8 類在哪一點追上 Haiku？ | 學習曲線（3 seeds mean ± std） |
| RQ2 OOS | OOS 偵測隨資料量怎麼變？**完全沒有 OOS 訓練資料時**（實務常態），只靠不確定性能攔下多少？ | OOS recall / precision / F1 / AUROC / AUPRC，以及兩個誤派指標。**OOS 誤派率** = 被派給任何 in-scope agent 的真 OOS ÷ 全部真 OOS（與門檻無關，等於 1 減 OOS recall）。**高信心 OOS 誤派率** = 被派給 in-scope agent 且信心 ≥ 門檻 τ（所以沒交給 LLM）的真 OOS ÷ 全部真 OOS，這才是「靜默誤派」，隨 τ 報成曲線。交給 LLM 只是多花錢，靜默誤派會讓系統執行錯的動作，是 production 最危險的錯 |
| RQ3 不確定性 | MSP、entropy、margin、temperature-scaled MSP，哪個最能預測錯誤？ | risk-coverage 曲線、AURC、reliability diagram |
| RQ4 LLM fallback | 在目標錯誤率下，小模型能自己處理多少？交給 Haiku 救回多少錯誤？離 oracle 上限差多少？ | coverage @ 目標 risk、LLM 呼叫率、錯誤回收率 |
| RQ5 成本 | 查詢量多大時，「本機推論 + 一次性訓練算力」比只用 LLM 便宜？一次性：訓練算力（實測時間 × 硬體換算）；經常性：API 費用（實測 token）與本機推論（實測延遲）。**CLINC150 是現成資料，不宣稱真實標註成本**；標註成本只做假設性敏感度分析（每筆標註假設 US$0.05 / 0.2 / 1 三檔） | 一張表 + 一句結論 |
| RQ6 生成式微調 | 同一個 routing 任務，LoRA 微調的小型生成模型 vs 全微調 encoder：準確率、校準、延遲、記憶體各差多少？QLoRA 相對 LoRA 省多少記憶體、掉多少準確率？ | encoder vs LoRA vs QLoRA 對照表 |
| RQ7 推論與服務 | encoder 匯出 ONNX 並 int8 量化後，準確率掉多少、延遲與模型大小省多少？包成服務後單機能撐多少 QPS？ | 延遲 p50/p95、吞吐量、大小、準確率差 |

### 3.1 優先序（RQ6 不得阻塞 RQ1 到 RQ5）

| 層 | 內容 | 規則 |
|---|---|---|
| Tier 1 必須完成 | RQ1 到 RQ5（encoder、OOS、不確定性、fallback、成本） | 這層完成就是完整的作品 |
| Tier 2 強烈建議 | RQ7（ONNX int8、FastAPI、Docker、壓測） | Tier 1 完成後做 |
| Tier 3 加分 | RQ6（Qwen LoRA／QLoRA） | 不得拖延 Tier 1、2；QLoRA 卡住就記錄原因並跳過，不影響其他 |

**四個 router**：LLM-only、small-only、uncertainty-aware hybrid、oracle（知道小模型哪些會錯，理論上限；方法沿用 τ-bench 專案的 oracle ceiling）。

**不做（v0.1）**：keyword 前置層、自建 RoutingBench、SetFit 每個刻度重跑（引用舊 repo 16-shot 70.2%）。

## 4. 設計決策

| 決策 | 選擇 | 為什麼不選另外的 |
|---|---|---|
| 標籤粒度 | 訓練 151 類（150 intent + oos），推論時對應到 8 類；兩層都報 | 8 類直接訓練：丟掉細粒度訊號，也無法對照 Larson |
| 資料量抽樣 | in-scope 每個 intent 抽 k 筆；**OOS 抽 ⌈2.5k⌉ 筆**（無條件進位，寫死對照表：k=1→3、5→13、10→25、25→63、50→125、100→250），維持 OOS+ 訓練集的比例（250 筆 OOS 對每個 intent 100 筆；這是 Larson OOS+ 設定的比例，不是真實流量的 OOS 比例），所以 k=100 的端點正好等於完整的 OOS+ 訓練集；測試集（18.2% 為 OOS）不抽樣；每個 seed 抽樣不同、固定可重現 | OOS 永遠給全量：曲線把「資料量」與「OOS 比例」混在一起；OOS 也抽 k 筆：端點不等於全量，無法對照原論文 |
| OOS 消融 | ModernBERT，in-scope 全量（k=100）下，OOS 訓練 0 筆 vs 250 筆（後者即曲線端點） | 只有一條曲線：分不出 OOS 能力來自 OOS 資料還是不確定性 |
| Encoder 模型 | **主模型 `answerdotai/ModernBERT-base`**（2024，Apache-2.0，約 149M）；**歷史基準 `bert-base-uncased`**（2018，也負責 AC2 流程檢查）。兩者都跑完整學習曲線；小刻度每次只要數十秒，多一條曲線成本低，卻能用實驗回答「為什麼不用 BERT」 | 只用 BERT：2026 年的作品選 2018 年的模型當主角，技術選型顯舊；DeBERTa-v3-small：刪除，範圍控制 |
| ModernBERT 相容性 | 開跑曲線前先試跑：MPS 上能否訓練（Flash Attention 與 unpadding 只支援 CUDA，MPS 走一般 attention）、Optimum 能否匯出 ONNX。匯出不行時，RQ7 改用 BERT，並在 README 寫明原因 | 不試跑直接開跑：跑到一半才發現不相容，浪費時間 |
| 8 類聚合 | 預設 argmax intent 再對應；summed 版一併報，用 validation 選 | summed 在結構上壓低 OOS（finance 38 個 intent vs oos 1 個） |
| 校準 | temperature scaling，只在 validation 上 fit | Platt 已在舊 repo 做過 |
| 不確定性訊號與門檻 | 全部由**已存檔的逐筆 logits** 離線計算；門檻只由 validation 決定 | 每種訊號各訓練一次：浪費，也不必要 |
| 便宜基準 | **多數類**（永遠猜最常見的 agent，不看語意的下限）＋ TF-IDF centroid 在每個刻度重算（秒級） | 沒有多數類：看不出 TF-IDF 的 74% 有多少只是類別分布偏差；沒有 TF-IDF：學習曲線缺「不微調」的對照 |
| LLM 對照 | 重跑 Haiku 4.5（temperature 0）於 val 3,100 + test 5,500，**逐筆預測與 token 用量存檔** | 舊 repo 沒存逐筆預測，無法組 cascade |
| 執行環境 | M4 MPS 訓練；CI 只跑 CPU 單元測試 + 極小模型煙霧測試 | CI 上訓練完整模型：慢、貴、不可重現 |
| 生成模型 | `Qwen/Qwen3-0.6B`（Apache-2.0）為首選，`Qwen/Qwen2.5-0.5B-Instruct`（Apache-2.0）為備案；以 30 分鐘試跑在 MPS 上確認 bf16 訓練可行後決定 | 1B 以上：16 GB 統一記憶體加上訓練開銷太緊，而且對「會不會做 LoRA」沒有多的證據 |
| 生成式微調形式 | SFT：用 chat template 讓模型輸出 agent 標籤；信心度 = 對 8 個候選標籤的序列 log-prob 做 softmax，存成與 encoder 相同格式，RQ3/RQ4 分析直接重用 | 接分類頭（SequenceClassification）：更簡單，但那不是業界說「LLM 微調」時指的技能；只生成不算機率：無法做校準與 fallback |
| 生成模型任務層級 | 只做 8 類 routing（不做 151 類 intent） | 151 個候選標籤逐一算 log-prob 太慢；與 Haiku 的比較本來就在 8 類 |
| LoRA 函式庫 | PEFT + TRL（`SFTTrainer`） | MLX-LM：Mac 上很快，但生態與業界主流不同，面試官較少見 |
| QLoRA 執行位置 | 先在 MPS 試 bitsandbytes 4-bit（官方有 macOS arm64 wheel，但 MPS 上的 4-bit 訓練是否可用**尚未確認**）；不行就在 Kaggle 免費 CUDA GPU 上跑，腳本與環境鎖版本進 repo | 放棄 QLoRA：少一個常見 JD 關鍵字，而且 RQ6 的記憶體比較會缺一半 |
| 推論優化 | encoder 用 Optimum 匯出 ONNX，ONNX Runtime 動態 int8 量化；CPU、batch 1 量延遲 | TensorRT：需要 NVIDIA GPU；只量 PyTorch：沒有優化可講 |
| 服務化 | FastAPI `/route` 端點（encoder ONNX + 門檻 + 可關閉的 Haiku fallback）、`/healthz`、Dockerfile；本機壓測 | 只交 notebook：沒有「上線」證據；上雲長期運行：L1 專案不值得維運成本 |

### 4.1 超參數協定（學習曲線，2026-09-23 定案）

所有 pilot 只看 validation，選出的值凍結後，整條曲線與兩個 encoder 共用。設定檔 `configs/curve.yaml`；pilot 只印出選定值，由人手動填入，程式不自動改設定。

| 項目 | 協定 |
|---|---|
| 訓練步數 | 實際步數 = max(S_min, 5 個 epoch 的步數)。S_min 從 {100, 200, 400} 選一個，所有 k 與兩個 encoder 共用。結果 JSON 記錄預定步數、實際步數與由哪一邊決定（`epochs` 或 `min_train_steps`） |
| S_min pilot（`make pilot-steps`） | k=5、seed 42，兩個 encoder 各用自己選定的 learning rate 都跑；選兩者 val 150 類 in-scope 準確率平均最高者，平手取較小的 S_min |
| learning rate pilot（`make pilot-lr`） | 每個 encoder 各自從 {1e-5, 2e-5, 5e-5} 選，k=100、seed 42；標準是 val 150 類 in-scope 準確率，平手看 val OOS recall，再平手取較小的 learning rate。k=100 的 epoch 步數（2,385）大於任何 S_min，所以這個 pilot 不設 S_min，先跑它 |
| AC2 重用 | BERT 的 5e-5 直接重用 AC2 seed 42 的 validation 數字；曲線上 BERT 的 k=100 點，若選中 5e-5，重用 AC2 的三個 run。重用前逐欄比對設定、訓練列數與步數，任何一項不同就重訓 |
| test 隔離 | pilot 不得產生或讀取 test 的任何結果：pilot 的評估函式在程式層級只接受 validation，傳入 test 就 raise，並有測試 |
| 信賴區間 | OOS recall 的信賴區間用 Wilson score interval（`metrics.wilson_interval`，預設 95%） |

**訓練次數估計**：ModernBERT 6 刻度 × 3 seeds + OOS 0 筆消融 × 3 seeds = 21 次；BERT 6 刻度 × 3 seeds = 18 次（其中 k=100 那 3 次就是 AC2）；共 39 次 encoder，k ≤ 25 的每次數十秒；另加 LoRA（全量 + 10-shot）× 3 seeds = 6 次、QLoRA 全量 × 1 seed = 1 次。encoder 小刻度每次數十秒、全量數分鐘到二十分鐘；LoRA 全量預估每次一小時上下，試跑後更新。

## 5. 驗收條件

| # | 條件 | 怎麼驗 |
|---|---|---|
| AC1 | 乾淨 clone 後 `make setup && make reproduce` 跑完整流程（下載資料含 checksum → 訓練 → 評估 → 產表） | 在新目錄實跑一次 |
| AC2 | **BERT 流程正確性檢查**：`bert-base-uncased` 在全量資料（k=100、OOS 250）訓練 151 類，150 類 in-scope 準確率在 3 seeds 下都 ≥ 95.7%。**這是從原論文 96.7% 推出的工程驗收門檻，不宣稱精確重現原論文**（實作、超參數、tokenizer、評估程式都可能與原論文不同）。第一次沒過先用 validation 調參，不下結論。BERT 是基準，不是主要結果 | `results/` JSON + report-check |
| AC3 | 每一次訓練都把 val 與 test 的**逐筆 logits** 存檔（含 split、seed、k、模型 revision），之後所有 RQ2–RQ5 分析只讀存檔，不重跑模型 | 測試：分析函式只接受存檔格式 |
| AC4 | **test 不參與任何調整**：T、門檻、聚合方式只由 validation 決定；有測試守住 | pytest（已有 `LeakageError`） |
| AC5 | RQ1 兩個模型 × 六個刻度 × 3 seeds 全跑完；抽樣有測試（每個 intent 正好 k 筆、OOS 筆數符合寫死的對照表、k=100 等於全量、不同 seed 抽樣不同、同 seed 相同）；多數類與 TF-IDF 基準在每個刻度都有；BERT 與 ModernBERT 另列一張效率表：參數量、訓練時間、訓練峰值記憶體、8 類準確率、OOS recall、ECE、CPU 推論延遲（回答「好是因為架構新，還是只因為模型大」；README 寫明 BERT 是歷史基準、ModernBERT 是主模型） | pytest + results JSON |
| AC6 | RQ4 報 test 上 coverage、selective risk、整體準確率、OOS recall、**OOS 誤派率與高信心 OOS 誤派率**、LLM 呼叫率、每 1K 成本，並列 oracle 上限與「不確定性抓到的可回收錯誤比例」；Haiku 與小模型用同一批查詢、同一個 8 類標籤空間、同一支解析程式；Haiku 花費 ≤ US$5 | results JSON 含實際 token 用量與花費 |
| AC7 | CI 綠燈：ruff、pytest、em dash 守門、commit hygiene | GitHub Actions |
| AC8 | README 首屏數字全部由 `make report` 從 JSON 產生，不手打；結論依結果寫；兩個層級不混比；README 明寫兩種 8 類聚合方式（argmax intent 再對應 vs 按 agent 加總機率），以及最終採用哪種是只用 validation 決定的 | report-check 比對 |
| AC9 | 模型上 Hugging Face Hub，model card 寫明限制（英文、CLINC 領域、OOS 數字） | **上架前需 Drew 同意** |
| AC10 | LoRA：生成模型全量與 10-shot 各 3 seeds，報 8 類準確率、OOS recall、ECE、推論延遲、訓練峰值記憶體；逐筆候選標籤機率以 AC3 同格式存檔，RQ3/RQ4 分析不改程式就能跑 | results JSON + 同一支分析指令 |
| AC11 | QLoRA：至少 1 次全量，與 LoRA 同超參數，報峰值記憶體與準確率差；執行環境（MPS 或 Kaggle CUDA）與相依版本寫進結果檔 | results JSON；Kaggle 路徑附可重跑的腳本 |
| AC12 | ONNX int8（主模型 ModernBERT；匯出不支援時改 BERT 並寫明）：8 類準確率相對 PyTorch 下降 ≤ 0.5pp；報 CPU batch 1 的 p50/p95 延遲、模型大小，並記錄硬體型號 | `make bench`，結果含硬體資訊 |
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

| 步驟 | 內容 | 層 |
|---|---|---|
| 1 | 骨架 + CI + 資料載入（階段 3–5）✅ 2026-09-23 | |
| 2 | BERT 全量 × 3 seeds，過 AC2（流程檢查，同時存 val/test logits） | Tier 1 |
| 3 | ModernBERT 相容性試跑 → 抽樣模組 → 兩條學習曲線 + OOS 消融 + 多數類 / TF-IDF 基準（RQ1、RQ2） | Tier 1 |
| 4 | Haiku 重跑 + 不確定性分析 + fallback + oracle（RQ3、RQ4） | Tier 1 |
| 5 | 成本表（RQ5）+ README 初版（Tier 1 在這裡就是完整作品） | Tier 1 |
| 6 | ONNX int8 + FastAPI + Docker + 壓測（RQ7），完成後更新 RQ5 的實測延遲 | Tier 2 |
| 7 | Qwen 試跑 → LoRA → QLoRA（RQ6） | Tier 3 |
| 8 | README 定稿 + model card（AC9 上架前問 Drew） | |

## 8. 第二個 benchmark（v0.2，Drew 2026-09-23 決定）

候選：Zhang et al. 2022 的 **BANKING77-OOS / CLINC-Single-Domain-OOS**。它區分 in-domain OOS（同領域但不支援的請求）與 out-of-domain OOS，比 CLINC 標準 OOS 更接近 agent routing 的真實難點，而且格式與本專案流程相同（單標籤分類 + OOS），不需要新的模型結構。
排在 v0.1 做完、AC2 通過之後（v0.2），用同一條流程重跑。

## 9. 凍結後的想法（v0.1 不做，記下來）

（目前為空。新想法寫在這裡，不改 §3 到 §5。）
