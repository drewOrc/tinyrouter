.PHONY: setup lint format test test-network smoke train evaluate ac2 pilot-lr pilot-steps baselines \
	curve oos-ablation verify-logits llm-smoke llm verify-llm analysis bench-cpu llm-latency cost figures report \
	reproduce reproduce-artifacts clean-checkpoints

CONFIG ?= configs/bert-base.yaml
SEED ?= 42

# Where output goes. Empty (the default) keeps each config's own paths,
# results/ and checkpoints/. `make reproduce` sets both to
# reproduction/<id>/... (and README_OUT for `make report`) so a rerun is
# written next to the committed results, never over them.
RESULTS_ROOT ?=
CHECKPOINT_ROOT ?=
README_OUT ?=
ROOT_ARGS := $(if $(RESULTS_ROOT),--results-root $(RESULTS_ROOT),)
LOCATION_ARGS := $(ROOT_ARGS) $(if $(CHECKPOINT_ROOT),--checkpoint-root $(CHECKPOINT_ROOT),)

# Load HF_TOKEN / ANTHROPIC_API_KEY from a gitignored .env if one exists.
ENV_FILE := $(wildcard .env)
UV_ENV := $(if $(ENV_FILE),--env-file $(ENV_FILE),)

# The workspace lives inside iCloud Drive. iCloud never syncs a path ending
# in `.nosync`, so the virtualenv and checkpoints live in `*.nosync`
# directories and the conventional names are symlinks to them. Without
# this, iCloud tries to upload a multi-GB torch install and every
# checkpoint, and can evict files mid-training. See README "iCloud".
NOSYNC_LINKS := .venv checkpoints reproduction

# No default target. On 2026-09-23 a zsh loop ran `make $t` with
# t="curve MODEL=bert"; make got one argument, read it as a variable
# assignment, ran the first target (setup) and exited 0, and two curves
# were reported done with no point run. A bare `make` (or only variable
# assignments) now fails instead of silently running setup.
# tests/test_makefile.py pins this.
ifeq ($(strip $(MAKECMDGOALS)),)
$(error no target given; a quoted "curve MODEL=bert" is one variable assignment, not a target)
endif

setup:
	@for name in $(NOSYNC_LINKS); do \
		if [ -L "$$name" ]; then continue; fi; \
		if [ -e "$$name" ]; then \
			echo "error: $$name exists and is not a symlink; move it to $$name.nosync and rerun make setup"; \
			exit 1; \
		fi; \
		mkdir -p "$$name.nosync" && ln -s "$$name.nosync" "$$name" && echo "linked $$name -> $$name.nosync"; \
	done
	uv sync --locked

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run python scripts/check_em_dash.py

format:
	uv run ruff check --fix .
	uv run ruff format .

# Offline unit tests. Tests that download from the Hugging Face Hub carry
# the `network` marker and are excluded by pyproject's addopts.
test:
	uv run pytest

test-network:
	uv run pytest -m network

# End to end on real data with a ~1 MB random BERT: download CLINC150
# (checksum verified), train one step on a few dozen rows on CPU, evaluate.
# Writes under the system temp dir, not checkpoints/.
smoke:
	uv run $(UV_ENV) python -m tinyrouter.smoke --config configs/smoke.yaml

train:
	uv run $(UV_ENV) python -m tinyrouter.train --config $(CONFIG) --seed $(SEED)

evaluate:
	uv run $(UV_ENV) python -m tinyrouter.evaluate --config $(CONFIG) --seed $(SEED)

# AC2: bert-base-uncased, full data, seeds 42/43/44 -> results/ac2.json; exit 1 on FAIL.
# Resumes: seeds already scored and archived are skipped. FORCE=1 reruns all.
# Weights of seeds 43 and 44 are deleted after their logits are archived.
# FORCE=1 also deletes seed 42's kept weights before retraining. After
# `make clean-checkpoints`, seed 42's weights are not rebuilt by a plain
# `make ac2` (its results already exist); getting them back takes FORCE=1,
# which retrains all three seeds.
ac2:
	uv run $(UV_ENV) python -m tinyrouter.ac2 --config configs/bert-base.yaml $(LOCATION_ARGS) $(if $(filter 1,$(FORCE)),--force,)

# Step 3 (docs/PLAN.md section 4, hyperparameter protocol). Order:
#   make pilot-lr     then copy each model's selected lr into configs/curve.yaml
#   make pilot-steps  then copy the selected S_min into configs/curve.yaml
#   make curve MODEL=bert ; make curve MODEL=modernbert ; make oos-ablation
# Pilots read and write validation numbers only (results/pilots/*.json) and
# never edit configs/curve.yaml. Curves and the ablation resume like ac2 and
# keep no weights, only logits. `curve` runs the cheap baselines first.
# `make baselines` runs them alone.
# Completion: each command's last line is `completed N/N ...` only after its
# index passed the checks in src/tinyrouter/completeness.py. `make curve`
# prints `completed 36/36 baseline points` before the curve starts, so a
# curve that fails later still has one completed line in its output. To
# decide a curve is done, match the whole line
# `completed 18/18 encoder points (<model>)` (the ablation:
# `completed 3/3 ablation points`), never just `completed`.
pilot-lr:
	uv run $(UV_ENV) python -m tinyrouter.pilots lr $(LOCATION_ARGS)

pilot-steps:
	uv run $(UV_ENV) python -m tinyrouter.pilots steps $(LOCATION_ARGS)

# Majority-class and TF-IDF centroid baselines on every (k, seed) sample; seconds.
baselines:
	uv run python -m tinyrouter.baselines $(ROOT_ARGS)

# Checks configs/curve.yaml is filled in, then runs the baselines, then the curve.
curve:
	@case "$(MODEL)" in bert|modernbert) ;; *) echo "usage: make curve MODEL=bert|modernbert"; exit 2;; esac
	uv run $(UV_ENV) python -m tinyrouter.curves --model $(MODEL) $(LOCATION_ARGS)

oos-ablation:
	uv run $(UV_ENV) python -m tinyrouter.curves --ablation $(LOCATION_ARGS)

# Every archive listed in results/logits-manifest.json is present and matches its SHA-256.
verify-logits:
	uv run python -m tinyrouter.archive $(ROOT_ARGS)

# Claude Haiku over CLINC150 in the 8-way routing space (docs/PLAN.md section 4,
# AC6). Needs ANTHROPIC_API_KEY (in .env or exported); without it they exit 2.
# Every call is journaled; a rerun calls only rows with no stored reply, and a
# change of model, prompt, temperature or max_tokens starts a fresh journal.
# A call is not started if it could take this run's identity past MAX_USD
# (default 5). Exit 1 when stopped by the cap or when a call failed after
# retries. `make llm-smoke` calls validation rows 0-19, writes under
# results/llm-smoke/ and prints the extrapolated cost of all 8,600 rows.
# Done means the whole last line `completed 8600/8600 llm predictions`.
llm-smoke:
	uv run --group llm $(UV_ENV) python -m tinyrouter.llm_run --smoke $(if $(MAX_USD),--max-usd $(MAX_USD),)

llm:
	uv run --group llm $(UV_ENV) python -m tinyrouter.llm_run $(if $(MAX_USD),--max-usd $(MAX_USD),) $(ROOT_ARGS)

# results/llm/haiku-8way.jsonl has every row once and matches its SHA-256 in
# results/llm-manifest.json and results/llm/haiku-8way.json.
verify-llm:
	uv run python -m tinyrouter.llm_run --verify $(ROOT_ARGS)

# RQ2 to RQ4 from the stored logits and Haiku predictions (docs/PLAN.md sections 3
# and 4, AC3, AC4, AC6): no training, no API calls. Needs the archives in
# results/logits/ and results/llm/haiku-8way.jsonl (GitHub Releases; check them
# with verify-logits and verify-llm). Writes results/analysis/{summary,curves,haiku}.json.
# Done means the whole last line `completed analysis (75/75 archives, 8600/8600 llm
# rows, 25 groups)`.
analysis:
	uv run python -m tinyrouter.analysis_run --quiet $(ROOT_ARGS)

# AC5 latency. bench-cpu: both encoders on CPU, batch 1, validation rows
# 0-499, 4 threads, the pretrained backbone with a 151-way head (latency
# depends on shapes, not weight values; see src/tinyrouter/latency.py);
# downloads the two base models; writes results/efficiency/cpu_latency.json.
# llm-latency: Haiku's per-call latency from results/llm/haiku-8way.jsonl
# (Release; check it with verify-llm); writes results/efficiency/haiku_latency.json.
bench-cpu:
	uv run $(UV_ENV) python -m tinyrouter.latency cpu $(ROOT_ARGS)

llm-latency:
	uv run python -m tinyrouter.latency haiku $(ROOT_ARGS)

# RQ5 from committed JSON only: measured numbers and assumed prices kept
# apart, break-even per scenario; writes results/cost/cost.json.
cost:
	uv run python -m tinyrouter.cost $(ROOT_ARGS)

# README figures from results/analysis/*.json; writes results/figures/*.png.
figures:
	uv run --group figures python -m tinyrouter.figures $(ROOT_ARGS)

# results/report.md and the README block between the BEGIN/END GENERATED
# markers, from committed JSON. tests/test_report.py fails when either is stale.
# Order after new results: analysis, bench-cpu, llm-latency, cost, figures, report.
report:
	uv run python -m tinyrouter.report $(if $(RESULTS_ROOT),--results-dir $(RESULTS_ROOT),) $(if $(README_OUT),--readme-out $(README_OUT),)

# AC1a (docs/PLAN.md section 5): download the three Releases into results/,
# check every file against the committed manifests, rebuild analysis, Haiku
# latency, cost, figures and report offline, and require results/ and
# README.md to be byte-identical to HEAD. No training, no API call, minutes.
# CPU latency is not rerun (machine-dependent); the committed file is used.
# Done means the last line `completed reproduce-artifacts: results/ and
# README.md byte-identical to HEAD`.
reproduce-artifacts: setup
	uv run python -m tinyrouter.reproduce artifacts

# AC1b (docs/PLAN.md section 5.1): the whole study again, about 6 to 7 hours
# on an Apple M4, Haiku about US$3 under its own US$5 reproduction-validation
# cap. Run it on a clean clone at a merged commit after `make setup`; it
# stops before anything else unless the tree is clean, HEAD is on
# origin/main, uv.lock is in sync, ANTHROPIC_API_KEY is set and 8.8 GiB are
# free. Everything goes under reproduction/<id>/ (id = HEAD[:12] unless
# REPRO_ID is set); results/ and README.md are never written. Resumes.
# Verdict and differences: reproduction/<id>/comparison.{json,md}.
reproduce:
	uv run $(UV_ENV) python -m tinyrouter.reproduce full $(if $(REPRO_ID),--id $(REPRO_ID),)

# Removes every trained model. Disk is tight (see docs/PLAN.md section 6).
clean-checkpoints:
	rm -rf checkpoints.nosync/*
