.PHONY: setup lint format test test-network smoke train evaluate ac2 verify-logits report clean-checkpoints

CONFIG ?= configs/bert-base.yaml
SEED ?= 42

# Load HF_TOKEN / ANTHROPIC_API_KEY from a gitignored .env if one exists.
ENV_FILE := $(wildcard .env)
UV_ENV := $(if $(ENV_FILE),--env-file $(ENV_FILE),)

# The workspace lives inside iCloud Drive. iCloud never syncs a path ending
# in `.nosync`, so the virtualenv and checkpoints live in `*.nosync`
# directories and the conventional names are symlinks to them. Without
# this, iCloud tries to upload a multi-GB torch install and every
# checkpoint, and can evict files mid-training. See README "iCloud".
NOSYNC_LINKS := .venv checkpoints

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
ac2:
	uv run $(UV_ENV) python -m tinyrouter.ac2 --config configs/bert-base.yaml $(if $(filter 1,$(FORCE)),--force,)

# Every archive listed in results/logits-manifest.json is present and matches its SHA-256.
verify-logits:
	uv run python -m tinyrouter.archive

report:
	uv run python -m tinyrouter.report

# Removes every trained model. Disk is tight (see docs/PLAN.md section 6).
clean-checkpoints:
	rm -rf checkpoints.nosync/*
