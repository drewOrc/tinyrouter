# Operations

TinyRouter is a research repository with no deployment target: nothing runs as a service, and nobody but its author depends on it. That sets how much operations it needs.

## CI

`.github/workflows/ci.yml` runs on every pull request, every push to `main`, weekly, and on demand.

| job | runs | required to merge |
|---|---|---|
| `test` | tracked-files guard, `uv sync --locked`, `make lint`, `make test` (offline) | yes |
| `commit-hygiene` | rejects tool-attribution trailers in commit messages (patterns in `.github/disallowed-trailers.txt`) | yes |
| `network` | `make test-network` and `make smoke` against the Hugging Face Hub | no |

`network` is not required because it depends on a third-party service. A Hub outage would otherwise block every merge, including ones that do not touch data loading, and a check people learn to override stops meaning anything. It still runs on every PR, and a red `network` on a PR that changes `data.py`, `smoke.py` or a config is a reason not to merge. The weekly run skips the cache, so it is the one that notices if a pinned upstream dataset or model disappears.

`main` is protected: changes go through a pull request, `test` and `commit-hygiene` must pass, and force pushes and branch deletion are refused.

## Rolling back

There is no deployment to roll back. A bad change is undone with `git revert <sha>` in a pull request, which goes through the same CI. History on `main` is never rewritten.

Results in `results/*.json` are produced by the code at a given commit, so reverting the code and re-running `make evaluate` regenerates them; the JSON records the model revision it was scored with.

## Model weights

Weights never go in git (`.gitignore` and the tracked-files guard both refuse `*.safetensors`, `*.bin`, `*.pt`, `*.onnx`). When a trained model is published to the Hugging Face Hub, anything that loads it pins a `revision` (a commit hash, not a branch name), the same way `configs/*.yaml` pin the base models. Rolling back a published model means pointing that `revision` at the previous commit; the Hub keeps every revision.
