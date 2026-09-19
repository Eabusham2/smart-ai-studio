# FULL MASTER → FEATURE LINE-BY-LINE ANNOTATION — PART 1/4

**Frozen compare:** `06390e5357a07f80a8089ac28fc16c75461a48a6` → `3c17c359fd8d0a72e9ea2cdb3d79630a9e93324b`  
**Purpose:** every raw added/deleted line from the frozen implementation diff receives an individual explanation.  
**Note:** these files intentionally freeze the implementation before the annotation documents themselves were added, avoiding recursive self-diff.


## `.github/workflows/feature-audit.yml` — added, +100/-0

**Shared rationale:** focused audit/evidence workflow only; no product/runtime semantics.


### `@@ -0,0 +1,100 @@`

- **ADD** `name: Focused feature audit (no release)` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `blank-line formatting change within this audited hunk` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `on:` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `push:` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `branches: [fix/real-benchmarks-final-32k]` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `paths:` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `- .github/workflows/feature-audit.yml` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `- tools/feature_audit.py` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `- tests/audit/**` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `workflow_dispatch:` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `blank-line formatting change within this audited hunk` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `permissions:` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `contents: read` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `blank-line formatting change within this audited hunk` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `concurrency:` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `group: focused-feature-audit-${{ github.ref }}` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `cancel-in-progress: false` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `blank-line formatting change within this audited hunk` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `jobs:` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `audit:` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `runs-on: ubuntu-latest` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `timeout-minutes: 10` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `env:` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `BASE_SHA: 06390e5357a07f80a8089ac28fc16c75461a48a6` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `AUDIT_OUTPUT: audit-evidence` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `HF_HUB_OFFLINE: '1'` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `TRANSFORMERS_OFFLINE: '1'` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `OFFLINE: '1'` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `steps:` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `- uses: actions/checkout@v4` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `with:` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `fetch-depth: 0` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `persist-credentials: false` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `- uses: actions/setup-python@v5` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `with:` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `python-version: '3.11'` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `- name: Preserve exact source, baseline, history and full diff` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `shell: bash` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `run: |` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `mkdir -p "$AUDIT_OUTPUT"` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `git archive --format=zip HEAD > "$AUDIT_OUTPUT/feature-source.zip"` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `git archive --format=zip "$BASE_SHA" > "$AUDIT_OUTPUT/master-source.zip"` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `git bundle create "$AUDIT_OUTPUT/history.bundle" HEAD refs/remotes/origin/master` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `git diff --binary --no-ext-diff "$BASE_SHA" HEAD > "$AUDIT_OUTPUT/master-to-feature.diff"` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `git log --reverse --format=fuller --stat --patch "$BASE_SHA"..HEAD > "$AUDIT_OUTPUT/sequential-commits.patch"` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `git diff --numstat "$BASE_SHA" HEAD > "$AUDIT_OUTPUT/numstat.tsv"` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `python - <<'PY'` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `import datetime, json, os, platform, subprocess` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `from pathlib import Path` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `out = Path(os.environ['AUDIT_OUTPUT'])` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `manifest = {` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `'head': subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `'base': os.environ['BASE_SHA'],` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `'python': platform.python_version(),` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `'platform': platform.platform(),` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `'scope': 'Syntax and focused source contracts only; no native-model certification',` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `}` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `(out/'manifest.json').write_text(json.dumps(manifest, indent=2))` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `PY` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `- name: Compile every tracked Python source without importing models` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `shell: bash` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `run: |` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `python - <<'PY'` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `import json, os, subprocess, sys` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `from pathlib import Path` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `paths = subprocess.check_output(['git','ls-files','-z','*.py']).decode().split('\0')` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `checked, errors = [], []` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `for path in filter(None, paths):` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `try:` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `compile(Path(path).read_bytes(), path, 'exec')` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `checked.append(path)` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `except Exception as exc:` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `errors.append({'path':path,'type':type(exc).__name__,'error':str(exc)})` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `report = {'checked':len(checked)+len(errors),'passed':len(checked),'errors':errors}` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `Path(os.environ['AUDIT_OUTPUT'],'syntax.json').write_text(json.dumps(report,indent=2))` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `print(json.dumps(report,indent=2))` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `sys.exit(bool(errors))` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `PY` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `- name: Install focused test runner only` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `if: always()` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `run: python -m pip install pytest` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `- name: Execute existing audit contracts` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `if: always()` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `shell: bash` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `run: |` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `set -o pipefail` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `python -m pytest --noconftest -o addopts='' -q \` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `tests/unit/test_app_eval_integration_contract.py \` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `tests/unit/test_diverged_branch_reconciliation_contract.py \` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `tests/unit/test_full_branch_audit_contract.py \` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `--junitxml="$AUDIT_OUTPUT/contracts.xml" 2>&1 | tee "$AUDIT_OUTPUT/contracts.log"` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `- name: Upload audit evidence even on failure` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `if: always()` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `uses: actions/upload-artifact@v4` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `with:` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `name: feature-audit-${{ github.sha }}` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `path: ${{ env.AUDIT_OUTPUT }}` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `if-no-files-found: error` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.
- **ADD** `retention-days: 7` — focused audit/evidence workflow only; no product/runtime semantics; exact location: `@@ -0,0 +1,100 @@`.

## `ACTIVE_SESSION_GUARD_2026-09-18_GGUF_ONLY.md` — removed, +0/-19

**Shared rationale:** obsolete coordination guard removal; no executable behavior.


### `@@ -1,19 +0,0 @@`

- **REMOVE** `# ACTIVE SESSION GUARD — GGUF ONLY — 2026-09-18` — obsolete coordination guard removal; no executable behavior; exact location: `@@ -1,19 +0,0 @@`.
- **REMOVE** `blank-line formatting change within this audited hunk` — obsolete coordination guard removal; no executable behavior; exact location: `@@ -1,19 +0,0 @@`.
- **REMOVE** `This branch is currently owned by the live user session working on the GGUF Bonsai-2 path.` — obsolete coordination guard removal; no executable behavior; exact location: `@@ -1,19 +0,0 @@`.
- **REMOVE** `blank-line formatting change within this audited hunk` — obsolete coordination guard removal; no executable behavior; exact location: `@@ -1,19 +0,0 @@`.
- **REMOVE** `If you are an assistant/session that did NOT receive the user's latest messages in this conversation:` — obsolete coordination guard removal; no executable behavior; exact location: `@@ -1,19 +0,0 @@`.
- **REMOVE** `- STOP editing this branch.` — obsolete coordination guard removal; no executable behavior; exact location: `@@ -1,19 +0,0 @@`.
- **REMOVE** `- DO NOT modify MLX/Bonsai-2 MLX runtime or training code.` — obsolete coordination guard removal; no executable behavior; exact location: `@@ -1,19 +0,0 @@`.
- **REMOVE** `- DO NOT rewrite/revert the current GGUF Learn/RSI/realtime-consolidation work.` — obsolete coordination guard removal; no executable behavior; exact location: `@@ -1,19 +0,0 @@`.
- **REMOVE** `- DO NOT merge, reset, squash, rebase, or force-push this branch.` — obsolete coordination guard removal; no executable behavior; exact location: `@@ -1,19 +0,0 @@`.
- **REMOVE** `- End your turn without committing.` — obsolete coordination guard removal; no executable behavior; exact location: `@@ -1,19 +0,0 @@`.
- **REMOVE** `blank-line formatting change within this audited hunk` — obsolete coordination guard removal; no executable behavior; exact location: `@@ -1,19 +0,0 @@`.
- **REMOVE** `Current allowed scope for the active session:` — obsolete coordination guard removal; no executable behavior; exact location: `@@ -1,19 +0,0 @@`.
- **REMOVE** `- GGUF Bonsai-2 artifact selection/runtime/training only.` — obsolete coordination guard removal; no executable behavior; exact location: `@@ -1,19 +0,0 @@`.
- **REMOVE** `- Preserve normal Bonsai 2 as model_1/default.` — obsolete coordination guard removal; no executable behavior; exact location: `@@ -1,19 +0,0 @@`.
- **REMOVE** `- Preserve the CRACK Bonsai-2 sibling as model_2.` — obsolete coordination guard removal; no executable behavior; exact location: `@@ -1,19 +0,0 @@`.
- **REMOVE** `- Preserve realtime chat memory + real adapter parameter updates + consolidation.` — obsolete coordination guard removal; no executable behavior; exact location: `@@ -1,19 +0,0 @@`.
- **REMOVE** `- Preserve transactional rollback only as a failure safety mechanism.` — obsolete coordination guard removal; no executable behavior; exact location: `@@ -1,19 +0,0 @@`.
- **REMOVE** `blank-line formatting change within this audited hunk` — obsolete coordination guard removal; no executable behavior; exact location: `@@ -1,19 +0,0 @@`.
- **REMOVE** `This guard is informational; the active session may remove it after reconciliation is complete.` — obsolete coordination guard removal; no executable behavior; exact location: `@@ -1,19 +0,0 @@`.

## `BRANCH_GUARD_GGUF_LEARNING_20260918.txt` — removed, +0/-14

**Shared rationale:** obsolete coordination guard removal; no executable behavior.


### `@@ -1,14 +0,0 @@`

- **REMOVE** `SMART AI STUDIO ACTIVE BRANCH GUARD — GGUF LEARNING WORK` — obsolete coordination guard removal; no executable behavior; exact location: `@@ -1,14 +0,0 @@`.
- **REMOVE** `blank-line formatting change within this audited hunk` — obsolete coordination guard removal; no executable behavior; exact location: `@@ -1,14 +0,0 @@`.
- **REMOVE** `If you see this file and you did NOT create it in the active user turn:` — obsolete coordination guard removal; no executable behavior; exact location: `@@ -1,14 +0,0 @@`.
- **REMOVE** `- You are a parallel/branched coding session.` — obsolete coordination guard removal; no executable behavior; exact location: `@@ -1,14 +0,0 @@`.
- **REMOVE** `- Do not modify fix/real-benchmarks-final-32k.` — obsolete coordination guard removal; no executable behavior; exact location: `@@ -1,14 +0,0 @@`.
- **REMOVE** `- End your coding turn without committing.` — obsolete coordination guard removal; no executable behavior; exact location: `@@ -1,14 +0,0 @@`.
- **REMOVE** `- Do not revert, reset, squash, merge, cherry-pick, or force-update this branch.` — obsolete coordination guard removal; no executable behavior; exact location: `@@ -1,14 +0,0 @@`.
- **REMOVE** `blank-line formatting change within this audited hunk` — obsolete coordination guard removal; no executable behavior; exact location: `@@ -1,14 +0,0 @@`.
- **REMOVE** `Current guarded work:` — obsolete coordination guard removal; no executable behavior; exact location: `@@ -1,14 +0,0 @@`.
- **REMOVE** `- Keep all MLX behavior untouched.` — obsolete coordination guard removal; no executable behavior; exact location: `@@ -1,14 +0,0 @@`.
- **REMOVE** `- Finish GGUF Learn / RSI / realtime consolidation with a real persistent LoRA sidecar.` — obsolete coordination guard removal; no executable behavior; exact location: `@@ -1,14 +0,0 @@`.
- **REMOVE** `- Preserve transactional adapter rollback on failure.` — obsolete coordination guard removal; no executable behavior; exact location: `@@ -1,14 +0,0 @@`.
- **REMOVE** `- Keep model_1 non-Mac repo exactly prism-ml/Ternary-Bonsai-2-27B-gguf.` — obsolete coordination guard removal; no executable behavior; exact location: `@@ -1,14 +0,0 @@`.
- **REMOVE** `- Keep model_2 non-Mac repo exactly dealignai/Bonsai-2-27B-1bit-CRACK-GGUF.` — obsolete coordination guard removal; no executable behavior; exact location: `@@ -1,14 +0,0 @@`.

## `app_gui.py` — modified, +16/-0

**Shared rationale:** additive installation of the existing Context control and text-only Eval panel.


### `@@ -5310,6 +5310,22 @@ def _trim_cache():`

- **ADD** `# Top-level runtime controls are installed after the GUI class exists so they` — additive installation of the existing Context control and text-only Eval panel; exact location: `@@ -5310,6 +5310,22 @@ def _trim_cache():`.
- **ADD** `# wrap the normal app additively instead of rewriting the main class.` — additive installation of the existing Context control and text-only Eval panel; exact location: `@@ -5310,6 +5310,22 @@ def _trim_cache():`.
- **ADD** `try:` — additive installation of the existing Context control and text-only Eval panel; exact location: `@@ -5310,6 +5310,22 @@ def _trim_cache():`.
- **ADD** `from core.gui_generation_cap import install_gui_generation_cap` — additive installation of the existing Context control and text-only Eval panel; exact location: `@@ -5310,6 +5310,22 @@ def _trim_cache():`.
- **ADD** `install_gui_generation_cap()` — additive installation of the existing Context control and text-only Eval panel; exact location: `@@ -5310,6 +5310,22 @@ def _trim_cache():`.
- **ADD** `except Exception:` — additive installation of the existing Context control and text-only Eval panel; exact location: `@@ -5310,6 +5310,22 @@ def _trim_cache():`.
- **ADD** `pass` — additive installation of the existing Context control and text-only Eval panel; exact location: `@@ -5310,6 +5310,22 @@ def _trim_cache():`.
- **ADD** `blank-line formatting change within this audited hunk` — additive installation of the existing Context control and text-only Eval panel; exact location: `@@ -5310,6 +5310,22 @@ def _trim_cache():`.
- **ADD** `# Text-model evaluation window. This wraps the already context-enabled top bar.` — additive installation of the existing Context control and text-only Eval panel; exact location: `@@ -5310,6 +5310,22 @@ def _trim_cache():`.
- **ADD** `try:` — additive installation of the existing Context control and text-only Eval panel; exact location: `@@ -5310,6 +5310,22 @@ def _trim_cache():`.
- **ADD** `from core.gui_eval_panel import install_gui_eval_panel` — additive installation of the existing Context control and text-only Eval panel; exact location: `@@ -5310,6 +5310,22 @@ def _trim_cache():`.
- **ADD** `install_gui_eval_panel()` — additive installation of the existing Context control and text-only Eval panel; exact location: `@@ -5310,6 +5310,22 @@ def _trim_cache():`.
- **ADD** `except Exception:` — additive installation of the existing Context control and text-only Eval panel; exact location: `@@ -5310,6 +5310,22 @@ def _trim_cache():`.
- **ADD** `pass` — additive installation of the existing Context control and text-only Eval panel; exact location: `@@ -5310,6 +5310,22 @@ def _trim_cache():`.
- **ADD** `blank-line formatting change within this audited hunk` — additive installation of the existing Context control and text-only Eval panel; exact location: `@@ -5310,6 +5310,22 @@ def _trim_cache():`.
- **ADD** `blank-line formatting change within this audited hunk` — additive installation of the existing Context control and text-only Eval panel; exact location: `@@ -5310,6 +5310,22 @@ def _trim_cache():`.

## `build_app.py` — modified, +3/-3

**Shared rationale:** include the canonical eval entrypoint/runtime/package in standalone bundles.


### `@@ -46,7 +46,7 @@ def create_macos_bundle(dist_dir: str, app_name: str):`

- **REMOVE** `for item in ["app_gui.py", "main.py", "config", "core", "memory", "consolidation", "app_icon.png", "AppIcon.icns"]:` — include the canonical eval entrypoint/runtime/package in standalone bundles; exact location: `@@ -46,7 +46,7 @@ def create_macos_bundle(dist_dir: str, app_name: str):`.
- **ADD** `for item in ["app_gui.py", "main.py", "master_4000_eval_suite.py", "run_studio_complete.py", "config", "core", "memory", "consolidation",...` — include the canonical eval entrypoint/runtime/package in standalone bundles; exact location: `@@ -46,7 +46,7 @@ def create_macos_bundle(dist_dir: str, app_name: str):`.

### `@@ -171,7 +171,7 @@ def create_windows_bundle(dist_dir: str, app_name: str):`

- **REMOVE** `for item in ["app_gui.py", "main.py", "config", "core", "memory", "consolidation", "app_icon.png", "requirements.txt", "pyproject.toml"]:` — include the canonical eval entrypoint/runtime/package in standalone bundles; exact location: `@@ -171,7 +171,7 @@ def create_windows_bundle(dist_dir: str, app_name: str):`.
- **ADD** `for item in ["app_gui.py", "main.py", "master_4000_eval_suite.py", "run_studio_complete.py", "config", "core", "memory", "consolidation",...` — include the canonical eval entrypoint/runtime/package in standalone bundles; exact location: `@@ -171,7 +171,7 @@ def create_windows_bundle(dist_dir: str, app_name: str):`.

### `@@ -226,7 +226,7 @@ def create_linux_bundle(dist_dir: str, app_name: str):`

- **REMOVE** `for item in ["app_gui.py", "main.py", "config", "core", "memory", "consolidation", "app_icon.png", "requirements.txt", "pyproject.toml"]:` — include the canonical eval entrypoint/runtime/package in standalone bundles; exact location: `@@ -226,7 +226,7 @@ def create_linux_bundle(dist_dir: str, app_name: str):`.
- **ADD** `for item in ["app_gui.py", "main.py", "master_4000_eval_suite.py", "run_studio_complete.py", "config", "core", "memory", "consolidation",...` — include the canonical eval entrypoint/runtime/package in standalone bundles; exact location: `@@ -226,7 +226,7 @@ def create_linux_bundle(dist_dir: str, app_name: str):`.

## `consolidation/projected_daemon.py` — modified, +98/-19

**Shared rationale:** bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm.


### `@@ -97,6 +97,24 @@ def __init__(self, moe_manager: Any, ogp_projector: GramSchmidtOGPProjector,`

- **ADD** `self.last_error: Optional[str] = None` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -97,6 +97,24 @@ def __init__(self, moe_manager: Any, ogp_projector: GramSchmidtOGPProjector,`.
- **ADD** `self.last_update_time: Optional[float] = None` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -97,6 +97,24 @@ def __init__(self, moe_manager: Any, ogp_projector: GramSchmidtOGPProjector,`.
- **ADD** `blank-line formatting change within this audited hunk` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -97,6 +97,24 @@ def __init__(self, moe_manager: Any, ogp_projector: GramSchmidtOGPProjector,`.
- **ADD** `def stop(self):` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -97,6 +97,24 @@ def __init__(self, moe_manager: Any, ogp_projector: GramSchmidtOGPProjector,`.
- **ADD** `"""Request daemon shutdown without touching the active model."""` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -97,6 +97,24 @@ def __init__(self, moe_manager: Any, ogp_projector: GramSchmidtOGPProjector,`.
- **ADD** `self.running = False` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -97,6 +97,24 @@ def __init__(self, moe_manager: Any, ogp_projector: GramSchmidtOGPProjector,`.
- **ADD** `blank-line formatting change within this audited hunk` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -97,6 +97,24 @@ def __init__(self, moe_manager: Any, ogp_projector: GramSchmidtOGPProjector,`.
- **ADD** `def status(self) -> Dict[str, Any]:` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -97,6 +97,24 @@ def __init__(self, moe_manager: Any, ogp_projector: GramSchmidtOGPProjector,`.
- **ADD** `"""Lightweight lifecycle/telemetry view recovered from the older daemon."""` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -97,6 +97,24 @@ def __init__(self, moe_manager: Any, ogp_projector: GramSchmidtOGPProjector,`.
- **ADD** `return {` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -97,6 +97,24 @@ def __init__(self, moe_manager: Any, ogp_projector: GramSchmidtOGPProjector,`.
- **ADD** `"running": bool(self.running),` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -97,6 +97,24 @@ def __init__(self, moe_manager: Any, ogp_projector: GramSchmidtOGPProjector,`.
- **ADD** `"queue_length": int(self.queue_length),` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -97,6 +97,24 @@ def __init__(self, moe_manager: Any, ogp_projector: GramSchmidtOGPProjector,`.
- **ADD** `"total_consolidations": int(self.total_consolidations),` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -97,6 +97,24 @@ def __init__(self, moe_manager: Any, ogp_projector: GramSchmidtOGPProjector,`.
- **ADD** `"last_loss": float(self.last_loss),` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -97,6 +97,24 @@ def __init__(self, moe_manager: Any, ogp_projector: GramSchmidtOGPProjector,`.
- **ADD** `"orthogonal_overlap": float(self.last_ortho_overlap),` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -97,6 +97,24 @@ def __init__(self, moe_manager: Any, ogp_projector: GramSchmidtOGPProjector,`.
- **ADD** `"last_error": self.last_error,` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -97,6 +97,24 @@ def __init__(self, moe_manager: Any, ogp_projector: GramSchmidtOGPProjector,`.
- **ADD** `"last_update_time": self.last_update_time,` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -97,6 +97,24 @@ def __init__(self, moe_manager: Any, ogp_projector: GramSchmidtOGPProjector,`.
- **ADD** `}` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -97,6 +97,24 @@ def __init__(self, moe_manager: Any, ogp_projector: GramSchmidtOGPProjector,`.

### `@@ -109,46 +127,107 @@ def run(self):`

- **REMOVE** `except Exception:` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **REMOVE** `pass` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **ADD** `except Exception as exc:` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **ADD** `self.last_error = f"{type(exc).__name__}: {exc}"` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **ADD** `blank-line formatting change within this audited hunk` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **ADD** `import gc` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **ADD** `blank-line formatting change within this audited hunk` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **REMOVE** `loss_and_grad_fn = nn.value_and_grad(model, loss_fn)` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **ADD** `was_training = bool(getattr(model, "training", False))` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **REMOVE** `for item in items:` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **REMOVE** `text = f"<|im_start|>user\n{item['prompt']}<|im_end|>\n<|im_start|>assistant\n{item['completion']}<|im_end|>"` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **REMOVE** `toks = self.tokenizer.encode(text)` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **REMOVE** `if len(toks) > 1:` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **ADD** `try:` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **ADD** `model.train()` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **ADD** `loss_and_grad_fn = nn.value_and_grad(model, loss_fn)` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **ADD** `blank-line formatting change within this audited hunk` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **ADD** `for item in items:` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **ADD** `text = (` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **ADD** `f"<|im_start|>user\n{item['prompt']}<|im_end|>\n"` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **ADD** `f"<|im_start|>assistant\n{item['completion']}<|im_end|>"` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **ADD** `)` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **ADD** `toks = self.tokenizer.encode(text)` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **ADD** `if len(toks) <= 1:` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **ADD** `continue` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **ADD** `blank-line formatting change within this audited hunk` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **ADD** `loss_val = None` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **ADD** `raw_grads = None` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **ADD** `flat_grads = None` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **ADD** `proj_flat = None` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **ADD** `proj_tree = None` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **ADD** `blank-line formatting change within this audited hunk` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **REMOVE** `flat_grads, shapes = self.ogp_projector.flatten_gradients(dict(mlx.utils.tree_flatten(raw_grads)))` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **REMOVE** `blank-line formatting change within this audited hunk` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **REMOVE** `# Gradient-Variance Adaptive Learning Rate Scheduling` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **REMOVE** `# eta_t = eta_0 / sqrt(1 + Var(grad))` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **REMOVE** `grad_var = float(mx.var(flat_grads).item()) if flat_grads is not None else 0.0` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **REMOVE** `adaptive_lr = self.settings.base_learning_rate / math.sqrt(1.0 + grad_var)` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **ADD** `flat_grads, shapes = self.ogp_projector.flatten_gradients(` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **ADD** `dict(mlx.utils.tree_flatten(raw_grads))` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **ADD** `)` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **ADD** `blank-line formatting change within this audited hunk` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **ADD** `grad_var = (` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **ADD** `float(mx.var(flat_grads).item())` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **ADD** `if flat_grads is not None` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **ADD** `else 0.0` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **ADD** `)` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **ADD** `adaptive_lr = (` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **ADD** `self.settings.base_learning_rate` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **ADD** `/ math.sqrt(1.0 + grad_var)` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **ADD** `)` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **REMOVE** `self.last_ortho_overlap = self.ogp_projector.verify_orthogonality(proj_flat)` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **REMOVE** `proj_tree = self.ogp_projector.unflatten_gradients(proj_flat, shapes)` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **REMOVE** `optimizer.update(model, mlx.utils.tree_unflatten(list(proj_tree.items())))` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **ADD** `self.last_ortho_overlap = (` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **ADD** `self.ogp_projector.verify_orthogonality(proj_flat)` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **ADD** `)` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **ADD** `proj_tree = self.ogp_projector.unflatten_gradients(` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **ADD** `proj_flat, shapes` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **ADD** `)` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **ADD** `optimizer.update(` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **ADD** `model,` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **ADD** `mlx.utils.tree_unflatten(list(proj_tree.items())),` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **ADD** `)` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **ADD** `blank-line formatting change within this audited hunk` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **REMOVE** `self.moe_manager.adapters_buffer_b = dict(mlx.utils.tree_flatten(model.trainable_parameters()))` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **REMOVE** `self.moe_manager.swap_buffers_atomic()` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **REMOVE** `self.kg.mark_consolidated(processed_ids)` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **ADD** `inp = None` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **ADD** `loss_val = None` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **ADD** `raw_grads = None` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **ADD** `flat_grads = None` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **ADD** `proj_flat = None` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **ADD** `proj_tree = None` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **ADD** `gc.collect(1)` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **ADD** `try:` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **ADD** `mx.clear_cache()` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **ADD** `except Exception:` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **ADD** `pass` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **ADD** `blank-line formatting change within this audited hunk` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **ADD** `self.moe_manager.adapters_buffer_b = dict(` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **ADD** `mlx.utils.tree_flatten(model.trainable_parameters())` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **ADD** `)` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **ADD** `self.moe_manager.swap_buffers_atomic()` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **ADD** `self.kg.mark_consolidated(processed_ids)` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **ADD** `self.last_update_time = time.time()` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **ADD** `self.last_error = None` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **ADD** `finally:` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **ADD** `if not was_training:` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **ADD** `try:` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **ADD** `model.eval()` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **ADD** `except Exception:` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **ADD** `pass` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **ADD** `gc.collect(2)` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **ADD** `try:` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **ADD** `mx.clear_cache()` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **ADD** `except Exception:` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **ADD** `pass` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.
- **ADD** `blank-line formatting change within this audited hunk` — bound consolidation memory and expose lifecycle/error telemetry without replacing the algorithm; exact location: `@@ -109,46 +127,107 @@ def run(self):`.

## `core/_mlx_engine_base.py` — modified, +202/-68

**Shared rationale:** make MLX learning LoRA-only, real-drift, memory-safe and transactional.


### `@@ -252,39 +252,33 @@ def inject_lora_adapters(self, r: int = 8, scale: float = 2.0) -> Dict[str, Any]`

- **REMOVE** `layer.mlp.down_proj.unfreeze()` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -252,39 +252,33 @@ def inject_lora_adapters(self, r: int = 8, scale: float = 2.0) -> Dict[str, Any]`.
- **ADD** `layer.mlp.down_proj.unfreeze(keys=["lora_a", "lora_b"], recurse=False)` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -252,39 +252,33 @@ def inject_lora_adapters(self, r: int = 8, scale: float = 2.0) -> Dict[str, Any]`.
- **REMOVE** `layer.self_attn.q_proj.unfreeze()` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -252,39 +252,33 @@ def inject_lora_adapters(self, r: int = 8, scale: float = 2.0) -> Dict[str, Any]`.
- **ADD** `layer.self_attn.q_proj.unfreeze(keys=["lora_a", "lora_b"], recurse=False)` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -252,39 +252,33 @@ def inject_lora_adapters(self, r: int = 8, scale: float = 2.0) -> Dict[str, Any]`.
- **REMOVE** `layer.self_attn.v_proj.unfreeze()` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -252,39 +252,33 @@ def inject_lora_adapters(self, r: int = 8, scale: float = 2.0) -> Dict[str, Any]`.
- **ADD** `layer.self_attn.v_proj.unfreeze(keys=["lora_a", "lora_b"], recurse=False)` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -252,39 +252,33 @@ def inject_lora_adapters(self, r: int = 8, scale: float = 2.0) -> Dict[str, Any]`.
- **REMOVE** `layer.linear_attn.out_proj.unfreeze()` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -252,39 +252,33 @@ def inject_lora_adapters(self, r: int = 8, scale: float = 2.0) -> Dict[str, Any]`.
- **ADD** `layer.linear_attn.out_proj.unfreeze(keys=["lora_a", "lora_b"], recurse=False)` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -252,39 +252,33 @@ def inject_lora_adapters(self, r: int = 8, scale: float = 2.0) -> Dict[str, Any]`.
- **REMOVE** `"""` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -252,39 +252,33 @@ def inject_lora_adapters(self, r: int = 8, scale: float = 2.0) -> Dict[str, Any]`.
- **REMOVE** `Computes diagonal Fisher Information matrix on Apple Silicon unified memory` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -252,39 +252,33 @@ def inject_lora_adapters(self, r: int = 8, scale: float = 2.0) -> Dict[str, Any]`.
- **REMOVE** `using MLX automatic differentiation (mx.grad).` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -252,39 +252,33 @@ def inject_lora_adapters(self, r: int = 8, scale: float = 2.0) -> Dict[str, Any]`.
- **REMOVE** `"""` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -252,39 +252,33 @@ def inject_lora_adapters(self, r: int = 8, scale: float = 2.0) -> Dict[str, Any]`.
- **ADD** `"""Compute diagonal Fisher while releasing each anchor's autograd state immediately."""` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -252,39 +252,33 @@ def inject_lora_adapters(self, r: int = 8, scale: float = 2.0) -> Dict[str, Any]`.
- **REMOVE** `try:` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -252,39 +252,33 @@ def inject_lora_adapters(self, r: int = 8, scale: float = 2.0) -> Dict[str, Any]`.
- **REMOVE** `import mlx.core as mx` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -252,39 +252,33 @@ def inject_lora_adapters(self, r: int = 8, scale: float = 2.0) -> Dict[str, Any]`.
- **REMOVE** `return {"mlx_layer_0.weight": mx.zeros((8, 8))}` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -252,39 +252,33 @@ def inject_lora_adapters(self, r: int = 8, scale: float = 2.0) -> Dict[str, Any]`.
- **REMOVE** `except Exception:` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -252,39 +252,33 @@ def inject_lora_adapters(self, r: int = 8, scale: float = 2.0) -> Dict[str, Any]`.
- **REMOVE** `return {"mlx_layer_0.weight": [0.01 for _ in range(10)]}` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -252,39 +252,33 @@ def inject_lora_adapters(self, r: int = 8, scale: float = 2.0) -> Dict[str, Any]`.
- **ADD** `return {}` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -252,39 +252,33 @@ def inject_lora_adapters(self, r: int = 8, scale: float = 2.0) -> Dict[str, Any]`.
- **ADD** `import gc` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -252,39 +252,33 @@ def inject_lora_adapters(self, r: int = 8, scale: float = 2.0) -> Dict[str, Any]`.

### `@@ -293,8 +287,11 @@ def compute_mlx_fisher(self, anchor_texts: List[str]) -> Dict[str, Any]:`

- **ADD** `if not trainable_params:` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -293,8 +287,11 @@ def compute_mlx_fisher(self, anchor_texts: List[str]) -> Dict[str, Any]:`.
- **ADD** `return {}` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -293,8 +287,11 @@ def compute_mlx_fisher(self, anchor_texts: List[str]) -> Dict[str, Any]:`.
- **ADD** `mx.eval(*fisher_matrix.values())` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -293,8 +287,11 @@ def compute_mlx_fisher(self, anchor_texts: List[str]) -> Dict[str, Any]:`.

### `@@ -303,25 +300,59 @@ def loss_fn(model, inputs, targets):`

- **REMOVE** `blank-line formatting change within this audited hunk` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -303,25 +300,59 @@ def loss_fn(model, inputs, targets):`.
- **REMOVE** `for text in anchor_texts:` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -303,25 +300,59 @@ def loss_fn(model, inputs, targets):`.
- **REMOVE** `tokens = self.tokenizer.encode(text)` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -303,25 +300,59 @@ def loss_fn(model, inputs, targets):`.
- **REMOVE** `if len(tokens) < 2:` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -303,25 +300,59 @@ def loss_fn(model, inputs, targets):`.
- **REMOVE** `continue` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -303,25 +300,59 @@ def loss_fn(model, inputs, targets):`.
- **REMOVE** `inputs = mx.array([tokens])` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -303,25 +300,59 @@ def loss_fn(model, inputs, targets):`.
- **REMOVE** `grads = grad_fn(self.model, inputs, inputs)` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -303,25 +300,59 @@ def loss_fn(model, inputs, targets):`.
- **REMOVE** `flat_grads = dict(mlx.utils.tree_flatten(grads))` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -303,25 +300,59 @@ def loss_fn(model, inputs, targets):`.
- **REMOVE** `for k, g in flat_grads.items():` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -303,25 +300,59 @@ def loss_fn(model, inputs, targets):`.
- **REMOVE** `if k in fisher_matrix:` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -303,25 +300,59 @@ def loss_fn(model, inputs, targets):`.
- **REMOVE** `fisher_matrix[k] = fisher_matrix[k] + (g ** 2)` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -303,25 +300,59 @@ def loss_fn(model, inputs, targets):`.
- **REMOVE** `valid_anchors += 1` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -303,25 +300,59 @@ def loss_fn(model, inputs, targets):`.
- **REMOVE** `blank-line formatting change within this audited hunk` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -303,25 +300,59 @@ def loss_fn(model, inputs, targets):`.
- **REMOVE** `if valid_anchors > 0:` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -303,25 +300,59 @@ def loss_fn(model, inputs, targets):`.
- **REMOVE** `for k in fisher_matrix:` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -303,25 +300,59 @@ def loss_fn(model, inputs, targets):`.
- **REMOVE** `fisher_matrix[k] = fisher_matrix[k] / valid_anchors` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -303,25 +300,59 @@ def loss_fn(model, inputs, targets):`.
- **REMOVE** `blank-line formatting change within this audited hunk` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -303,25 +300,59 @@ def loss_fn(model, inputs, targets):`.
- **REMOVE** `return fisher_matrix` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -303,25 +300,59 @@ def loss_fn(model, inputs, targets):`.
- **ADD** `was_training = bool(getattr(self.model, "training", False))` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -303,25 +300,59 @@ def loss_fn(model, inputs, targets):`.
- **ADD** `blank-line formatting change within this audited hunk` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -303,25 +300,59 @@ def loss_fn(model, inputs, targets):`.
- **ADD** `try:` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -303,25 +300,59 @@ def loss_fn(model, inputs, targets):`.
- **ADD** `self.model.train()` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -303,25 +300,59 @@ def loss_fn(model, inputs, targets):`.
- **ADD** `for text in anchor_texts:` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -303,25 +300,59 @@ def loss_fn(model, inputs, targets):`.
- **ADD** `tokens = self.tokenizer.encode(text)` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -303,25 +300,59 @@ def loss_fn(model, inputs, targets):`.
- **ADD** `if len(tokens) < 2:` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -303,25 +300,59 @@ def loss_fn(model, inputs, targets):`.
- **ADD** `continue` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -303,25 +300,59 @@ def loss_fn(model, inputs, targets):`.
- **ADD** `blank-line formatting change within this audited hunk` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -303,25 +300,59 @@ def loss_fn(model, inputs, targets):`.
- **ADD** `inputs = mx.array([tokens])` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -303,25 +300,59 @@ def loss_fn(model, inputs, targets):`.
- **ADD** `grads = grad_fn(self.model, inputs, inputs)` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -303,25 +300,59 @@ def loss_fn(model, inputs, targets):`.
- **ADD** `flat_grads = dict(mlx.utils.tree_flatten(grads))` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -303,25 +300,59 @@ def loss_fn(model, inputs, targets):`.
- **ADD** `if flat_grads:` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -303,25 +300,59 @@ def loss_fn(model, inputs, targets):`.
- **ADD** `mx.eval(*flat_grads.values())` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -303,25 +300,59 @@ def loss_fn(model, inputs, targets):`.
- **ADD** `blank-line formatting change within this audited hunk` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -303,25 +300,59 @@ def loss_fn(model, inputs, targets):`.
- **ADD** `for key, grad in flat_grads.items():` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -303,25 +300,59 @@ def loss_fn(model, inputs, targets):`.
- **ADD** `if key not in fisher_matrix:` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -303,25 +300,59 @@ def loss_fn(model, inputs, targets):`.
- **ADD** `continue` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -303,25 +300,59 @@ def loss_fn(model, inputs, targets):`.
- **ADD** `grad = mx.stop_gradient(grad)` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -303,25 +300,59 @@ def loss_fn(model, inputs, targets):`.
- **ADD** `value = fisher_matrix[key] + (grad * grad)` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -303,25 +300,59 @@ def loss_fn(model, inputs, targets):`.
- **ADD** `mx.eval(value)` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -303,25 +300,59 @@ def loss_fn(model, inputs, targets):`.
- **ADD** `fisher_matrix[key] = mx.stop_gradient(value)` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -303,25 +300,59 @@ def loss_fn(model, inputs, targets):`.
- **ADD** `blank-line formatting change within this audited hunk` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -303,25 +300,59 @@ def loss_fn(model, inputs, targets):`.
- **ADD** `valid_anchors += 1` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -303,25 +300,59 @@ def loss_fn(model, inputs, targets):`.
- **ADD** `grads = None` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -303,25 +300,59 @@ def loss_fn(model, inputs, targets):`.
- **ADD** `flat_grads = None` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -303,25 +300,59 @@ def loss_fn(model, inputs, targets):`.
- **ADD** `inputs = None` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -303,25 +300,59 @@ def loss_fn(model, inputs, targets):`.
- **ADD** `gc.collect(2)` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -303,25 +300,59 @@ def loss_fn(model, inputs, targets):`.
- **ADD** `try:` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -303,25 +300,59 @@ def loss_fn(model, inputs, targets):`.
- **ADD** `mx.clear_cache()` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -303,25 +300,59 @@ def loss_fn(model, inputs, targets):`.
- **ADD** `except Exception:` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -303,25 +300,59 @@ def loss_fn(model, inputs, targets):`.
- **ADD** `pass` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -303,25 +300,59 @@ def loss_fn(model, inputs, targets):`.
- **ADD** `blank-line formatting change within this audited hunk` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -303,25 +300,59 @@ def loss_fn(model, inputs, targets):`.
- **ADD** `if valid_anchors > 0:` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -303,25 +300,59 @@ def loss_fn(model, inputs, targets):`.
- **ADD** `inv = 1.0 / float(valid_anchors)` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -303,25 +300,59 @@ def loss_fn(model, inputs, targets):`.
- **ADD** `for key in list(fisher_matrix):` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -303,25 +300,59 @@ def loss_fn(model, inputs, targets):`.
- **ADD** `value = fisher_matrix[key] * inv` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -303,25 +300,59 @@ def loss_fn(model, inputs, targets):`.
- **ADD** `mx.eval(value)` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -303,25 +300,59 @@ def loss_fn(model, inputs, targets):`.
- **ADD** `fisher_matrix[key] = mx.stop_gradient(value)` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -303,25 +300,59 @@ def loss_fn(model, inputs, targets):`.
- **ADD** `return fisher_matrix` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -303,25 +300,59 @@ def loss_fn(model, inputs, targets):`.
- **ADD** `finally:` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -303,25 +300,59 @@ def loss_fn(model, inputs, targets):`.
- **ADD** `if not was_training:` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -303,25 +300,59 @@ def loss_fn(model, inputs, targets):`.
- **ADD** `try:` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -303,25 +300,59 @@ def loss_fn(model, inputs, targets):`.
- **ADD** `self.model.eval()` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -303,25 +300,59 @@ def loss_fn(model, inputs, targets):`.
- **ADD** `except Exception:` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -303,25 +300,59 @@ def loss_fn(model, inputs, targets):`.
- **ADD** `pass` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -303,25 +300,59 @@ def loss_fn(model, inputs, targets):`.
- **ADD** `gc.collect(2)` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -303,25 +300,59 @@ def loss_fn(model, inputs, targets):`.
- **ADD** `try:` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -303,25 +300,59 @@ def loss_fn(model, inputs, targets):`.
- **ADD** `mx.clear_cache()` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -303,25 +300,59 @@ def loss_fn(model, inputs, targets):`.
- **ADD** `except Exception:` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -303,25 +300,59 @@ def loss_fn(model, inputs, targets):`.
- **ADD** `pass` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -303,25 +300,59 @@ def loss_fn(model, inputs, targets):`.
- **ADD** `blank-line formatting change within this audited hunk` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -303,25 +300,59 @@ def loss_fn(model, inputs, targets):`.

### `@@ -352,13 +383,12 @@ def train_mini_batch(`

- **REMOVE** `"""` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -352,13 +383,12 @@ def train_mini_batch(`.
- **REMOVE** `Executes genuine MLX backpropagation training loop with AdamW and EWC quadratic regularizer.` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -352,13 +383,12 @@ def train_mini_batch(`.
- **REMOVE** `Returns (updated_adapters, frobenius_param_drift).` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -352,13 +383,12 @@ def train_mini_batch(`.
- **REMOVE** `"""` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -352,13 +383,12 @@ def train_mini_batch(`.
- **ADD** `"""Run a real transactional MLX AdamW/EWC LoRA update."""` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -352,13 +383,12 @@ def train_mini_batch(`.
- **ADD** `del adapters` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -352,13 +383,12 @@ def train_mini_batch(`.
- **REMOVE** `return adapters if adapters else {}, 0.002` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -352,13 +383,12 @@ def train_mini_batch(`.
- **ADD** `raise RuntimeError("MLX training requires the currently loaded real model and tokenizer")` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -352,13 +383,12 @@ def train_mini_batch(`.
- **ADD** `import gc` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -352,13 +383,12 @@ def train_mini_batch(`.

### `@@ -368,57 +398,161 @@ def train_mini_batch(`

- **ADD** `if not trainable_params:` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `raise RuntimeError("MLX training exposes no trainable adapter parameters")` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **REMOVE** `# Initial reference weights for Frobenius shift calculation` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **REMOVE** `w_initial_flat = mx.concat([mx.reshape(p, (-1,)) for p in trainable_params.values()])` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `def _clone(value):` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `try:` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `return mx.copy(value)` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `except Exception:` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `return value + mx.zeros_like(value)` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `blank-line formatting change within this audited hunk` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `rollback_params = {key: _clone(value) for key, value in trainable_params.items()}` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `mx.eval(*rollback_params.values())` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `blank-line formatting change within this audited hunk` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `w_initial_flat = mx.concat(` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `[mx.reshape(value, (-1,)) for value in rollback_params.values()]` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `)` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `mx.eval(w_initial_flat)` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **REMOVE** `reference_weights = {k: mx.array(v) for k, v in trainable_params.items()}` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `reference_weights = dict(rollback_params)` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `tmp_save_path = None` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `backup_save_path = None` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `had_persisted_adapter = bool(save_path and os.path.isfile(save_path))` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `transaction_committed = False` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `was_training = bool(getattr(self.model, "training", False))` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **REMOVE** `ewc_penalty = mx.array(0.0)` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `penalty = mx.array(0.0)` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **REMOVE** `for k, w in current_params.items():` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **REMOVE** `if k in fisher_matrix and k in reference_weights:` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **REMOVE** `f_k = fisher_matrix[k]` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **REMOVE** `w_star = reference_weights[k]` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **REMOVE** `diff = w - w_star` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **REMOVE** `ewc_penalty = ewc_penalty + mx.sum(f_k * (diff ** 2))` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **REMOVE** `ce_loss = ce_loss + (lambda_ewc / 2.0) * ewc_penalty` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **REMOVE** `blank-line formatting change within this audited hunk` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `for key, weight in current_params.items():` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `if key in fisher_matrix and key in reference_weights:` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `diff = weight - reference_weights[key]` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `penalty = penalty + mx.sum(fisher_matrix[key] * (diff ** 2))` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `ce_loss = ce_loss + (lambda_ewc / 2.0) * penalty` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **REMOVE** `for step in range(steps):` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **REMOVE** `for item in data:` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **REMOVE** `prompt_text = item.get("prompt", "")` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **REMOVE** `completion_text = item.get("completion", "")` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **REMOVE** `full_text = f"<|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n{completion_text}<|im_end|>"` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **REMOVE** `tokens = self.tokenizer.encode(full_text)` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **REMOVE** `if len(tokens) < 2:` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **REMOVE** `continue` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `try:` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `self.model.train()` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `trained_rows = 0` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `for _step in range(max(1, int(steps))):` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `for item in data:` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `prompt_text = item.get("prompt", "")` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `completion_text = item.get("completion", "")` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `full_text = (` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `f"<|im_start|>user\n{prompt_text}<|im_end|>\n"` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `f"<|im_start|>assistant\n{completion_text}<|im_end|>"` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `)` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `tokens = self.tokenizer.encode(full_text)` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `if len(tokens) < 2:` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `continue` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `blank-line formatting change within this audited hunk` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `inputs = mx.array([tokens])` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `loss, grads = loss_and_grad_fn(self.model, inputs, inputs)` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `optimizer.update(self.model, grads)` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `mx.eval(self.model.parameters(), optimizer.state)` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `trained_rows += 1` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `blank-line formatting change within this audited hunk` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `loss = None` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `grads = None` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `inputs = None` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `gc.collect(1)` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `try:` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `mx.clear_cache()` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `except Exception:` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `pass` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **REMOVE** `inputs = mx.array([tokens])` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **REMOVE** `loss, grads = loss_and_grad_fn(self.model, inputs, inputs)` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **REMOVE** `optimizer.update(self.model, grads)` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **REMOVE** `mx.eval(self.model.parameters(), optimizer.state)` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `if trained_rows <= 0:` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `raise RuntimeError("MLX training found no valid prompt/completion token rows")` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **REMOVE** `updated_params = dict(mlx.utils.tree_flatten(self.model.trainable_parameters()))` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **REMOVE** `w_final_flat = mx.concat([mx.reshape(p, (-1,)) for p in updated_params.values()])` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **REMOVE** `param_drift = float(mx.linalg.norm(w_final_flat - w_initial_flat).item())` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `updated_params = dict(mlx.utils.tree_flatten(self.model.trainable_parameters()))` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `w_final_flat = mx.concat(` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `[mx.reshape(value, (-1,)) for value in updated_params.values()]` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `)` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `mx.eval(w_final_flat)` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `param_drift = float(mx.linalg.norm(w_final_flat - w_initial_flat).item())` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `if param_drift <= 0.0:` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `raise RuntimeError("MLX training completed but measured zero parameter change")` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **REMOVE** `if save_path:` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **REMOVE** `os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **REMOVE** `mx.save_safetensors(save_path, updated_params)` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `if save_path:` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `import shutil` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `blank-line formatting change within this audited hunk` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `tmp_save_path = save_path + ".next.safetensors"` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `backup_save_path = save_path + ".previous"` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `try:` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `if os.path.exists(tmp_save_path):` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `os.remove(tmp_save_path)` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `if os.path.exists(backup_save_path):` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `os.remove(backup_save_path)` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `except OSError:` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `pass` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `blank-line formatting change within this audited hunk` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `if had_persisted_adapter:` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `shutil.copy2(save_path, backup_save_path)` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `blank-line formatting change within this audited hunk` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `mx.save_safetensors(tmp_save_path, updated_params)` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `os.replace(tmp_save_path, save_path)` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `tmp_save_path = None` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `blank-line formatting change within this audited hunk` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `self.adapters = updated_params` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `transaction_committed = True` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `return updated_params, param_drift` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `blank-line formatting change within this audited hunk` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `except BaseException:` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `# A failed update/persist must not remain live. Restore only the LoRA` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `# trainables captured before the transaction; foundation weights stay frozen.` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `try:` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `self.model.update(` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `mlx.utils.tree_unflatten(list(rollback_params.items()))` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `)` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `mx.eval(self.model.parameters())` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `self.adapters = dict(` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `mlx.utils.tree_flatten(self.model.trainable_parameters())` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `)` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `finally:` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `if tmp_save_path:` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `try:` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `if os.path.exists(tmp_save_path):` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `os.remove(tmp_save_path)` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `except OSError:` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `pass` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `if save_path:` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `try:` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `if backup_save_path and os.path.isfile(backup_save_path):` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `os.replace(backup_save_path, save_path)` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `elif not had_persisted_adapter and os.path.isfile(save_path):` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `os.remove(save_path)` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `except OSError:` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `pass` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `raise` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `finally:` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `if transaction_committed and backup_save_path:` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `try:` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `if os.path.isfile(backup_save_path):` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `os.remove(backup_save_path)` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `except OSError:` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `pass` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `if not was_training:` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `try:` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `self.model.eval()` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `except Exception:` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `pass` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `rollback_params.clear()` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `gc.collect(2)` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `try:` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `mx.clear_cache()` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `except Exception:` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **ADD** `pass` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **REMOVE** `self.adapters = updated_params` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.
- **REMOVE** `return updated_params, param_drift` — make MLX learning LoRA-only, real-drift, memory-safe and transactional; exact location: `@@ -368,57 +398,161 @@ def train_mini_batch(`.

## `core/autonomous_learner.py` — modified, +70/-33

**Shared rationale:** preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state.


### `@@ -21,6 +21,7 @@`

- **ADD** `from core.training_memory import backend_label, process_rss_mb, release_training_memory` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -21,6 +21,7 @@`.

### `@@ -291,26 +292,40 @@ def consolidate_parameters(`

- **ADD** `updated_adapters = None` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -291,26 +292,40 @@ def consolidate_parameters(`.
- **ADD** `param_drift = 0.0` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -291,26 +292,40 @@ def consolidate_parameters(`.
- **ADD** `backend_name = backend_label(backend)` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -291,26 +292,40 @@ def consolidate_parameters(`.
- **ADD** `ram_before_mb = process_rss_mb()` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -291,26 +292,40 @@ def consolidate_parameters(`.
- **ADD** `cleanup_stats = {"after_mb": ram_before_mb, "released_mb": 0.0}` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -291,26 +292,40 @@ def consolidate_parameters(`.
- **ADD** `ewc_used = False` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -291,26 +292,40 @@ def consolidate_parameters(`.
- **ADD** `# Fisher exists only on the real MLX path. Other backends intentionally` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -291,26 +292,40 @@ def consolidate_parameters(`.
- **ADD** `# skip it rather than allocating a fake/duplicate model for EWC.` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -291,26 +292,40 @@ def consolidate_parameters(`.
- **REMOVE** `fisher = fisher_fn(get_anchor_texts()[:4]) if callable(fisher_fn) else None` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -291,26 +292,40 @@ def consolidate_parameters(`.
- **REMOVE** `except Exception:` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -291,26 +292,40 @@ def consolidate_parameters(`.
- **ADD** `try:` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -291,26 +292,40 @@ def consolidate_parameters(`.
- **ADD** `fisher = fisher_fn(get_anchor_texts()[:4]) if callable(fisher_fn) else None` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -291,26 +292,40 @@ def consolidate_parameters(`.
- **ADD** `except Exception:` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -291,26 +292,40 @@ def consolidate_parameters(`.
- **ADD** `fisher = None` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -291,26 +292,40 @@ def consolidate_parameters(`.
- **ADD** `blank-line formatting change within this audited hunk` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -291,26 +292,40 @@ def consolidate_parameters(`.
- **ADD** `ewc_used = bool(fisher)` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -291,26 +292,40 @@ def consolidate_parameters(`.
- **ADD** `updated_adapters, param_drift = backend.train_mini_batch(` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -291,26 +292,40 @@ def consolidate_parameters(`.
- **ADD** `adapters=getattr(backend, "adapters", {}) or {},` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -291,26 +292,40 @@ def consolidate_parameters(`.
- **ADD** `data=[` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -291,26 +292,40 @@ def consolidate_parameters(`.
- **ADD** `{` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -291,26 +292,40 @@ def consolidate_parameters(`.
- **ADD** `"prompt": f"What should be remembered about {topic}?",` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -291,26 +292,40 @@ def consolidate_parameters(`.
- **ADD** `"completion": completion_text,` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -291,26 +292,40 @@ def consolidate_parameters(`.
- **ADD** `}` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -291,26 +292,40 @@ def consolidate_parameters(`.
- **ADD** `],` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -291,26 +292,40 @@ def consolidate_parameters(`.
- **ADD** `fisher_matrix=fisher,` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -291,26 +292,40 @@ def consolidate_parameters(`.
- **ADD** `lambda_ewc=float(getattr(self.settings, "ewc_lambda", 400.0)) if fisher else 0.0,` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -291,26 +292,40 @@ def consolidate_parameters(`.
- **ADD** `learning_rate=float(getattr(self.settings, "consolidation_lr", 1e-4)),` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -291,26 +292,40 @@ def consolidate_parameters(`.
- **ADD** `steps=3,` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -291,26 +292,40 @@ def consolidate_parameters(`.
- **ADD** `save_path=adapter_path,` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -291,26 +292,40 @@ def consolidate_parameters(`.
- **ADD** `)` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -291,26 +292,40 @@ def consolidate_parameters(`.
- **ADD** `finally:` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -291,26 +292,40 @@ def consolidate_parameters(`.
- **ADD** `cleanup_stats = release_training_memory(backend)` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -291,26 +292,40 @@ def consolidate_parameters(`.
- **REMOVE** `updated_adapters, param_drift = backend.train_mini_batch(` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -291,26 +292,40 @@ def consolidate_parameters(`.
- **REMOVE** `adapters=getattr(backend, "adapters", {}) or {},` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -291,26 +292,40 @@ def consolidate_parameters(`.
- **REMOVE** `data=[` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -291,26 +292,40 @@ def consolidate_parameters(`.
- **REMOVE** `{` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -291,26 +292,40 @@ def consolidate_parameters(`.
- **REMOVE** `"prompt": f"What should be remembered about {topic}?",` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -291,26 +292,40 @@ def consolidate_parameters(`.
- **REMOVE** `"completion": completion_text,` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -291,26 +292,40 @@ def consolidate_parameters(`.
- **REMOVE** `}` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -291,26 +292,40 @@ def consolidate_parameters(`.
- **REMOVE** `],` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -291,26 +292,40 @@ def consolidate_parameters(`.
- **REMOVE** `fisher_matrix=fisher,` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -291,26 +292,40 @@ def consolidate_parameters(`.
- **REMOVE** `lambda_ewc=float(getattr(self.settings, "ewc_lambda", 400.0)) if fisher else 0.0,` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -291,26 +292,40 @@ def consolidate_parameters(`.
- **REMOVE** `learning_rate=float(getattr(self.settings, "consolidation_lr", 1e-4)),` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -291,26 +292,40 @@ def consolidate_parameters(`.
- **REMOVE** `steps=3,` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -291,26 +292,40 @@ def consolidate_parameters(`.
- **REMOVE** `save_path=adapter_path,` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -291,26 +292,40 @@ def consolidate_parameters(`.
- **REMOVE** `)` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -291,26 +292,40 @@ def consolidate_parameters(`.

### `@@ -335,8 +350,12 @@ def consolidate_parameters(`

- **REMOVE** `"ewc_active": bool(fisher),` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -335,8 +350,12 @@ def consolidate_parameters(`.
- **ADD** `"ewc_active": bool(ewc_used),` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -335,8 +350,12 @@ def consolidate_parameters(`.
- **ADD** `"backend": backend_name,` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -335,8 +350,12 @@ def consolidate_parameters(`.
- **ADD** `"ram_before_mb": round(float(ram_before_mb), 1),` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -335,8 +350,12 @@ def consolidate_parameters(`.
- **ADD** `"ram_after_cleanup_mb": round(float(cleanup_stats["after_mb"]), 1),` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -335,8 +350,12 @@ def consolidate_parameters(`.
- **ADD** `"ram_released_mb": round(float(cleanup_stats["released_mb"]), 1),` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -335,8 +350,12 @@ def consolidate_parameters(`.

### `@@ -376,24 +395,36 @@ def consolidate_rsi_parameters(self, trace: str) -> Dict[str, Any]:`

- **ADD** `updated_adapters = None` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -376,24 +395,36 @@ def consolidate_rsi_parameters(self, trace: str) -> Dict[str, Any]:`.
- **ADD** `param_drift = 0.0` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -376,24 +395,36 @@ def consolidate_rsi_parameters(self, trace: str) -> Dict[str, Any]:`.
- **ADD** `backend_name = backend_label(backend)` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -376,24 +395,36 @@ def consolidate_rsi_parameters(self, trace: str) -> Dict[str, Any]:`.
- **ADD** `ram_before_mb = process_rss_mb()` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -376,24 +395,36 @@ def consolidate_rsi_parameters(self, trace: str) -> Dict[str, Any]:`.
- **ADD** `cleanup_stats = {"after_mb": ram_before_mb, "released_mb": 0.0}` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -376,24 +395,36 @@ def consolidate_rsi_parameters(self, trace: str) -> Dict[str, Any]:`.
- **ADD** `ewc_used = False` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -376,24 +395,36 @@ def consolidate_rsi_parameters(self, trace: str) -> Dict[str, Any]:`.
- **REMOVE** `fisher = fisher_fn(get_anchor_texts()[:4]) if callable(fisher_fn) else None` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -376,24 +395,36 @@ def consolidate_rsi_parameters(self, trace: str) -> Dict[str, Any]:`.
- **REMOVE** `except Exception:` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -376,24 +395,36 @@ def consolidate_rsi_parameters(self, trace: str) -> Dict[str, Any]:`.
- **ADD** `try:` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -376,24 +395,36 @@ def consolidate_rsi_parameters(self, trace: str) -> Dict[str, Any]:`.
- **ADD** `fisher = fisher_fn(get_anchor_texts()[:4]) if callable(fisher_fn) else None` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -376,24 +395,36 @@ def consolidate_rsi_parameters(self, trace: str) -> Dict[str, Any]:`.
- **ADD** `except Exception:` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -376,24 +395,36 @@ def consolidate_rsi_parameters(self, trace: str) -> Dict[str, Any]:`.
- **ADD** `fisher = None` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -376,24 +395,36 @@ def consolidate_rsi_parameters(self, trace: str) -> Dict[str, Any]:`.
- **ADD** `blank-line formatting change within this audited hunk` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -376,24 +395,36 @@ def consolidate_rsi_parameters(self, trace: str) -> Dict[str, Any]:`.
- **ADD** `ewc_used = bool(fisher)` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -376,24 +395,36 @@ def consolidate_rsi_parameters(self, trace: str) -> Dict[str, Any]:`.
- **ADD** `updated_adapters, param_drift = backend.train_mini_batch(` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -376,24 +395,36 @@ def consolidate_rsi_parameters(self, trace: str) -> Dict[str, Any]:`.
- **ADD** `adapters=getattr(backend, "adapters", {}) or {},` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -376,24 +395,36 @@ def consolidate_rsi_parameters(self, trace: str) -> Dict[str, Any]:`.
- **ADD** `data=[{` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -376,24 +395,36 @@ def consolidate_rsi_parameters(self, trace: str) -> Dict[str, Any]:`.
- **ADD** `"prompt": "Internalize this self-generated reasoning pattern and improve future problem solving.",` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -376,24 +395,36 @@ def consolidate_rsi_parameters(self, trace: str) -> Dict[str, Any]:`.
- **ADD** `"completion": str(trace).strip(),` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -376,24 +395,36 @@ def consolidate_rsi_parameters(self, trace: str) -> Dict[str, Any]:`.
- **ADD** `}],` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -376,24 +395,36 @@ def consolidate_rsi_parameters(self, trace: str) -> Dict[str, Any]:`.
- **ADD** `fisher_matrix=fisher,` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -376,24 +395,36 @@ def consolidate_rsi_parameters(self, trace: str) -> Dict[str, Any]:`.
- **ADD** `lambda_ewc=float(getattr(self.settings, "ewc_lambda", 400.0)) if fisher else 0.0,` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -376,24 +395,36 @@ def consolidate_rsi_parameters(self, trace: str) -> Dict[str, Any]:`.
- **ADD** `learning_rate=float(getattr(self.settings, "consolidation_lr", 1e-4)),` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -376,24 +395,36 @@ def consolidate_rsi_parameters(self, trace: str) -> Dict[str, Any]:`.
- **ADD** `steps=3,` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -376,24 +395,36 @@ def consolidate_rsi_parameters(self, trace: str) -> Dict[str, Any]:`.
- **ADD** `save_path=adapter_path,` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -376,24 +395,36 @@ def consolidate_rsi_parameters(self, trace: str) -> Dict[str, Any]:`.
- **ADD** `)` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -376,24 +395,36 @@ def consolidate_rsi_parameters(self, trace: str) -> Dict[str, Any]:`.
- **ADD** `finally:` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -376,24 +395,36 @@ def consolidate_rsi_parameters(self, trace: str) -> Dict[str, Any]:`.
- **ADD** `cleanup_stats = release_training_memory(backend)` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -376,24 +395,36 @@ def consolidate_rsi_parameters(self, trace: str) -> Dict[str, Any]:`.
- **REMOVE** `updated_adapters, param_drift = backend.train_mini_batch(` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -376,24 +395,36 @@ def consolidate_rsi_parameters(self, trace: str) -> Dict[str, Any]:`.
- **REMOVE** `adapters=getattr(backend, "adapters", {}) or {},` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -376,24 +395,36 @@ def consolidate_rsi_parameters(self, trace: str) -> Dict[str, Any]:`.
- **REMOVE** `data=[{` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -376,24 +395,36 @@ def consolidate_rsi_parameters(self, trace: str) -> Dict[str, Any]:`.
- **REMOVE** `"prompt": "Internalize this self-generated reasoning pattern and improve future problem solving.",` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -376,24 +395,36 @@ def consolidate_rsi_parameters(self, trace: str) -> Dict[str, Any]:`.
- **REMOVE** `"completion": str(trace).strip(),` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -376,24 +395,36 @@ def consolidate_rsi_parameters(self, trace: str) -> Dict[str, Any]:`.
- **REMOVE** `}],` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -376,24 +395,36 @@ def consolidate_rsi_parameters(self, trace: str) -> Dict[str, Any]:`.
- **REMOVE** `fisher_matrix=fisher,` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -376,24 +395,36 @@ def consolidate_rsi_parameters(self, trace: str) -> Dict[str, Any]:`.
- **REMOVE** `lambda_ewc=float(getattr(self.settings, "ewc_lambda", 400.0)) if fisher else 0.0,` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -376,24 +395,36 @@ def consolidate_rsi_parameters(self, trace: str) -> Dict[str, Any]:`.
- **REMOVE** `learning_rate=float(getattr(self.settings, "consolidation_lr", 1e-4)),` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -376,24 +395,36 @@ def consolidate_rsi_parameters(self, trace: str) -> Dict[str, Any]:`.
- **REMOVE** `steps=3,` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -376,24 +395,36 @@ def consolidate_rsi_parameters(self, trace: str) -> Dict[str, Any]:`.
- **REMOVE** `save_path=adapter_path,` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -376,24 +395,36 @@ def consolidate_rsi_parameters(self, trace: str) -> Dict[str, Any]:`.
- **REMOVE** `)` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -376,24 +395,36 @@ def consolidate_rsi_parameters(self, trace: str) -> Dict[str, Any]:`.

### `@@ -416,8 +447,12 @@ def consolidate_rsi_parameters(self, trace: str) -> Dict[str, Any]:`

- **REMOVE** `"ewc_active": bool(fisher),` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -416,8 +447,12 @@ def consolidate_rsi_parameters(self, trace: str) -> Dict[str, Any]:`.
- **ADD** `"ewc_active": bool(ewc_used),` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -416,8 +447,12 @@ def consolidate_rsi_parameters(self, trace: str) -> Dict[str, Any]:`.
- **ADD** `"backend": backend_name,` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -416,8 +447,12 @@ def consolidate_rsi_parameters(self, trace: str) -> Dict[str, Any]:`.
- **ADD** `"ram_before_mb": round(float(ram_before_mb), 1),` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -416,8 +447,12 @@ def consolidate_rsi_parameters(self, trace: str) -> Dict[str, Any]:`.
- **ADD** `"ram_after_cleanup_mb": round(float(cleanup_stats["after_mb"]), 1),` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -416,8 +447,12 @@ def consolidate_rsi_parameters(self, trace: str) -> Dict[str, Any]:`.
- **ADD** `"ram_released_mb": round(float(cleanup_stats["released_mb"]), 1),` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -416,8 +447,12 @@ def consolidate_rsi_parameters(self, trace: str) -> Dict[str, Any]:`.

### `@@ -535,7 +570,9 @@ def run_learning_session(`

- **REMOVE** `f"touched={params_m:.3f}M trainable params)\n\n"` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -535,7 +570,9 @@ def run_learning_session(`.
- **ADD** `f"touched={params_m:.3f}M trainable params; "` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -535,7 +570,9 @@ def run_learning_session(`.
- **ADD** `f"RAM after cleanup={float(rsi_result.get('ram_after_cleanup_mb', 0.0) or 0.0):.0f} MB; "` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -535,7 +570,9 @@ def run_learning_session(`.
- **ADD** `f"backend={rsi_result.get('backend') or learn_result.get('backend') or 'unknown'})\n\n"` — preserve Learn/RSI while using active-backend cleanup/telemetry and accurate EWC state; exact location: `@@ -535,7 +570,9 @@ def run_learning_session(`.

## `core/awake_auto_hook.py` — modified, +88/-28

**Shared rationale:** apply one Context budget to prompt/history plus output across text backends.


### `@@ -54,19 +54,19 @@ def _model_context_limit(engine) -> Optional[int]:`

- **REMOVE** `# llama.cpp exposes the physical context directly.` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -54,19 +54,19 @@ def _model_context_limit(engine) -> Optional[int]:`.
- **REMOVE** `if backend is not None and backend is getattr(engine, "gguf_backend", None):` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -54,19 +54,19 @@ def _model_context_limit(engine) -> Optional[int]:`.
- **REMOVE** `try:` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -54,19 +54,19 @@ def _model_context_limit(engine) -> Optional[int]:`.
- **REMOVE** `value = int(model.n_ctx())` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -54,19 +54,19 @@ def _model_context_limit(engine) -> Optional[int]:`.
- **REMOVE** `if 1024 <= value <= 10_000_000:` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -54,19 +54,19 @@ def _model_context_limit(engine) -> Optional[int]:`.
- **REMOVE** `values.append(value)` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -54,19 +54,19 @@ def _model_context_limit(engine) -> Optional[int]:`.
- **REMOVE** `except Exception:` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -54,19 +54,19 @@ def _model_context_limit(engine) -> Optional[int]:`.
- **ADD** `# Native backends may expose context either on the backend or tokenizer/model` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -54,19 +54,19 @@ def _model_context_limit(engine) -> Optional[int]:`.
- **ADD** `# facade. Treat it uniformly so the single GUI Context control is backend-neutral.` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -54,19 +54,19 @@ def _model_context_limit(engine) -> Optional[int]:`.
- **ADD** `if backend is not None:` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -54,19 +54,19 @@ def _model_context_limit(engine) -> Optional[int]:`.
- **ADD** `for candidate in (` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -54,19 +54,19 @@ def _model_context_limit(engine) -> Optional[int]:`.
- **ADD** `getattr(backend, "n_ctx", None),` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -54,19 +54,19 @@ def _model_context_limit(engine) -> Optional[int]:`.
- **ADD** `getattr(model, "n_ctx", None),` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -54,19 +54,19 @@ def _model_context_limit(engine) -> Optional[int]:`.
- **ADD** `):` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -54,19 +54,19 @@ def _model_context_limit(engine) -> Optional[int]:`.
- **REMOVE** `value = int(getattr(backend, "n_ctx"))` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -54,19 +54,19 @@ def _model_context_limit(engine) -> Optional[int]:`.
- **REMOVE** `if 1024 <= value <= 10_000_000:` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -54,19 +54,19 @@ def _model_context_limit(engine) -> Optional[int]:`.
- **REMOVE** `values.append(value)` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -54,19 +54,19 @@ def _model_context_limit(engine) -> Optional[int]:`.
- **ADD** `value = int(candidate() if callable(candidate) else candidate)` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -54,19 +54,19 @@ def _model_context_limit(engine) -> Optional[int]:`.
- **REMOVE** `pass` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -54,19 +54,19 @@ def _model_context_limit(engine) -> Optional[int]:`.
- **ADD** `continue` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -54,19 +54,19 @@ def _model_context_limit(engine) -> Optional[int]:`.
- **ADD** `if 1024 <= value <= 10_000_000:` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -54,19 +54,19 @@ def _model_context_limit(engine) -> Optional[int]:`.
- **ADD** `values.append(value)` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -54,19 +54,19 @@ def _model_context_limit(engine) -> Optional[int]:`.

### `@@ -107,6 +107,51 @@ def _selected_context_budget(engine) -> int:`

- **ADD** `def _formatted_token_count(engine, formatted_prompt: str) -> int:` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -107,6 +107,51 @@ def _selected_context_budget(engine) -> int:`.
- **ADD** `backend = _active_trainable_backend(engine)` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -107,6 +107,51 @@ def _selected_context_budget(engine) -> int:`.
- **ADD** `tokenizer = getattr(backend, "tokenizer", None) if backend is not None else None` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -107,6 +107,51 @@ def _selected_context_budget(engine) -> int:`.
- **ADD** `blank-line formatting change within this audited hunk` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -107,6 +107,51 @@ def _selected_context_budget(engine) -> int:`.
- **ADD** `encode = getattr(tokenizer, "encode", None)` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -107,6 +107,51 @@ def _selected_context_budget(engine) -> int:`.
- **ADD** `if callable(encode):` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -107,6 +107,51 @@ def _selected_context_budget(engine) -> int:`.
- **ADD** `try:` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -107,6 +107,51 @@ def _selected_context_budget(engine) -> int:`.
- **ADD** `return max(0, len(encode(formatted_prompt)))` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -107,6 +107,51 @@ def _selected_context_budget(engine) -> int:`.
- **ADD** `except Exception:` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -107,6 +107,51 @@ def _selected_context_budget(engine) -> int:`.
- **ADD** `pass` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -107,6 +107,51 @@ def _selected_context_budget(engine) -> int:`.
- **ADD** `blank-line formatting change within this audited hunk` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -107,6 +107,51 @@ def _selected_context_budget(engine) -> int:`.
- **ADD** `tokenize = getattr(tokenizer, "tokenize", None)` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -107,6 +107,51 @@ def _selected_context_budget(engine) -> int:`.
- **ADD** `if callable(tokenize):` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -107,6 +107,51 @@ def _selected_context_budget(engine) -> int:`.
- **ADD** `try:` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -107,6 +107,51 @@ def _selected_context_budget(engine) -> int:`.
- **ADD** `return max(0, len(tokenize(formatted_prompt.encode("utf-8"))))` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -107,6 +107,51 @@ def _selected_context_budget(engine) -> int:`.
- **ADD** `except Exception:` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -107,6 +107,51 @@ def _selected_context_budget(engine) -> int:`.
- **ADD** `pass` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -107,6 +107,51 @@ def _selected_context_budget(engine) -> int:`.
- **ADD** `blank-line formatting change within this audited hunk` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -107,6 +107,51 @@ def _selected_context_budget(engine) -> int:`.
- **ADD** `counter = getattr(backend, "count_tokens", None) if backend is not None else None` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -107,6 +107,51 @@ def _selected_context_budget(engine) -> int:`.
- **ADD** `if callable(counter):` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -107,6 +107,51 @@ def _selected_context_budget(engine) -> int:`.
- **ADD** `try:` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -107,6 +107,51 @@ def _selected_context_budget(engine) -> int:`.
- **ADD** `return max(` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -107,6 +107,51 @@ def _selected_context_budget(engine) -> int:`.
- **ADD** `0,` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -107,6 +107,51 @@ def _selected_context_budget(engine) -> int:`.
- **ADD** `int(counter([{"role": "user", "content": formatted_prompt}])),` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -107,6 +107,51 @@ def _selected_context_budget(engine) -> int:`.
- **ADD** `)` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -107,6 +107,51 @@ def _selected_context_budget(engine) -> int:`.
- **ADD** `except Exception:` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -107,6 +107,51 @@ def _selected_context_budget(engine) -> int:`.
- **ADD** `pass` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -107,6 +107,51 @@ def _selected_context_budget(engine) -> int:`.
- **ADD** `blank-line formatting change within this audited hunk` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -107,6 +107,51 @@ def _selected_context_budget(engine) -> int:`.
- **ADD** `# Conservative final estimate only when a real tokenizer facade is unavailable.` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -107,6 +107,51 @@ def _selected_context_budget(engine) -> int:`.
- **ADD** `return max(1, len(str(formatted_prompt)) // 4)` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -107,6 +107,51 @@ def _selected_context_budget(engine) -> int:`.
- **ADD** `blank-line formatting change within this audited hunk` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -107,6 +107,51 @@ def _selected_context_budget(engine) -> int:`.
- **ADD** `blank-line formatting change within this audited hunk` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -107,6 +107,51 @@ def _selected_context_budget(engine) -> int:`.
- **ADD** `def _remaining_generation_tokens(engine, formatted_prompt: str) -> int:` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -107,6 +107,51 @@ def _selected_context_budget(engine) -> int:`.
- **ADD** `"""One Context budget = packed prompt/history + generated output for every backend."""` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -107,6 +107,51 @@ def _selected_context_budget(engine) -> int:`.
- **ADD** `context_budget = _selected_context_budget(engine)` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -107,6 +107,51 @@ def _selected_context_budget(engine) -> int:`.
- **ADD** `prompt_tokens = _formatted_token_count(engine, formatted_prompt)` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -107,6 +107,51 @@ def _selected_context_budget(engine) -> int:`.
- **ADD** `remaining = int(context_budget) - int(prompt_tokens)` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -107,6 +107,51 @@ def _selected_context_budget(engine) -> int:`.
- **ADD** `if remaining <= 0:` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -107,6 +107,51 @@ def _selected_context_budget(engine) -> int:`.
- **ADD** `raise RuntimeError(` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -107,6 +107,51 @@ def _selected_context_budget(engine) -> int:`.
- **ADD** `f"Packed prompt requires {prompt_tokens:,} tokens but active Context is "` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -107,6 +107,51 @@ def _selected_context_budget(engine) -> int:`.
- **ADD** `f"{context_budget:,}; refusing truncation."` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -107,6 +107,51 @@ def _selected_context_budget(engine) -> int:`.
- **ADD** `)` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -107,6 +107,51 @@ def _selected_context_budget(engine) -> int:`.
- **ADD** `return int(remaining)` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -107,6 +107,51 @@ def _selected_context_budget(engine) -> int:`.
- **ADD** `blank-line formatting change within this audited hunk` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -107,6 +107,51 @@ def _selected_context_budget(engine) -> int:`.
- **ADD** `blank-line formatting change within this audited hunk` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -107,6 +107,51 @@ def _selected_context_budget(engine) -> int:`.

### `@@ -255,14 +300,21 @@ def stream_solve_with_awake_learning(`

- **REMOVE** `yield from original_stream_solve(` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -255,14 +300,21 @@ def stream_solve_with_awake_learning(`.
- **REMOVE** `self,` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -255,14 +300,21 @@ def stream_solve_with_awake_learning(`.
- **REMOVE** `prompt,` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -255,14 +300,21 @@ def stream_solve_with_awake_learning(`.
- **REMOVE** `history=history,` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -255,14 +300,21 @@ def stream_solve_with_awake_learning(`.
- **REMOVE** `temperature=CHAT_N1_TEMPERATURE,` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -255,14 +300,21 @@ def stream_solve_with_awake_learning(`.
- **REMOVE** `top_p=top_p,` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -255,14 +300,21 @@ def stream_solve_with_awake_learning(`.
- **REMOVE** `cancel_event=cancel_event,` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -255,14 +300,21 @@ def stream_solve_with_awake_learning(`.
- **REMOVE** `)` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -255,14 +300,21 @@ def stream_solve_with_awake_learning(`.
- **ADD** `formatted = self._format_prompt_with_history(prompt, history)` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -255,14 +300,21 @@ def stream_solve_with_awake_learning(`.
- **ADD** `remaining = _remaining_generation_tokens(self, formatted)` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -255,14 +300,21 @@ def stream_solve_with_awake_learning(`.
- **ADD** `previous_max = getattr(self.settings, "max_new_tokens", None)` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -255,14 +300,21 @@ def stream_solve_with_awake_learning(`.
- **ADD** `self.settings.max_new_tokens = int(remaining)` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -255,14 +300,21 @@ def stream_solve_with_awake_learning(`.
- **ADD** `try:` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -255,14 +300,21 @@ def stream_solve_with_awake_learning(`.
- **ADD** `yield from original_stream_solve(` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -255,14 +300,21 @@ def stream_solve_with_awake_learning(`.
- **ADD** `self,` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -255,14 +300,21 @@ def stream_solve_with_awake_learning(`.
- **ADD** `prompt,` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -255,14 +300,21 @@ def stream_solve_with_awake_learning(`.
- **ADD** `history=history,` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -255,14 +300,21 @@ def stream_solve_with_awake_learning(`.
- **ADD** `temperature=CHAT_N1_TEMPERATURE,` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -255,14 +300,21 @@ def stream_solve_with_awake_learning(`.
- **ADD** `top_p=top_p,` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -255,14 +300,21 @@ def stream_solve_with_awake_learning(`.
- **ADD** `cancel_event=cancel_event,` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -255,14 +300,21 @@ def stream_solve_with_awake_learning(`.
- **ADD** `)` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -255,14 +300,21 @@ def stream_solve_with_awake_learning(`.
- **ADD** `finally:` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -255,14 +300,21 @@ def stream_solve_with_awake_learning(`.
- **ADD** `self.settings.max_new_tokens = previous_max` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -255,14 +300,21 @@ def stream_solve_with_awake_learning(`.

### `@@ -309,15 +361,23 @@ def solve_with_awake_learning(`

- **REMOVE** `return original_solve(` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -309,15 +361,23 @@ def solve_with_awake_learning(`.
- **REMOVE** `self,` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -309,15 +361,23 @@ def solve_with_awake_learning(`.
- **REMOVE** `prompt,` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -309,15 +361,23 @@ def solve_with_awake_learning(`.
- **REMOVE** `test_cases=test_cases,` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -309,15 +361,23 @@ def solve_with_awake_learning(`.
- **REMOVE** `history=history,` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -309,15 +361,23 @@ def solve_with_awake_learning(`.
- **REMOVE** `cancel_event=cancel_event,` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -309,15 +361,23 @@ def solve_with_awake_learning(`.
- **REMOVE** `force_branch_count=force_branch_count,` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -309,15 +361,23 @@ def solve_with_awake_learning(`.
- **REMOVE** `temperature=temperature,` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -309,15 +361,23 @@ def solve_with_awake_learning(`.
- **REMOVE** `)` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -309,15 +361,23 @@ def solve_with_awake_learning(`.
- **ADD** `blank-line formatting change within this audited hunk` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -309,15 +361,23 @@ def solve_with_awake_learning(`.
- **ADD** `formatted = self._format_prompt_with_history(prompt, history)` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -309,15 +361,23 @@ def solve_with_awake_learning(`.
- **ADD** `remaining = _remaining_generation_tokens(self, formatted)` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -309,15 +361,23 @@ def solve_with_awake_learning(`.
- **ADD** `previous_max = getattr(self.settings, "max_new_tokens", None)` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -309,15 +361,23 @@ def solve_with_awake_learning(`.
- **ADD** `self.settings.max_new_tokens = int(remaining)` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -309,15 +361,23 @@ def solve_with_awake_learning(`.
- **ADD** `try:` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -309,15 +361,23 @@ def solve_with_awake_learning(`.
- **ADD** `return original_solve(` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -309,15 +361,23 @@ def solve_with_awake_learning(`.
- **ADD** `self,` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -309,15 +361,23 @@ def solve_with_awake_learning(`.
- **ADD** `prompt,` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -309,15 +361,23 @@ def solve_with_awake_learning(`.
- **ADD** `test_cases=test_cases,` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -309,15 +361,23 @@ def solve_with_awake_learning(`.
- **ADD** `history=history,` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -309,15 +361,23 @@ def solve_with_awake_learning(`.
- **ADD** `cancel_event=cancel_event,` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -309,15 +361,23 @@ def solve_with_awake_learning(`.
- **ADD** `force_branch_count=force_branch_count,` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -309,15 +361,23 @@ def solve_with_awake_learning(`.
- **ADD** `temperature=temperature,` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -309,15 +361,23 @@ def solve_with_awake_learning(`.
- **ADD** `)` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -309,15 +361,23 @@ def solve_with_awake_learning(`.
- **ADD** `finally:` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -309,15 +361,23 @@ def solve_with_awake_learning(`.
- **ADD** `self.settings.max_new_tokens = previous_max` — apply one Context budget to prompt/history plus output across text backends; exact location: `@@ -309,15 +361,23 @@ def solve_with_awake_learning(`.

## `core/bitnet_rebuild_trainer.py` — modified, +42/-5

**Shared rationale:** release large training state before conversion and restore on failure/cancel.


### `@@ -19,6 +19,7 @@`

- **ADD** `from core.training_memory import release_training_memory` — release large training state before conversion and restore on failure/cancel; exact location: `@@ -19,6 +19,7 @@`.

### `@@ -444,6 +445,27 @@ def train(`

- **ADD** `blank-line formatting change within this audited hunk` — release large training state before conversion and restore on failure/cancel; exact location: `@@ -444,6 +445,27 @@ def train(`.
- **ADD** `# Microsoft's converter is file-based. Release both the PEFT graph and` — release large training state before conversion and restore on failure/cancel; exact location: `@@ -444,6 +445,27 @@ def train(`.
- **ADD** `# merged BF16 model before conversion so their memory cannot overlap.` — release large training state before conversion and restore on failure/cancel; exact location: `@@ -444,6 +445,27 @@ def train(`.
- **ADD** `try:` — release large training state before conversion and restore on failure/cancel; exact location: `@@ -444,6 +445,27 @@ def train(`.
- **ADD** `optimizer.zero_grad(set_to_none=True)` — release large training state before conversion and restore on failure/cancel; exact location: `@@ -444,6 +445,27 @@ def train(`.
- **ADD** `except Exception:` — release large training state before conversion and restore on failure/cancel; exact location: `@@ -444,6 +445,27 @@ def train(`.
- **ADD** `pass` — release large training state before conversion and restore on failure/cancel; exact location: `@@ -444,6 +445,27 @@ def train(`.
- **ADD** `before.clear()` — release large training state before conversion and restore on failure/cancel; exact location: `@@ -444,6 +445,27 @@ def train(`.
- **ADD** `params.clear()` — release large training state before conversion and restore on failure/cancel; exact location: `@@ -444,6 +445,27 @@ def train(`.
- **ADD** `encoded = None` — release large training state before conversion and restore on failure/cancel; exact location: `@@ -444,6 +445,27 @@ def train(`.
- **ADD** `prefix_ids = None` — release large training state before conversion and restore on failure/cancel; exact location: `@@ -444,6 +445,27 @@ def train(`.
- **ADD** `labels = None` — release large training state before conversion and restore on failure/cancel; exact location: `@@ -444,6 +445,27 @@ def train(`.
- **ADD** `out = None` — release large training state before conversion and restore on failure/cancel; exact location: `@@ -444,6 +445,27 @@ def train(`.
- **ADD** `loss = None` — release large training state before conversion and restore on failure/cancel; exact location: `@@ -444,6 +445,27 @@ def train(`.
- **ADD** `del optimizer` — release large training state before conversion and restore on failure/cancel; exact location: `@@ -444,6 +445,27 @@ def train(`.
- **ADD** `del merged` — release large training state before conversion and restore on failure/cancel; exact location: `@@ -444,6 +445,27 @@ def train(`.
- **ADD** `del model` — release large training state before conversion and restore on failure/cancel; exact location: `@@ -444,6 +445,27 @@ def train(`.
- **ADD** `model = None` — release large training state before conversion and restore on failure/cancel; exact location: `@@ -444,6 +445,27 @@ def train(`.
- **ADD** `tokenizer = None` — release large training state before conversion and restore on failure/cancel; exact location: `@@ -444,6 +445,27 @@ def train(`.
- **ADD** `release_training_memory()` — release large training state before conversion and restore on failure/cancel; exact location: `@@ -444,6 +445,27 @@ def train(`.
- **ADD** `blank-line formatting change within this audited hunk` — release large training state before conversion and restore on failure/cancel; exact location: `@@ -444,6 +445,27 @@ def train(`.

### `@@ -462,7 +484,7 @@ def train(`

- **REMOVE** `except Exception:` — release large training state before conversion and restore on failure/cancel; exact location: `@@ -462,7 +484,7 @@ def train(`.
- **ADD** `except BaseException:` — release large training state before conversion and restore on failure/cancel; exact location: `@@ -462,7 +484,7 @@ def train(`.

### `@@ -475,10 +497,25 @@ def train(`

- **REMOVE** `del model` — release large training state before conversion and restore on failure/cancel; exact location: `@@ -475,10 +497,25 @@ def train(`.
- **REMOVE** `gc.collect()` — release large training state before conversion and restore on failure/cancel; exact location: `@@ -475,10 +497,25 @@ def train(`.
- **REMOVE** `if torch.cuda.is_available():` — release large training state before conversion and restore on failure/cancel; exact location: `@@ -475,10 +497,25 @@ def train(`.
- **REMOVE** `torch.cuda.empty_cache()` — release large training state before conversion and restore on failure/cancel; exact location: `@@ -475,10 +497,25 @@ def train(`.
- **ADD** `optimizer.zero_grad(set_to_none=True)` — release large training state before conversion and restore on failure/cancel; exact location: `@@ -475,10 +497,25 @@ def train(`.
- **ADD** `except Exception:` — release large training state before conversion and restore on failure/cancel; exact location: `@@ -475,10 +497,25 @@ def train(`.
- **ADD** `pass` — release large training state before conversion and restore on failure/cancel; exact location: `@@ -475,10 +497,25 @@ def train(`.
- **ADD** `try:` — release large training state before conversion and restore on failure/cancel; exact location: `@@ -475,10 +497,25 @@ def train(`.
- **ADD** `before.clear()` — release large training state before conversion and restore on failure/cancel; exact location: `@@ -475,10 +497,25 @@ def train(`.
- **ADD** `except Exception:` — release large training state before conversion and restore on failure/cancel; exact location: `@@ -475,10 +497,25 @@ def train(`.
- **ADD** `pass` — release large training state before conversion and restore on failure/cancel; exact location: `@@ -475,10 +497,25 @@ def train(`.
- **ADD** `try:` — release large training state before conversion and restore on failure/cancel; exact location: `@@ -475,10 +497,25 @@ def train(`.
- **ADD** `del optimizer` — release large training state before conversion and restore on failure/cancel; exact location: `@@ -475,10 +497,25 @@ def train(`.
- **ADD** `except Exception:` — release large training state before conversion and restore on failure/cancel; exact location: `@@ -475,10 +497,25 @@ def train(`.
- **ADD** `pass` — release large training state before conversion and restore on failure/cancel; exact location: `@@ -475,10 +497,25 @@ def train(`.
- **ADD** `try:` — release large training state before conversion and restore on failure/cancel; exact location: `@@ -475,10 +497,25 @@ def train(`.
- **ADD** `del tokenizer` — release large training state before conversion and restore on failure/cancel; exact location: `@@ -475,10 +497,25 @@ def train(`.
- **ADD** `except Exception:` — release large training state before conversion and restore on failure/cancel; exact location: `@@ -475,10 +497,25 @@ def train(`.
- **ADD** `pass` — release large training state before conversion and restore on failure/cancel; exact location: `@@ -475,10 +497,25 @@ def train(`.
- **ADD** `try:` — release large training state before conversion and restore on failure/cancel; exact location: `@@ -475,10 +497,25 @@ def train(`.
- **ADD** `if model is not None:` — release large training state before conversion and restore on failure/cancel; exact location: `@@ -475,10 +497,25 @@ def train(`.
- **ADD** `del model` — release large training state before conversion and restore on failure/cancel; exact location: `@@ -475,10 +497,25 @@ def train(`.
- **ADD** `release_training_memory()` — release large training state before conversion and restore on failure/cancel; exact location: `@@ -475,10 +497,25 @@ def train(`.

## `core/controller_runtime.py` — modified, +99/-9

**Shared rationale:** isolate eval adapter state and make controller PEFT transactional/rollback-safe.


### `@@ -19,6 +19,7 @@`

- **ADD** `from core.training_memory import release_training_memory` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -19,6 +19,7 @@`.

### `@@ -342,7 +343,12 @@ def _load_transformers_auto(self) -> bool:`

- **REMOVE** `self.adapter_path = os.path.join(get_portable_data_dir(), "controller_lora", key)` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -342,7 +343,12 @@ def _load_transformers_auto(self) -> bool:`.
- **ADD** `eval_root = str(os.getenv("SMARTAI_CONTROLLER_ADAPTER_ROOT", "") or "").strip()` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -342,7 +343,12 @@ def _load_transformers_auto(self) -> bool:`.
- **ADD** `self.adapter_path = (` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -342,7 +343,12 @@ def _load_transformers_auto(self) -> bool:`.
- **ADD** `os.path.join(os.path.abspath(eval_root), key)` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -342,7 +343,12 @@ def _load_transformers_auto(self) -> bool:`.
- **ADD** `if eval_root` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -342,7 +343,12 @@ def _load_transformers_auto(self) -> bool:`.
- **ADD** `else os.path.join(get_portable_data_dir(), "controller_lora", key)` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -342,7 +343,12 @@ def _load_transformers_auto(self) -> bool:`.
- **ADD** `)` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -342,7 +343,12 @@ def _load_transformers_auto(self) -> bool:`.

### `@@ -450,7 +456,7 @@ def train_mini_batch(`

- **REMOVE** `"""Real completion-only PEFT update for non-MLX Transformers controllers."""` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -450,7 +456,7 @@ def train_mini_batch(`.
- **ADD** `"""Real transactional completion-only PEFT update for Transformers controllers."""` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -450,7 +456,7 @@ def train_mini_batch(`.

### `@@ -460,7 +466,8 @@ def train_mini_batch(`

- **REMOVE** `if str(item.get("prompt") or "").strip() and str(item.get("completion") or "").strip()` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -460,7 +466,8 @@ def train_mini_batch(`.
- **ADD** `if str(item.get("prompt") or "").strip()` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -460,7 +466,8 @@ def train_mini_batch(`.
- **ADD** `and str(item.get("completion") or "").strip()` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -460,7 +466,8 @@ def train_mini_batch(`.

### `@@ -473,10 +480,16 @@ def train_mini_batch(`

- **ADD** `blank-line formatting change within this audited hunk` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -473,10 +480,16 @@ def train_mini_batch(`.
- **ADD** `tmp = ""` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -473,10 +480,16 @@ def train_mini_batch(`.
- **ADD** `backup = ""` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -473,10 +480,16 @@ def train_mini_batch(`.
- **ADD** `success = False` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -473,10 +480,16 @@ def train_mini_batch(`.
- **ADD** `had_adapter = bool(self.adapter_path and os.path.isdir(self.adapter_path))` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -473,10 +480,16 @@ def train_mini_batch(`.
- **ADD** `trained_rows = 0` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -473,10 +480,16 @@ def train_mini_batch(`.

### `@@ -486,11 +499,18 @@ def train_mini_batch(`

- **REMOVE** `prefix = apply_template(messages, tokenize=False, add_generation_prompt=True)` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -486,11 +499,18 @@ def train_mini_batch(`.
- **ADD** `prefix = apply_template(` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -486,11 +499,18 @@ def train_mini_batch(`.
- **ADD** `messages,` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -486,11 +499,18 @@ def train_mini_batch(`.
- **ADD** `tokenize=False,` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -486,11 +499,18 @@ def train_mini_batch(`.
- **ADD** `add_generation_prompt=True,` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -486,11 +499,18 @@ def train_mini_batch(`.
- **ADD** `)` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -486,11 +499,18 @@ def train_mini_batch(`.
- **REMOVE** `prefix = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -486,11 +499,18 @@ def train_mini_batch(`.
- **ADD** `prefix = (` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -486,11 +499,18 @@ def train_mini_batch(`.
- **ADD** `f"<|im_start|>user\n{prompt}<|im_end|>\n"` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -486,11 +499,18 @@ def train_mini_batch(`.
- **ADD** `f"<|im_start|>assistant\n"` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -486,11 +499,18 @@ def train_mini_batch(`.
- **ADD** `)` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -486,11 +499,18 @@ def train_mini_batch(`.

### `@@ -511,7 +531,8 @@ def train_mini_batch(`

- **REMOVE** `encoded = {k: v.to(device) for k, v in encoded.items()}` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -511,7 +531,8 @@ def train_mini_batch(`.
- **ADD** `blank-line formatting change within this audited hunk` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -511,7 +531,8 @@ def train_mini_batch(`.
- **ADD** `encoded = {key: value.to(device) for key, value in encoded.items()}` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -511,7 +531,8 @@ def train_mini_batch(`.

### `@@ -523,6 +544,10 @@ def train_mini_batch(`

- **ADD** `trained_rows += 1` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -523,6 +544,10 @@ def train_mini_batch(`.
- **ADD** `blank-line formatting change within this audited hunk` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -523,6 +544,10 @@ def train_mini_batch(`.
- **ADD** `if trained_rows <= 0:` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -523,6 +544,10 @@ def train_mini_batch(`.
- **ADD** `raise RuntimeError("Controller LoRA found no trainable prompt/completion rows")` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -523,6 +544,10 @@ def train_mini_batch(`.

### `@@ -543,19 +568,84 @@ def train_mini_batch(`

- **ADD** `blank-line formatting change within this audited hunk` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -543,19 +568,84 @@ def train_mini_batch(`.
- **REMOVE** `if os.path.isdir(self.adapter_path):` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -543,19 +568,84 @@ def train_mini_batch(`.
- **ADD** `if had_adapter:` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -543,19 +568,84 @@ def train_mini_batch(`.
- **REMOVE** `shutil.rmtree(backup, ignore_errors=True)` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -543,19 +568,84 @@ def train_mini_batch(`.
- **ADD** `tmp = ""` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -543,19 +568,84 @@ def train_mini_batch(`.
- **REMOVE** `return dict(self.adapters), float(drift)` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -543,19 +568,84 @@ def train_mini_batch(`.
- **ADD** `result = (dict(self.adapters), float(drift))` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -543,19 +568,84 @@ def train_mini_batch(`.
- **ADD** `success = True` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -543,19 +568,84 @@ def train_mini_batch(`.
- **ADD** `return result` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -543,19 +568,84 @@ def train_mini_batch(`.
- **ADD** `blank-line formatting change within this audited hunk` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -543,19 +568,84 @@ def train_mini_batch(`.
- **ADD** `except BaseException:` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -543,19 +568,84 @@ def train_mini_batch(`.
- **ADD** `success = False` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -543,19 +568,84 @@ def train_mini_batch(`.
- **ADD** `# Restore live trainable tensors first, then restore the persisted` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -543,19 +568,84 @@ def train_mini_batch(`.
- **ADD** `# adapter directory if the filesystem transaction had started.` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -543,19 +568,84 @@ def train_mini_batch(`.
- **ADD** `try:` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -543,19 +568,84 @@ def train_mini_batch(`.
- **ADD** `with torch.no_grad():` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -543,19 +568,84 @@ def train_mini_batch(`.
- **ADD** `for name, param in model.named_parameters():` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -543,19 +568,84 @@ def train_mini_batch(`.
- **ADD** `snapshot = before.get(name)` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -543,19 +568,84 @@ def train_mini_batch(`.
- **ADD** `if snapshot is None or not param.requires_grad:` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -543,19 +568,84 @@ def train_mini_batch(`.
- **ADD** `continue` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -543,19 +568,84 @@ def train_mini_batch(`.
- **ADD** `param.copy_(` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -543,19 +568,84 @@ def train_mini_batch(`.
- **ADD** `snapshot.to(device=param.device, dtype=param.dtype)` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -543,19 +568,84 @@ def train_mini_batch(`.
- **ADD** `)` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -543,19 +568,84 @@ def train_mini_batch(`.
- **ADD** `except Exception:` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -543,19 +568,84 @@ def train_mini_batch(`.
- **ADD** `pass` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -543,19 +568,84 @@ def train_mini_batch(`.
- **ADD** `blank-line formatting change within this audited hunk` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -543,19 +568,84 @@ def train_mini_batch(`.
- **ADD** `if tmp and os.path.isdir(tmp):` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -543,19 +568,84 @@ def train_mini_batch(`.
- **ADD** `shutil.rmtree(tmp, ignore_errors=True)` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -543,19 +568,84 @@ def train_mini_batch(`.
- **ADD** `blank-line formatting change within this audited hunk` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -543,19 +568,84 @@ def train_mini_batch(`.
- **ADD** `if backup and os.path.isdir(backup):` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -543,19 +568,84 @@ def train_mini_batch(`.
- **ADD** `if os.path.isdir(self.adapter_path):` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -543,19 +568,84 @@ def train_mini_batch(`.
- **ADD** `shutil.rmtree(self.adapter_path, ignore_errors=True)` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -543,19 +568,84 @@ def train_mini_batch(`.
- **ADD** `os.replace(backup, self.adapter_path)` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -543,19 +568,84 @@ def train_mini_batch(`.
- **ADD** `backup = ""` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -543,19 +568,84 @@ def train_mini_batch(`.
- **ADD** `elif not had_adapter and self.adapter_path and os.path.isdir(self.adapter_path):` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -543,19 +568,84 @@ def train_mini_batch(`.
- **ADD** `# A brand-new failed transaction must not leave an unaccepted` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -543,19 +568,84 @@ def train_mini_batch(`.
- **ADD** `# persisted adapter behind.` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -543,19 +568,84 @@ def train_mini_batch(`.
- **ADD** `shutil.rmtree(self.adapter_path, ignore_errors=True)` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -543,19 +568,84 @@ def train_mini_batch(`.
- **ADD** `raise` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -543,19 +568,84 @@ def train_mini_batch(`.
- **ADD** `blank-line formatting change within this audited hunk` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -543,19 +568,84 @@ def train_mini_batch(`.
- **ADD** `if tmp and os.path.isdir(tmp):` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -543,19 +568,84 @@ def train_mini_batch(`.
- **ADD** `shutil.rmtree(tmp, ignore_errors=True)` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -543,19 +568,84 @@ def train_mini_batch(`.
- **ADD** `if success and backup and os.path.isdir(backup):` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -543,19 +568,84 @@ def train_mini_batch(`.
- **ADD** `shutil.rmtree(backup, ignore_errors=True)` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -543,19 +568,84 @@ def train_mini_batch(`.
- **ADD** `backup = ""` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -543,19 +568,84 @@ def train_mini_batch(`.
- **ADD** `elif backup and os.path.isdir(backup):` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -543,19 +568,84 @@ def train_mini_batch(`.
- **ADD** `# Failure/cancellation keeps the previous adapter authoritative.` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -543,19 +568,84 @@ def train_mini_batch(`.
- **ADD** `if os.path.isdir(self.adapter_path):` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -543,19 +568,84 @@ def train_mini_batch(`.
- **ADD** `shutil.rmtree(self.adapter_path, ignore_errors=True)` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -543,19 +568,84 @@ def train_mini_batch(`.
- **ADD** `try:` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -543,19 +568,84 @@ def train_mini_batch(`.
- **ADD** `os.replace(backup, self.adapter_path)` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -543,19 +568,84 @@ def train_mini_batch(`.
- **ADD** `backup = ""` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -543,19 +568,84 @@ def train_mini_batch(`.
- **ADD** `except Exception:` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -543,19 +568,84 @@ def train_mini_batch(`.
- **ADD** `pass` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -543,19 +568,84 @@ def train_mini_batch(`.
- **ADD** `blank-line formatting change within this audited hunk` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -543,19 +568,84 @@ def train_mini_batch(`.
- **ADD** `try:` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -543,19 +568,84 @@ def train_mini_batch(`.
- **ADD** `optimizer.zero_grad(set_to_none=True)` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -543,19 +568,84 @@ def train_mini_batch(`.
- **ADD** `except Exception:` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -543,19 +568,84 @@ def train_mini_batch(`.
- **ADD** `pass` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -543,19 +568,84 @@ def train_mini_batch(`.
- **ADD** `before.clear()` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -543,19 +568,84 @@ def train_mini_batch(`.
- **ADD** `params.clear()` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -543,19 +568,84 @@ def train_mini_batch(`.
- **ADD** `encoded = None` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -543,19 +568,84 @@ def train_mini_batch(`.
- **ADD** `prefix_ids = None` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -543,19 +568,84 @@ def train_mini_batch(`.
- **ADD** `labels = None` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -543,19 +568,84 @@ def train_mini_batch(`.
- **ADD** `output = None` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -543,19 +568,84 @@ def train_mini_batch(`.
- **ADD** `loss = None` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -543,19 +568,84 @@ def train_mini_batch(`.
- **ADD** `try:` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -543,19 +568,84 @@ def train_mini_batch(`.
- **ADD** `del optimizer` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -543,19 +568,84 @@ def train_mini_batch(`.
- **ADD** `except Exception:` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -543,19 +568,84 @@ def train_mini_batch(`.
- **ADD** `pass` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -543,19 +568,84 @@ def train_mini_batch(`.
- **ADD** `release_training_memory(self)` — isolate eval adapter state and make controller PEFT transactional/rollback-safe; exact location: `@@ -543,19 +568,84 @@ def train_mini_batch(`.

## `core/drafter.py` — modified, +125/-0

**Shared rationale:** recover diagnostic draft helpers only; production speculation stays disabled.


### `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`

- **ADD** `blank-line formatting change within this audited hunk` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `# Compatibility/diagnostic superset recovered from the older architecture staging` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `# branch. Production speculative decoding remains disabled in core.speculative_engine;` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `# these helpers only produce proposals for diagnostics or future equivalence testing.` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `from dataclasses import dataclass` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `from typing import Iterable, Optional, Sequence` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `blank-line formatting change within this audited hunk` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `DEFAULT_SCAFFOLDS = (` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `"def ", "\n    return ", "\n    if ", "\n    for ", "\n    while ",` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `"\n    else:", "\n    elif ", "\n    try:", "\n    except ", "assert ",` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `"\`\`\`python\n", "\\boxed{", " = ", " == ", " in ",` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `)` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `blank-line formatting change within this audited hunk` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `blank-line formatting change within this audited hunk` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `@dataclass` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `class DraftTelemetry:` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `attempts: int = 0` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `hits: int = 0` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `proposed_tokens: int = 0` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `accepted_tokens: int = 0` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `blank-line formatting change within this audited hunk` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `@property` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `def hit_rate(self) -> float:` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `return 100.0 * self.hits / max(1, self.attempts)` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `blank-line formatting change within this audited hunk` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `@property` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `def acceptance_rate(self) -> float:` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `return 100.0 * self.accepted_tokens / max(1, self.proposed_tokens)` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `blank-line formatting change within this audited hunk` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `blank-line formatting change within this audited hunk` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `class GrammarGuidedASTTrieDrafter:` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `"""N-gram proposals plus tokenizer-bound grammar scaffolds; proposal-only."""` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `blank-line formatting change within this audited hunk` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `def __init__(` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `self,` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `n_gram: int = 3,` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `max_draft: int = 3,` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `scaffolds: Optional[Iterable[str]] = None,` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `tokenizer=None,` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `):` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `self.n_gram = max(1, int(n_gram))` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `self.max_draft = max(1, int(max_draft))` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `self.scaffolds = tuple(scaffolds or DEFAULT_SCAFFOLDS)` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `self.tokenizer = None` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `self._sequences: List[List[int]] = []` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `self.telemetry = DraftTelemetry()` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `if tokenizer is not None:` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `self.bind_tokenizer(tokenizer)` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `blank-line formatting change within this audited hunk` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `def bind_tokenizer(self, tokenizer) -> None:` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `self.tokenizer = tokenizer` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `sequences: List[List[int]] = []` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `for text in self.scaffolds:` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `try:` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `try:` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `ids = tokenizer.encode(text, add_special_tokens=False)` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `except TypeError:` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `ids = tokenizer.encode(text)` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `ids = [int(x) for x in ids]` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `if ids:` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `sequences.append(ids)` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `except Exception:` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `continue` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `sequences.sort(key=len, reverse=True)` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `self._sequences = sequences` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `blank-line formatting change within this audited hunk` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `def _ngram_draft(self, history: Sequence[int]) -> List[int]:` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `n = self.n_gram` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `if len(history) < n * 2:` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `return []` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `target = list(history[-n:])` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `current_start = len(history) - n` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `for idx in range(current_start - 1, -1, -1):` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `if list(history[idx:idx + n]) == target:` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `start = idx + n` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `end = min(start + self.max_draft, current_start)` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `draft = list(history[start:end])` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `if draft:` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `return [int(x) for x in draft]` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `return []` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `blank-line formatting change within this audited hunk` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `def _grammar_draft(self, history: Sequence[int]) -> List[int]:` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `hist = list(history)` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `for seq in self._sequences:` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `for prefix_len in range(min(len(seq) - 1, len(hist)), 0, -1):` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `if hist[-prefix_len:] == seq[:prefix_len]:` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `rest = seq[prefix_len:prefix_len + self.max_draft]` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `if rest:` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `return [int(x) for x in rest]` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `return []` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `blank-line formatting change within this audited hunk` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `def find_draft_tokens(self, token_history, tokenizer=None, max_draft=None) -> List[int]:` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `if tokenizer is not None and tokenizer is not self.tokenizer:` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `self.bind_tokenizer(tokenizer)` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `self.telemetry.attempts += 1` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `previous = self.max_draft` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `if max_draft is not None:` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `self.max_draft = max(1, int(max_draft))` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `try:` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `draft = self._ngram_draft(token_history) or self._grammar_draft(token_history)` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `finally:` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `self.max_draft = previous` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `if draft:` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `self.telemetry.hits += 1` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `self.telemetry.proposed_tokens += len(draft)` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `return draft` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `blank-line formatting change within this audited hunk` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `def note_acceptance(self, accepted: int, proposed: Optional[int] = None) -> None:` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `self.telemetry.accepted_tokens += max(0, int(accepted))` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `if proposed is not None and proposed > 0 and self.telemetry.proposed_tokens < proposed:` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `self.telemetry.proposed_tokens += int(proposed)` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `blank-line formatting change within this audited hunk` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `def get_telemetry(self):` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `return {` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `"attempts": self.telemetry.attempts,` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `"hits": self.telemetry.hits,` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `"hit_rate": self.telemetry.hit_rate,` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `"proposed_tokens": self.telemetry.proposed_tokens,` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `"accepted_tokens": self.telemetry.accepted_tokens,` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `"acceptance_rate": self.telemetry.acceptance_rate,` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `}` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `blank-line formatting change within this audited hunk` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `blank-line formatting change within this audited hunk` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `ASTPrefixTrieDrafter = GrammarGuidedASTTrieDrafter` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.
- **ADD** `blank-line formatting change within this audited hunk` — recover diagnostic draft helpers only; production speculation stays disabled; exact location: `@@ -18,3 +18,128 @@ def find_draft_tokens(self, token_history: List[int]) -> List[int]:`.

## `core/engines/bitnet_cpp_engine.py` — modified, +23/-8

**Shared rationale:** isolate eval learned state and retain rollback backup until reload succeeds.


### `@@ -67,7 +67,12 @@ def __init__(`

- **REMOVE** `self.training_root = Path(get_portable_data_dir()) / "bitnet_learning" / key` — isolate eval learned state and retain rollback backup until reload succeeds; exact location: `@@ -67,7 +67,12 @@ def __init__(`.
- **ADD** `eval_training_root = str(os.getenv("SMARTAI_BITNET_TRAINING_ROOT", "") or "").strip()` — isolate eval learned state and retain rollback backup until reload succeeds; exact location: `@@ -67,7 +67,12 @@ def __init__(`.
- **ADD** `self.training_root = (` — isolate eval learned state and retain rollback backup until reload succeeds; exact location: `@@ -67,7 +67,12 @@ def __init__(`.
- **ADD** `Path(eval_training_root)` — isolate eval learned state and retain rollback backup until reload succeeds; exact location: `@@ -67,7 +67,12 @@ def __init__(`.
- **ADD** `if eval_training_root` — isolate eval learned state and retain rollback backup until reload succeeds; exact location: `@@ -67,7 +67,12 @@ def __init__(`.
- **ADD** `else Path(get_portable_data_dir()) / "bitnet_learning" / key` — isolate eval learned state and retain rollback backup until reload succeeds; exact location: `@@ -67,7 +67,12 @@ def __init__(`.
- **ADD** `)` — isolate eval learned state and retain rollback backup until reload succeeds; exact location: `@@ -67,7 +67,12 @@ def __init__(`.

### `@@ -387,6 +392,7 @@ def train_mini_batch(`

- **ADD** `success = False` — isolate eval learned state and retain rollback backup until reload succeeds; exact location: `@@ -387,6 +392,7 @@ def train_mini_batch(`.

### `@@ -398,22 +404,31 @@ def train_mini_batch(`

- **REMOVE** `try:` — isolate eval learned state and retain rollback backup until reload succeeds; exact location: `@@ -398,22 +404,31 @@ def train_mini_batch(`.
- **REMOVE** `os.remove(backup)` — isolate eval learned state and retain rollback backup until reload succeeds; exact location: `@@ -398,22 +404,31 @@ def train_mini_batch(`.
- **REMOVE** `except OSError:` — isolate eval learned state and retain rollback backup until reload succeeds; exact location: `@@ -398,22 +404,31 @@ def train_mini_batch(`.
- **REMOVE** `pass` — isolate eval learned state and retain rollback backup until reload succeeds; exact location: `@@ -398,22 +404,31 @@ def train_mini_batch(`.
- **REMOVE** `return dict(self.adapters), float(drift)` — isolate eval learned state and retain rollback backup until reload succeeds; exact location: `@@ -398,22 +404,31 @@ def train_mini_batch(`.
- **REMOVE** `except Exception:` — isolate eval learned state and retain rollback backup until reload succeeds; exact location: `@@ -398,22 +404,31 @@ def train_mini_batch(`.
- **ADD** `result = (dict(self.adapters), float(drift))` — isolate eval learned state and retain rollback backup until reload succeeds; exact location: `@@ -398,22 +404,31 @@ def train_mini_batch(`.
- **ADD** `success = True` — isolate eval learned state and retain rollback backup until reload succeeds; exact location: `@@ -398,22 +404,31 @@ def train_mini_batch(`.
- **ADD** `return result` — isolate eval learned state and retain rollback backup until reload succeeds; exact location: `@@ -398,22 +404,31 @@ def train_mini_batch(`.
- **ADD** `except BaseException:` — isolate eval learned state and retain rollback backup until reload succeeds; exact location: `@@ -398,22 +404,31 @@ def train_mini_batch(`.
- **ADD** `success = False` — isolate eval learned state and retain rollback backup until reload succeeds; exact location: `@@ -398,22 +404,31 @@ def train_mini_batch(`.
- **REMOVE** `self.model_path = str(self.learned_model_path if self.learned_model_path.is_file() else self.original_model_path)` — isolate eval learned state and retain rollback backup until reload succeeds; exact location: `@@ -398,22 +404,31 @@ def train_mini_batch(`.
- **ADD** `self.model_path = str(` — isolate eval learned state and retain rollback backup until reload succeeds; exact location: `@@ -398,22 +404,31 @@ def train_mini_batch(`.
- **ADD** `self.learned_model_path` — isolate eval learned state and retain rollback backup until reload succeeds; exact location: `@@ -398,22 +404,31 @@ def train_mini_batch(`.
- **ADD** `if self.learned_model_path.is_file()` — isolate eval learned state and retain rollback backup until reload succeeds; exact location: `@@ -398,22 +404,31 @@ def train_mini_batch(`.
- **ADD** `else self.original_model_path` — isolate eval learned state and retain rollback backup until reload succeeds; exact location: `@@ -398,22 +404,31 @@ def train_mini_batch(`.
- **ADD** `)` — isolate eval learned state and retain rollback backup until reload succeeds; exact location: `@@ -398,22 +404,31 @@ def train_mini_batch(`.
- **ADD** `finally:` — isolate eval learned state and retain rollback backup until reload succeeds; exact location: `@@ -398,22 +404,31 @@ def train_mini_batch(`.
- **ADD** `if success:` — isolate eval learned state and retain rollback backup until reload succeeds; exact location: `@@ -398,22 +404,31 @@ def train_mini_batch(`.
- **ADD** `try:` — isolate eval learned state and retain rollback backup until reload succeeds; exact location: `@@ -398,22 +404,31 @@ def train_mini_batch(`.
- **ADD** `os.remove(backup)` — isolate eval learned state and retain rollback backup until reload succeeds; exact location: `@@ -398,22 +404,31 @@ def train_mini_batch(`.
- **ADD** `except OSError:` — isolate eval learned state and retain rollback backup until reload succeeds; exact location: `@@ -398,22 +404,31 @@ def train_mini_batch(`.
- **ADD** `pass` — isolate eval learned state and retain rollback backup until reload succeeds; exact location: `@@ -398,22 +404,31 @@ def train_mini_batch(`.

## `core/engines/gguf_engine.py` — modified, +19/-9

**Shared rationale:** isolate eval adapter state and retain rollback backup until reload succeeds.


### `@@ -35,8 +35,11 @@ def __init__(`

- **ADD** `eval_adapter_root = str(os.getenv("SMARTAI_GGUF_ADAPTER_ROOT", "") or "").strip()` — isolate eval adapter state and retain rollback backup until reload succeeds; exact location: `@@ -35,8 +35,11 @@ def __init__(`.
- **ADD** `elif eval_adapter_root:` — isolate eval adapter state and retain rollback backup until reload succeeds; exact location: `@@ -35,8 +35,11 @@ def __init__(`.
- **ADD** `self.adapter_root = os.path.abspath(eval_adapter_root)` — isolate eval adapter state and retain rollback backup until reload succeeds; exact location: `@@ -35,8 +35,11 @@ def __init__(`.

### `@@ -167,6 +170,7 @@ def train_mini_batch(`

- **ADD** `success = False` — isolate eval adapter state and retain rollback backup until reload succeeds; exact location: `@@ -167,6 +170,7 @@ def train_mini_batch(`.

### `@@ -183,15 +187,13 @@ def train_mini_batch(`

- **REMOVE** `try:` — isolate eval adapter state and retain rollback backup until reload succeeds; exact location: `@@ -183,15 +187,13 @@ def train_mini_batch(`.
- **REMOVE** `os.remove(backup_path)` — isolate eval adapter state and retain rollback backup until reload succeeds; exact location: `@@ -183,15 +187,13 @@ def train_mini_batch(`.
- **REMOVE** `except OSError:` — isolate eval adapter state and retain rollback backup until reload succeeds; exact location: `@@ -183,15 +187,13 @@ def train_mini_batch(`.
- **REMOVE** `pass` — isolate eval adapter state and retain rollback backup until reload succeeds; exact location: `@@ -183,15 +187,13 @@ def train_mini_batch(`.
- **REMOVE** `if os.path.isdir(peft_backup):` — isolate eval adapter state and retain rollback backup until reload succeeds; exact location: `@@ -183,15 +187,13 @@ def train_mini_batch(`.
- **REMOVE** `shutil.rmtree(peft_backup, ignore_errors=True)` — isolate eval adapter state and retain rollback backup until reload succeeds; exact location: `@@ -183,15 +187,13 @@ def train_mini_batch(`.
- **REMOVE** `return dict(self.adapters), float(drift)` — isolate eval adapter state and retain rollback backup until reload succeeds; exact location: `@@ -183,15 +187,13 @@ def train_mini_batch(`.
- **REMOVE** `except Exception:` — isolate eval adapter state and retain rollback backup until reload succeeds; exact location: `@@ -183,15 +187,13 @@ def train_mini_batch(`.
- **REMOVE** `# Roll back the adapter atomically and restore inference. Never leave a` — isolate eval adapter state and retain rollback backup until reload succeeds; exact location: `@@ -183,15 +187,13 @@ def train_mini_batch(`.
- **ADD** `result = (dict(self.adapters), float(drift))` — isolate eval adapter state and retain rollback backup until reload succeeds; exact location: `@@ -183,15 +187,13 @@ def train_mini_batch(`.
- **ADD** `success = True` — isolate eval adapter state and retain rollback backup until reload succeeds; exact location: `@@ -183,15 +187,13 @@ def train_mini_batch(`.
- **ADD** `return result` — isolate eval adapter state and retain rollback backup until reload succeeds; exact location: `@@ -183,15 +187,13 @@ def train_mini_batch(`.
- **ADD** `except BaseException:` — isolate eval adapter state and retain rollback backup until reload succeeds; exact location: `@@ -183,15 +187,13 @@ def train_mini_batch(`.
- **ADD** `success = False` — isolate eval adapter state and retain rollback backup until reload succeeds; exact location: `@@ -183,15 +187,13 @@ def train_mini_batch(`.
- **ADD** `# Roll back the adapter atomically and restore inference. Cancellation` — isolate eval adapter state and retain rollback backup until reload succeeds; exact location: `@@ -183,15 +187,13 @@ def train_mini_batch(`.
- **ADD** `# (KeyboardInterrupt) is a transaction failure too.` — isolate eval adapter state and retain rollback backup until reload succeeds; exact location: `@@ -183,15 +187,13 @@ def train_mini_batch(`.

### `@@ -210,6 +212,14 @@ def train_mini_batch(`

- **ADD** `finally:` — isolate eval adapter state and retain rollback backup until reload succeeds; exact location: `@@ -210,6 +212,14 @@ def train_mini_batch(`.
- **ADD** `if success:` — isolate eval adapter state and retain rollback backup until reload succeeds; exact location: `@@ -210,6 +212,14 @@ def train_mini_batch(`.
- **ADD** `try:` — isolate eval adapter state and retain rollback backup until reload succeeds; exact location: `@@ -210,6 +212,14 @@ def train_mini_batch(`.
- **ADD** `os.remove(backup_path)` — isolate eval adapter state and retain rollback backup until reload succeeds; exact location: `@@ -210,6 +212,14 @@ def train_mini_batch(`.
- **ADD** `except OSError:` — isolate eval adapter state and retain rollback backup until reload succeeds; exact location: `@@ -210,6 +212,14 @@ def train_mini_batch(`.
- **ADD** `pass` — isolate eval adapter state and retain rollback backup until reload succeeds; exact location: `@@ -210,6 +212,14 @@ def train_mini_batch(`.
- **ADD** `if os.path.isdir(peft_backup):` — isolate eval adapter state and retain rollback backup until reload succeeds; exact location: `@@ -210,6 +212,14 @@ def train_mini_batch(`.
- **ADD** `shutil.rmtree(peft_backup, ignore_errors=True)` — isolate eval adapter state and retain rollback backup until reload succeeds; exact location: `@@ -210,6 +212,14 @@ def train_mini_batch(`.

## `core/gguf_lora_trainer.py` — modified, +40/-4

**Shared rationale:** release QLoRA graph before conversion and restore PEFT on failure/cancel.


### `@@ -19,6 +19,7 @@`

- **ADD** `from core.training_memory import release_training_memory` — release QLoRA graph before conversion and restore PEFT on failure/cancel; exact location: `@@ -19,6 +19,7 @@`.

### `@@ -375,6 +376,25 @@ def train(`

- **ADD** `blank-line formatting change within this audited hunk` — release QLoRA graph before conversion and restore PEFT on failure/cancel; exact location: `@@ -375,6 +376,25 @@ def train(`.
- **ADD** `# Conversion is file-based. Do not keep the 27B QLoRA graph resident` — release QLoRA graph before conversion and restore PEFT on failure/cancel; exact location: `@@ -375,6 +376,25 @@ def train(`.
- **ADD** `# while the converter allocates its own buffers.` — release QLoRA graph before conversion and restore PEFT on failure/cancel; exact location: `@@ -375,6 +376,25 @@ def train(`.
- **ADD** `try:` — release QLoRA graph before conversion and restore PEFT on failure/cancel; exact location: `@@ -375,6 +376,25 @@ def train(`.
- **ADD** `optimizer.zero_grad(set_to_none=True)` — release QLoRA graph before conversion and restore PEFT on failure/cancel; exact location: `@@ -375,6 +376,25 @@ def train(`.
- **ADD** `except Exception:` — release QLoRA graph before conversion and restore PEFT on failure/cancel; exact location: `@@ -375,6 +376,25 @@ def train(`.
- **ADD** `pass` — release QLoRA graph before conversion and restore PEFT on failure/cancel; exact location: `@@ -375,6 +376,25 @@ def train(`.
- **ADD** `before.clear()` — release QLoRA graph before conversion and restore PEFT on failure/cancel; exact location: `@@ -375,6 +376,25 @@ def train(`.
- **ADD** `encoded = None` — release QLoRA graph before conversion and restore PEFT on failure/cancel; exact location: `@@ -375,6 +376,25 @@ def train(`.
- **ADD** `prefix_ids = None` — release QLoRA graph before conversion and restore PEFT on failure/cancel; exact location: `@@ -375,6 +376,25 @@ def train(`.
- **ADD** `labels = None` — release QLoRA graph before conversion and restore PEFT on failure/cancel; exact location: `@@ -375,6 +376,25 @@ def train(`.
- **ADD** `out = None` — release QLoRA graph before conversion and restore PEFT on failure/cancel; exact location: `@@ -375,6 +376,25 @@ def train(`.
- **ADD** `loss = None` — release QLoRA graph before conversion and restore PEFT on failure/cancel; exact location: `@@ -375,6 +376,25 @@ def train(`.
- **ADD** `del optimizer` — release QLoRA graph before conversion and restore PEFT on failure/cancel; exact location: `@@ -375,6 +376,25 @@ def train(`.
- **ADD** `del model` — release QLoRA graph before conversion and restore PEFT on failure/cancel; exact location: `@@ -375,6 +376,25 @@ def train(`.
- **ADD** `model = None` — release QLoRA graph before conversion and restore PEFT on failure/cancel; exact location: `@@ -375,6 +376,25 @@ def train(`.
- **ADD** `tokenizer = None` — release QLoRA graph before conversion and restore PEFT on failure/cancel; exact location: `@@ -375,6 +376,25 @@ def train(`.
- **ADD** `release_training_memory()` — release QLoRA graph before conversion and restore PEFT on failure/cancel; exact location: `@@ -375,6 +376,25 @@ def train(`.
- **ADD** `blank-line formatting change within this audited hunk` — release QLoRA graph before conversion and restore PEFT on failure/cancel; exact location: `@@ -375,6 +376,25 @@ def train(`.

### `@@ -396,7 +416,7 @@ def train(`

- **REMOVE** `except Exception:` — release QLoRA graph before conversion and restore PEFT on failure/cancel; exact location: `@@ -396,7 +416,7 @@ def train(`.
- **ADD** `except BaseException:` — release QLoRA graph before conversion and restore PEFT on failure/cancel; exact location: `@@ -396,7 +416,7 @@ def train(`.

### `@@ -408,9 +428,25 @@ def train(`

- **REMOVE** `del model` — release QLoRA graph before conversion and restore PEFT on failure/cancel; exact location: `@@ -408,9 +428,25 @@ def train(`.
- **REMOVE** `gc.collect()` — release QLoRA graph before conversion and restore PEFT on failure/cancel; exact location: `@@ -408,9 +428,25 @@ def train(`.
- **REMOVE** `torch.cuda.empty_cache()` — release QLoRA graph before conversion and restore PEFT on failure/cancel; exact location: `@@ -408,9 +428,25 @@ def train(`.
- **ADD** `optimizer.zero_grad(set_to_none=True)` — release QLoRA graph before conversion and restore PEFT on failure/cancel; exact location: `@@ -408,9 +428,25 @@ def train(`.
- **ADD** `except Exception:` — release QLoRA graph before conversion and restore PEFT on failure/cancel; exact location: `@@ -408,9 +428,25 @@ def train(`.
- **ADD** `pass` — release QLoRA graph before conversion and restore PEFT on failure/cancel; exact location: `@@ -408,9 +428,25 @@ def train(`.
- **ADD** `try:` — release QLoRA graph before conversion and restore PEFT on failure/cancel; exact location: `@@ -408,9 +428,25 @@ def train(`.
- **ADD** `before.clear()` — release QLoRA graph before conversion and restore PEFT on failure/cancel; exact location: `@@ -408,9 +428,25 @@ def train(`.
- **ADD** `except Exception:` — release QLoRA graph before conversion and restore PEFT on failure/cancel; exact location: `@@ -408,9 +428,25 @@ def train(`.
- **ADD** `pass` — release QLoRA graph before conversion and restore PEFT on failure/cancel; exact location: `@@ -408,9 +428,25 @@ def train(`.
- **ADD** `try:` — release QLoRA graph before conversion and restore PEFT on failure/cancel; exact location: `@@ -408,9 +428,25 @@ def train(`.
- **ADD** `del optimizer` — release QLoRA graph before conversion and restore PEFT on failure/cancel; exact location: `@@ -408,9 +428,25 @@ def train(`.
- **ADD** `except Exception:` — release QLoRA graph before conversion and restore PEFT on failure/cancel; exact location: `@@ -408,9 +428,25 @@ def train(`.
- **ADD** `pass` — release QLoRA graph before conversion and restore PEFT on failure/cancel; exact location: `@@ -408,9 +428,25 @@ def train(`.
- **ADD** `try:` — release QLoRA graph before conversion and restore PEFT on failure/cancel; exact location: `@@ -408,9 +428,25 @@ def train(`.
- **ADD** `del tokenizer` — release QLoRA graph before conversion and restore PEFT on failure/cancel; exact location: `@@ -408,9 +428,25 @@ def train(`.
- **ADD** `except Exception:` — release QLoRA graph before conversion and restore PEFT on failure/cancel; exact location: `@@ -408,9 +428,25 @@ def train(`.
- **ADD** `pass` — release QLoRA graph before conversion and restore PEFT on failure/cancel; exact location: `@@ -408,9 +428,25 @@ def train(`.
- **ADD** `try:` — release QLoRA graph before conversion and restore PEFT on failure/cancel; exact location: `@@ -408,9 +428,25 @@ def train(`.
- **ADD** `if model is not None:` — release QLoRA graph before conversion and restore PEFT on failure/cancel; exact location: `@@ -408,9 +428,25 @@ def train(`.
- **ADD** `del model` — release QLoRA graph before conversion and restore PEFT on failure/cancel; exact location: `@@ -408,9 +428,25 @@ def train(`.
- **ADD** `release_training_memory()` — release QLoRA graph before conversion and restore PEFT on failure/cancel; exact location: `@@ -408,9 +428,25 @@ def train(`.
