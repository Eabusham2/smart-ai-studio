"""Optional flagship DeepSWE v1.1 sidecar for the master evaluation.

This is intentionally separate from SWE-bench Verified. When enabled at startup it
runs all 113 official DeepSWE v1.1 tasks from the pinned upstream repository using
Pier + mini-swe-agent's official text-based configuration. Candidate agent runs have
verification disabled; only the answer-blind selected final patch is replayed into a
fresh official task and verified once. The normal 4K suite is unchanged when the
user answers N (the default).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DEEPSWE_REPO = "https://github.com/datacurve-ai/deep-swe.git"
DEEPSWE_PIN = "0b9fabbb63b9104d678fe965e1632f2dd9eaa2ea"
MINISWE_CONFIG_PIN = "04d809ceab9df28f9adaed044884180159172930"
MINISWE_CONFIG_URL = (
    "https://raw.githubusercontent.com/SWE-agent/mini-swe-agent/"
    f"{MINISWE_CONFIG_PIN}/src/minisweagent/config/mini_textbased.yaml"
)
DEEPSWE_TASK_COUNT = 113
DEEPSWE_CONTEXT_TOKENS = int(os.environ.get("SMARTAI_DEEPSWE_CONTEXT", "226000"))
DEEPSWE_MAX_OUTPUT_TOKENS = int(os.environ.get("SMARTAI_DEEPSWE_MAX_OUTPUT", "8192"))
DEEPSWE_CACHE = Path("eval_datasets") / "deepswe-v1.1"
DEEPSWE_RESULTS = Path("eval_results") / "deepswe_flagship"
DEEPSWE_STATE = DEEPSWE_RESULTS / "state.json"
SPLIT_NAME = "DeepSWE-Flagship-v1.1-113"


def _ask_enabled() -> bool:
    try:
        answer = input(
            "Run flagship DeepSWE v1.1 too? 113 long-horizon agent tasks, very expensive [y/N]: "
        ).strip().lower()
    except (EOFError, KeyboardInterrupt):
        answer = ""
    return answer in {"y", "yes"}


def _run(cmd: List[str], *, cwd: Optional[Path] = None, env: Optional[Dict[str, str]] = None,
         timeout: Optional[float] = None) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"STDOUT:\n{proc.stdout[-4000:]}\nSTDERR:\n{proc.stderr[-4000:]}"
        )
    return proc


def _ensure_tools() -> str:
    if shutil.which("git") is None:
        raise RuntimeError("Flagship DeepSWE requires git")
    if shutil.which("docker") is None:
        raise RuntimeError("Flagship DeepSWE requires Docker")
    pier = shutil.which("pier")
    if pier:
        return pier
    uv = shutil.which("uv")
    if not uv:
        raise RuntimeError(
            "Flagship DeepSWE requires Pier. Install uv/Pier, or run `uv tool install 'datacurve-pier>0.3.0'`."
        )
    _run([uv, "tool", "install", "datacurve-pier>0.3.0"], timeout=600)
    pier = shutil.which("pier")
    if not pier:
        candidate = Path.home() / ".local" / "bin" / "pier"
        if candidate.exists():
            pier = str(candidate)
    if not pier:
        raise RuntimeError("Pier installation completed but the `pier` executable was not found")
    return pier


def _ensure_official_repo() -> Tuple[Path, List[Path]]:
    DEEPSWE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    if not (DEEPSWE_CACHE / ".git").exists():
        if DEEPSWE_CACHE.exists():
            shutil.rmtree(DEEPSWE_CACHE)
        _run(["git", "clone", "--no-tags", DEEPSWE_REPO, str(DEEPSWE_CACHE)], timeout=1200)
    _run(["git", "fetch", "origin", DEEPSWE_PIN], cwd=DEEPSWE_CACHE, timeout=600)
    _run(["git", "checkout", "--detach", DEEPSWE_PIN], cwd=DEEPSWE_CACHE, timeout=120)
    _run(["git", "reset", "--hard", DEEPSWE_PIN], cwd=DEEPSWE_CACHE, timeout=120)
    tasks = sorted(
        p for p in (DEEPSWE_CACHE / "tasks").iterdir()
        if p.is_dir() and (p / "task.toml").exists() and (p / "instruction.md").exists()
    )
    if len(tasks) != DEEPSWE_TASK_COUNT:
        raise RuntimeError(
            f"Pinned DeepSWE v1.1 expected {DEEPSWE_TASK_COUNT} tasks, found {len(tasks)}; refusing substitution"
        )
    return DEEPSWE_CACHE, tasks


def _load_state() -> Dict[str, Any]:
    try:
        data = json.loads(DEEPSWE_STATE.read_text())
        if isinstance(data, dict) and data.get("benchmark_commit") == DEEPSWE_PIN:
            return data
    except Exception:
        pass
    return {
        "benchmark_commit": DEEPSWE_PIN,
        "context_tokens": DEEPSWE_CONTEXT_TOKENS,
        "baseline": {},
        "rsi": {},
        "final": {},
    }


def _save_state(state: Dict[str, Any]) -> None:
    DEEPSWE_RESULTS.mkdir(parents=True, exist_ok=True)
    tmp = DEEPSWE_STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True))
    tmp.replace(DEEPSWE_STATE)


def _task_instruction(task_dir: Path) -> str:
    return (task_dir / "instruction.md").read_text(encoding="utf-8", errors="replace").strip()


def _append_system_rule(config_text: str, universal_rule: str, stage_rule: str = "",
                        prior_patch: str = "") -> str:
    marker = "  instance_template: |"
    if marker not in config_text:
        raise RuntimeError("Official mini_textbased.yaml schema changed; refusing to guess system-template placement")
    additions = [
        "",
        "    Smart AI Studio reasoning constraints:",
        *["    " + line for line in universal_rule.strip().splitlines() if line.strip()],
    ]
    if stage_rule.strip():
        additions.extend("    " + line for line in stage_rule.strip().splitlines())
    if prior_patch.strip():
        additions.append("    Review your own prior selected patch below; no verifier result or hidden answer is provided:")
        additions.extend("    " + line for line in prior_patch.splitlines())
    return config_text.replace(marker, "\n".join(additions) + "\n" + marker, 1)


def _official_agent_config(universal_rule: str, stage: str, prior_patch: str = "") -> Path:
    DEEPSWE_RESULTS.mkdir(parents=True, exist_ok=True)
    cache = DEEPSWE_RESULTS / f"mini_textbased_{MINISWE_CONFIG_PIN}.yaml"
    if not cache.exists():
        req = urllib.request.Request(MINISWE_CONFIG_URL, headers={"User-Agent": "SmartAIStudio/DeepSWE"})
        with urllib.request.urlopen(req, timeout=30) as response:
            cache.write_bytes(response.read())
    source = cache.read_text(encoding="utf-8")
    stage_rule = ""
    if stage.startswith("rsi"):
        stage_rule = (
            "This is an independent recursive-self-improvement attempt. Critique only your own reasoning/patches. "
            "Never assume, request, or infer verifier output, a hidden reference solution, or the correct answer."
        )
    elif stage.startswith("phase4"):
        stage_rule = (
            "This is one independent Pro-search branch. Solve the task independently; no verifier result or hidden answer exists in context."
        )
    rendered = _append_system_rule(source, universal_rule, stage_rule, prior_patch)
    fd, path = tempfile.mkstemp(prefix="smartai_deepswe_", suffix=".yaml", dir=DEEPSWE_RESULTS)
    os.close(fd)
    Path(path).write_text(rendered, encoding="utf-8")
    return Path(path)


class _LocalModelBridge:
    def __init__(self, owner, runtime_module, phase4_module):
        self.owner = owner
        self.runtime_module = runtime_module
        self.phase4_module = phase4_module
        self.server: Optional[ThreadingHTTPServer] = None
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()

    def start(self) -> str:
        bridge = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                return

            def _json(self, code: int, payload: Dict[str, Any]):
                body = json.dumps(payload).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path.rstrip("/") in {"/v1/models", "/models"}:
                    self._json(200, {"object": "list", "data": [{"id": "smart-ai-studio", "object": "model"}]})
                else:
                    self._json(404, {"error": {"message": "not found"}})

            def do_POST(self):
                if self.path.rstrip("/") not in {"/v1/chat/completions", "/chat/completions"}:
                    self._json(404, {"error": {"message": "not found"}})
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    req = json.loads(self.rfile.read(length) or b"{}")
                    messages = req.get("messages") or []
                    if not isinstance(messages, list) or not messages:
                        raise ValueError("messages must be a non-empty list")
                    tok = bridge.owner.engine.tokenizer
                    if hasattr(tok, "apply_chat_template"):
                        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    else:
                        prompt = "\n".join(
                            f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages
                        ) + "\nassistant:"
                    prompt_tokens = len(tok.encode(prompt))
                    detected = bridge.runtime_module._model_context_limit(bridge.owner.engine)
                    effective_context = min(
                        DEEPSWE_CONTEXT_TOKENS,
                        int(detected) if detected else DEEPSWE_CONTEXT_TOKENS,
                    )
                    if prompt_tokens >= effective_context:
                        self._json(400, {"error": {"message": (
                            f"DeepSWE prompt is {prompt_tokens} tokens, exceeding the allowed context "
                            f"({effective_context}; configured ceiling {DEEPSWE_CONTEXT_TOKENS}). No compaction performed."
                        )}})
                        return
                    requested = int(req.get("max_tokens") or req.get("max_completion_tokens") or DEEPSWE_MAX_OUTPUT_TOKENS)
                    max_tokens = max(1, min(DEEPSWE_MAX_OUTPUT_TOKENS, requested, effective_context - prompt_tokens))
                    temperature = float(req.get("temperature", 0.2) or 0.2)
                    top_p = float(req.get("top_p", 0.92) or 0.92)
                    with bridge.lock:
                        output = bridge.phase4_module._generate_branches_same_model(
                            bridge.owner,
                            prompt,
                            [temperature],
                            max_tokens=max_tokens,
                            top_p=top_p,
                        )[0]
                    completion_tokens = len(tok.encode(output))
                    now = int(time.time())
                    self._json(200, {
                        "id": f"chatcmpl-smartai-{now}",
                        "object": "chat.completion",
                        "created": now,
                        "model": str(req.get("model") or "smart-ai-studio"),
                        "choices": [{
                            "index": 0,
                            "message": {"role": "assistant", "content": output},
                            "finish_reason": "stop",
                        }],
                        "usage": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": prompt_tokens + completion_tokens,
                        },
                    })
                except Exception as exc:
                    self._json(500, {"error": {"message": repr(exc)}})

        self.server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
        port = int(self.server.server_address[1])
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        host = os.environ.get("SMARTAI_DEEPSWE_AGENT_HOST", "host.docker.internal")
        return f"http://{host}:{port}/v1"

    def stop(self):
        if self.server:
            self.server.shutdown()
            self.server.server_close()
        if self.thread:
            self.thread.join(timeout=2)


def _find_nonempty(root: Path, name: str) -> Path:
    matches = [p for p in root.rglob(name) if p.is_file() and p.stat().st_size > 0]
    if not matches:
        raise RuntimeError(f"Pier produced no non-empty {name} under {root}")
    return max(matches, key=lambda p: p.stat().st_mtime)


def _pier_env() -> Dict[str, str]:
    env = os.environ.copy()
    root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def _generate_patch(pier: str, task_dir: Path, base_url: str, universal_rule: str,
                    stage: str, temperature: float, prior_patch: str = "") -> str:
    config = _official_agent_config(universal_rule, stage, prior_patch)
    jobs_dir = DEEPSWE_RESULTS / "pier_jobs" / stage
    jobs_dir.mkdir(parents=True, exist_ok=True)
    tag = hashlib.sha1(f"{task_dir.name}:{stage}:{temperature}:{time.time_ns()}".encode()).hexdigest()[:10]
    job = f"{task_dir.name}-{tag}"
    kwargs = json.dumps({"max_tokens": DEEPSWE_MAX_OUTPUT_TOKENS, "temperature": float(temperature), "top_p": 0.92})
    cmd = [
        pier, "run", "-p", str(task_dir),
        "--agent", "mini-swe-agent",
        "--model", "openai/smart-ai-studio",
        "--env", "docker",
        "--n-concurrent", "1",
        "--disable-verification",
        "--yes",
        "--jobs-dir", str(jobs_dir),
        "--job-name", job,
        "--agent-env", f"OPENAI_BASE_URL={base_url}",
        "--agent-env", "OPENAI_API_KEY=smartai-local",
        "--agent-env", "MSWEA_API_KEY=smartai-local",
        "--agent-kwarg", "model_class=litellm_textbased",
        "--agent-kwarg", f"config_file={config}",
        "--agent-kwarg", f"model_kwargs={kwargs}",
    ]
    try:
        _run(cmd, env=_pier_env(), timeout=None)
        patch_path = _find_nonempty(jobs_dir / job, "model.patch")
        return patch_path.read_text(encoding="utf-8", errors="replace")
    finally:
        try:
            config.unlink()
        except OSError:
            pass


def _verify_selected_patch(pier: str, task_dir: Path, patch: str, stage: str) -> bool:
    jobs_dir = DEEPSWE_RESULTS / "pier_verify" / stage
    jobs_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".patch", prefix="smartai_selected_", delete=False,
                                     dir=DEEPSWE_RESULTS, encoding="utf-8") as f:
        f.write(patch)
        patch_path = Path(f.name).resolve()
    tag = hashlib.sha1(f"{task_dir.name}:{stage}:{time.time_ns()}".encode()).hexdigest()[:10]
    job = f"verify-{task_dir.name}-{tag}"
    cmd = [
        pier, "run", "-p", str(task_dir),
        "--agent-import-path", "eval.deepswe_patch_replay_agent:PatchReplayAgent",
        "--agent-kwarg", f"patch_file={patch_path}",
        "--env", "docker",
        "--n-concurrent", "1",
        "--enable-verification",
        "--yes",
        "--jobs-dir", str(jobs_dir),
        "--job-name", job,
    ]
    try:
        _run(cmd, env=_pier_env(), timeout=None)
        reward_path = _find_nonempty(jobs_dir / job, "reward.json")
        reward = json.loads(reward_path.read_text(encoding="utf-8"))
        return float(reward.get("reward", 0.0)) == 1.0
    finally:
        try:
            patch_path.unlink()
        except OSError:
            pass


def _blind_select(phase4_module, owner, patches: List[str]) -> Tuple[str, int, str]:
    if not patches:
        return "", 0, "no branches"
    fingerprints = ["PATCH_SHA256:" + hashlib.sha256(p.encode()).hexdigest() for p in patches]
    _, idx, _, selection = phase4_module._choose_without_ground_truth(
        owner,
        "FlagshipSWE-BlindSelection",
        {"prompt": ""},
        fingerprints,
    )
    return patches[idx], idx, selection


def _score(values: Dict[str, bool]) -> float:
    return 100.0 * sum(v is True for v in values.values()) / max(1, DEEPSWE_TASK_COUNT)


def install(runtime_module, phase4_module, prompt_module, cls) -> None:
    original_run = cls.run_full_suite
    original_eval_all = cls._evaluate_all_splits
    original_rsi = phase4_module._run_rsi_self_improvement

    def run_with_optional_deepswe(self):
        self._deepswe_enabled = _ask_enabled()
        self._deepswe_bridge = None
        if not self._deepswe_enabled:
            print("[*] DeepSWE flagship: skipped (N). Normal suite unchanged.", flush=True)
            return original_run(self)

        pier = _ensure_tools()
        _, tasks = _ensure_official_repo()
        state = _load_state()
        bridge = _LocalModelBridge(self, runtime_module, phase4_module)
        base_url = bridge.start()
        detected = runtime_module._model_context_limit(self.engine)
        effective = min(DEEPSWE_CONTEXT_TOKENS, int(detected) if detected else DEEPSWE_CONTEXT_TOKENS)
        print(
            f"[*] DeepSWE flagship enabled: {len(tasks)} official v1.1 tasks | "
            f"configured context {DEEPSWE_CONTEXT_TOKENS:,} | effective model ceiling {effective:,} | "
            f"max output/agent step {DEEPSWE_MAX_OUTPUT_TOKENS:,}",
            flush=True,
        )
        self._deepswe_pier = pier
        self._deepswe_tasks = tasks
        self._deepswe_state = state
        self._deepswe_base_url = base_url
        self._deepswe_bridge = bridge
        try:
            return original_run(self)
        finally:
            bridge.stop()

    def eval_all_with_deepswe(self, splits, cache, phase, start, total):
        scores = original_eval_all(self, splits, cache, phase, start, total)
        if not getattr(self, "_deepswe_enabled", False):
            return scores

        state = self._deepswe_state
        tasks = self._deepswe_tasks
        pier = self._deepswe_pier
        base_url = self._deepswe_base_url
        universal = prompt_module.GLOBAL_SYSTEM_SUFFIX

        if phase == "Phase 1: Baseline":
            baseline = state.setdefault("baseline", {})
            durations: List[float] = []
            for idx, task_dir in enumerate(tasks, 1):
                task_id = task_dir.name
                if task_id in baseline:
                    continue
                t0 = time.perf_counter()
                patch = _generate_patch(pier, task_dir, base_url, universal, "baseline", 0.20)
                ok = _verify_selected_patch(pier, task_dir, patch, "baseline")
                baseline[task_id] = bool(ok)
                _save_state(state)
                durations.append(time.perf_counter() - t0)
                avg = sum(durations) / len(durations)
                eta = avg * max(0, len(tasks) - idx)
                print(
                    f"[DeepSWE Phase 1] {idx}/{len(tasks)} ({idx/len(tasks)*100:.2f}%) | "
                    f"{task_id}: {'PASS' if ok else 'FAIL'} | ETA {int(eta)}s",
                    flush=True,
                )
            scores[SPLIT_NAME] = _score(baseline)

        elif phase == "Phase 4: Post-Consolidation":
            baseline = state.get("baseline", {})
            if len(baseline) != DEEPSWE_TASK_COUNT:
                raise RuntimeError(
                    "DeepSWE Phase-4 retake requires a complete DeepSWE baseline from the same opt-in state"
                )
            misses = [t for t in tasks if baseline.get(t.name) is False]
            final = state.setdefault("final", {})
            for idx, task_dir in enumerate(misses, 1):
                task_id = task_dir.name
                if task_id in final:
                    continue
                instruction = _task_instruction(task_dir)
                entropy = phase4_module._normalized_entropy(self, instruction)
                mode, branch_count = phase4_module._pro_router(self).route(
                    entropy, has_test_cases=True
                )
                temperatures = phase4_module.get_ladder_temperatures(branch_count)
                patches = [
                    _generate_patch(
                        pier, task_dir, base_url, universal,
                        f"phase4-b{bidx+1}", temp,
                    )
                    for bidx, temp in enumerate(temperatures)
                ]
                selected, winner_idx, selection = _blind_select(phase4_module, self, patches)
                ok = _verify_selected_patch(pier, task_dir, selected, "phase4-selected")
                final[task_id] = bool(ok)
                self._last_phase4_pro_meta = {
                    "mode": mode,
                    "entropy": float(entropy),
                    "branch_count": branch_count,
                    "temperatures": temperatures,
                    "winning_branch": winner_idx + 1,
                    "verified": bool(ok),
                    "selection": selection,
                    "all_branch_tokens": None,
                }
                _save_state(state)
                print(
                    f"[DeepSWE Phase 4 Pro] {idx}/{len(misses)} | {task_id} | "
                    f"{mode} | branches={branch_count} | selected={winner_idx+1} | "
                    f"{'PASS' if ok else 'FAIL'}",
                    flush=True,
                )
            combined = {
                task.name: (
                    True if baseline.get(task.name) is True else bool(final.get(task.name, False))
                )
                for task in tasks
            }
            scores[SPLIT_NAME] = _score(combined)

        return scores

    def rsi_with_deepswe(self, splits, cache):
        normal_seeded = original_rsi(self, splits, cache)
        if not getattr(self, "_deepswe_enabled", False):
            return normal_seeded

        state = self._deepswe_state
        baseline = state.get("baseline", {})
        misses = [t for t in self._deepswe_tasks if baseline.get(t.name) is False]
        rsi_state = state.setdefault("rsi", {})
        seeded = 0
        temperatures = [0.20, 0.38, 0.58, 0.82]
        universal = prompt_module.GLOBAL_SYSTEM_SUFFIX

        for item_idx, task_dir in enumerate(misses, 1):
            task_id = task_dir.name
            if rsi_state.get(task_id, {}).get("passed") is True:
                continue
            previous_patch = ""
            item_result = {"passed": False, "rounds": []}
            for round_idx in (1, 2):
                patches = [
                    _generate_patch(
                        self._deepswe_pier,
                        task_dir,
                        self._deepswe_base_url,
                        universal,
                        f"rsi-r{round_idx}-b{bidx+1}",
                        temp,
                        previous_patch if round_idx == 2 else "",
                    )
                    for bidx, temp in enumerate(temperatures)
                ]
                selected, winner_idx, selection = _blind_select(phase4_module, self, patches)
                # Verification happens only now, after all candidate generation and answer-blind selection.
                passed = _verify_selected_patch(
                    self._deepswe_pier,
                    task_dir,
                    selected,
                    f"rsi-r{round_idx}-selected",
                )
                item_result["rounds"].append({
                    "round": round_idx,
                    "winning_branch": winner_idx + 1,
                    "selection": selection,
                    "passed": bool(passed),
                })
                previous_patch = selected
                if passed:
                    item_result["passed"] = True
                    try:
                        self.engine.kg.log_interaction(
                            phase4_module.RSI_SESSION_ID,
                            _task_instruction(task_dir),
                            selected,
                            1.0,
                            1.0,
                            domain=f"RSI::{SPLIT_NAME}",
                        )
                        seeded += 1
                    except Exception:
                        pass
                    break
            rsi_state[task_id] = item_result
            _save_state(state)
            print(
                f"[DeepSWE RSI] {item_idx}/{len(misses)} | {task_id} | "
                f"{'VERIFIED SELF-CORRECTION' if item_result['passed'] else 'no verified correction'}",
                flush=True,
            )
        return normal_seeded + seeded

    cls.run_full_suite = run_with_optional_deepswe
    cls._evaluate_all_splits = eval_all_with_deepswe
    phase4_module._run_rsi_self_improvement = rsi_with_deepswe
