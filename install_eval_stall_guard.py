#!/usr/bin/env python3
from pathlib import Path
import re

P = Path("master_4000_eval_suite.py")
MARK = "# === EYAD OVERNIGHT STALL GUARD ==="

if not P.exists():
    raise SystemExit("[!] master_4000_eval_suite.py not found")

src = P.read_text(encoding="utf-8")

PATCH = r'''
# === EYAD OVERNIGHT STALL GUARD ===
# Supervises whole evaluation items and redraws the original V5 telemetry line every 10s.
# V5 prompts, generation, scoring, 4096/EOS and evaluation stages are not changed.
import collections as _wd_collections
import json as _wd_json
import os as _wd_os
import subprocess as _wd_subprocess
import sys as _wd_sys
import time as _wd_time
from datetime import timedelta as _wd_timedelta
from pathlib import Path as _WdPath

_WD_HEARTBEAT = _WdPath("eval_results/.eval_item_heartbeat.json")
_WD_STALLS = _WdPath("eval_results/.eval_item_stalls.json")
_WD_METRICS = _WdPath("eval_results/.eval_display_metrics.json")
_WD_CHECKPOINT = _WdPath("eval_results/eval_checkpoint_4000.json")
_WD_TELEMETRY = _WdPath("eval_results/telemetry_stream.jsonl")
_WD_TOTAL = 4014
_WD_TIMEOUT = int(_wd_os.environ.get("EVAL_ITEM_WATCHDOG_SECONDS", "1200"))
_WD_MAX_STALLS = int(_wd_os.environ.get("EVAL_ITEM_MAX_STALLS", "2"))
_WD_STATUS_EVERY = float(_wd_os.environ.get("EVAL_STATUS_SECONDS", "10"))
_WD_CHILD = "EYAD_MASTER_EVAL_CHILD"
_WD_RUNTIME = {}


def _wd_atomic_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(_wd_json.dumps(obj), encoding="utf-8")
    _wd_os.replace(tmp, path)


def _wd_read(path):
    try:
        return _wd_json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _wd_checkpoint_snapshot(phase_hint=None):
    chk = _wd_read(_WD_CHECKPOINT)
    cache = chk.get("completed_items", {}) if isinstance(chk, dict) else {}
    phase = str(chk.get("phase") or phase_hint or "Phase 1: Baseline")
    prefix = phase + "_"
    vals = []
    if isinstance(cache, dict):
        for key, value in cache.items():
            if not isinstance(key, str) or key.startswith("__"):
                continue
            if key.startswith(prefix):
                vals.append(value is True)
    return phase, len(vals), sum(vals)


def _wd_seed_tps():
    env = _wd_os.environ.get("EVAL_LAST_REAL_TPS")
    try:
        if env and float(env) > 0:
            return float(env)
    except Exception:
        pass
    old = _wd_read(_WD_METRICS)
    try:
        x = float(old.get("last_tps", 0) or 0)
        if x > 0:
            return x
    except Exception:
        pass
    try:
        if _WD_TELEMETRY.exists():
            for line in reversed(_WD_TELEMETRY.read_text(encoding="utf-8", errors="ignore").splitlines()):
                try:
                    rec = _wd_json.loads(line)
                    x = float(rec.get("tok_per_sec", 0) or 0)
                    if x > 0:
                        return x
                except Exception:
                    continue
    except Exception:
        pass
    return 0.0


def _wd_count(key):
    return int(_wd_read(_WD_STALLS).get(key, 0) or 0)


def _wd_set_count(key, n):
    d = _wd_read(_WD_STALLS)
    if n <= 0:
        d.pop(key, None)
    else:
        d[key] = int(n)
    _wd_atomic_json(_WD_STALLS, d)


def _wd_save_metrics(split_name):
    r = _WD_RUNTIME
    old = _wd_read(_WD_METRICS)
    _wd_atomic_json(_WD_METRICS, {
        "phase": r.get("phase", "Phase 1: Baseline"),
        "done": int(r.get("done", 0)),
        "correct": int(r.get("correct", 0)),
        "last_tps": float(r.get("last_tps", 0.0) or 0.0),
        "durations": list(r.get("durations", []))[-20:],
        "split": split_name or old.get("split", ""),
        "updated": _wd_time.time(),
    })


def _wd_init_runtime():
    phase, done, correct = _wd_checkpoint_snapshot()
    old = _wd_read(_WD_METRICS)
    durations = []
    try:
        durations = [float(x) for x in old.get("durations", [])[-20:] if float(x) > 0]
    except Exception:
        durations = []
    _WD_RUNTIME.clear()
    _WD_RUNTIME.update({
        "phase": phase,
        "done": done,
        "correct": correct,
        "last_tps": _wd_seed_tps(),
        "durations": _wd_collections.deque(durations, maxlen=20),
    })
    _wd_save_metrics(None)


_wd_original_eval_single = Master4000EvaluationEngine._evaluate_single_item


def _wd_eval_single(self, split_name, item):
    item_id = str(item.get("id", "unknown"))
    key = f"{split_name}:{item_id}"

    phase_now, done_now, correct_now = _wd_checkpoint_snapshot(_WD_RUNTIME.get("phase"))
    if phase_now != _WD_RUNTIME.get("phase"):
        _WD_RUNTIME["phase"] = phase_now
        _WD_RUNTIME["done"] = done_now
        _WD_RUNTIME["correct"] = correct_now
        _WD_RUNTIME["durations"] = _wd_collections.deque(maxlen=20)

    if _wd_count(key) >= _WD_MAX_STALLS:
        print(f"\n[!] Watchdog: skipping repeatedly stalled item {key}; recorded failed.", flush=True)
        _wd_set_count(key, 0)
        return False

    _wd_atomic_json(_WD_HEARTBEAT, {
        "active": True,
        "pid": _wd_os.getpid(),
        "key": key,
        "split": split_name,
        "item_id": item_id,
        "started": _wd_time.time(),
    })
    _wd_save_metrics(split_name)

    completed = False
    t0 = _wd_time.perf_counter()
    try:
        out = _wd_original_eval_single(self, split_name, item)
        completed = True
        _wd_set_count(key, 0)

        dur = max(0.001, _wd_time.perf_counter() - t0)
        _WD_RUNTIME["done"] = int(_WD_RUNTIME.get("done", 0)) + 1
        if bool(out):
            _WD_RUNTIME["correct"] = int(_WD_RUNTIME.get("correct", 0)) + 1
        _WD_RUNTIME["durations"].append(dur)
        try:
            real_tps = float(getattr(self, "last_tok_per_sec", 0.0) or 0.0)
            if real_tps > 0:
                _WD_RUNTIME["last_tps"] = real_tps
        except Exception:
            pass
        _wd_save_metrics(split_name)
        return out
    finally:
        if completed:
            _wd_atomic_json(_WD_HEARTBEAT, {
                "active": False,
                "pid": _wd_os.getpid(),
                "key": key,
                "split": split_name,
                "item_id": item_id,
                "ended": _wd_time.time(),
            })


Master4000EvaluationEngine._evaluate_single_item = _wd_eval_single


def _wd_render_original_line(hb):
    m = _wd_read(_WD_METRICS)
    phase = str(m.get("phase") or "Phase 1: Baseline")
    split_name = str(hb.get("split") or m.get("split") or "")
    done = max(0, min(_WD_TOTAL, int(m.get("done", 0) or 0)))
    correct = max(0, int(m.get("correct", 0) or 0))
    acc = (100.0 * correct / done) if done else 0.0
    tps = max(0.0, float(m.get("last_tps", 0.0) or 0.0))
    left = max(0, _WD_TOTAL - done)

    try:
        durs = [float(x) for x in m.get("durations", []) if float(x) > 0]
    except Exception:
        durs = []

    if durs:
        eta = str(_wd_timedelta(seconds=int((sum(durs) / len(durs)) * left)))
    elif tps > 0:
        eta = str(_wd_timedelta(seconds=int(left * 4096 / tps)))
    else:
        eta = "calculating"

    pct = 100.0 * done / _WD_TOTAL
    n = 25
    fill = int(n * done / _WD_TOTAL)
    bar = "█" * fill + "░" * (n - fill)
    return (
        f"[{bar}] {pct:5.1f}% | {done}/{_WD_TOTAL} | Acc: {acc:5.1f}% | "
        f"{tps:4.1f} t/s | ETA: {eta} | {phase} | {split_name}"
    )


def _wd_child_run():
    _wd_init_runtime()
    runner = Master4000EvaluationEngine(max_duration_hours=72.0)
    runner.run_full_suite()


def _wd_supervise():
    env = dict(_wd_os.environ)
    env[_WD_CHILD] = "1"

    while True:
        _wd_atomic_json(_WD_HEARTBEAT, {"active": False})
        child = _wd_subprocess.Popen(
            [_wd_sys.executable, "-u", _wd_os.path.abspath(__file__)],
            env=env,
        )
        stalled = False
        last_status = 0.0

        try:
            while child.poll() is None:
                _wd_time.sleep(1)
                hb = _wd_read(_WD_HEARTBEAT)
                if not hb.get("active") or int(hb.get("pid", -1)) != child.pid:
                    continue

                now = _wd_time.time()
                age = now - float(hb.get("started", now))

                if now - last_status >= _WD_STATUS_EVERY:
                    _wd_sys.stdout.write("\r" + _wd_render_original_line(hb) + "   ")
                    _wd_sys.stdout.flush()
                    last_status = now

                if age < _WD_TIMEOUT:
                    continue

                key = str(hb.get("key", "unknown"))
                n = _wd_count(key) + 1
                _wd_set_count(key, n)
                print(
                    f"\n[!] Watchdog: {key} stuck {int(age)}s; "
                    f"restarting from checkpoint ({n}/{_WD_MAX_STALLS}).",
                    flush=True,
                )
                child.kill()
                child.wait()
                stalled = True
                break
        except KeyboardInterrupt:
            print("", flush=True)
            try:
                child.send_signal(2)
                child.wait(timeout=10)
            except Exception:
                try:
                    child.kill()
                except Exception:
                    pass
            return 130

        if stalled:
            _wd_time.sleep(2)
            continue
        return int(child.returncode or 0)
# === END EYAD OVERNIGHT STALL GUARD ===
'''

MAIN = r'''
if __name__ == "__main__":
    if _wd_os.environ.get(_WD_CHILD) == "1":
        _wd_child_run()
    else:
        raise SystemExit(_wd_supervise())
'''

if MARK in src:
    base = src[:src.index(MARK)]
else:
    main = re.search(r'(?m)^if __name__ == ["\']__main__["\']:\s*$', src)
    if not main:
        raise SystemExit("[!] __main__ block not found; source untouched")
    base = src[:main.start()]

new_src = base + PATCH + "\n" + MAIN
compile(new_src, str(P), "exec")
P.write_text(new_src, encoding="utf-8")
print("[✓] Original V5 telemetry format restored.")
print("[✓] Full progress line redraws every 10s using real checkpoint/last-run metrics.")
print("[✓] V5 prompts/generation/scoring/stages remain untouched.")
print("[✓] 20-minute stall watchdog remains active.")
