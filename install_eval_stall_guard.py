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
# Supervises whole items and refreshes the evaluator's REAL telemetry line every 10s.
# Does not change V5 prompts, _fast_generate, token limits, EOS handling, scoring, or stages.
import json as _wd_json
import math as _wd_math
import os as _wd_os
import re as _wd_re
import signal as _wd_signal
import subprocess as _wd_subprocess
import sys as _wd_sys
import time as _wd_time
from pathlib import Path as _WdPath
try:
    import psutil as _wd_psutil
except Exception:
    _wd_psutil = None

_WD_STATE = _WdPath("eval_results/.eval_live_state.json")
_WD_STALLS = _WdPath("eval_results/.eval_item_stalls.json")
_WD_TIMEOUT = int(_wd_os.environ.get("EVAL_ITEM_WATCHDOG_SECONDS", "1200"))
_WD_MAX_STALLS = int(_wd_os.environ.get("EVAL_ITEM_MAX_STALLS", "2"))
_WD_STATUS_EVERY = float(_wd_os.environ.get("EVAL_STATUS_SECONDS", "10"))
_WD_CHILD = "EYAD_MASTER_EVAL_CHILD"


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


def _wd_merge(**updates):
    d = _wd_read(_WD_STATE)
    d.update(updates)
    _wd_atomic_json(_WD_STATE, d)
    return d


def _wd_count(key):
    return int(_wd_read(_WD_STALLS).get(key, 0) or 0)


def _wd_set_count(key, n):
    d = _wd_read(_WD_STALLS)
    if n <= 0:
        d.pop(key, None)
    else:
        d[key] = int(n)
    _wd_atomic_json(_WD_STALLS, d)


def _wd_fmt_elapsed(seconds):
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _wd_bar(done, total, width=25):
    if total <= 0:
        return "░" * width
    fill = max(0, min(width, int(width * done / total)))
    return "█" * fill + "░" * (width - fill)


class _WdTee:
    """Pass child output through unchanged and remember the evaluator's latest real telemetry line."""
    def __init__(self, real):
        self.real = real

    def write(self, text):
        n = self.real.write(text)
        self.real.flush()
        for piece in _wd_re.split(r"[\r\n]+", text):
            piece = piece.strip()
            if not piece:
                continue
            m = _wd_re.search(r"Checkpoint Loaded:\s*(\d+)\s*/\s*(\d+).*?\(([\d.]+)%\)", piece)
            if m:
                _wd_merge(completed=int(m.group(1)), total=int(m.group(2)), percent=float(m.group(3)))
            if "t/s" in piece and "ETA:" in piece and _wd_re.search(r"\d+\s*/\s*\d+", piece):
                m2 = _wd_re.search(r"(\d+)\s*/\s*(\d+)", piece)
                mt = _wd_re.search(r"([\d.]+)\s*t/s", piece)
                ma = _wd_re.search(r"Acc:\s*([\d.]+)%", piece)
                updates = {"progress_line": piece, "progress_time": _wd_time.time()}
                if m2:
                    updates.update(completed=int(m2.group(1)), total=int(m2.group(2)), percent=100.0 * int(m2.group(1)) / max(1, int(m2.group(2))))
                if mt:
                    updates["last_tps"] = float(mt.group(1))
                if ma:
                    updates["accuracy"] = float(ma.group(1))
                _wd_merge(**updates)
        return n

    def flush(self):
        return self.real.flush()

    def __getattr__(self, name):
        return getattr(self.real, name)


_wd_original_eval_single = Master4000EvaluationEngine._evaluate_single_item


def _wd_eval_single(self, split_name, item):
    item_id = str(item.get("id", "unknown"))
    key = f"{split_name}:{item_id}"

    if _wd_count(key) >= _WD_MAX_STALLS:
        print(f"\n[!] Watchdog: skipping repeatedly stalled item {key}; recorded failed.", flush=True)
        _wd_set_count(key, 0)
        return False

    last_tps = float(getattr(self, "last_tok_per_sec", 0.0) or 0.0)
    _wd_merge(
        active=True,
        pid=_wd_os.getpid(),
        key=key,
        split=split_name,
        item_id=item_id,
        started=_wd_time.time(),
        last_tps=last_tps if _wd_math.isfinite(last_tps) and last_tps > 0 else _wd_read(_WD_STATE).get("last_tps", 0.0),
    )

    completed = False
    try:
        out = _wd_original_eval_single(self, split_name, item)
        completed = True
        _wd_set_count(key, 0)
        return out
    finally:
        if completed:
            new_tps = float(getattr(self, "last_tok_per_sec", 0.0) or 0.0)
            _wd_merge(
                active=False,
                pid=_wd_os.getpid(),
                key=key,
                split=split_name,
                item_id=item_id,
                ended=_wd_time.time(),
                last_tps=new_tps if _wd_math.isfinite(new_tps) and new_tps > 0 else _wd_read(_WD_STATE).get("last_tps", 0.0),
            )


Master4000EvaluationEngine._evaluate_single_item = _wd_eval_single


def _wd_child_run():
    _wd_sys.stdout = _WdTee(_wd_sys.stdout)
    runner = Master4000EvaluationEngine(max_duration_hours=72.0)
    runner.run_full_suite()


def _wd_render(child, state, age):
    split_name = str(state.get("split", "?"))
    item_id = str(state.get("item_id", "?"))
    done = int(state.get("completed", 0) or 0)
    total = int(state.get("total", 4014) or 4014)
    pct = float(state.get("percent", 100.0 * done / max(1, total)) or 0.0)
    acc = state.get("accuracy")
    tps = float(state.get("last_tps", 0.0) or 0.0)

    ram = None
    if _wd_psutil is not None:
        try:
            ram = _wd_psutil.Process(child.pid).memory_info().rss / (1024 ** 3)
        except Exception:
            pass

    # If the evaluator has already emitted a real full telemetry line, retain its real
    # accuracy/tps/ETA/progress fields and only append the currently-running item/elapsed.
    real = str(state.get("progress_line", "") or "").strip()
    if real:
        real = _wd_re.sub(r"\s*\|\s*Current:.*$", "", real)
        suffix = f" | Current: {item_id} | elapsed {_wd_fmt_elapsed(age)}"
        if ram is not None and "RAM:" not in real:
            suffix += f" | RAM: {ram:.2f} GB"
        return real + suffix

    # Before the first newly-completed item, reconstruct the same telemetry shape from
    # the loaded checkpoint. Unknown fields stay explicitly unknown rather than fake.
    acc_text = f"{float(acc):5.1f}%" if acc is not None else "  --.-%"
    tps_text = f"{tps:4.1f} t/s" if tps > 0 else " --.- t/s"
    ram_text = f"{ram:.2f} GB" if ram is not None else "-- GB"
    return (
        f"[{_wd_bar(done, total)}] {pct:5.1f}% | {done}/{total} | Acc: {acc_text} | "
        f"{tps_text} | ETA: calculating | RAM: {ram_text} | {split_name} | "
        f"Current: {item_id} | elapsed {_wd_fmt_elapsed(age)}"
    )


def _wd_supervise():
    env = dict(_wd_os.environ)
    env[_WD_CHILD] = "1"

    while True:
        old = _wd_read(_WD_STATE)
        _wd_atomic_json(_WD_STATE, {
            "active": False,
            "completed": old.get("completed", 0),
            "total": old.get("total", 4014),
            "percent": old.get("percent", 0.0),
            "accuracy": old.get("accuracy"),
            "last_tps": old.get("last_tps", 0.0),
            "progress_line": old.get("progress_line", ""),
        })
        child = _wd_subprocess.Popen([_wd_sys.executable, "-u", _wd_os.path.abspath(__file__)], env=env)
        stalled = False
        last_status = 0.0
        last_split = None

        try:
            while child.poll() is None:
                _wd_time.sleep(1)
                state = _wd_read(_WD_STATE)
                if not state.get("active") or int(state.get("pid", -1)) != child.pid:
                    continue

                now = _wd_time.time()
                started = float(state.get("started", now))
                age = now - started
                key = str(state.get("key", "unknown"))
                split_name = str(state.get("split", "?"))

                if split_name != last_split:
                    print(f"\n▶ STARTING SPLIT: {split_name}", flush=True)
                    last_split = split_name
                    last_status = 0.0

                if now - last_status >= _WD_STATUS_EVERY:
                    line = _wd_render(child, state, age)
                    print("\r" + line + "   ", end="", flush=True)
                    last_status = now

                if age < _WD_TIMEOUT:
                    continue

                n = _wd_count(key) + 1
                _wd_set_count(key, n)
                print(f"\n[!] Watchdog: {key} stuck {int(age)}s; restarting from checkpoint ({n}/{_WD_MAX_STALLS}).", flush=True)
                child.kill()
                child.wait()
                stalled = True
                break
        except KeyboardInterrupt:
            print("", flush=True)
            try:
                child.send_signal(_wd_signal.SIGINT)
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
print("[✓] Full evaluator telemetry refresh installed; V5 prompts/generator/evaluator untouched.")
print("[✓] Full progress telemetry is now redrawn every 10 seconds while an item runs.")
print("[✓] 20-minute stall recovery remains active.")
