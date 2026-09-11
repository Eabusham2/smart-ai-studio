#!/usr/bin/env python3
from pathlib import Path
import re

P = Path("master_4000_eval_suite.py")
MARK = "# === EYAD OVERNIGHT STALL GUARD ==="
END_MARK = "# === END EYAD OVERNIGHT STALL GUARD ==="

if not P.exists():
    raise SystemExit("[!] master_4000_eval_suite.py not found")

src = P.read_text(encoding="utf-8")

PATCH = r'''
# === EYAD OVERNIGHT STALL GUARD ===
# Supervises whole evaluation items without changing V5 prompts/generation/evaluation behavior.
import json as _wd_json
import os as _wd_os
import subprocess as _wd_subprocess
import sys as _wd_sys
import time as _wd_time
from pathlib import Path as _WdPath

_WD_HEARTBEAT = _WdPath("eval_results/.eval_item_heartbeat.json")
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


_wd_original_eval_single = Master4000EvaluationEngine._evaluate_single_item


def _wd_eval_single(self, split_name, item):
    item_id = str(item.get("id", "unknown"))
    key = f"{split_name}:{item_id}"

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

    completed = False
    try:
        out = _wd_original_eval_single(self, split_name, item)
        completed = True
        _wd_set_count(key, 0)
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


def _wd_child_run():
    runner = Master4000EvaluationEngine(max_duration_hours=72.0)
    runner.run_full_suite()


def _wd_supervise():
    env = dict(_wd_os.environ)
    env[_WD_CHILD] = "1"

    while True:
        _wd_atomic_json(_WD_HEARTBEAT, {"active": False})
        child = _wd_subprocess.Popen([
            _wd_sys.executable, "-u", _wd_os.path.abspath(__file__)
        ], env=env)
        stalled = False
        last_status = 0.0
        last_split = None

        try:
            while child.poll() is None:
                _wd_time.sleep(1)
                hb = _wd_read(_WD_HEARTBEAT)
                if not hb.get("active") or int(hb.get("pid", -1)) != child.pid:
                    continue

                now = _wd_time.time()
                started = float(hb.get("started", now))
                age = now - started
                key = str(hb.get("key", "unknown"))
                split_name = str(hb.get("split", key.split(":", 1)[0]))
                item_id = str(hb.get("item_id", key.split(":", 1)[-1]))

                if split_name != last_split:
                    print(f"\n▶ STARTING SPLIT: {split_name}", flush=True)
                    last_split = split_name
                    last_status = 0.0

                if now - last_status >= _WD_STATUS_EVERY:
                    remaining = max(0, _WD_TIMEOUT - age)
                    print(
                        f"\r[working] {split_name} | {item_id} | elapsed {_wd_fmt_elapsed(age)} "
                        f"| stall guard {_wd_fmt_elapsed(remaining)} remaining   ",
                        end="",
                        flush=True,
                    )
                    last_status = now

                if age < _WD_TIMEOUT:
                    continue

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
    # The previous installer put the guard immediately before the final __main__ block,
    # so replace only that generated tail and leave all reconstructed V5 code intact.
    base = src[:src.index(MARK)]
else:
    main = re.search(r'(?m)^if __name__ == ["\']__main__["\']:\s*$', src)
    if not main:
        raise SystemExit("[!] __main__ block not found; source untouched")
    base = src[:main.start()]

new_src = base + PATCH + "\n" + MAIN
compile(new_src, str(P), "exec")
P.write_text(new_src, encoding="utf-8")
print("[✓] Overnight stall guard updated; V5 generator/evaluator/prompts untouched.")
print("[✓] Status heartbeat every 10s + explicit split-start announcements active.")
print("[✓] 20-minute item watchdog, auto-restart, 2-stall skip still active.")
