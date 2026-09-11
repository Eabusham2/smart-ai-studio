#!/usr/bin/env python3
from pathlib import Path
import re

P = Path("master_4000_eval_suite.py")
MARK = "# === EYAD OVERNIGHT STALL GUARD ==="

if not P.exists():
    raise SystemExit("[!] master_4000_eval_suite.py not found")

src = P.read_text(encoding="utf-8")
if MARK in src:
    print("[i] Overnight stall guard already installed.")
    raise SystemExit(0)

main = re.search(r'(?m)^if __name__ == ["\']__main__["\']:\s*$', src)
if not main:
    raise SystemExit("[!] __main__ block not found; source untouched")

PATCH = r'''
# === EYAD OVERNIGHT STALL GUARD ===
# Keeps the existing V5 generator/evaluator untouched. Only supervises whole items.
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


_wd_original_eval_single = Master4000EvaluationEngine._evaluate_single_item


def _wd_eval_single(self, split_name, item):
    item_id = str(item.get("id", "unknown"))
    key = f"{split_name}:{item_id}"

    if _wd_count(key) >= _WD_MAX_STALLS:
        print(f"\n[!] Watchdog: skipping repeatedly stalled item {key}; recorded failed.")
        _wd_set_count(key, 0)
        return False

    _wd_atomic_json(_WD_HEARTBEAT, {
        "active": True,
        "pid": _wd_os.getpid(),
        "key": key,
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

        try:
            while child.poll() is None:
                _wd_time.sleep(5)
                hb = _wd_read(_WD_HEARTBEAT)
                if not hb.get("active") or int(hb.get("pid", -1)) != child.pid:
                    continue
                age = _wd_time.time() - float(hb.get("started", _wd_time.time()))
                if age < _WD_TIMEOUT:
                    continue

                key = str(hb.get("key", "unknown"))
                n = _wd_count(key) + 1
                _wd_set_count(key, n)
                print(f"\n[!] Watchdog: {key} stuck {int(age)}s; restarting from checkpoint ({n}/{_WD_MAX_STALLS}).")
                child.kill()
                child.wait()
                stalled = True
                break
        except KeyboardInterrupt:
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

new_src = src[:main.start()] + PATCH + "\n" + MAIN
compile(new_src, str(P), "exec")
P.write_text(new_src, encoding="utf-8")
print("[✓] Overnight stall guard installed; V5 generator/evaluator left untouched.")
print("[✓] 20-minute item watchdog, auto-restart, 2-stall skip active.")
