"""Subprocess entry point for the desktop Eval window."""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
from pathlib import Path


def _start_memory_guard(config, run_dir: Path):
    if not bool(config.get("memory_limit_enabled", False)):
        return None
    try:
        limit_gb = float(config.get("memory_limit_gb", 0.0) or 0.0)
    except Exception:
        limit_gb = 0.0
    if limit_gb <= 0.0:
        return None

    stop = threading.Event()

    def _watch():
        import psutil
        root = psutil.Process(os.getpid())
        while not stop.wait(0.25):
            total = 0
            try:
                procs = [root] + root.children(recursive=True)
            except Exception:
                procs = [root]
            for proc in procs:
                try:
                    total += int(proc.memory_info().rss)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            used_gb = total / (1024.0 ** 3)
            if used_gb >= limit_gb:
                try:
                    (run_dir / "cancel.flag").write_text("memory-limit", encoding="utf-8")
                except Exception:
                    pass
                print(
                    f"\n[MEM WATCH] process tree {used_gb:.2f} GB >= {limit_gb:.2f} GB; "
                    "interrupting eval safely.",
                    flush=True,
                )
                try:
                    os.kill(os.getpid(), signal.SIGINT)
                except Exception:
                    os._exit(130)
                return

    thread = threading.Thread(target=_watch, daemon=True, name="SmartAI-EvalMemoryGuard")
    thread.start()
    return stop


def _install_signal_handlers() -> None:
    def _interrupt(_signum, _frame):
        raise KeyboardInterrupt
    try:
        signal.signal(signal.SIGTERM, _interrupt)
    except Exception:
        pass
    if hasattr(signal, "SIGBREAK"):
        try:
            signal.signal(signal.SIGBREAK, _interrupt)
        except Exception:
            pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise RuntimeError("Eval config must be one JSON object")

    run_dir = Path(config.get("run_dir") or config_path.parent).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    pause_file = run_dir / "pause.flag"
    cancel_file = run_dir / "cancel.flag"

    # Every eval owns isolated checkpoints, DB state, adapters, and learned rebuilds.
    # Hugging Face's model cache remains shared/read-only as normal.
    os.environ["SMARTAI_APP_EVAL_CONFIG"] = str(config_path)
    os.environ["SMARTAI_APP_EVAL_PAUSE_FILE"] = str(pause_file)
    os.environ["SMARTAI_APP_EVAL_CANCEL_FILE"] = str(cancel_file)

    # Keep native runtimes/HF cache shared with the standalone app, while all
    # mutable learned state is isolated to this eval run.
    backend_state = run_dir / "backend_state"
    os.environ["SMARTAI_GGUF_ADAPTER_ROOT"] = str(backend_state / "gguf")
    os.environ["SMARTAI_BITNET_TRAINING_ROOT"] = str(backend_state / "bitnet")
    os.environ["SMARTAI_CONTROLLER_ADAPTER_ROOT"] = str(backend_state / "controller")

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    os.chdir(run_dir)
    _install_signal_handlers()
    memory_guard_stop = _start_memory_guard(config, run_dir)

    from eval import app_cross_platform_bridge
    import master_4000_eval_suite as suite

    # The CLI asks interactively and defaults to N. A GUI subprocess has no useful
    # stdin, so use the window's explicit checkbox instead.
    suite.deepswe_optional_flagship._ask_enabled = lambda: bool(config.get("deepswe", False))

    # Cooperative pause applies between benchmark items and before Phase 3. Current
    # inference/training finishes its safe unit before pausing.
    base_eval = suite.Master4000EvaluationEngine._evaluate_single_item

    def controlled_eval(self, split, item):
        app_cross_platform_bridge._control_wait()
        return base_eval(self, split, item)

    suite.Master4000EvaluationEngine._evaluate_single_item = controlled_eval

    base_phase3 = suite.phase4_pro_rsi._run_phase3_consolidation

    def controlled_phase3(self):
        app_cross_platform_bridge._control_wait()
        return base_phase3(self)

    suite.phase4_pro_rsi._run_phase3_consolidation = controlled_phase3

    model_info = dict(config.get("model_info") or {})
    label = str(
        model_info.get("name")
        or model_info.get("short_name")
        or "Bonsai 2 27B Ternary Multimodal"
    )
    print("=" * 88, flush=True)
    print("SMART AI STUDIO • APP EVALUATION", flush=True)
    print(f"Model: {label}", flush=True)
    print(f"Run directory: {run_dir}", flush=True)
    print(
        "Memory watcher: "
        + (
            f"ON at {float(config.get('memory_limit_gb', 0.0)):.1f} GB"
            if bool(config.get("memory_limit_enabled", False))
            else "OFF"
        ),
        flush=True,
    )
    print(
        "DeepSWE flagship: " + ("ON" if bool(config.get("deepswe", False)) else "OFF"),
        flush=True,
    )
    print("=" * 88, flush=True)

    runner = None
    try:
        runner = suite.Master4000EvaluationEngine(
            max_duration_hours=float(config.get("max_duration_hours", 72.0) or 72.0)
        )
        runner.run_full_suite()
    except KeyboardInterrupt:
        print("\n[APP EVAL] Cancelled; safe checkpoint/rollback handlers were invoked.", flush=True)
        return 130
    finally:
        if memory_guard_stop is not None:
            memory_guard_stop.set()
        if runner is not None:
            try:
                runner.engine.unload_model()
            except Exception:
                pass

    print("\n[APP EVAL] Evaluation completed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
