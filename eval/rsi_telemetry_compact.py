"""Keep full RSI telemetry in JSONL while shortening console heartbeats."""
from __future__ import annotations

import json
import os
import time
from datetime import timedelta
from typing import Any

import psutil


def _fmt_eta(value: Any) -> str:
    if value is None:
        return "calculating"
    try:
        seconds = max(0, int(value))
    except Exception:
        return "calculating"
    return str(timedelta(seconds=seconds))


def _fmt_tps(value: Any) -> str:
    try:
        tps = float(value)
        if tps > 0.0:
            return f"{tps:.1f}t/s"
    except Exception:
        pass
    return "calculating"


def install(stage_module) -> None:
    if getattr(stage_module, "_rsi_compact_console_installed", False):
        return

    original_emit = stage_module._emit

    def emit(stage: str, event: str, **fields: Any) -> None:
        if str(stage) != "RSI":
            return original_emit(stage, event, **fields)

        ram_gb = round(psutil.virtual_memory().used / (1024 ** 3), 3)
        payload = {
            "ts": time.time(),
            "stage": stage,
            "event": event,
            "ram_gb": ram_gb,
            **fields,
        }
        os.makedirs(os.path.dirname(stage_module.STAGE_TELEMETRY_LOG), exist_ok=True)
        with open(stage_module.STAGE_TELEMETRY_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, sort_keys=True, default=str) + "\n")

        tps = _fmt_tps(fields.get("tps", 0.0))
        if event == "start":
            total = int(fields.get("attempt_cap", fields.get("eligible_reasoning_misses", 0)) or 0)
            deferred = int(fields.get("dialogue_deferred", 0) or 0)
            print(f"[RSI] Start | Items: {total} | Deferred memory: {deferred} | TPS: {tps} | RAM: {ram_gb:.1f}GB", flush=True)
            return

        if event == "progress" and int(fields.get("done", 0) or 0) == 0 and fields.get("status") == "working":
            return

        if event in {"heartbeat", "progress"}:
            done = int(fields.get("done", 0) or 0)
            total = max(0, int(fields.get("total", 0) or 0))
            pct = float(fields.get("percent", 0.0) or 0.0)
            verified = int(fields.get("verified", 0) or 0)
            split = str(fields.get("last_split", fields.get("split", "")) or "")
            item = str(fields.get("last_item", fields.get("item", "")) or "")
            current = f"{split}/{item}" if split and item else (split or item or "working")
            eta = _fmt_eta(fields.get("eta_seconds"))
            print(
                f"[RSI] {current:<28} | Item {done}/{total} ({pct:5.2f}%) | "
                f"Verified: {verified} | TPS: {tps} | ETA: {eta} | RAM: {ram_gb:.1f}GB",
                flush=True,
            )
            return

        if event == "end":
            verified = int(fields.get("verified_traces", 0) or 0)
            seconds = float(fields.get("seconds", 0.0) or 0.0)
            print(f"[RSI] Done | Verified: {verified} | TPS: {tps} | Time: {timedelta(seconds=int(seconds))} | RAM: {ram_gb:.1f}GB", flush=True)
            return

        compact = " | ".join(f"{k}={v}" for k, v in fields.items())
        suffix = f" | {compact}" if compact else ""
        print(f"[RSI] {event}{suffix} | TPS: {tps} | RAM: {ram_gb:.1f}GB", flush=True)

    stage_module._emit = emit
    stage_module._rsi_compact_console_installed = True
