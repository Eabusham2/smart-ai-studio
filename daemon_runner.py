#!/usr/bin/env python3
"""
Standalone Sleep Consolidation Daemon Runner.
Continuously monitors system idle periods and triggers Elastic Weight Consolidation (EWC)
cycles to weave daytime episodic memories permanently into Slow-LoRA synaptic parameters.
"""

import argparse
import os
import signal
import sys
import time
from datetime import datetime
from typing import Optional

from config.settings import Settings, get_settings
from consolidation.daemon import SleepConsolidationDaemon
from memory.db import EpisodicMemoryDB


class SleepDaemonRunner:
    def __init__(
        self,
        settings: Optional[Settings] = None,
        poll_interval_seconds: int = 60,
        idle_threshold_minutes: int = 30
    ):
        self.settings = settings or get_settings()
        self.poll_interval = poll_interval_seconds
        self.idle_threshold_seconds = idle_threshold_minutes * 60
        self.db = EpisodicMemoryDB(db_path=self.settings.database_path)
        self.daemon = SleepConsolidationDaemon(
            db_path=self.settings.database_path,
            lora_adapter_path=self.settings.lora_adapter_path,
            use_mock=self.settings.use_mock,
            settings=self.settings
        )
        self.running = True

    def get_last_interaction_time(self) -> Optional[datetime]:
        """Retrieves timestamp of the most recent user interaction."""
        with self.db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT created_at FROM interactions ORDER BY id DESC LIMIT 1")
            row = cursor.fetchone()
            if row and row[0]:
                try:
                    return datetime.fromisoformat(row[0])
                except Exception:
                    pass
        return None

    def run_once(self) -> dict:
        """Executes a single sleep consolidation cycle."""
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Running single consolidation cycle...")
        result = self.daemon.run_consolidation_cycle()
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Consolidation Result: {result}")
        return result

    def start_loop(self):
        """Continuous background idle watcher loop."""
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 🌙 Biological Sleep Daemon started.")
        print(f"  • Database: {self.settings.database_path}")
        print(f"  • Idle Threshold: {self.idle_threshold_seconds // 60} minutes")
        print(f"  • Poll Interval: {self.poll_interval} seconds")
        print(f"  • Mock Mode: {self.settings.use_mock}")

        def handle_signal(sig, frame):
            print("\nShutting down Sleep Daemon...")
            self.running = False

        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)

        while self.running:
            try:
                unconsolidated = self.db.fetch_surprise_replay_data(limit=1, unconsolidated_only=True)
                if unconsolidated:
                    last_time = self.get_last_interaction_time()
                    if last_time:
                        idle_elapsed = (datetime.utcnow() - last_time).total_seconds()
                        if idle_elapsed >= self.idle_threshold_seconds:
                            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] System idle ({idle_elapsed:.0f}s >= {self.idle_threshold_seconds}s). Awakening Sleep Consolidation...")
                            self.daemon.run_consolidation_cycle()
                        else:
                            remaining = self.idle_threshold_seconds - idle_elapsed
                            # Periodic idle heartbeat
                            pass

                time.sleep(self.poll_interval)
            except Exception as e:
                print(f"[Error in Sleep Daemon loop]: {e}")
                time.sleep(self.poll_interval)

        print("Sleep Daemon stopped.")


def main():
    parser = argparse.ArgumentParser(description="Autonomous Sleep Consolidation Daemon Runner")
    parser.add_argument("--once", action="store_true", help="Execute one consolidation cycle and exit")
    parser.add_argument("--interval", type=int, default=60, help="Polling interval in seconds")
    parser.add_argument("--idle-threshold", type=int, default=30, help="Inactivity threshold in minutes")
    parser.add_argument("--mock", action="store_true", help="Run with mock updates for testing")
    parser.add_argument("--backend", type=str, choices=["auto", "torch", "mlx", "mock"], help="Engine backend")
    parser.add_argument("--mlx", action="store_true", help="Force Apple Silicon native MLX backend")
    parser.add_argument("--db-path", type=str, default="memory.db", help="SQLite database path")
    parser.add_argument("--adapter-path", type=str, default="./consolidated_slow_lora", help="Slow-LoRA adapter path")

    args = parser.parse_args()

    overrides = {
        "database_path": args.db_path,
        "lora_adapter_path": args.adapter_path,
        "idle_sleep_threshold_minutes": args.idle_threshold
    }
    if args.mock:
        overrides["use_mock"] = True
    if args.mlx:
        overrides["backend"] = "mlx"
    elif args.backend:
        overrides["backend"] = args.backend

    settings = get_settings(**overrides)
    runner = SleepDaemonRunner(
        settings=settings,
        poll_interval_seconds=args.interval,
        idle_threshold_minutes=args.idle_threshold
    )

    if args.once:
        runner.run_once()
    else:
        runner.start_loop()


if __name__ == "__main__":
    main()
