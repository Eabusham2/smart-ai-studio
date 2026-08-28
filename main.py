#!/usr/bin/env python3
"""
Interactive CLI for the Local 1.58-Bit 27B Autonomous Reasoning System.
Provides test-time Pro reasoning, sandbox verification, episodic SQLite logging,
and manual sleep consolidation triggering.
"""

import argparse
import sys
import time
from typing import Optional

from config.settings import Settings, get_settings
from consolidation.daemon import SleepConsolidationDaemon
from core.pro_engine import ProReasoningEngine
from memory.db import EpisodicMemoryDB

# Rich terminal formatting with plain-text fallback
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.markdown import Markdown
    from rich.text import Text
    console = Console()
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    console = None


def print_banner():
    if HAS_RICH:
        banner = """[bold cyan]🧠 Smart AI Studio — Autonomous Reasoning System[/bold cyan]
[dim]Test-Time Pro Search Engine • RLVR Sandbox • Biological Sleep Consolidation (EWC-LoRA)[/dim]"""
        console.print(Panel(banner, border_style="cyan"))
    else:
        print("=" * 70)
        print(" Smart AI Studio — Autonomous Reasoning System ")
        print(" Test-Time Pro Search • RLVR Sandbox • Biological Sleep (EWC-LoRA) ")
        print("=" * 70)


def print_stats(db: EpisodicMemoryDB):
    stats = db.get_stats()
    if HAS_RICH:
        table = Table(title="Episodic Memory & Consolidation Statistics", border_style="bright_blue")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="bold green")
        table.add_row("Total Interactions", str(stats["total_interactions"]))
        table.add_row("Verified Traces (Reward = 1.0)", str(stats["verified_count"]))
        table.add_row("Unconsolidated Memories", str(stats["unconsolidated_verified"]))
        table.add_row("Sleep Consolidation Cycles", str(stats["consolidation_cycles"]))
        table.add_row("Avg Verified Surprise Score", str(stats["average_verified_surprise"]))
        console.print(table)
    else:
        print("\n--- Episodic Memory Statistics ---")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        print("----------------------------------\n")


def display_reasoning_result(prompt: str, completion: str, meta: dict, db_row_id: int):
    if HAS_RICH:
        # Header info
        mode_color = "green" if meta["verified"] else "yellow"
        info_text = (
            f"[bold]Mode:[/bold] [{mode_color}]{meta['mode']}[/{mode_color}]  |  "
            f"[bold]Entropy H(Y):[/bold] {meta['entropy']:.3f}  |  "
            f"[bold]Branches:[/bold] {meta['branch_count']}  |  "
            f"[bold]Latency:[/bold] {meta['execution_time_ms']:.1f}ms  |  "
            f"[bold]Surprise:[/bold] {meta['surprise_score']:.2f}"
        )
        console.print(Panel(info_text, title="Reasoning Execution Profile", border_style="blue"))

        # Verifier details
        if meta.get("verifier_details"):
            console.print(f"[dim]🔍 Verifier: {meta['verifier_details']}[/dim]")

        # Solution output
        console.print(Panel(Markdown(completion), title="Optimal Solution", border_style="green"))
        console.print(f"[dim green]✓ Trace logged to episodic memory (ID #{db_row_id})[/dim green]\n")
    else:
        print("\n" + "-" * 50)
        print(f"Mode: {meta['mode']} | Entropy: {meta['entropy']} | Branches: {meta['branch_count']} | Latency: {meta['execution_time_ms']:.1f}ms")
        print(f"Surprise Score: {meta['surprise_score']} | Verified: {meta['verified']}")
        if meta.get("verifier_details"):
            print(f"Verifier: {meta['verifier_details']}")
        print("-" * 50)
        print(completion)
        print(f"\n[Trace logged to episodic memory: ID #{db_row_id}]\n")


def process_query(
    engine: ProReasoningEngine,
    db: EpisodicMemoryDB,
    prompt: str,
    test_cases: Optional[str] = None
):
    if HAS_RICH:
        console.print(f"\n[bold yellow]⚡ Routing and exploring trajectories...[/bold yellow]")
    else:
        print("\nRouting and exploring trajectories...")

    completion, meta = engine.solve(prompt=prompt, test_cases=test_cases)

    # Store in episodic memory
    row_id = db.log_interaction(
        prompt=prompt,
        completion=completion,
        raw_branches=meta.get("raw_branches"),
        verified_reward=meta.get("verified_reward", 0.0),
        surprise_score=meta.get("surprise_score", 0.0),
        mode=meta.get("mode", "Instant"),
        entropy=meta.get("entropy", 0.0),
        winning_branch=meta.get("winning_branch", 0),
        test_cases=test_cases
    )

    display_reasoning_result(prompt, completion, meta, row_id)


def main():
    parser = argparse.ArgumentParser(description="Smart AI Studio CLI")
    parser.add_argument("--prompt", type=str, help="Single query prompt to execute")
    parser.add_argument("--tests", type=str, help="Deterministic test assertions for RLVR sandbox")
    parser.add_argument("--small-model", action="store_true", help="Use lightweight local model fallback")
    parser.add_argument("--model-path", type=str, help="Path/ID for base model checkpoint")
    parser.add_argument("--backend", choices=["auto", "mlx", "gguf", "bitnet", "torch", "mock"], default="auto", help="Inference backend (default: auto)")
    parser.add_argument("--mlx", action="store_true", help="Force Apple Silicon native MLX backend")
    parser.add_argument("--live", action="store_true", help="Execute live 27B Ternary AI with 4-bit KV caching for 16GB M1 Mac")
    parser.add_argument("--kv-bits", type=int, default=4, help="KV-cache quantization bitwidth (default: 4)")
    parser.add_argument("--adapter-path", type=str, help="Path to Slow-LoRA adapter")
    parser.add_argument("--auto-download", action="store_true", default=True, help="Auto-download missing model weights")
    parser.add_argument("--no-auto-download", dest="auto_download", action="store_false", help="Disable auto-downloading")
    parser.add_argument("--dry-run-hardware", action="store_true", help="Profile host hardware accelerators and exit")
    parser.add_argument("--ui", action="store_true", help="Launch native Smart AI Desktop Application")
    parser.add_argument("--app", "--gui", dest="app", action="store_true", help="Launch native Desktop Application GUI")
    parser.add_argument("--db-path", "--database-path", "--db", dest="db_path", default="memory.db", help="SQLite database path (default: memory.db)")
    parser.add_argument("--stats", action="store_true", help="Show episodic memory statistics and exit")
    parser.add_argument("--sleep", action="store_true", help="Trigger a sleep consolidation cycle and exit")

    args = parser.parse_args()

    if args.dry_run_hardware:
        from core.hardware import detect_system_hardware, resolve_optimal_backend
        hw = detect_system_hardware()
        print("\n=== 🖥️ Host System Hardware Profile ===")
        for k, v in hw.to_dict().items():
            print(f"  • {k}: {v}")
        print(f"\n  ► Resolved Optimal Backend for Ternary Model: {resolve_optimal_backend('ternary')}")
        print(f"  ► Resolved Optimal Backend for Vision Model:  {resolve_optimal_backend('multimodal_vision')}\n")
        return

    # Settings overrides
    overrides = {"database_path": args.db_path, "auto_download": args.auto_download}
    if args.live:
        overrides["live_mode"] = True
        overrides["base_model_path"] = "prism-ml/Ternary-Bonsai-27B-mlx-2bit"
        overrides["mlx_model_path"] = "prism-ml/Ternary-Bonsai-27B-mlx-2bit"
        overrides["kv_bits"] = args.kv_bits
    if args.mlx:
        overrides["backend"] = "mlx"
    elif args.backend:
        overrides["backend"] = args.backend
    if args.small_model:
        overrides["small_model"] = True
    if args.model_path:
        overrides["base_model_path"] = args.model_path
    if args.adapter_path:
        overrides["lora_adapter_path"] = args.adapter_path

    settings = get_settings(**overrides)
    db = EpisodicMemoryDB(db_path=settings.database_path)

    if args.app:
        from app_gui import launch_app
        launch_app(settings=settings)
        return

    if args.ui:
        from app_gui import launch_app
        launch_app(settings=settings)
        return

    if args.stats:
        print_stats(db)
        return

    if args.sleep:
        if HAS_RICH:
            console.print("[bold blue]🌙 Initiating Biological Sleep Consolidation Cycle...[/bold blue]")
        else:
            print("Initiating Biological Sleep Consolidation Cycle...")

        daemon = SleepConsolidationDaemon(
            db_path=settings.database_path,
            lora_adapter_path=settings.lora_adapter_path,
            use_mock=settings.use_mock,
            settings=settings
        )
        res = daemon.run_consolidation_cycle()
        if HAS_RICH:
            console.print(Panel(str(res), title="Consolidation Cycle Summary", border_style="purple"))
        else:
            print("Consolidation Result:", res)
        return

    print_banner()

    engine = ProReasoningEngine(settings=settings)

    if args.prompt:
        process_query(engine, db, args.prompt, args.tests)
        return

    # Interactive Loop
    if HAS_RICH:
        console.print("[dim]Type your query, or commands: [bold]:stats[/bold], [bold]:sleep[/bold], [bold]:tests <assertions>[/bold], [bold]:exit[/bold][/dim]\n")
    else:
        print("Type your query, or commands: :stats, :sleep, :tests <assertions>, :exit\n")

    current_test_cases = None

    while True:
        try:
            user_input = input("reasoning-engine > ").strip()
            if not user_input:
                continue

            if user_input.lower() in (":exit", ":quit", "exit", "quit"):
                break
            elif user_input == ":stats":
                print_stats(db)
                continue
            elif user_input == ":sleep":
                daemon = SleepConsolidationDaemon(
                    db_path=settings.database_path,
                    lora_adapter_path=settings.lora_adapter_path,
                    use_mock=settings.use_mock,
                    settings=settings
                )
                res = daemon.run_consolidation_cycle()
                if HAS_RICH:
                    console.print(Panel(str(res), title="Consolidation Cycle Summary", border_style="purple"))
                else:
                    print("Consolidation Result:", res)
                continue
            elif user_input.startswith(":tests "):
                current_test_cases = user_input[7:].strip()
                print(f"[Set active test assertions: {current_test_cases}]")
                continue
            elif user_input == ":tests":
                print(f"[Current test assertions: {current_test_cases}]")
                continue
            elif user_input == ":clear_tests":
                current_test_cases = None
                print("[Cleared test assertions]")
                continue

            process_query(engine, db, user_input, current_test_cases)

        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break


if __name__ == "__main__":
    main()
