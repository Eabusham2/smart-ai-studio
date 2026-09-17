"""Standalone addon verification using canonical runtime components.

This file used to duplicate its own lossy H2O token-compaction implementation.
Nothing in the product imports this module, but it remains useful as a manual
integrity check. Reuse the canonical components instead so this diagnostic cannot
reintroduce context dropping or drift away from the live runtime.
"""
from core.h2o_cache import H2OKVCacheArena
from run_studio_complete import (
    GRPOTrainingEngine,
    HierarchicalMoERouter,
    StagedRoundRobinLoRATrainer,
    SymbolicMCTSSearchEngine,
)


def run_addon_subsystem_verification():
    print("\n" + "=" * 90)
    print("🚀 EXECUTING SMART AI STUDIO: UNIFIED ADDON INTEGRITY VERIFICATION")
    print("=" * 90)
    results = []

    h2o = H2OKVCacheArena(sink_size=4, heavy_size=8, max_budget=16)
    for _ in range(5):
        h2o.register_step_attention(
            [1.0 if i in [0, 1, 2, 3, 10, 15, 20] else 0.1 for i in range(32)]
        )
    retained_indices = h2o.compute_compacted_indices(32)
    lossless_h2o = retained_indices == list(range(32))
    results.append(
        (
            "1. Lossless H2O Compatibility Policy",
            lossless_h2o,
            f"Retained {len(retained_indices)}/32 tokens (no context dropped)",
        )
    )

    class MockSandbox:
        @staticmethod
        def evaluate_dsl_expression(expr: str):
            if ">>~fold(1) <#>scale(2)" in expr:
                return [4, 8, 12, 16]
            return [1, 2, 3, 4]

    mcts_engine = SymbolicMCTSSearchEngine(sandbox=MockSandbox(), max_simulations=8)
    best_expr, q_val, visits = mcts_engine.search_best_invariant("[2, 4, 6, 8]")
    results.append(
        (
            "2. Symbolic MCTS Invariant Tree Search",
            visits >= 8 and q_val > 0.0,
            f"Discovered Invariant: {best_expr} (Q={q_val:.2f}, Visits={visits})",
        )
    )

    grpo = GRPOTrainingEngine(model=None, tokenizer=None, sandbox=MockSandbox(), group_size=4)
    advantages = grpo.compute_group_advantages([1.0, 0.0, 1.0, 0.0])
    results.append(
        (
            "3. GRPO Advantage Normalization",
            len(advantages) == 4 and abs(sum(advantages)) < 1e-4,
            f"Normalized Group Advantages: {advantages}",
        )
    )

    class MockLayer:
        def __init__(self):
            self.frozen = False

        def freeze(self):
            self.frozen = True

        def unfreeze(self):
            self.frozen = False

    class MockModel:
        def __init__(self):
            self.layers = [MockLayer() for _ in range(60)]

    rr_trainer = StagedRoundRobinLoRATrainer(model=MockModel(), total_layers=60, chunk_size=6)
    rr_trainer.unfreeze_active_chunk(1)
    chunk1_active = all(not rr_trainer.model.layers[i].frozen for i in range(6, 12))
    chunk0_frozen = all(rr_trainer.model.layers[i].frozen for i in range(0, 6))
    results.append(
        (
            "4. Interleaved 6-Layer Plasticity Chunking",
            chunk1_active and chunk0_frozen,
            "Verified 6/60 layers unfreeze in staged blocks.",
        )
    )

    moe_router = HierarchicalMoERouter(model=None)
    math_exp = moe_router.route_prompt("Solve AIME competition math equation with \\boxed{} answer")
    code_exp = moe_router.route_prompt("Write a python class to apply a git diff patch in sandbox")
    lore_exp = moe_router.route_prompt("What is the official currency of the Balehan empire?")
    results.append(
        (
            "5. Sparse MoE-LoRA Cosine Router",
            math_exp == "math" and code_exp == "code" and lore_exp == "lore",
            f"Routed: Math->{math_exp}, Code->{code_exp}, Lore->{lore_exp}",
        )
    )

    print("=" * 90)
    print(f"{'STATUS':<10} | {'SUBSYSTEM / INVARIANT':<45} | {'DETAILS'}")
    print("=" * 90)
    for name, passed, detail in results:
        status_str = "\033[92m[✓ PASS]\033[0m" if passed else "\033[91m[✗ FAIL]\033[0m"
        print(f"{status_str:<19} | {name:<45} | {detail}")
    print("=" * 90)
    total_passed = sum(1 for _, passed, _ in results if passed)
    print(
        f"📊 ADDON SUBSYSTEM SCORE: {total_passed}/{len(results)} COMPLETED "
        f"({(total_passed / len(results)) * 100:.1f}%)\n"
    )


if __name__ == "__main__":
    run_addon_subsystem_verification()
