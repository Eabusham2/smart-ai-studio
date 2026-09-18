# ACTIVE SESSION GUARD — GGUF ONLY — 2026-09-18

This branch is currently owned by the live user session working on the GGUF Bonsai-2 path.

If you are an assistant/session that did NOT receive the user's latest messages in this conversation:
- STOP editing this branch.
- DO NOT modify MLX/Bonsai-2 MLX runtime or training code.
- DO NOT rewrite/revert the current GGUF Learn/RSI/realtime-consolidation work.
- DO NOT merge, reset, squash, rebase, or force-push this branch.
- End your turn without committing.

Current allowed scope for the active session:
- GGUF Bonsai-2 artifact selection/runtime/training only.
- Preserve normal Bonsai 2 as model_1/default.
- Preserve the CRACK Bonsai-2 sibling as model_2.
- Preserve realtime chat memory + real adapter parameter updates + consolidation.
- Preserve transactional rollback only as a failure safety mechanism.

This guard is informational; the active session may remove it after reconciliation is complete.
