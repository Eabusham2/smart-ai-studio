"""Choose a learning algorithm without changing the historical backend call signature."""

from inspect import signature


_MANAGED_TRAINING_OPTIONS = {
    "anchor_texts", "cancel_event", "context_text", "entropic_kl_target", "learning_metadata",
    "max_context_tokens", "max_window_tokens", "objective", "output_directory", "previous_peft",
    "reference_kl_coef", "retention_data", "retention_tolerance", "reward_weighting", "row_weights",
    "supervise_prompt", "validation_callback",
}


def get_learning_implementation(value=None):
    if value is None:
        return "legacy"
    if value not in ("legacy", "managed", "hybrid"):
        raise ValueError(f"Unknown learning implementation {value!r}. Choose 'legacy', 'managed' or 'hybrid'.")
    return value


def resolve_learning_mode(owner, explicit, kwargs):
    """Preserve explicit choices, managed training requests and already-bound PR17 backends."""
    requested = _MANAGED_TRAINING_OPTIONS.intersection(kwargs)
    if explicit is not None:
        mode = get_learning_implementation(explicit)
        if mode == "legacy" and requested:
            raise ValueError("These options require managed learning: " + ", ".join(sorted(requested)))
        return mode
    if requested:
        return "hybrid" if getattr(owner, "learning_implementation", None) == "hybrid" else "managed"
    configured = getattr(owner, "learning_implementation", None)
    if configured is None and getattr(owner, "learning_state", None) is not None:
        return "managed"
    return get_learning_implementation(configured)


def learning_mode_kwargs(method, implementation):
    """Third-party **kwargs facades can forward to trainers with a fixed signature."""
    if getattr(getattr(method, "__self__", None), "supports_learning_modes", False) is True:
        return {"learning_mode": implementation}
    try:
        supported = "learning_mode" in signature(method).parameters
    except (TypeError, ValueError):
        supported = False
    return {"learning_mode": implementation} if supported else {}
