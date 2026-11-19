"""Bounded exact-token memoization and isolated MLX prefill snapshots.

No model is imported or loaded here. Cache support is deliberately an allowlist:
native KV/quantized KV and whole ArraysCache recurrent state. Unknown, rotating,
TurboQuant and externally mutated caches retain the normal full-prefill path.
Snapshots never contain generated tokens and are never trimmed. Reuse requires
the same chunk schedule, so recurrent/quantized prefill boundaries are preserved.
"""
from __future__ import annotations

import copy
import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass


def tokenizer_signature(tokenizer):
    """Detect replacement and public encoding configuration/vocabulary changes.

    Integrations which mutate hidden tokenizer internals must call the backend's
    invalidate_prompt_reuse hook (or increment tokenizer._smartai_revision).
    """
    attrs = ("bos_token", "eos_token", "bos_token_id", "eos_token_id",
             "pad_token_id", "add_bos_token", "add_eos_token", "chat_template",
             "special_tokens_map", "init_kwargs", "_smartai_revision")
    config = {name: getattr(tokenizer, name, None) for name in attrs}
    added = getattr(tokenizer, "get_added_vocab", None)
    if callable(added):
        config["added_vocab"] = added()
    encode = getattr(tokenizer, "encode", None)
    config["encode"] = id(getattr(encode, "__func__", encode))
    payload = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return (id(tokenizer), id(getattr(tokenizer, "_tokenizer", None)),
            hashlib.sha256(payload).hexdigest())


class TokenizerMemo:
    """LRU bounded by entries, token count AND retained Python/text bytes."""
    def __init__(self, max_entries=128, max_tokens=65536, max_bytes=8 * 1024 ** 2):
        self.max_entries = max(0, int(max_entries))
        self.max_tokens = max(0, int(max_tokens))
        self.max_bytes = max(0, int(max_bytes))
        self.entries = OrderedDict()
        self.signature = None
        self.tokens = self.nbytes = self.hits = self.misses = 0

    def clear(self):
        self.entries.clear()
        self.tokens = self.nbytes = 0
        self.signature = None

    def encode(self, tokenizer, prompt, *, enabled=True):
        signature = tokenizer_signature(tokenizer)
        if signature != self.signature:
            self.clear()
            self.signature = signature
        if enabled and prompt in self.entries:
            tokens, _ = self.entries[prompt]
            self.entries.move_to_end(prompt)
            self.hits += 1
            return list(tokens)
        self.misses += 1
        # The original encode defaults are preserved; no BOS/template rewriting.
        result = tokenizer.encode(prompt)
        tokens = tuple(int(token) for token in result)
        size = len(prompt.encode("utf-8")) + len(prompt) * 4 + len(tokens) * 36 + 128
        if (enabled and self.max_entries and len(tokens) <= self.max_tokens
                and size <= self.max_bytes):
            while self.entries and (len(self.entries) >= self.max_entries
                    or self.tokens + len(tokens) > self.max_tokens
                    or self.nbytes + size > self.max_bytes):
                _, (old, old_size) = self.entries.popitem(last=False)
                self.tokens -= len(old)
                self.nbytes -= old_size
            self.entries[prompt] = (tokens, size)
            self.tokens += len(tokens)
            self.nbytes += size
        return list(tokens)


def _cache_contracts():
    try:
        from mlx_lm.models import cache as native
    except ImportError:
        return {}
    specs = {
        "KVCache": {"keys", "values", "offset", "step"},
        "QuantizedKVCache": {"keys", "values", "offset", "step", "group_size", "bits"},
        "ArraysCache": {"cache", "left_padding", "lengths"},
    }
    return {getattr(native, name): (name, attrs) for name, attrs in specs.items()
            if isinstance(getattr(native, name, None), type)}


def _tree_arrays(value):
    if value is None or type(value) in (str, int, float, bool):
        return []
    if type(value) in (list, tuple):
        return [a for item in value for a in _tree_arrays(item)]
    if type(value) is dict and all(type(key) is str for key in value):
        return [a for item in value.values() for a in _tree_arrays(item)]
    import mlx.core as mx
    if type(value) is mx.array:
        return [value]
    raise ValueError("unsupported cache state object")


def cache_nbytes(cache, token_count):
    """Validate complete state, exact offsets and known unbatched cache classes."""
    if type(cache) is not list or not cache or type(token_count) is not int or token_count < 0:
        raise ValueError("invalid prompt cache")
    contracts = _cache_contracts()
    total = 0
    for layer in cache:
        contract = contracts.get(type(layer))
        if contract is None:
            raise ValueError("cache class has no verified snapshot contract")
        name, attrs = contract
        state = vars(layer)
        if not set(state).issubset(attrs):
            raise ValueError("unknown cache metadata")
        if name != "ArraysCache":
            if type(getattr(layer, "offset", None)) is not int or layer.offset != token_count:
                raise ValueError("cache offset does not match exact token prefix")
            keys, values = getattr(layer, "keys", None), getattr(layer, "values", None)
            if token_count and (keys is None or values is None):
                raise ValueError("cache tensors missing")
            for array in _tree_arrays((keys, values)):
                if len(array.shape) != 4 or array.shape[0] != 1 or array.shape[-2] < token_count:
                    raise ValueError("cache tensor shape does not match prefix")
        else:
            if not isinstance(getattr(layer, "cache", None), list):
                raise ValueError("invalid recurrent cache")
            if getattr(layer, "left_padding", None) is not None or getattr(layer, "lengths", None) is not None:
                raise ValueError("batched recurrent cache is unsupported")
            if token_count and not any(x is not None for x in layer.cache):
                raise ValueError("missing recurrent state")
        arrays = _tree_arrays(state)
        for array in arrays:
            if array.shape and array.shape[0] != 1:
                raise ValueError("batched cache is unsupported")
            size = int(array.nbytes)
            if size < 0:
                raise ValueError("invalid array size")
            total += size
    return total


def cache_policy_signature(cache):
    """Precision/layout options are identity, not a speed-tuning parameter."""
    return tuple((type(layer), getattr(layer, "bits", None),
                  getattr(layer, "group_size", None), getattr(layer, "step", None),
                  len(layer.cache) if type(layer).__name__ == "ArraysCache" else None)
                 for layer in cache)


def clone_cache(cache, token_count):
    """Clone every array/container; reject deceptive or incomplete deepcopy hooks."""
    expected = cache_nbytes(cache, token_count)
    cloned = copy.deepcopy(cache)
    if cloned is cache or len(cloned) != len(cache):
        raise ValueError("prompt cache did not clone")
    if cache_nbytes(cloned, token_count) != expected:
        raise ValueError("cloned prompt cache changed size")
    for before, after in zip(cache, cloned):
        if before is after or type(before) is not type(after):
            raise ValueError("cache layer was aliased")
        def check_tree(left, right):
            if type(left) is not type(right):
                raise ValueError("cache metadata type changed")
            if type(left) is dict:
                if left is right or left.keys() != right.keys():
                    raise ValueError("cache metadata aliased or incomplete")
                for key in left:
                    check_tree(left[key], right[key])
            elif type(left) in (list, tuple):
                if len(left) != len(right) or (type(left) is list and left is right):
                    raise ValueError("cache container aliased or incomplete")
                for a, b in zip(left, right):
                    check_tree(a, b)
            elif left is None or type(left) in (str, int, float, bool):
                if left != right:
                    raise ValueError("cache metadata changed")
        check_tree(vars(before), vars(after))
        left, right = _tree_arrays(vars(before)), _tree_arrays(vars(after))
        if len(left) != len(right):
            raise ValueError("cloned arrays missing")
        for a, b in zip(left, right):
            if a is b or a.shape != b.shape or a.dtype != b.dtype:
                raise ValueError("cache array was aliased or changed")
    return cloned


@dataclass(frozen=True)
class PromptSnapshot:
    tokens: tuple
    cache: list
    nbytes: int
    prefill_step_size: int


class ExactPrefixStore:
    def __init__(self, max_entries=2, max_tokens=8192, max_bytes=256 * 1024 ** 2,
                 memory_admission=None):
        self.max_entries = max(0, int(max_entries))
        self.max_tokens = max(0, int(max_tokens))
        self.max_bytes = max(0, int(max_bytes))
        self.entries = OrderedDict()
        self.nbytes = self.hits = self.misses = 0
        self.memory_admission = memory_admission

    def clear(self):
        self.entries.clear()
        self.nbytes = 0

    def lookup(self, tokens, prefill_step_size):
        tokens = tuple(tokens)
        if type(prefill_step_size) is not int or prefill_step_size <= 0:
            self.misses += 1
            return None
        for key, snapshot in sorted(list(self.entries.items()), key=lambda item: len(item[1].tokens), reverse=True):
            n = len(snapshot.tokens)
            # Keep one token for next-token logits, and preserve original chunks.
            if (snapshot.prefill_step_size != prefill_step_size or not 0 < n < len(tokens)
                    or tokens[:n] != snapshot.tokens
                    or (n != len(tokens) - 1 and n % prefill_step_size)):
                continue
            if self.memory_admission is not None and not self.memory_admission(snapshot.nbytes):
                self.clear()
                self.misses += 1
                return None
            try:
                cloned = clone_cache(snapshot.cache, n)
            except Exception:
                self.nbytes -= snapshot.nbytes
                del self.entries[key]
                continue
            self.entries.move_to_end(key)
            self.hits += 1
            return cloned, n
        self.misses += 1
        return None

    def remember(self, tokens, cache, prefill_step_size):
        tokens = tuple(tokens)
        if (not tokens or not self.max_entries or len(tokens) > self.max_tokens
                or any(type(token) is not int or token < 0 for token in tokens)
                or type(prefill_step_size) is not int or prefill_step_size <= 0):
            return False
        try:
            size = cache_nbytes(cache, len(tokens)) + len(tokens) * 36
            if size > self.max_bytes:
                return False
        except Exception:
            return False
        key = (tokens, int(prefill_step_size))
        old = self.entries.pop(key, None)
        if old:
            self.nbytes -= old.nbytes
        old = None
        while self.entries and (len(self.entries) >= self.max_entries or self.nbytes + size > self.max_bytes):
            _, old = self.entries.popitem(last=False)
            self.nbytes -= old.nbytes
            old = None
        # Retire replaced snapshots before allocating the next copy. Retention
        # limits alone do not bound peak memory while both copies are alive.
        if self.memory_admission is not None and not self.memory_admission(size):
            self.clear()
            return False
        try:
            cloned = clone_cache(cache, len(tokens))
        except Exception:
            return False
        self.entries[key] = PromptSnapshot(tokens, cloned, size, int(prefill_step_size))
        self.nbytes += size
        return True
