"""Serve the already loaded model without entering desktop chat or tools.

The API owns a request, not a second model. Client histories are formatted with
the selected model's chat template and never enter the Studio conversation DB.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import inspect
import json
import math
from pathlib import Path
import threading


class FormattedPrompt(str):
    """An already-templated request; VLM paths must not wrap it as one user turn."""
    _smartai_formatted_chat = True

    def __new__(cls, value, max_tokens=None, *, api_request=True):
        result = super().__new__(cls, value)
        result._smartai_api_max_tokens = max_tokens
        result._smartai_api_request = bool(api_request)
        return result


class LoadedModelAdapter:
    def __init__(self, engine, *, selected_model=None, busy=None, pro_defaults=None):
        self.engine = engine
        self.selected_model = selected_model or (lambda: {})
        self.busy = busy or (lambda: "")
        self._request_lock = threading.Lock()
        self._last_learning = {"status": "not_run"}
        self.pro_defaults = dict(pro_defaults or {})
        self._last_pro = {"status": "not_run"}

    def model_info(self):
        if not getattr(self.engine, "is_model_loaded", False):
            return None
        info = self.selected_model() or {}
        if info.get("model_type", "text") != "text":
            return None
        return {
            "id": "smart-ai-studio",
            "name": getattr(self.engine, "active_model_name", None) or info.get("name", "Loaded model"),
            "backend": getattr(self.engine, "active_backend", None),
            "capabilities": self._capabilities(),
        }

    def _learning_mode(self):
        service = getattr(self.engine, "awake_consolidator", None)
        return getattr(service, "learning_implementation",
                       getattr(getattr(self.engine, "settings", None), "learning_implementation", "off"))

    def _learning_ready(self, backend):
        service = getattr(self.engine, "awake_consolidator", None)
        check = getattr(service, "_real_training_ready", None)
        try:
            return bool(getattr(service, "engine", None) is backend and callable(check) and check())
        except Exception:
            return False

    def _sampling_defaults(self, backend, supported, thinking=True):
        """Prefer the resident model's published sampling policy, not desktop Pro temperatures."""
        info = self.selected_model() or {}
        config = getattr(getattr(backend, "model", None), "generation_config", None)
        raw = {}
        if config is not None:
            try:
                raw.update(config if isinstance(config, dict) else config.to_dict())
            except (AttributeError, TypeError, ValueError):
                pass
        # MLX packs commonly ship a separate local generation_config.json.
        for source in (getattr(backend, "_mtp_source", None), getattr(backend, "_mlx_loaded_source", None),
                       getattr(backend, "model_path", None), getattr(self.engine, "active_model_path", None)):
            if not isinstance(source, (str, Path)):
                continue
            try:
                candidate = Path(source).expanduser() / "generation_config.json"
                if candidate.is_file():
                    with candidate.open("r", encoding="utf-8") as handle:
                        value = json.load(handle)
                    if isinstance(value, dict):
                        raw.update(value)
                    break
            except (OSError, ValueError, TypeError):
                continue
        defaults = {"temperature": 0.65, "top_p": 0.92}
        raw.update(getattr(backend, "sampling_defaults", None) or {})
        for source, key, low, high in (
            ("temperature", "temperature", 0, 2), ("top_p", "top_p", 0, 1),
            ("top_k", "top_k", 0, 200000), ("min_p", "min_p", 0, 1),
            ("typical_p", "typical_p", 0, 1),
            ("repetition_penalty", "repeat_penalty", .01, 10),
            ("repeat_penalty", "repeat_penalty", .01, 10),
            ("presence_penalty", "presence_penalty", -2, 2),
            ("frequency_penalty", "frequency_penalty", -2, 2),
        ):
            value = raw.get(source)
            if key in {"temperature", "top_p"} or key in supported:
                if type(value) in (int, float) and math.isfinite(value) and low <= value <= high:
                    if key != "top_k" or type(value) is int:
                        defaults[key] = value
        # The two built-in Bonsai 2.5 presets share the publisher's Qwen3.8
        # sampling recommendations; these override generic llama.cpp defaults.
        if info.get("bonsai25_mtp"):
            defaults.update(temperature=1.0 if thinking else .7, top_p=.95 if thinking else .8)
            defaults.update({k: v for k, v in {
                "top_k": 20, "min_p": 0.0, "repeat_penalty": 1.0,
                "presence_penalty": 0.0 if thinking else 1.5, "frequency_penalty": 0.0,
            }.items() if k in supported})
        return defaults

    def _learn_completed_reply(self, backend, request, raw, starts_thinking, cancel_event):
        """Learn one new ordinary exchange without archiving or projecting client history."""
        mode = self._learning_mode()
        if request.learning_mode == "off" or mode == "off":
            self._last_learning = {"status": "disabled", "mode": mode}
            return
        if not self._learning_ready(backend):
            self._last_learning = {"status": "unavailable", "mode": mode}
            return
        if cancel_event.is_set():
            return
        # Tools and control messages belong to the harness. Never train on
        # incomplete tool calls, reasoning text or an unpaired tool result.
        from core.gui_thinking import reasoning_segments
        from core.context_memory import working_history, completed_groups
        text = "".join(value for thought, value in reasoning_segments(raw, starts_in_thought=starts_thinking)
                       if not thought).strip()
        if (not text or "<tool_call>" in text.casefold() or "<media_call>" in text.casefold()
                or request.tool_choice == "required" or isinstance(request.tool_choice, dict) or not request.messages):
            self._last_learning = {"status": "skipped", "mode": mode}
            return
        last = request.messages[-1]
        if last.get("role") != "user" or not str(last.get("content", "")).strip():
            self._last_learning = {"status": "skipped", "mode": mode}
            return
        turns = [{"role": "user", "content": last["content"]},
                 {"role": "assistant", "content": text}]
        if not completed_groups(working_history(turns)):
            self._last_learning = {"status": "skipped", "mode": mode}
            return
        service = self.engine.awake_consolidator
        try:
            if mode == "legacy":
                # The original consolidator marks its own desktop traces when
                # a DB is attached. Use the same algorithm without that DB.
                previous_db = service.db
                try:
                    service.db = None
                    accepted = service.consolidate_chunk_sync(turns, cancel_event=cancel_event)
                finally:
                    service.db = previous_db
                self._last_learning = {"status": "updated" if accepted else "not_committed", "mode": mode}
            else:
                result = service.train_pairs([{"prompt": last["content"], "completion": text}],
                                             mode="awake", cancel_event=cancel_event)
                drift = result.get("parameter_drift_l2")
                if result.get("status") != "success" or type(drift) not in (int, float) or not math.isfinite(drift) or drift <= 0:
                    self._last_learning = {"status": "not_committed", "mode": mode}
                    return
                self._last_learning = {"status": "updated", "mode": mode,
                                       "parameter_drift_l2": drift,
                                       "trainable_parameters_touched": result.get("trainable_parameters_touched")}
        except Exception:
            # Inference already completed. Do not turn a rejected/failed
            # training candidate into a fabricated success or lost answer.
            self._last_learning = {"status": "not_committed", "mode": mode}

    def _runtime_options(self, backend):
        """Only advertise controls explicitly implemented by the selected runtime."""
        native = getattr(backend, "runtime", None) == "prism_speculative"
        try:
            params = inspect.signature(backend.stream_generate_tokens).parameters
        except (TypeError, ValueError, AttributeError):
            params = {}
        mlx = "api_sampling" in params and "api_speedups" in params
        sampling = set()
        if native:
            sampling = set(getattr(backend, "sampling_defaults", {})) & {
                "top_k", "min_p", "typical_p", "repeat_penalty", "repeat_last_n",
                "frequency_penalty", "presence_penalty"}
            sampling.add("seed")
        elif mlx:
            sampling = {"top_k", "min_p", "repeat_penalty", "repeat_last_n",
                        "frequency_penalty", "presence_penalty"}
        info = self.selected_model() or {}
        tokenizer = getattr(backend, "tokenizer", None)
        processor = getattr(backend, "processor", None)
        template = getattr(tokenizer, "chat_template", None) or getattr(processor, "chat_template", None)
        thinking = bool(info.get("bonsai25_mtp") or
                        (isinstance(template, str) and "enable_thinking" in template))
        speedups = {"enabled", "prompt_reuse", "tokenizer_memo", "metal_warmup"} if mlx else set()
        if mlx and getattr(backend, "_vlm_runtime", None) is not None:
            speedups.add("speculative_mode")
            if getattr(getattr(backend, "_mtp", None), "head", None) is not None:
                speedups.add("max_draft_tokens")
        return native, mlx, sampling, thinking, speedups

    def _capabilities(self):
        try:
            backend = self._backend()
        except ValueError:
            return {}
        native, mlx, sampling, thinking, speedups = self._runtime_options(backend)
        configured = getattr(self.engine, "lossless_speed_options", None) or {}
        actual = getattr(backend, "_lossless_speedups", None) or {}
        active = {key: bool(actual.get(key, configured.get(key, False))) if mlx else False
                  for key in ("prompt_reuse", "tokenizer_memo", "metal_warmup")}
        active["persistent_cache"] = False
        active["branch_prefill"] = False
        if mlx:
            mtp = getattr(backend, "_mtp", None)
            active["speculative_mode"] = "mtp" if getattr(mtp, "head", None) is not None else "none"
            if active["speculative_mode"] == "mtp":
                active["max_draft_tokens"] = mtp.max_draft_tokens
                active["prompt_reuse"] = False  # MTP owns the prefill state.
        elif native:
            plan = getattr(backend, "_plan", None)
            active["speculative_mode"] = getattr(plan, "active_mode", "none")
            active["max_draft_tokens"] = getattr(plan, "max_draft_tokens", None)
        defaults = self._sampling_defaults(backend, sampling)
        return {
            "stateless": True, "client_owned_history": True, "client_owned_tools": True,
            "context_limit": self._context_limit(backend),
            "sampling_options": ["temperature", "top_p", *sorted(sampling)],
            "sampling_defaults": defaults,
            "sampling_defaults_non_thinking": self._sampling_defaults(backend, sampling, thinking=False),
            "learning_modes": ["legacy", "managed", "hybrid", "off"],
            "learning_mode_default": self._learning_mode(),
            "learning_ready": self._learning_ready(backend),
            "learning_last_result": dict(self._last_learning),
            "learning_mode_switch_requires_reload": True,
            "pro": {"supported": True, "defaults": self._pro_config({}),
                    "routing": ["auto", "fixed"], "max_branches": 64,
                    "max_best_work_passes": 999, "last_result": dict(self._last_pro)},
            "enable_thinking_supported": thinking,
            "include_reasoning_protocols": ["chat"],
            "request_speedups": sorted(speedups),
            "active_speedups": active,
            "load_time_only": ["model", "context_capacity", "device", "offload", "kv_cache_precision",
                               "runtime", "memory_limit", "native_gguf_speculation"],
            "disabled": ["desktop_history", "desktop_tools", "desktop_trace_logging",
                         "persistent_api_prompt_cache", "server_side_conversation"],
        }

    def _pro_config(self, overrides):
        settings = getattr(self.engine, "settings", None)
        passes = getattr(settings, "best_work_passes", 2)
        if type(passes) is not int or passes < 1:
            passes = 2
        count = (getattr(self.engine, "lossless_speed_options", None) or {}).get("native_branch_count", 1)
        if type(count) is not int or not 1 <= count <= 64:
            count = 1
        return {"enabled": False, "routing": "auto", "branch_count": count,
                "best_work_passes": passes, **self.pro_defaults, **overrides}

    def _decode_stream(self, backend, options, cancel_event):
        if backend is self.engine:
            return self._torch_stream(backend, options, cancel_event)
        options = dict(options)
        if "cancelled" in inspect.signature(backend.stream_generate_tokens).parameters:
            options["cancelled"] = cancel_event.is_set
        return backend.stream_generate_tokens(**options)

    def _validate_execution(self, backend, request):
        from core.local_api import APIError
        native, mlx, sampling, thinking, speedups = self._runtime_options(backend)
        unsupported = set(request.sampling) - sampling
        # A standard client may transmit zero-valued neutral penalties.
        unsupported -= {key for key in ("frequency_penalty", "presence_penalty")
                        if request.sampling.get(key) == 0}
        if unsupported:
            raise APIError(400, "Sampling option unavailable for the loaded runtime: " + ", ".join(sorted(unsupported)),
                           code="unsupported_option")
        if request.enable_thinking is not None and not thinking:
            raise APIError(400, "The loaded chat template does not expose enable_thinking.", code="unsupported_option")
        unsupported = set(request.speedups) - speedups
        if unsupported:
            raise APIError(400, "Speedup requires a compatible loaded runtime or reload: " + ", ".join(sorted(unsupported)),
                           code="unsupported_option")
        head = getattr(getattr(backend, "_mtp", None), "head", None)
        if (request.speedups.get("speculative_mode") == "mtp" or "max_draft_tokens" in request.speedups) and head is None:
            raise APIError(400, "Load the verified MTP head in Studio before requesting MTP.", code="unsupported_option")
        return native, mlx

    @contextmanager
    def _lease(self, request=None, cancel_event=None):
        from core.local_api import APIError
        reason = self.busy()
        if reason:
            raise APIError(409, str(reason), code="model_busy")
        if not self._request_lock.acquire(blocking=False):
            raise APIError(409, "The model is serving another request.", code="model_busy")
        model_lock = getattr(self.engine, "_model_lock", None)
        acquired = False
        previous_mode, previous_scope, switched = None, None, False
        try:
            if model_lock is not None:
                acquired = model_lock.acquire(blocking=False)
                if not acquired:
                    raise APIError(409, "A model operation is in progress.", code="model_busy")
            if not self.model_info():
                raise APIError(503, "Load a text model in Smart AI Studio first.", code="model_not_loaded")
            previous_mode = self._learning_mode()
            requested = getattr(request, "learning_mode", None)
            if requested not in (None, "off", previous_mode):
                switch = getattr(self.engine, "set_learning_implementation", None)
                if previous_mode not in {"legacy", "managed", "hybrid"} or not callable(switch):
                    raise APIError(400, "This model cannot switch learning implementations.", code="unsupported_option")
                previous_scope = getattr(self.engine, "learning_scope", None)
                if getattr(previous_scope, "pin", None) or getattr(previous_scope, "kind", "model") not in {"model", "session", "project"}:
                    raise APIError(400, "A pinned or document learning scope cannot be switched by an API request.", code="unsupported_option")
                # The real implementation switch reloads its own checkpoint and
                # rolls back on failure. The production model lock is reentrant.
                switch(requested, cancel_event=cancel_event)
                switched = True
            backend = self._backend()
            if requested not in (None, "off") and not self._learning_ready(backend):
                raise APIError(400, "The requested learning implementation has no verified trainer on this model.", code="unsupported_option")
            prepare = getattr(backend, "prepare_request", None)
            if callable(prepare):
                prepare()
            yield backend
        finally:
            try:
                if switched:
                    # Restore the app selection, not the just-trained checkpoint
                    # of a different implementation. It stays saved in its own store.
                    self.engine.set_learning_implementation(previous_mode, preferred_scope=previous_scope)
            finally:
                if acquired:
                    model_lock.release()
                self._request_lock.release()

    def _backend(self):
        from core.local_api import APIError
        backend = getattr(self.engine, {
            "mlx": "mlx_backend", "gguf": "gguf_backend", "bitnet": "bitnet_backend",
            "controller": "controller_backend",
        }.get(getattr(self.engine, "active_backend", None), "_missing_backend"), None)
        if backend is None and getattr(self.engine, "model", None) is not None:
            return self.engine
        if backend is None or not callable(getattr(backend, "stream_generate_tokens", None)):
            raise APIError(503, "The loaded model has no live streaming backend.", code="backend_unavailable")
        return backend

    @staticmethod
    def _messages(request):
        messages = deepcopy(request.messages)
        system_parts = []
        conversation = []
        for message in messages:
            # Qwen templates expect structured function arguments, although
            # OpenAI sends them on the wire as a JSON-encoded string.
            for call in message.get("tool_calls", []):
                function = call.get("function", {})
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    try:
                        function["arguments"] = json.loads(arguments)
                    except ValueError:
                        pass
            if message.get("role") in {"system", "developer"}:
                if message.get("content"):
                    system_parts.append(message["content"])
            else:
                conversation.append(message)
        choice = request.tool_choice
        if choice == "required":
            instruction = "Respond with at least one call to the supplied tools."
        elif isinstance(choice, dict):
            name = (choice.get("function") or {}).get("name") or choice.get("name")
            instruction = f"Respond by calling the supplied tool {json.dumps(name)}." if name else ""
        else:
            instruction = ""
        if instruction:
            system_parts.append(instruction)
        # Bonsai's official template accepts exactly one leading system turn.
        # Harnesses may send several system/developer messages, including
        # updated developer instructions later in their complete history.
        if system_parts:
            conversation.insert(0, {"role": "system", "content": "\n\n".join(system_parts)})
        return conversation

    def _format_prompt(self, backend, request):
        from core.local_api import APIError
        messages = self._messages(request)
        tools = [] if request.tool_choice == "none" else deepcopy(request.tools)
        for tool in tools:
            for key in ("original_type", "format", "namespace", "original_name"):
                tool.pop(key, None)
        tokenizer = getattr(backend, "tokenizer", None)
        processor = getattr(backend, "processor", None)
        candidates = [tokenizer, getattr(processor, "tokenizer", None), processor]
        template = next((getattr(obj, "apply_chat_template", None) for obj in candidates
                         if callable(getattr(obj, "apply_chat_template", None))), None)
        if callable(template):
            options = {"tokenize": False, "add_generation_prompt": True}
            if tools:
                options["tools"] = tools
            if request.enable_thinking is not None:
                options["enable_thinking"] = request.enable_thinking
            try:
                prompt = template(messages, **options)
            except Exception as error:
                raise APIError(400, f"The selected model's chat template rejected this request: {error}",
                               code="chat_template_error") from error
        elif callable(getattr(backend, "_request", None)):
            # llama-server owns the GGUF template, including its native tool
            # syntax. Never substitute a guessed ChatML conversation.
            payload = {"messages": messages, "add_generation_prompt": True}
            if tools:
                payload["tools"] = tools
            if request.enable_thinking is not None:
                payload["chat_template_kwargs"] = {"enable_thinking": request.enable_thinking}
            try:
                result = backend._request("/apply-template", payload)
                prompt = result.get("prompt")
            except Exception as error:
                raise APIError(501, "The selected native runtime cannot format this chat request: " + str(error),
                               code="chat_template_unavailable") from error
        else:
            raise APIError(501, "This backend does not expose its chat template. Select a Bonsai 2.5 MLX or MTP GGUF model.",
                           code="chat_template_unavailable")
        if not isinstance(prompt, str) or not prompt:
            raise APIError(400, "The selected model produced an empty chat template.", code="chat_template_error")
        return FormattedPrompt(prompt, request.max_tokens)

    @staticmethod
    def _prompt_tokens(backend, prompt, speedups=None):
        count_native = getattr(backend, "api_prompt_count", None)
        if callable(count_native):
            return count_native(prompt, speedups or {})
        encode = getattr(getattr(backend, "tokenizer", None), "encode", None)
        if not callable(encode):
            encode = getattr(getattr(getattr(backend, "processor", None), "tokenizer", None), "encode", None)
        if callable(encode):
            ids = encode(prompt)
            if isinstance(ids, dict):
                ids = ids.get("input_ids")
            if ids is not None:
                return len(ids)
        tokenize = getattr(getattr(backend, "model", None), "tokenize", None)
        if callable(tokenize):
            return len(tokenize(prompt.encode("utf-8"), add_bos=True, special=True))
        return None

    def _context_limit(self, backend):
        values = []
        info = self.selected_model() or {}
        for value in (getattr(backend, "context_budget_tokens", None), getattr(backend, "n_ctx", None),
                      getattr(self.engine, "_context_budget_tokens", None), info.get("max_context")):
            if type(value) is int and 0 < value <= 10_000_000:
                values.append(value)
        for obj in (getattr(backend, "config", None), getattr(getattr(backend, "model", None), "config", None),
                    getattr(getattr(backend, "model", None), "args", None), getattr(backend, "tokenizer", None)):
            for name in ("max_position_embeddings", "max_seq_len", "max_sequence_length", "context_length", "model_max_length"):
                value = obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)
                if type(value) is int and 0 < value <= 10_000_000:
                    values.append(value)
        return min(values) if values else None

    def count_tokens(self, request):
        from core.local_api import APIError
        with self._lease() as backend:
            self._validate_execution(backend, request)
            count = self._prompt_tokens(backend, self._format_prompt(backend, request), request.speedups)
            if count is None:
                raise APIError(501, "The loaded runtime does not expose an exact text tokenizer.")
            return count

    def generate(self, request, cancel_event):
        from core.generation_metrics import RequestMetrics, observe_generations, valid_count
        from core.local_api import APIError, GenerationChunk
        with self._lease(request, cancel_event) as backend:
            if cancel_event.is_set():
                return
            native, mlx = self._validate_execution(backend, request)
            prompt = self._format_prompt(backend, request)
            prompt_count = self._prompt_tokens(backend, prompt, request.speedups)
            context = self._context_limit(backend)
            if context is not None and (type(prompt_count) is not int or prompt_count < 0):
                raise APIError(501, "Cannot verify the native input token count against the loaded Context.",
                               code="tokenizer_unavailable")
            if request.context_limit is not None:
                if context is None or request.context_limit > context:
                    raise APIError(400, "The requested context_limit exceeds the verified loaded Context.", code="context_length_exceeded")
                context = request.context_limit
            remaining = context - prompt_count if context and prompt_count is not None else None
            if remaining is not None and remaining <= 0:
                raise APIError(400, "The request fills the loaded context. Shorten the input or increase Context in Studio.",
                               code="context_length_exceeded")
            if request.max_tokens is not None and remaining is not None and request.max_tokens > remaining:
                raise APIError(400, "Input plus requested output exceeds the loaded Context; no messages were truncated.",
                               code="context_length_exceeded")
            maximum = request.max_tokens or remaining
            if maximum is None:
                raise APIError(400, "Specify max_tokens: this backend does not expose its remaining Context.")
            # Carry the effective request limit through the MLX desktop
            # all-remaining-context shim, even when max_tokens was omitted.
            prompt._smartai_api_max_tokens = maximum
            prefix = prompt.rstrip().casefold()
            from core.gui_thinking import _OPEN
            starts_thinking = prefix.endswith(tuple(_OPEN))
            _, _, supported, _, _ = self._runtime_options(backend)
            defaults = self._sampling_defaults(backend, supported, thinking=starts_thinking)
            temperature = defaults.pop("temperature") if request.temperature is None else request.temperature
            top_p = defaults.pop("top_p") if request.top_p is None else request.top_p
            # Remember an explicit temperature before filling in response metadata;
            # Pro otherwise uses its established search ladder.
            explicit_temperature = request.temperature
            request.temperature, request.top_p = temperature, top_p
            sampling = {key: value for key, value in defaults.items() if key in supported}
            sampling.update(request.sampling)
            options = {"prompt": prompt, "max_tokens": maximum, "temperature": temperature, "top_p": top_p}
            if native:
                options.update({key: value for key, value in sampling.items()
                                if key == "seed" or key in backend.sampling_defaults})
            elif mlx:
                options.update(api_sampling=sampling, api_speedups=dict(request.speedups))
            pro = self._pro_config(request.pro)
            if pro["enabled"]:
                from core.local_api_pro import run_pro
                self._last_pro = {"status": "running"}
                try:
                    result = run_pro(self, backend, request, prompt, prompt_count, maximum,
                                     context, options, pro, explicit_temperature, cancel_event)
                except InterruptedError:
                    self._last_pro = {"status": "cancelled"}
                    if cancel_event.is_set():
                        return
                    raise
                except Exception:
                    self._last_pro = {"status": "error"}
                    raise
                if cancel_event.is_set():
                    self._last_pro = {"status": "cancelled"}
                    return
                self._last_pro = result["report"]
                chosen_temp = self._last_pro.get("selected_temperature")
                if chosen_temp is not None:
                    request.temperature = chosen_temp
                # Pro buffers candidates and emits only the selected text. Its
                # reasoning markers are explicit, even for an implicit template.
                yield GenerationChunk(prompt_tokens=prompt_count, starts_in_thought=False)
                text = result["text"]
                for offset in range(0, len(text), 256):
                    if cancel_event.is_set():
                        return
                    yield GenerationChunk(text=text[offset:offset + 256])
                if cancel_event.is_set():
                    return
                if (result["finish_reason"] == "stop" and valid_count(result["completion_tokens"])
                        and not result["tool_call"] and not result["stop_sequence"]
                        and request.learning_mode != "off" and self._learning_ready(backend)):
                    self._learn_completed_reply(backend, request, result["answer"], False, cancel_event)
                elif request.learning_mode == "off":
                    self._last_learning = {"status": "disabled", "mode": "off"}
                else:
                    self._last_learning = {"status": "skipped" if self._learning_ready(backend) else "unavailable",
                                           "mode": self._learning_mode()}
                yield GenerationChunk(prompt_tokens=prompt_count, completion_tokens=result["completion_tokens"],
                                      finish_reason=result["finish_reason"])
                return
            self._last_pro = {"status": "disabled"}
            yield GenerationChunk(prompt_tokens=prompt_count, starts_in_thought=starts_thinking)
            learn = request.learning_mode != "off" and self._learning_ready(backend)
            raw_parts, raw_size = [], 0
            measurements = RequestMetrics()
            # Only the harness may execute a model-requested tool.
            stream = self._decode_stream(backend, options, cancel_event)
            try:
                with observe_generations(measurements.update):
                    try:
                        for piece in stream:
                            if cancel_event.is_set():
                                break
                            if not isinstance(piece, str):
                                raise RuntimeError("The model backend emitted a non-text chunk.")
                            if learn:
                                raw_size += len(piece.encode("utf-8"))
                                if raw_size <= 16 * 1024 * 1024:
                                    raw_parts.append(piece)
                                else:
                                    raw_parts.clear()
                                    learn = False
                            live = measurements.snapshot()
                            current = live.get("context") or {}
                            yield GenerationChunk(text=piece,
                                                  prompt_tokens=current.get("prompt_tokens"),
                                                  completion_tokens=live.get("tokens"))
                    finally:
                        close = getattr(stream, "close", None)
                        if callable(close):
                            close()
            except APIError:
                raise
            except Exception as error:
                if cancel_event.is_set():
                    return
                raise APIError(500, f"Model generation failed: {error}", error_type="server_error") from error
            snapshot = measurements.snapshot()
            usage = snapshot.get("context") or {}
            measured_prompt = usage.get("prompt_tokens")
            completion = snapshot.get("tokens")
            if cancel_event.is_set():
                return
            finish = "length" if valid_count(completion) and completion >= maximum else "stop"
            if finish == "stop" and learn and valid_count(completion):
                self._learn_completed_reply(backend, request, "".join(raw_parts), starts_thinking, cancel_event)
            elif request.learning_mode == "off":
                self._last_learning = {"status": "disabled", "mode": "off"}
            elif not self._learning_ready(backend):
                self._last_learning = {"status": "unavailable", "mode": self._learning_mode()}
            else:
                self._last_learning = {"status": "skipped", "mode": self._learning_mode()}
            yield GenerationChunk(
                prompt_tokens=measured_prompt if valid_count(measured_prompt) else prompt_count,
                completion_tokens=completion if valid_count(completion) else None,
                finish_reason=finish,
            )

    @staticmethod
    def _torch_stream(engine, options, cancel_event):
        from core.transformers_stream import stream_transformers
        inputs = engine.tokenizer(options["prompt"], return_tensors="pt")
        if getattr(engine, "device", None) in {"cuda", "mps"} and hasattr(inputs, "to"):
            inputs = inputs.to(engine.device)
        temperature = options["temperature"]
        sampling = {"max_new_tokens": options["max_tokens"], "do_sample": temperature > 0,
                    "pad_token_id": engine.tokenizer.eos_token_id}
        if temperature > 0:
            sampling.update(temperature=temperature, top_p=options["top_p"])
        yield from stream_transformers(engine.model, engine.tokenizer, inputs, sampling,
                                       owner=engine, cancelled=cancel_event.is_set)
