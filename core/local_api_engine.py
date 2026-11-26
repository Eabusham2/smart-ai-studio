"""Serve the already loaded model without entering desktop tools or learning.

The API owns a request, not a second model. Client histories are formatted with
the selected model's chat template and never enter the Studio conversation DB.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import inspect
import json
import threading


class FormattedPrompt(str):
    """An already-templated request; VLM paths must not wrap it as one user turn."""
    _smartai_formatted_chat = True

    def __new__(cls, value, max_tokens=None):
        result = super().__new__(cls, value)
        result._smartai_api_max_tokens = max_tokens
        return result


class LoadedModelAdapter:
    def __init__(self, engine, *, selected_model=None, busy=None):
        self.engine = engine
        self.selected_model = selected_model or (lambda: {})
        self.busy = busy or (lambda: "")
        self._request_lock = threading.Lock()

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
        }

    @contextmanager
    def _lease(self):
        from core.local_api import APIError
        reason = self.busy()
        if reason:
            raise APIError(409, str(reason), code="model_busy")
        if not self._request_lock.acquire(blocking=False):
            raise APIError(409, "The model is serving another request.", code="model_busy")
        model_lock = getattr(self.engine, "_model_lock", None)
        acquired = False
        try:
            if model_lock is not None:
                acquired = model_lock.acquire(blocking=False)
                if not acquired:
                    raise APIError(409, "A model operation is in progress.", code="model_busy")
            if not self.model_info():
                raise APIError(503, "Load a text model in Smart AI Studio first.", code="model_not_loaded")
            backend = self._backend()
            prepare = getattr(backend, "prepare_request", None)
            if callable(prepare):
                prepare()
            yield backend
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
    def _prompt_tokens(backend, prompt):
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
                      info.get("max_context")):
            if type(value) is int and 0 < value <= 10_000_000:
                values.append(value)
        for obj in (getattr(backend, "config", None), getattr(getattr(backend, "model", None), "config", None),
                    getattr(getattr(backend, "model", None), "args", None), getattr(backend, "tokenizer", None)):
            for name in ("max_position_embeddings", "max_seq_len", "model_max_length"):
                value = obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)
                if type(value) is int and 0 < value <= 10_000_000:
                    values.append(value)
        return min(values) if values else None

    def count_tokens(self, request):
        from core.local_api import APIError
        with self._lease() as backend:
            count = self._prompt_tokens(backend, self._format_prompt(backend, request))
            if count is None:
                raise APIError(501, "The loaded runtime does not expose an exact text tokenizer.")
            return count

    def generate(self, request, cancel_event):
        from core.generation_metrics import RequestMetrics, observe_generations, valid_count
        from core.local_api import APIError, GenerationChunk
        with self._lease() as backend:
            if cancel_event.is_set():
                return
            prompt = self._format_prompt(backend, request)
            prompt_count = self._prompt_tokens(backend, prompt)
            context = self._context_limit(backend)
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
            options = {"prompt": prompt, "max_tokens": maximum,
                       "temperature": request.temperature, "top_p": request.top_p}
            prefix = prompt.rstrip().casefold()
            yield GenerationChunk(prompt_tokens=prompt_count,
                                  starts_in_thought=prefix.endswith(("<think>", "<thinking>")))
            measurements = RequestMetrics()
            if backend is self.engine:
                stream = self._torch_stream(backend, options, cancel_event)
            else:
                parameters = inspect.signature(backend.stream_generate_tokens).parameters
                if "cancelled" in parameters:
                    options["cancelled"] = cancel_event.is_set
                # The protocol parser recognizes the model's XML/JSON tool
                # syntax. Studio's desktop dispatcher is never invoked here.
                stream = backend.stream_generate_tokens(**options)
            try:
                with observe_generations(measurements.update):
                    try:
                        for piece in stream:
                            if cancel_event.is_set():
                                break
                            if not isinstance(piece, str):
                                raise RuntimeError("The model backend emitted a non-text chunk.")
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
            yield GenerationChunk(
                prompt_tokens=measured_prompt if valid_count(measured_prompt) else prompt_count,
                completion_tokens=completion if valid_count(completion) else None,
                finish_reason="length" if valid_count(completion) and completion >= maximum else "stop",
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
