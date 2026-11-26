"""Real localhost protocol tests; no weights, network services or tool execution."""
from contextlib import contextmanager
import http.client
import json
import socket
import threading
import time
import unittest

from core.local_api import APIError, GenerationChunk, LocalModelAPIServer, normalize_request


MODEL = "smart-ai-studio"
TOOLS = [{"type": "function", "function": {"name": "read_file", "description": "Read a file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]


def wait_for(predicate, timeout=2):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        threading.Event().wait(0.01)
    return bool(predicate())


def events(raw):
    parsed = []
    for block in raw.decode().split("\n\n"):
        data = next((line[6:] for line in block.splitlines() if line.startswith("data: ")), None)
        if data is not None:
            parsed.append(data if data == "[DONE]" else json.loads(data))
    return parsed


class APITests(unittest.TestCase):
    def setUp(self):
        self.requests = []
        self.chunks = ["Hello", " world", GenerationChunk(prompt_tokens=12, completion_tokens=2)]
        self.loaded = True
        self.busy = False
        def generate(request, cancel):
            self.requests.append(request)
            yield from self.chunks
        self.server = LocalModelAPIServer(generate, lambda: {"id": MODEL, "name": "Bonsai test"} if self.loaded else None,
                                          port=0, busy=lambda: self.busy)
        self.server.start()

    def tearDown(self):
        self.assertTrue(self.server.stop())

    def request(self, path="/v1/chat/completions", body=None, *, headers=None, method="POST", raw=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.server.port, timeout=4)
        try:
            if raw is None and method == "POST":
                raw = json.dumps(body if body is not None else {"model": MODEL, "messages": [{"role": "user", "content": "Hi"}]}).encode()
            headers = {"Content-Type": "application/json", **(headers or {})}
            conn.request(method, path, raw, headers)
            response = conn.getresponse()
            data = response.read()
            return response.status, dict(response.getheaders()), data
        finally:
            conn.close()

    def json_request(self, *args, **kwargs):
        status, headers, raw = self.request(*args, **kwargs)
        return status, json.loads(raw)

    def test_models_and_health_use_the_loaded_model(self):
        status, models = self.json_request("/v1/models", method="GET")
        self.assertEqual(status, 200)
        self.assertEqual(models["data"][0]["id"], MODEL)
        self.assertEqual(models["data"][0]["display_name"], "Bonsai test")
        self.loaded = False
        self.assertEqual(self.json_request("/v1/models", method="GET")[1]["data"], [])
        self.assertFalse(self.json_request("/healthz", method="GET")[1]["model_loaded"])

    def test_chat_completion_and_exact_usage(self):
        status, response = self.json_request()
        self.assertEqual(status, 200)
        self.assertEqual(response["choices"][0]["message"]["content"], "Hello world")
        self.assertEqual(response["choices"][0]["finish_reason"], "stop")
        self.assertEqual(response["usage"], {"prompt_tokens": 12, "completion_tokens": 2, "total_tokens": 14})
        self.assertEqual(self.requests[0].messages, [{"role": "user", "content": "Hi"}])

    def test_unknown_usage_is_not_invented_from_words(self):
        self.chunks = ["many words of unmeasured output"]
        self.assertIsNone(self.json_request()[1]["usage"])
        status, response = self.json_request("/v1/messages", {"model": MODEL, "max_tokens": 10, "messages": [{"role": "user", "content": "Hi"}]})
        self.assertEqual(status, 200)
        self.assertEqual(response["usage"], {"input_tokens": None, "output_tokens": None})

    def test_chat_stream_text_usage_and_termination(self):
        status, headers, raw = self.request(body={"model": MODEL, "messages": [{"role": "user", "content": "Hi"}], "stream": True, "stream_options": {"include_usage": True}})
        self.assertEqual(status, 200)
        self.assertIn("text/event-stream", headers["Content-Type"])
        data = events(raw)
        self.assertEqual(data[0]["choices"][0]["delta"]["role"], "assistant")
        self.assertEqual("".join(event["choices"][0]["delta"].get("content", "") for event in data[:-1] if event.get("choices")), "Hello world")
        self.assertEqual(data[-1], "[DONE]")
        self.assertEqual(data[-2]["usage"]["total_tokens"], 14)

    def test_chat_stream_is_live_before_generation_finishes(self):
        gate = threading.Event()
        def generate(request, cancel):
            yield "First token"
            while not gate.wait(0.01) and not cancel.is_set():
                pass
            yield " second token"
        self.server.generate = generate
        conn = http.client.HTTPConnection("127.0.0.1", self.server.port, timeout=3)
        try:
            conn.request("POST", "/v1/chat/completions", json.dumps({"model": MODEL, "messages": [{"role": "user", "content": "Hi"}], "stream": True}), {"Content-Type": "application/json"})
            response = conn.getresponse()
            lines = []
            while not any("First token" in x for x in lines):
                lines.append(response.readline().decode())
            self.assertFalse(gate.is_set())
            self.assertEqual(self.server.active_requests, 1)
            gate.set()
            self.assertIn(b"second token", response.read())
        finally:
            gate.set()
            conn.close()

    def test_chat_tool_call_and_tool_result_round_trip(self):
        self.chunks = ['I will inspect it.\n<tool_', 'call>{"name":"read_file","arguments":{"path":"a.py"}}</tool_call>']
        status, response = self.json_request(body={"model": MODEL, "messages": [{"role": "user", "content": "Read a.py"}], "tools": TOOLS})
        self.assertEqual(status, 200)
        message = response["choices"][0]["message"]
        call = message["tool_calls"][0]
        self.assertEqual(response["choices"][0]["finish_reason"], "tool_calls")
        self.assertEqual(json.loads(call["function"]["arguments"]), {"path": "a.py"})
        self.chunks = ["The file is empty."]
        history = [{"role": "user", "content": "Read a.py"}, message, {"role": "tool", "tool_call_id": call["id"], "content": ""}]
        self.assertEqual(self.json_request(body={"model": MODEL, "messages": history, "tools": TOOLS})[0], 200)
        self.assertEqual(self.requests[-1].messages[-1], history[-1])
        self.assertEqual(self.requests[-1].messages[-2]["tool_calls"][0]["id"], call["id"])

    def test_quoted_calls_are_literal_and_thought_calls_are_suppressed(self):
        literal = '<tool_call>{"name":"read_file","arguments":{"path":"literal"}}</tool_call>'
        self.chunks = ["<think>reason ", literal, "</think>Example:\n```\n", literal, "\n```\nDone"]
        status, response = self.json_request(body={"model": MODEL, "messages": [{"role": "user", "content": "Explain the format"}], "tools": TOOLS})
        self.assertEqual(status, 200)
        message = response["choices"][0]["message"]
        self.assertNotIn("tool_calls", message)
        self.assertNotIn("reason", message["content"])
        self.assertIn(literal, message["content"])

    def test_tool_argument_text_is_preserved_byte_for_byte(self):
        arguments = {"path": 'literal <think> and </tool_call> and ``` and "quotes"'}
        text = '<tool_call>' + json.dumps({"name": "read_file", "arguments": arguments}) + '</tool_call>'
        self.chunks = list(text)
        status, response = self.json_request(body={"model": MODEL, "messages": [{"role": "user", "content": "Read"}], "tools": TOOLS})
        self.assertEqual(status, 200, response)
        self.assertEqual(json.loads(response["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]), arguments)

    def test_bonsai_xml_tool_calls_preserve_typed_parameters(self):
        tools = [{"type": "function", "function": {"name": "inspect", "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "count": {"type": "integer"}, "recursive": {"type": "boolean"}, "options": {"type": "object"}}}}}]
        raw = '<tool_call>\n<function=inspect>\n<parameter=path>\n <think>literal</think> </tool_call> \n</parameter>\n<parameter=count>\n5\n</parameter>\n<parameter=recursive>\ntrue\n</parameter>\n<parameter=options>\n{"a": [1, 2]}\n</parameter>\n</function>\n</tool_call>'
        self.chunks = list(raw)
        status, response = self.json_request(body={"model": MODEL, "messages": [{"role": "user", "content": "Inspect"}], "tools": tools})
        self.assertEqual(status, 200, response)
        arguments = json.loads(response["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])
        self.assertEqual(arguments, {"path": " <think>literal</think> </tool_call> ", "count": 5, "recursive": True, "options": {"a": [1, 2]}})

    def test_implicit_template_thoughts_are_hidden_including_fake_tool_calls(self):
        self.chunks = [GenerationChunk(starts_in_thought=True), 'private <tool_call>{"name":"read_file","arguments":{}}</tool_call>', "</think>Public answer"]
        status, response = self.json_request(body={"model": MODEL, "messages": [{"role": "user", "content": "Hi"}], "tools": TOOLS})
        self.assertEqual(status, 200, response)
        self.assertEqual(response["choices"][0]["message"], {"role": "assistant", "content": "Public answer"})

    def test_parallel_tools_have_separate_ids_and_stream_indexes(self):
        self.chunks = ['<tool_call>{"name":"read_file","arguments":{"path":"a"}}</tool_call>', '\n<tool_call><function=read_file><parameter=path>b</parameter></function></tool_call>']
        body = {"model": MODEL, "messages": [{"role": "user", "content": "Read both"}], "tools": TOOLS, "stream": True}
        status, _, raw = self.request(body=body)
        self.assertEqual(status, 200)
        tool_deltas = [call for event in events(raw) if isinstance(event, dict) for choice in event.get("choices", []) for call in choice["delta"].get("tool_calls", [])]
        initial = [call for call in tool_deltas if "id" in call]
        self.assertEqual([call["index"] for call in initial], [0, 1])
        self.assertNotEqual(initial[0]["id"], initial[1]["id"])

    def test_stop_sequences_split_across_chunks_are_not_leaked(self):
        self.chunks = ["alpha<EN", "D>never show"]
        status, response = self.json_request(body={"model": MODEL, "messages": [{"role": "user", "content": "Go"}], "stop": ["<END>"]})
        self.assertEqual(status, 200)
        self.assertEqual(response["choices"][0]["message"]["content"], "alpha")

    def test_responses_stream_order_and_final_output(self):
        status, _, raw = self.request("/v1/responses", {"model": MODEL, "instructions": "Be brief", "input": "Hi", "stream": True, "store": False})
        self.assertEqual(status, 200)
        data = events(raw)
        types = [e["type"] for e in data]
        self.assertEqual(types[:4], ["response.created", "response.in_progress", "response.output_item.added", "response.content_part.added"])
        self.assertEqual(types[-4:], ["response.output_text.done", "response.content_part.done", "response.output_item.done", "response.completed"])
        self.assertEqual([e["sequence_number"] for e in data], list(range(len(data))))
        self.assertEqual(data[-1]["response"]["output"][0]["content"][0]["text"], "Hello world")
        self.assertEqual(self.requests[0].messages[0], {"role": "system", "content": "Be brief"})

    def test_responses_function_call_output_round_trip(self):
        self.chunks = ['<tool_call>{"name":"read_file","arguments":{"path":"a.py"}}</tool_call>']
        tools = [{"type": "function", **TOOLS[0]["function"]}]
        status, response = self.json_request("/v1/responses", {"model": MODEL, "input": "Read", "tools": tools})
        self.assertEqual(status, 200)
        item = response["output"][0]
        self.assertEqual(item["type"], "function_call")
        self.chunks = ["Done"]
        follow = [{"role": "user", "content": [{"type": "input_text", "text": "Read"}]}, item,
                  {"type": "function_call_output", "call_id": item["call_id"], "output": "file text"}]
        self.assertEqual(self.json_request("/v1/responses", {"model": MODEL, "input": follow, "tools": tools})[0], 200)
        self.assertEqual(self.requests[-1].messages[-1], {"role": "tool", "tool_call_id": item["call_id"], "content": "file text"})

    def test_codex_custom_namespaced_tool_preserves_raw_patch(self):
        patch = "*** Begin Patch\n*** Add File: x.txt\n+hi\n*** End Patch"
        tools = [{"type": "namespace", "name": "functions", "tools": [{"type": "custom", "name": "apply_patch", "description": "Apply a patch", "format": {"type": "grammar", "syntax": "lark", "definition": "ignored at the adapter boundary"}}]}]
        self.chunks = ["<tool_call>" + json.dumps({"name": "functions.apply_patch", "arguments": {"input": patch}}) + "</tool_call>"]
        status, _, raw = self.request("/v1/responses", {"model": MODEL, "input": "Add x.txt", "tools": tools, "stream": True})
        self.assertEqual(status, 200)
        data = events(raw)
        item = data[-1]["response"]["output"][0]
        self.assertEqual(item["type"], "custom_tool_call")
        self.assertEqual(item["namespace"], "functions")
        self.assertEqual(item["name"], "apply_patch")
        self.assertEqual(item["input"], patch)
        self.assertIn("response.custom_tool_call_input.delta", [event["type"] for event in data])
        self.chunks = ["Done"]
        status, _ = self.json_request("/v1/responses", {"model": MODEL, "input": [{"role": "user", "content": "Add"}, item, {"type": "custom_tool_call_output", "call_id": item["call_id"], "output": "Applied"}], "tools": tools})
        self.assertEqual(status, 200)
        self.assertEqual(self.requests[-1].messages[1]["tool_calls"][0]["function"]["name"], "functions.apply_patch")
        self.assertEqual(json.loads(self.requests[-1].messages[1]["tool_calls"][0]["function"]["arguments"])["input"], patch)

    def test_responses_length_finishes_as_incomplete(self):
        self.chunks = ["Truncated", GenerationChunk(finish_reason="length")]
        status, response = self.json_request("/v1/responses", {"model": MODEL, "input": "Hi"})
        self.assertEqual(status, 200)
        self.assertEqual(response["status"], "incomplete")
        self.assertEqual(response["incomplete_details"], {"reason": "max_output_tokens"})

    def test_anthropic_text_stream_order_and_usage(self):
        status, _, raw = self.request("/v1/messages?beta=true", {"model": MODEL, "system": [{"type": "text", "text": "System", "cache_control": {"type": "ephemeral"}}], "max_tokens": 20, "messages": [{"role": "user", "content": "Hi"}], "stream": True})
        self.assertEqual(status, 200)
        data = events(raw)
        self.assertEqual([e["type"] for e in data], ["message_start", "content_block_start", "content_block_delta", "content_block_delta", "content_block_stop", "message_delta", "message_stop"])
        self.assertEqual(data[-2]["delta"]["stop_reason"], "end_turn")
        self.assertEqual(data[-2]["usage"], {"input_tokens": 12, "output_tokens": 2})

    def test_anthropic_tools_and_results_round_trip(self):
        self.chunks = ['<tool_call>{"name":"read_file","arguments":{"path":"a.py"}}</tool_call>']
        tools = [{"name": "read_file", "input_schema": TOOLS[0]["function"]["parameters"]}]
        status, response = self.json_request("/v1/messages", {"model": MODEL, "max_tokens": 20, "messages": [{"role": "user", "content": "Read"}], "tools": tools})
        self.assertEqual(status, 200)
        block = response["content"][0]
        self.assertEqual(block["type"], "tool_use")
        self.assertEqual(response["stop_reason"], "tool_use")
        self.chunks = ["Done"]
        history = [{"role": "user", "content": "Read"}, {"role": "assistant", "content": response["content"]}, {"role": "user", "content": [{"type": "tool_result", "tool_use_id": block["id"], "content": [{"type": "text", "text": "Error"}], "is_error": True}]}]
        status, _ = self.json_request("/v1/messages", {"model": MODEL, "max_tokens": 20, "messages": history, "tools": tools})
        self.assertEqual(status, 200)
        self.assertEqual(self.requests[-1].messages[-1]["tool_call_id"], block["id"])
        self.assertEqual(self.requests[-1].messages[-1]["content"], "Tool error: Error")

    def test_exact_anthropic_token_count_and_unavailable_error(self):
        body = {"model": MODEL, "messages": [{"role": "user", "content": "Hi"}]}
        self.assertEqual(self.json_request("/v1/messages/count_tokens", body)[0], 501)
        self.server.count_tokens = lambda request: 17
        self.assertEqual(self.json_request("/v1/messages/count_tokens", body), (200, {"input_tokens": 17}))
        self.server.count_tokens = lambda request: None
        self.assertEqual(self.json_request("/v1/messages/count_tokens", body)[0], 501)

    def test_key_auth_for_bearer_and_anthropic_headers(self):
        self.server.api_key = "test-secret"
        self.assertEqual(self.json_request()[0], 401)
        self.assertEqual(self.json_request(headers={"Authorization": "Bearer wrong"})[0], 401)
        self.assertEqual(self.json_request(headers={"Authorization": "Bearer test-secret"})[0], 200)
        self.assertEqual(self.json_request(headers={"x-api-key": "test-secret"})[0], 200)

    def test_nonloopback_requires_key_and_invalid_host_rejected(self):
        with self.assertRaisesRegex(ValueError, "API key"):
            LocalModelAPIServer(None, None, host="0.0.0.0")
        with self.assertRaisesRegex(ValueError, "IP address"):
            LocalModelAPIServer(None, None, host="arbitrary.test")
        self.assertEqual(self.json_request(headers={"Host": "attacker.test"})[0], 403)
        self.assertEqual(self.json_request(headers={"Origin": "https://attacker.test"})[0], 403)

    def test_no_loaded_model_and_wrong_model_errors(self):
        self.loaded = False
        self.assertEqual(self.json_request()[0], 503)
        self.loaded = True
        self.assertEqual(self.json_request(body={"model": "not-loaded", "messages": [{"role": "user", "content": "Hi"}]})[0], 404)

    def test_busy_request_is_rejected_before_generation(self):
        self.busy = True
        status, response = self.json_request()
        self.assertEqual(status, 409)
        self.assertEqual(response["error"]["code"], "model_busy")
        self.assertEqual(self.requests, [])

    def test_concurrent_generations_do_not_enter_same_model(self):
        entered, release = threading.Event(), threading.Event()
        def generate(request, cancel):
            entered.set()
            while not release.wait(0.01) and not cancel.is_set():
                pass
            yield "done"
        self.server.generate = generate
        results = []
        worker = threading.Thread(target=lambda: results.append(self.json_request()))
        worker.start()
        try:
            self.assertTrue(entered.wait(2))
            self.assertEqual(self.json_request()[0], 409)
        finally:
            release.set()
            worker.join(3)
        self.assertEqual(results[0][0], 200)

    def test_disconnect_cancels_silent_prefill(self):
        entered, observed = threading.Event(), threading.Event()
        def generate(request, cancel):
            entered.set()
            if cancel.wait(3):
                observed.set()
            yield "never consumed"
        self.server.generate = generate
        conn = socket.create_connection(("127.0.0.1", self.server.port))
        body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": "Hi"}], "stream": True}).encode()
        conn.sendall(f"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n".encode() + body)
        self.assertTrue(entered.wait(2))
        conn.close()
        self.assertTrue(observed.wait(2), "Disconnect must cancel without waiting for the first token")
        self.assertTrue(wait_for(lambda: self.server.active_requests == 0))

    def test_stop_waits_for_generator_cleanup_before_model_reuse(self):
        entered, cleanup, release = threading.Event(), threading.Event(), threading.Event()
        def generate(request, cancel):
            entered.set()
            try:
                cancel.wait(3)
                yield ""
            finally:
                cleanup.set()
                release.wait(3)
        self.server.generate = generate
        result = []
        worker = threading.Thread(target=lambda: result.append(self.json_request()))
        worker.start()
        try:
            self.assertTrue(entered.wait(2))
            self.assertFalse(self.server.stop(timeout=0))
            self.assertFalse(self.server.is_running)
            self.assertTrue(cleanup.wait(2))
            self.assertEqual(self.server.active_requests, 1)
            with self.assertRaisesRegex(RuntimeError, "still stopping"):
                self.server.start()
        finally:
            release.set()
            worker.join(3)
        self.assertTrue(self.server.stop())

    def test_errors_during_stream_use_error_events(self):
        def generate(request, cancel):
            yield "start"
            raise APIError(400, "Context full", code="context_length_exceeded")
        self.server.generate = generate
        status, _, raw = self.request("/v1/responses", {"model": MODEL, "input": "Hi", "stream": True})
        self.assertEqual(status, 200)
        data = events(raw)
        self.assertEqual(data[-1]["type"], "error")
        self.assertEqual(data[-1]["code"], "context_length_exceeded")
        self.assertEqual([event["sequence_number"] for event in data], list(range(len(data))))
        self.assertNotIn("response.completed", [e["type"] for e in data])

    def test_unknown_tool_and_required_tool_failure_are_explicit(self):
        self.chunks = ['<tool_call>{"name":"not_supplied","arguments":{}}</tool_call>']
        body = {"model": MODEL, "messages": [{"role": "user", "content": "Read"}], "tools": TOOLS}
        self.assertEqual(self.json_request(body=body)[0], 502)
        self.chunks = ["I did not call a tool"]
        self.assertEqual(self.json_request(body={**body, "tool_choice": "required"})[0], 502)

    def test_invalid_json_large_body_and_unsupported_features(self):
        self.assertEqual(self.json_request(raw=b"not json")[0], 400)
        self.assertEqual(self.json_request(raw=b'{"temperature":NaN}')[0], 400)
        self.assertEqual(self.json_request(raw=b"[]")[0], 400)
        self.server.max_body_bytes = 20
        self.assertEqual(self.json_request()[0], 413)
        self.server.max_body_bytes = 8 * 1024 * 1024
        base = {"model": MODEL, "messages": [{"role": "user", "content": "Hi"}]}
        for key, value in (("n", 2), ("logprobs", True), ("stream", "true"), ("text", "invalid"),
                           ("tool_choice", []), ("tool_choice", {"type": "function", "function": "bad"}),
                           ("parallel_tool_calls", "bad"), ("messages", [{"role": [], "content": "x"}]),
                           ("temperature", -1), ("max_tokens", 0),
                           ("response_format", {"type": "json_schema"})):
            with self.subTest(key=key):
                self.assertEqual(self.json_request(body={**base, key: value})[0], 400)
        self.assertEqual(self.json_request("/v1/responses", {"model": MODEL, "input": "Hi", "previous_response_id": "old"})[0], 400)
        self.assertEqual(self.json_request(body={**base, "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://example.test/a.png"}}]}]})[0], 400)

    def test_max_output_bytes_cancels_and_returns_error(self):
        self.server.max_output_bytes = 2
        self.assertEqual(self.json_request()[0], 413)


if __name__ == "__main__":
    unittest.main()
