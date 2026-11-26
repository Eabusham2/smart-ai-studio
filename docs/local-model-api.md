# Serve the loaded model to other programs

Smart AI Studio can expose its resident text model through a local HTTP server.
Load Bonsai 2.5 or another compatible text backend, then open **API Server** in
the top bar. Keep `127.0.0.1` and port `1234` for clients on the same computer,
optionally set an API key, and select **Start**. The dialog shows the URL and has
**Copy URL**, **Stop**, and a key generator.

The public model ID is always `smart-ai-studio`; `/v1/models` reports the actual
loaded model name. Choosing Bonsai or CRACK in Studio determines which weights
answer the requests.

| Interface | URL |
|---|---|
| OpenAI-compatible base URL | `http://127.0.0.1:1234/v1` |
| Model list | `GET /v1/models` |
| Chat Completions | `POST /v1/chat/completions` |
| Responses, including HTTP SSE | `POST /v1/responses` |
| Anthropic base URL | `http://127.0.0.1:1234` |
| Anthropic Messages | `POST /v1/messages` |
| Exact input-token count | `POST /v1/messages/count_tokens` |
| Health | `GET /health` or `GET /healthz` |

## Quick check

With the server started and no key selected:

```bash
curl http://127.0.0.1:1234/v1/models
curl http://127.0.0.1:1234/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"smart-ai-studio","messages":[{"role":"user","content":"Say hello."}],"max_tokens":128,"stream":true}'
```

When a key is set, add `Authorization: Bearer YOUR_KEY` or `x-api-key: YOUR_KEY`.
The examples below use `local` only as a client-side placeholder for a loopback
server with authentication disabled. Replace it with the dialog's actual key
when authentication is enabled. Keys are not written into app preferences.

## Codex

Merge the following provider into `~/.codex/config.toml`. The example assumes
Studio loaded a 32,768-token context; change `model_context_window` to the actual
loaded window and keep the compaction threshold below it:

```toml
model = "smart-ai-studio"
model_provider = "smartai"
web_search = "disabled"
model_context_window = 32768
model_auto_compact_token_limit = 24000

[model_providers.smartai]
name = "Smart AI Studio"
base_url = "http://127.0.0.1:1234/v1"
env_key = "SMARTAI_API_KEY"
wire_api = "responses"
requires_openai_auth = false
supports_websockets = false
```

Start Codex from the same terminal:

```bash
export SMARTAI_API_KEY=local
codex
```

This selects HTTP Responses with full conversation replay. Current Codex sends
`store: false` automatically in its
[Responses request builder](https://github.com/openai/codex/blob/main/codex-rs/core/src/client.rs);
no additional `store` setting belongs in the TOML above. Direct API callers must
leave `store` and `background` false and omit `previous_response_id`. The server
supports ordinary functions and Codex custom/namespace tools such as
`apply_patch`; the client executes them. Hosted web search is unavailable.
See the official [Codex provider configuration](https://developers.openai.com/codex/config-reference).

## Claude Code

Use the server root, without `/v1`, for Anthropic routing. Route the normal and
smaller helper model choices to the one loaded model. Set the context value
below to Studio's actual loaded window; 32,768 is an example. The output limit
reserves room for the prompt and tool definitions:

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:1234
export ANTHROPIC_AUTH_TOKEN=local
export ANTHROPIC_MODEL=smart-ai-studio
export ANTHROPIC_DEFAULT_OPUS_MODEL=smart-ai-studio
export ANTHROPIC_DEFAULT_SONNET_MODEL=smart-ai-studio
export ANTHROPIC_DEFAULT_HAIKU_MODEL=smart-ai-studio
export CLAUDE_CODE_SUBAGENT_MODEL=smart-ai-studio
export CLAUDE_CODE_SUBAGENT_MODEL_FORCE=1
export CLAUDE_CODE_MAX_CONTEXT_TOKENS=32768
export CLAUDE_CODE_MAX_OUTPUT_TOKENS=4096
export ENABLE_TOOL_SEARCH=false
export CLAUDE_CODE_DISABLE_STRUCTURED_OUTPUTS=1
export CLAUDE_CODE_DISABLE_THINKING=1
claude --model smart-ai-studio
```

Run `/status` in Claude Code to confirm that its Anthropic base URL is
`http://127.0.0.1:1234` and its credential source is `ANTHROPIC_AUTH_TOKEN`.
Existing settings can override shell exports. The compatibility variables load
MCP tools upfront and omit provider-specific structured-output/thinking
parameters. Bonsai can still reason internally through its own chat template.

This selects the loaded local model; it does not run Anthropic's Claude weights.
Use ordinary client tools and text requests. Claude-specific hosted features,
vision/audio inputs, and encrypted reasoning blocks are not part of this API.
Only one generation can run at once, so avoid simultaneous subagents sharing
this endpoint. The gateway and model variables are documented in
[Claude Code's gateway guide](https://code.claude.com/docs/en/llm-gateway-connect)
and [environment reference](https://code.claude.com/docs/en/env-vars). MCP tool
deferral behind gateways is described in the
[tool-search guide](https://code.claude.com/docs/en/mcp#tool-search-through-an-llm-gateway).

## Hermes

Run `hermes model`, choose **Custom endpoint**, select Chat Completions if asked
for the transport, and enter the OpenAI base URL, the key (or `local`), and
`smart-ai-studio` as the model. Its `~/.hermes/config.yaml` model section can use:

```yaml
model:
  default: smart-ai-studio
  provider: custom
  base_url: http://127.0.0.1:1234/v1
  api_key: local
  context_length: 65536
```

Current Hermes requires at least 64,000 context tokens for agent tools. Before
starting the server, unload the model, set Studio's Context to at least 65,536,
and reload it **only if the machine has enough memory**. Set Hermes to the same
actual window. Declaring a larger window in Hermes does not enlarge Studio's
loaded model. Keep requests serial and use text tools. See
[Hermes custom-provider documentation](https://hermes-agent.nousresearch.com/docs/integrations/providers).

## Model ownership and request behavior

The server uses the already loaded backend. It does not allocate another model,
enter the desktop Pro/Best Work solver, run Studio's tool registry, or learn from
API conversations. Histories and tool schemas are rendered using the actual
loaded model's chat template. Multiple system/developer messages are combined
into one leading system turn for templates that require it; their text is kept.

Starting the server reserves the model. Desktop generation, switching/unloading,
learning, evaluation, speedup changes and Context changes wait until you stop
serving. A second overlapping API request receives `409 model_busy`. Stopping
the server or disconnecting a client cancels generation; unloading and app exit
wait for backend cleanup to finish. No cancelled request is silently restarted.

The client controls `temperature`, `top_p`, output-token limit, supported stop
strings and tool choice. The API respects the request's output limit separately
from Studio's unlimited-until-context desktop behavior. Exact prompt/output
counts come from the tokenizer and backend metrics; unavailable counts remain
unknown. Input plus requested output that exceeds the loaded context is a
`400 context_length_exceeded`, with no silent history truncation.

Text streaming, JSON and Qwen XML function calls, tool results and multiple tool
calls are supported across the three protocols. Tools run in the client. This
version does not provide embeddings, vision/audio API input, hosted tools,
stored Responses, WebSockets or constrained JSON/grammar output. Unsupported
requests receive explicit errors where detectable; choosing a protocol does
not make the local model implement every provider-specific behavior.

Loopback is the default. A nonlocal listen address requires a key; choose the
host/port deliberately before Start. The server provides HTTP, not TLS. It
rejects browser cross-origin requests and limits request sizes/connections.
`SMARTAI_API_HOST`, `SMARTAI_API_PORT`, and `SMARTAI_API_KEY` can supply the dialog's
initial values.

## Validation scope

Offline tests exercise real localhost HTTP/SSE, authentication, full histories,
tools, typed XML arguments, cancellation during silent prefill, busy leases and
desktop shutdown. Separate smoke tests parsed synthetic responses with the
official OpenAI and Anthropic SDKs, including final streamed tool messages.
These tests do not download pretrained weights or claim a completed Bonsai 27B
task inside Codex, Claude Code or Hermes. Actual agent quality and speed depend
on the selected weights, context, tools and hardware.
