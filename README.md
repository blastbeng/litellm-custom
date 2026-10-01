# litellm-custom

A personal LLM router built on **LiteLLM Proxy** with lightweight Python
gateways: a single OpenAI-compatible endpoint (`:4000`) that aggregates local
models (llama-swap on a dedicated PC) and cloud providers (synthetic.new,
OpenRouter, Groq, Gemini, Ollama Cloud, inference4free), with **dynamic**
wildcard model listing, fallback policy and `reasoning_effort` handling.

```
                     ┌────────────────────────── host (blast) ──────────────────────────┐
 client (AiderDesk   │                                                                  │
 / curl / any        │   ┌─────────────┐   CONNECT    ┌──────────────────────────┐     │
 OpenAI SDK) ────────┼──►│  litellm    │─────────────►│  llmtrim (MITM :43117)   │─────┼──► internet
 localhost:4000      │   │  :4000      │              │  tracks/compresses LLM   │     │   (api.groq.com,
                     │   └──┬───┬───┬──┘              └──────────────────────────┘     │   openrouter.ai,
                     │      │   │   │                                                  │   generativelanguage,
                     │      │   │   └────────────► llama-gate ──► blastpc:11435          │   api.synthetic.new,
                     │      │   │                  (TCP gate, PC off → 503)         │   ollama.com)
                     │      │   │                                                  │
                     │      │   ├────► openrouter-gw ─┐                           │
                     │      │   ├────► groq-gw ───────┤ (via llmtrim)             │
                     │      │   ├────► gemini-gw ─────┤                           │
                     │      │   └────► inference4free-gw ─┴──► 192.168.1.13:18010   │
                     │      │                                                      │
                     └────────────────────────────────────────────────────────────────┘
```

## Components

| Service | File | Role |
|---|---|---|
| `litellm` | `config/config.yaml` | LiteLLM proxy/router (port 4000) |
| `llama-gate` | `llama-gate/gate.py` | TCP gate to llama-swap (PC off → 503 in ~2s) |
| `ollama-gate` | `llama-gate/gate.py` (same, env-driven) | TCP gate to Ollama OpenAI-compat API (`blastpc:11434`) |
| `ollama-gw` | `provider-gw/gw.py` (generic) | Ollama listing with `ID_PREFIX=ollama/`, no cache (models visible only while the PC is online) |
| `openrouter-gw` | `openrouter-gw/gw.py` | OpenRouter listing: path `/v1/*`→`/api/v1/*` + id prefix `openrouter/` |
| `groq-gw`, `gemini-gw`, `inference4free-gw` | `provider-gw/gw.py` (generic) | listing with `ID_PREFIX`, `PATH_PRE`, last-good cache |
| `reasoning_clamp.py` | litellm callback | `reasoning_effort` cascade/clamp per provider + fallback policy |
| `llmtrim` (external) | `/opt/docker/compose/llmtrim` | MITM proxy: tracks/compresses LLM traffic |

## How wildcard routing works

Each provider is exposed with **a single wildcard** entry in `model_list`:

```yaml
- model_name: groq/*
  litellm_params:
    model: openai/*          # the capture replaces "*" with the native id
    api_base: http://groq-gw/openai/v1
```

- Public name `groq/openai/gpt-oss-120b` → the upstream receives the native
  id `openai/gpt-oss-120b` (the capture strips the prefix).
- With `litellm_settings.check_provider_endpoint: true`, `/v1/models` and
  `/model/info` expose the endpoint's **real** model list: if a provider's
  models change, litellm stays up to date (essential for `inference4free/*`,
  whose list is dynamic).

### Why the gateways exist

LiteLLM (v1.104.0) has three behaviors the gateways work around:

1. **Listing always hits `{origin}/v1/models`**, ignoring the api_base path
   → different path schemes need `PATH_PRE` (Groq: `/openai`,
   Gemini: `/v1beta/openai`, OpenRouter: `/api`).
2. **Rename of ids whose first segment matches a known provider**
   (`openai/...`, `deepseek/...`, ...) in the listing → the gateway prefixes
   ids with `ID_PREFIX` (e.g. `groq/`, `inference4free/`): the rename no
   longer fires and the capture restores the native id at runtime.
3. **Single timeout** (httpx, connect included) → "PC off" cannot be
   distinguished from "PC on with a huge model loading":
   `llama-gate` solves it (503 in ~2s if the PC doesn't answer TCP,
   otherwise a transparent pipe so litellm can wait up to 30 min).

### Last-good cache (provider-gw, `MODELS_CACHE=1`)

For unstable upstreams (deepseek4free is a project under development): if
`/models` fails, the gateway serves the **last valid model list** saved on
disk (header `X-Models-Cache: stale`). Models remain always visible in
litellm even when the endpoint is down.

## Fallback policy

- **`synthetic/*`** → **`local-model`** (llama-swap local model) first,
  then **`ollama/local-model`** (the Ollama model with the same name on
  the Windows PC) — only on
  provider errors (busy/offline/rate-limit). The `ollama/local-model`
  group exists only while the Windows PC is online (dynamic listing via
  `ollama-gw` + TCP gate): when the PC is off the gate 503s in ~2s.
- **`local-model`** itself (requested directly) → **`ollama/local-model`**
  only. **`small-model`** is a standalone model (invoke it directly): it is
  **not** a fallback of any model.
- **Everything else** → no fallback: the error goes straight to the client.
- **`ollama/*`** (any local Ollama model) and **`openrouter/free`** never
  fall back (not even to themselves).

⚠️ Fallback keys in litellm do **not** support wildcards
(`get_fallback_model_group`: exact match / provider-stripped / `"*"` only),
so the per-model policy is implemented in the `reasoning_clamp.py` callback
via **per-request** `litellm_params.fallbacks`, which override the config
list (`router.py`: `kwargs.get("fallbacks", self.fallbacks)`).
`config.yaml` only keeps the explicit disable entries.

## `reasoning_effort` handling (cascade)

Rule: default `xhigh` where valid, otherwise fall down (`high`, `medium`, ...).

| Provider | Default | Notes |
|---|---|---|
| `synthetic/*`, `openrouter/*`, `ollama-cloud/*` | `xhigh` (in config) | accepted |
| `gemini/*` | `high` (callback) | `xhigh` not valid on Gemini; `none/minimal/low/medium` respected; `xhigh/max` → clamped to `high` |
| `groq/*` | `high` (callback) | only for reasoning models (`openai/gpt-oss-*`, `qwen/*`); for the others (`allam-2-7b`, whisper, ...) the parameter is **removed** (they would 400) |
| `inference4free/*` | none | endpoint not validable per-model |

The client-provided value always wins (clamped if invalid).

## Deployment

```bash
cp config/config.example.yaml config/config.yaml   # then fill in the keys
sudo systemctl restart docker-compose@litellm
```

`docker-compose.yml` expects:
- `llmtrim` listening on `:43117` with its CA at
  `/opt/docker/compose/llmtrim/data/.llmtrim/ca-bundle.pem`
  (the bundle is the system root CA store + the llmtrim CA, kept in sync by
  `llmtrim-ca-sync`);
- llama-swap on `blastpc` (192.168.1.29, port 11435) for the local model;
- the `deepseek4free` service on 192.168.1.13:18010 (optional).

All outbound HTTPS traffic from litellm and the gateways goes through
llmtrim (`HTTPS_PROXY`/`GW_PROXY`), so provider calls are intercepted,
tracked and compressed just like direct client traffic. The local model is
in `NO_PROXY` (LAN traffic).

### Note on llama-server (local model)

The context reported by `llama-server` is **`--ctx-size / -np`**: with
`-np 2` and `--ctx-size 262144` each slot exposes 131072 tokens and litellm
rejects large requests with `ContextWindowExceededError`. With `-np 1` the

**Note on 5-minute client timeouts (AiderDesk and similar AI-SDK clients).**
`Headers Timeout Error` after exactly 5 minutes is **not litellm**: litellm's
per-model timeout for the local model is 1800s. It is the client's HTTP stack
(undici, used by the Vercel AI SDK) whose `headersTimeout` defaults to
300s — the local model can easily take longer than that before the first
response bytes when it is busy or prefilling a huge context. Fix it in the
client (AiderDesk provider settings: set timeout to `false` or > 300000 ms),
not in the proxy.
full context is available (262144), consistent with `model_info` in config.

## Files

```
config/config.yaml        # REAL config (keys) — NOT committed
config/config.example.yaml
docker-compose.yml
reasoning_clamp.py        # litellm callback (reasoning + fallback policy)
llama-gate/gate.py
openrouter-gw/gw.py
provider-gw/gw.py         # generic gateway (ID_PREFIX / PATH_PRE / cache)
```

## Security

- `config/config.yaml` is in `.gitignore`: it contains real keys. Use
  `config/config.example.yaml` as the template.
- No keys are present in the code or the compose file.