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
                     │      │   │   └────────────► llama-gate ──► blastpc:11435      │   api.synthetic.new,
                     │      │   │        (TCP gate, http|https auto, PC off → 503)   │   ollama.com)
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
| `llama-gate` | `llama-gate/gate.py` | TCP gate to llama-swap — `GATE_TLS=auto`: tries TLS, falls back to plain HTTP on `WRONG_VERSION_NUMBER` (llama-swap runs **plain HTTP** since 2026-10-02, previously `https` self-signed with gate-side TLS termination; litellm always stays plain HTTP; PC off → 503 in ~2s) |
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
   `llama-gate` solves it (503 in ~2s if the PC doesn't answer TCP/TLS,
   otherwise a transparent pipe so litellm can wait up to 30 min).

### Last-good cache (provider-gw, `MODELS_CACHE=1`)

For unstable upstreams (deepseek4free is a project under development): if
`/models` fails, the gateway serves the **last valid model list** saved on
disk (header `X-Models-Cache: stale`). Models remain always visible in
litellm even when the endpoint is down.

## Fallback policy

**`small-model` is the final fallback of every model** (user request
2026-10-02): it is reached **only after every other fallback in the chain
has failed**.

**Local Ollama was removed on 2026-10-04** (`ollama/*`, `ollama-gate`,
`ollama-gw`): the only local stack left is llama-swap behind `llama-gate`.
**`ollama-cloud/*` (ollama.com) is untouched** — it is a cloud provider.

- **`synthetic/*`** → **`local-model`** (llama-swap local model) first,
  then **`small-model`** as the final resort.
- **`synthetic/syn:small:text`** and **`synthetic/syn:small:vision`** →
  **`small-model`** only (user request 2026-10-04: **aligned**, a single
  fallback, capacity-matched small → small).
  `small-model` is text-only on llama-swap, so a vision request may 400 on
  it: that is the explicit choice behind aligning both to one target.
- **`synthetic/hf:nomic-ai/nomic-embed-text-v1.5`** (embedding served by
  synthetic.new on `/v1/embeddings`, user request 2026-10-04) →
  **`embedding-model`** (llama-swap: Qwen3-Embedding-0.6b on CPU, context
  16384, same `llama-gate` as the other local models). Single target: the
  other llama-swap models are chat models and would 400 on an embeddings
  request, so there is no further step.
- **`local-model`** itself (requested directly) → **`small-model`**.
- **`small-model`** → no fallback at all: it is the last resort of
  everyone else, retrying it alone would make no sense.
- **`embedding-model`** (requested directly) → no fallback: it is the only
  embedding-capable model in the chain, any other target would 400 on an
  embeddings request.
- **Everything else** (`groq/*`, `gemini/*`, `openrouter/*`,
  `openrouter/free`, `ollama-cloud/*`, `inference4free/*`,
  and any future prefix) → **`small-model`** directly.

⚠️ Fallback keys in litellm do **not** support wildcards
(`get_fallback_model_group`: exact match / provider-stripped / `"*"` only),
so the per-model policy is implemented in the `reasoning_clamp.py` callback
via **per-request** `litellm_params.fallbacks`, which override the config
list (`router.py`: `kwargs.get("fallbacks", self.fallbacks)`).
`config.yaml` keeps the catch-all `"*": ["small-model"]` entry as a safety
net for requests that skip the callback.

## `reasoning_effort` handling (cascade)

Rule: default `xhigh` where valid, otherwise fall down (`high`, `medium`, ...).

| Provider | Default | Notes |
|---|---|---|
| `synthetic/*`, `openrouter/*`, `ollama-cloud/*` | `xhigh` (in config) | accepted |
| `gemini/*` | `high` (callback) | `xhigh` not valid on Gemini; `none/minimal/low/medium` respected; `xhigh/max` → clamped to `high` |
| `groq/*` | `high` (callback) | only for reasoning models (`openai/gpt-oss-*`, `qwen/*`); for the others (`allam-2-7b`, whisper, ...) the parameter is **removed** (they would 400) |
| `inference4free/*` | none | endpoint not validable per-model |

The client-provided value always wins (clamped if invalid).

## Context clamp (`max_tokens`)

The context limit is enforced **by the upstream, not by litellm**: on the local
PC the big models are served by **Strata** behind llama-swap, which counts the
prompt with the **real tokenizer** and answers **400** —
`prompt (N tokens) + max tokens (M) exceeds the context (C); requests are
never truncated` — and litellm wraps that 400 as `BadRequestError`. Strata can
shorten `max_tokens` to the room left **itself** (exact, no estimation):
add `"fit_max_tokens": true` to `strata-<model>.json` on `blastpc` (or the
About tab → Model settings in its web page; a prompt that leaves no room at
all is still refused). With
`enable_pre_call_checks: false` litellm performs **no** context check at all,
so a big-prompt request kills the **whole fallback chain**
(`synthetic/*` → `local-model` → `small-model`, all 131072).

`reasoning_clamp.py` therefore clamps `max_tokens` (and
`max_completion_tokens`) to `context − prompt`, and — when the request
carries a fallback chain — to the **smallest context in the chain**: the
budget must fit every target, including `small-model` (131072), the final
fallback of every model. The pre-call hook runs **once
per client request** (`proxy/utils.py`), not once per fallback attempt, so the
budget computed on the requested model is **inherited by every fallback
target**. The `litellm_params` defaults (`max_tokens`) are merged into every
attempt **after** the hook (invisible to it): this is why the cloud
wildcards carry **no** `max_tokens` default and `synthetic/*` is capped at
32768.
The prompt size is **estimated conservatively** (`chars / 2.8` + per-message
overhead + **tool schemas** + images, plus a **fixed + proportional safety
margin** — 1024 + 3% of the estimate): overestimating is safe (shorter
completion), underestimating produces the upstream 400. Measured on real
~100k-token transcripts `chars / 2.8` alone lands **~1.4–1.8% below** the
upstream count and tool schemas are worth ~80 tokens each — the old fixed
margin lost there and the clamp handed out a `max_tokens` the upstream
refused (2026-10-06 incident, issue #545 text). If the prompt alone saturates
the context the clamp is useless — there is no room left for the answer — and
the **overflow cascade** below takes over.

Contexts live in `CONTEXT_BY_PREFIX` in `reasoning_clamp.py` and **must stay
aligned with `model_info.max_input_tokens` in `config.yaml`** (131072 for
`local-model`, `small-model`, `synthetic/*`, `inference4free/*`;
16384 for `embedding-model`; 262144 for `openrouter/*`, `groq/*`, `gemini/*`,
`ollama-cloud/*`).

## Context overflow cascade (`B → C → E → 413`)

When `prompt + 1024 + margin > context` the request cannot fit **any** model in
the chain, so shrinking `max_tokens` cannot save it: the prompt itself has to
shrink. `async_pre_call_hook` then runs a cascade and, after each level,
re-estimates the prompt and re-applies the clamp:

| Level | What it does | Lossy? |
|---|---|---|
| **B** | Compact with an **available 262k model** (`groq/openai/gpt-oss-120b` → `gemini/models/gemini-2.5-flash-lite` → `openrouter/auto`, tried in order, 45 s each, no retries, no fallbacks). The 131072 local models are not used as compactors: the body barely fits in them. | Faithful summary |
| **C** | `litellm.compression.compress()` — BM25 scoring, low-relevance messages replaced by stubs, system/last-user/last-assistant protected. **Zero LLM calls**, target = 55 % of the context. | **Yes** — the `litellm_content_retrieve` tool is *not* injected and the originals are *not* kept: nothing in this stack serves that tool, so a stub is gone for good. That is why C comes after B. |
| **E** | Map/reduce: the body is cut into ≤ 90k-token chunks (max 8), each summarised by `small-model` (180 s each), the partials are re-placed before the current request. Last resort: lossy **and** it costs N calls on the local stack that is already under pressure. | Yes |
| **F** | Nothing freed space → **413** with a structured detail (`model`, `context_tokens`, `prompt_tokens_estimated`, `tried`, `hint`) instead of firing the request and letting it die on the fallback chain. | — |

What is preserved at every level: the **system** messages verbatim, and the
**tail from the last `user` message onward** verbatim (cutting there keeps the
current tool exchanges intact — an orphaned `tool` result would 400).

The cascade runs **only in the async hook** (B and E make LLM calls); the sync
`pre_call_hook` applies policy + clamp only. Internal calls are marked with
litellm's own `INTERNAL_CALL_ORIGIN_METADATA_KEY`, which is the recursion
guard (a compaction call must never be compacted again) and keeps them out of
spend logs, rate limits and cooldowns.

## Deployment

```bash
cp config/config.example.yaml config/config.yaml   # then fill in the keys
sudo systemctl restart docker-compose@litellm
```

`docker-compose.yml` expects:
- `llmtrim` listening on `:43117`, publishing its CA bundle at
  `/opt/docker/compose/llmtrim/data/.llmtrim/trust/ca-bundle.pem`
  (system root CA store + the llmtrim CA, kept in sync by `llmtrim-ca-sync`);
- llama-swap on `blastpc` (192.168.1.29, port 11435, **plain HTTP** since
  2026-10-02 — `llama-gate` runs with `GATE_TLS=auto`, so it also works if
  llama-swap goes back to `https` self-signed: it terminates the TLS itself;
  litellm always talks plain HTTP to the gate);
- the `deepseek4free` service on 192.168.1.13:18010 (optional).

All outbound HTTPS traffic from litellm and the gateways goes through
llmtrim (`HTTPS_PROXY`/`GW_PROXY`), so provider calls are intercepted,
tracked and compressed just like direct client traffic. The local model is
in `NO_PROXY` (LAN traffic).

### CA trust: mount the **directory**, never the file

`llmtrim` regenerates its MITM CA whenever its interception set widens
(`LLMTRIM_EXTRA_HOSTS`) or the image updates, and `llmtrim-ca-sync` rebuilds
the bundle with `cat … > bundle.tmp && mv` — a **new inode**. A bind-mount of a
**single file** stays pinned to the inode the file had when the container
started, so the replacement is invisible until the container is recreated:
every provider call then fails with
`[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get
local issuer certificate` (aiohttp/httpx wrap it as "Connection error", and
`/v1/models` listings fail the same way).

That is why litellm mounts `…/data/.llmtrim/trust` → `/data/.llmtrim` and the
gateways mount it → `/data/llmtrim` (`GW_CA=/data/llmtrim/ca-bundle.pem`):
a **directory** mount follows file replacements, so a CA rotation is live
without restarting anything. `trust/` contains **only** the public bundle —
`ca.key` (0600) is deliberately outside every container.

### Note on llama-server (local model)

The context reported by `llama-server` is **`--ctx-size / -np`**: with
`-np 2` and `--ctx-size 262144` each slot exposes 131072 tokens, and the
upstream rejects oversized requests with its own 400 (see
**Context clamp** above). With `-np 1` the full context is available (262144),
consistent with `model_info` in config. A context flip 262144 ↔ 131072 must be
kept in sync in **three** places: `config.yaml` (`model_info`),
`config.example.yaml`, and `CONTEXT_BY_PREFIX` in `reasoning_clamp.py`.

**Note on 5-minute client timeouts (AiderDesk and similar AI-SDK clients).**
`Headers Timeout Error` after exactly 5 minutes is **not litellm**: litellm's
per-model timeout for the local model is 1800s. It is the client's HTTP stack
(undici, used by the Vercel AI SDK) whose `headersTimeout` defaults to
300s — the local model can easily take longer than that before the first
response bytes when it is busy or prefilling a huge context. Fix it in the
client (AiderDesk provider settings: set timeout to `false` or > 300000 ms),
not in the proxy.

## Files

```
config/config.yaml        # REAL config (keys) — NOT committed
config/config.example.yaml
docker-compose.yml
reasoning_clamp.py        # litellm callback (reasoning + fallbacks + clamp + overflow cascade)
llama-gate/gate.py
openrouter-gw/gw.py
provider-gw/gw.py         # generic gateway (ID_PREFIX / PATH_PRE / cache)
```

## Security

- `config/config.yaml` is in `.gitignore`: it contains real keys. Use
  `config/config.example.yaml` as the template.
- No keys are present in the code or the compose file.