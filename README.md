# litellm-custom

A personal LLM router built on **LiteLLM Proxy** with lightweight Python
gateways: a single OpenAI-compatible endpoint (`:4000`) that aggregates local
models (llama-swap and Ollama on a dedicated PC), cloud providers
(synthetic.new, OpenRouter, Groq, Gemini, Ollama Cloud, Cloudflare Workers AI), with **dynamic**
wildcard model listing, fallback policy and `reasoning_effort` handling.

```
                     ┌────────────────────────── host (blast) ──────────────────────────┐
 client (AiderDesk   │                                                                  │
 / curl / any        │   ┌──────────────────────────────┐                               │
 OpenAI SDK) ────────┼──►│  litellm  :4000              │──────────────► internet       │
 localhost:4000      │   │  + headroom callback         │   diretto     (api.groq.com,   │
                     │   │    (compressione in-process) │               openrouter.ai,     │
                     │   └──┬───┬───┬───────────────────┘               generativelanguage,│
                     │      │   │   │                                   api.synthetic.new,│
                     │      │   │   ├────────────► llama-gate ──► blastpc:11435          │
                     │      │   │   │      (TCP gate, http|https auto, PC off → 503)      │
                     │      │   │   └────────────► ollama-gw ──► ollama-gate ──► 11434    │
                     │      │   │        (TCP gate, plain HTTP, PC off → 503)             │
                     │      │   │                                                      │
                     │      │   ├────► openrouter-gw ─┐                               │
                     │      │   ├────► groq-gw ───────┤ (diretto: nessun proxy)        │
                     │      │   ├────► gemini-gw ─────┤                               │
                     │      │   └────► cloudflare-gw ─┴──► internet (api.*)             │
                     │      │                                                          │
                     └────────────────────────────────────────────────────────────────┘
```

`llmtrim` (MITM proxy su :43117) e' stato **disattivato** il 2026-11-14: non
esiste piu' un hop proxy tra litellm e i provider (`HTTPS_PROXY`/`SSL_CERT_FILE`
e i `GW_PROXY` dei gateway sono stati rimossi — con il proxy spento ogni
chiamata cloud falliva con `OpenAIError: Connection error`). La compressione
del contesto ora la fa **headroom** dentro litellm (vedi sotto).
```

## Components

| Service | File | Role |
|---|---|---|
| `litellm` | `config/config.yaml` | LiteLLM proxy/router (port 4000) |
| `llama-gate` | `llama-gate/gate.py` | TCP gate to llama-swap — `GATE_TLS=auto`: tries TLS, falls back to plain HTTP on `WRONG_VERSION_NUMBER` (llama-swap runs **plain HTTP** since 2026-10-02, previously `https` self-signed with gate-side TLS termination; litellm always stays plain HTTP; PC off → 503 in ~2s) |
| `ollama-gate` | `llama-gate/gate.py` (same, env-driven) | TCP gate to Ollama's OpenAI-compatible API (`blastpc:11434`) — `GATE_TLS=0`, Ollama serves **plain HTTP** on that port, so the TLS probe is skipped; PC off → 503 in ~2s |
| `ollama-gw` | `provider-gw/gw.py` (generic) | Ollama local listing with `ID_PREFIX=ollama/` + `STRIP_LATEST=1`, no cache (models visible only while the PC is online; PC off → the gate 503s) |
| `openrouter-gw` | `openrouter-gw/gw.py` | OpenRouter listing: path `/v1/*`→`/api/v1/*` + id prefix `openrouter/` |
| `groq-gw`, `gemini-gw` | `provider-gw/gw.py` (generic) | listing with `ID_PREFIX`, `PATH_PRE`, last-good cache |
| `cloudflare-gw` | `cloudflare-gw/gw.py` (dedicated) | Workers AI listing from the REST search endpoint, rewritten as OpenAI `/v1/models`: id prefix `cloudflare/`, property mapping (`context_length`, `pricing`, `reasoning_parameters`), free-plan / deprecated / `-lora` filters, last-good cache |
| `reasoning_clamp.py` | litellm callback | `reasoning_effort` cascade/clamp per provider + fallback policy |
| `headroom_callback.py` | litellm callback | `HeadroomUserClamp` (headroom-ai): comprime **in-process** tool output e messaggi utente strutturati (JSON/CSV) con `compress_user_messages=True`, fail-open (un errore logga un warning e la richiesta prosegue invariata). Tre strati limitano la perdita dei tool output (2026-11-19): (1) gli ultimi `HR_PROTECT_LAST` messaggi (default 6) restano verbatim — il tool output che l'agente sta leggendo e' sempre integro; (2) i tool di lettura (`*file_read*`, `*grep*`, `*glob*`, …) sono esclusi dal path lossy perché AiderDesk manda nomi qualificati (`power---file_read`) che i default upstream non matchano; (3) i messaggi `user` che sembrano dump di log restano protetti (`ProtectRecentAndLogs`). Il resto della storia resta compresso lossy: su 50 messaggi/116k token il risparmio misurato e' -72% (era -76% senza finestra). Prosa e codice restano intatti e **Kompress/ML non e' installato** (nessuna AI locale). litellm 1.109 rifiuta il path alla classe (serve un'istanza, come `rclamp`) e invoca l'hook solo se `async_pre_call_hook` e' definito nella classe concreta: per questo la sottoclasse lo ri-dichiara. Senza `transformers` il callback registra la **stima** come backend tokenizer HuggingFace (`_force_estimation_without_transformers`): headroom 0.41.0 importa `transformers` fuori dal `try` e per i modelli qwen/deepseek/llama l'eccezione annullava l'intera compressione ("Compression failed … No module named 'transformers'") |
| `Dockerfile.litellm` | build di `litellm-headroom:local` | litellm upstream **+ headroom-ai** nel venv `/app/.venv`. litellm resta pinnato alla versione del base image (headroom-ai lo aggiornerebbe: senza constraint pip passa 1.109.0 → 1.109.1). Buildata da `manage-docker` (e a mano con `docker compose build litellm`) |
| `llmtrim` (external) | `/opt/docker/compose/llmtrim` | **disattivato il 2026-11-14** — nessun hop proxy tra litellm e i provider |

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
  models change, litellm stays up to date (essential for `ollama/*` and
  `openrouter/*`, whose lists follow the upstream).

### Why the gateways exist

LiteLLM (v1.104.0) has four behaviors the gateways work around:

1. **Listing always hits `{origin}/v1/models`**, ignoring the api_base path
   → different path schemes need `PATH_PRE` (Groq: `/openai`,
   Gemini: `/v1beta/openai`, OpenRouter: `/api`).
2. **Rename of ids whose first segment matches a known provider**
   (`openai/...`, `deepseek/...`, ...) in the listing → the gateway prefixes
   ids with `ID_PREFIX` (e.g. `groq/`, `ollama/`): the rename no
   longer fires and the capture restores the native id at runtime
   (`openrouter/` is mandatory: OpenRouter ids contain slashes and the
   `:free` suffix).
3. **Single timeout** (httpx, connect included) → "PC off" cannot be
   distinguished from "PC on with a huge model loading":
   `llama-gate` solves it (503 in ~2s if the PC doesn't answer TCP/TLS,
   otherwise a transparent pipe so litellm can wait up to 30 min).
4. **One `api_key` per deployment** → an upstream with its **own** secret
   cannot be authenticated without copying that secret into this repo. The
   gateway can read it from the upstream project's `.env`
   (`UPSTREAM_KEY_FILE` + `UPSTREAM_KEY_NAME`, re-read every 60s) and
   **replaces** litellm's placeholder `Authorization: Bearer dummy`.
   Opt-in and currently unused: every live gateway keeps passing the
   client's header through (groq, gemini, ollama, openrouter).

### Last-good cache (provider-gw, `MODELS_CACHE=1`)

For upstreams that can be briefly unreachable: if `/models` fails, the
gateway serves the **last valid model list** saved on disk (header
`X-Models-Cache: stale`). Models remain visible in litellm even when the
endpoint is down. Each gateway has its **own** cache file
(`/cache/<id-prefix>-models.json`).

### Cloudflare Workers AI (`cloudflare-gw`)

Workers AI is the one upstream whose listing cannot be proxied by the
generic gateway:

- **No OpenAI-compatible listing.** `GET /ai/v1/models` answers
  `7001 "GET not supported for requested URI"`; chat (`/ai/v1/chat/completions`)
  is OpenAI-compatible (streaming included), the *catalogue* is not. It only
  exists at `GET /ai/models/search`, REST + paginated, with metadata nested in
  `properties[]`.
- **Requested path.** `cloudflare-gw` rewrites `GET /v1/models` (and
  `/models`) into the search call, then maps the result to the same shape
  litellm and `reasoning_clamp` already consume from every other provider:

  | Workers AI property | Listing field |
  |---|---|
  | `id` (native `@cf/vendor/model`) | `id` with `ID_PREFIX` prepended |
  | `properties.context_window` | `context_length` |
  | `properties.price` (per-M USD, list of `{unit, price}`) | `pricing.prompt/completion` (USD per token) |
  | `properties.reasoning_effort.supported_efforts` | `reasoning_parameters.efforts` |

  Because the listing carries `context_length`, the per-model context window
  reaches `_DYN_CTX` on the litellm side: the `model_info` ceiling on the
  `cloudflare/*` wildcard is only a pre-sweep fallback.
- **Passthrough.** Everything except the listing is forwarded untouched (body,
  query string, SSE): the wildcard capture strips `cloudflare/` and litellm
  strips the provider prefix `openai/`, so Workers AI receives the native
  `@cf/vendor/model` id. `@cf/...` does not collide with any known litellm
  provider, so no rename corruption (unlike OpenRouter).
- **Filters (account-dependent).** `cloudflare-gw` drops from the listing:
  models with `require_workers_paid=true` (a Workers **Free** account gets
  `403 "not available on the Workers Free plan"`), models whose
  `planned_deprecation_date` is already past (`5028 "Model has been
  deprecated"`), and `-lora` deployments (LoRA bases, not general purpose
  models). Knobs: `DROP_PAID`, `DROP_DEPRECATED`, `DROP_LORA` (`0` disables).
  On a paid account set `DROP_PAID=0`.
- **`reasoning_effort` is validated per model and the 400 is hard.**
  `gpt-oss-120b` rejects `xhigh`, `qwen3.8-27b` rejects `high`, models without
  `supported_efforts` still accept a varying set. `reasoning_clamp` therefore
  sends the parameter **only** when the listing exposes `supported_efforts`
  and the value is in it, otherwise removes it and lets the Cloudflare default
  apply; no static `reasoning_effort` is set in `litellm_params`.
- **Cache.** Last-good listing is written *after* the mapping (ids already in
  public form) and served with `X-Models-Cache: stale` when Cloudflare is
  unreachable (`MODELS_CACHE=0` disables).

### Quota notices served as a 200

Several free backends enforce per-IP quotas but answer **HTTP 200** with the
quota text as the assistant message (`该ip请求过多… 60 次/ora`, seen on free
backends). To litellm that is a *success*: no retry, no fallback, and the
client gets unusable text instead of an answer.

* **litellm side (`reasoning_clamp`)**: `async_post_call_success_hook` looks at
  the first `UPSTREAM_NOTICE_SCAN_CHARS` (300) chars of a non-streaming answer
  and turns a notice into a **429** with an explicit message; in streaming the
  same check runs on the first chunks from
  `async_post_call_streaming_iterator_hook`, which drop the garbage and emit a
  single explicit chunk (the SSE status is already gone at that point). Normal
  responses pass through untouched.

### Quarantine of models that keep breaking (`reasoning_clamp`)

A model that fails repeatedly (rate limit, 5xx, timeout, connection error, or
a quota notice served as a 200) is temporarily **hidden from the listings** so
that an external client does not pick it, and the compaction cascade stops
using it as a worker. One sliding-window counter per **public model name**,
fed by both client requests (`async_post_call_failure_hook`) and the internal
cascade attempts:

| Knob | Value | Meaning |
|---|---|---|
| `QUAR_FAIL_THRESHOLD` | 3 | failures inside the window |
| `QUAR_FAIL_WINDOW_S` | 600 | sliding window (10 min) |
| `QUAR_BASE_S` | 900 | first quarantine: 15 min |
| `QUAR_MAX_S` | 21600 | escalation cap: each relapse doubles it, up to 6 h |

Effects and guarantees:

* the name disappears from `/v1/models`, `/model/info` and
  `/model_group/info` (and `/v1/models/{id}` answers 404) through
  `async_filter_listed_models`; it comes back **by itself** when the TTL
  expires — no config change, the wildcard stays;
* it is **not** chosen by `_worker_pool` while quarantined (no wasted hop);
* it is still **callable** if a client asks for it explicitly: hiding is not
  blocking, so a healthy model is re-enabled by its first success
  (`_quar_note_ok`) without waiting for the TTL;
* only provider faults count — 400/401/404 and context-overflow are our
  errors and never quarantine a model;
* `auto` and every `*/auto` are **never** quarantined (nor counted): they are
  the fallback by definition, hiding them would empty the fallback list;
* state is in RAM only: a litellm restart clears all quarantines.

## Media models (llama-swap image / video / audio)

Since 2026-10-16 llama-swap also serves **image generation, video
generation, TTS and music** models. They are exposed by **four partial
wildcards** that follow llama-swap's listing live — there is no per-model
entry, so adding/removing/renaming a model in llama-swap makes it
appear/disappear here without touching this config:

| wildcard | upstream endpoint | native ids (listing 2026-10-16) |
|---|---|---|
| `llama-swap/img-*` | `POST /v1/images/generations` | `img-edit-qwen`, `img-flux-dev`, `img-flux-schnell`, `img-qwen-image`, `img-sdxl-light`, `img-zimage-turbo` |
| `llama-swap/vid-*` | `POST /v1/images/generations` | `vid-ltx25`, `vid-wan22-14b`, `vid-wan22-5b` |
| `llama-swap/tts-*` | `POST /v1/audio/speech` | `tts-chatterbox`, `tts-kokoro`, `tts-pocket`, `tts-qwen3`, `tts-supertonic`, `tts-voxcpm2`, `tts-voxtral` |
| `llama-swap/music-*` | `POST /v1/audio/speech` | `music-ace-step`, `music-minimax3`, `music-stable-audio` |

Public name `llama-swap/img-flux-dev`; the upstream receives the **native**
id `img-flux-dev` (`PatternMatchRouter` capture → `openai/img-flux-dev`, then
the `openai` provider strips its own prefix). Same `llama-gate` as the chat
models: PC off → 503 in ~2s, huge model loading → up to 30 min.

- **The `llama-swap/` prefix is mandatory**: `get_known_models_from_wildcard`
  does `split("/", 1)` and expands **nothing** without a `/`.
- **Video goes through the image endpoint**: llama-swap has no `/v1/videos`
  (it would 404, and the OpenAI video route is async/`Sora`-shaped);
  stable-diffusion.cpp serves video from `/v1/images/generations` too.
- **The chat models of llama-swap keep their unprefixed names**
  (`local-model`, `small-model`, `embedding-model`), with their explicit
  entries and real contexts — that is why a single `llama-swap/*` wildcard is
  *not* used: it would swallow them too.
- **No fallback** (user request 2026-10-16) — see the fallback policy below.
  They are also **excluded from the context-cascade worker pool**
  (`_worker_pool`), where their local-hardware tier would otherwise put them
  first in line as compression workers.
- `mode` in `model_info` (`image_generation` / `audio_speech`) documents the
  use for listings/UIs/cost tracking; the **endpoint is chosen by the route
  the client calls**, not by `mode`.
- Voices are per model: `GET /v1/audio/voices?model=<id>` on llama-swap
  (e.g. `tts-kokoro` → `["if_sara","im_nicola"]`).

## Fallback policy

**`small-model` is no longer a fallback of any model** (user request
2026-10-25). It remains a standalone model that can be called directly, but no
chain contains it any more: when a chain is exhausted, the upstream error
reaches the client unchanged.

**Local Ollama was restored on 2026-10-11** (`ollama/*`, `ollama-gate`,
`ollama-gw`): the Windows PC serves its OpenAI-compatible API on
`blastpc:11434` behind the same TCP gate as llama-swap. `ollama/small-model`
is the local step of **two** chains only — `synthetic/syn:small:text` (user
request 2026-10-12) and `synthetic/syn:small:vision` (user request 2026-10-14,
chains identical).
**`ollama-cloud/*` (ollama.com) is a separate cloud provider** and is
untouched.

- **`synthetic/*`** → **no fallback** (user request 2026-10-28: `local-model`
  removed from every `synthetic/*` chain, right after `small-model` on
  2026-10-25). Only the three exact chains below remain.
- **`synthetic/syn:small:text`** → **`ollama/small-model`** →
  **`openrouter/free`** (user request 2026-10-12; since 2026-10-14
  `syn:small:vision` has the *identical* chain).
  `ollama/small-model` — the Ollama model on the **same PC**
  (`192.168.1.29:11434`, via `ollama-gw` + `ollama-gate`) — comes **first**:
  an **independent server**, so it answers when llama-swap is busy or 429s (its
  one-request-at-a-time serialization is exactly why this step exists). Its
  static context is 131072, so a larger prompt can 400 here — one wasted hop,
  the chain continues. The name must exist in Ollama on the PC, otherwise the
  upstream 404s: also just a failed hop. The **free cloud** model is the
  safety net for when the local PC is off. `ollama/small-model` and
  `openrouter/free` are **concrete names resolved by the `ollama/*` and
  `openrouter/*` wildcards** in the model_list (litellm `get_model_list` →
  `_get_wildcard_deployments` → `PatternMatchRouter.route`), **not**
  fictional prefixes.
- **`synthetic/syn:small:vision`** → **`ollama/small-model`** →
  **`openrouter/free`** — the *identical* chain of
  `synthetic/syn:small:text` (user request 2026-10-14: same steps, same order;
  defined as an alias so the two cannot diverge again). The local target is
  text-only, so a vision request may 400 there: one wasted hop, the chain
  continues.
- **`synthetic/hf:nomic-ai/nomic-embed-text-v1.5`** (embedding served by
  synthetic.new on `/v1/embeddings`, user request 2026-10-04) →
  **`embedding-model`** (llama-swap: Qwen3-Embedding-0.6b on CPU, context
  32768, 1024-dim vectors, same `llama-gate` as the other local models). Single target: the
  other llama-swap models are chat models and would 400 on an embeddings
  request, so there is no further step.
- **`local-model`** itself (requested directly) → **no fallback** (user
  request 2026-10-25: `small-model` removed from every chain).
- **`small-model`** → no fallback: standalone model, never a fallback.
- **`embedding-model`** (requested directly) → no fallback: it is the only
  embedding-capable model in the chain, any other target would 400 on an
  embeddings request.
- **`llama-swap/*` media models** (`img-*`, `vid-*`, `tts-*`, `music-*`) →
  **no fallback at all** (user request 2026-10-16). An image, a video, a
  speech sample or a music track cannot be produced by a chat model, so any
  hop in a chain could only 400 upstream — and would silently turn a failed
  generation into a bogus chat answer. The callback returns `[]` for **every**
  `llama-swap/...` name, so no config catch-all can pull in a chat model (the
  per-request `fallbacks: []` overrides it); a generation error surfaces as-is.
  The same prefix check keeps these models out of the context-cascade
  **worker pool** (`_worker_pool`), where their local-hardware tier would
  otherwise rank them first as compression workers.
- **Everything else** (`groq/*`, `gemini/*`, `cloudflare/*`, `openrouter/*`,
  `openrouter/free`, `ollama/*`, `ollama-cloud/*`, and any future prefix)
  → **no fallback**.

⚠️ Fallback keys in litellm do **not** support wildcards
(`get_fallback_model_group`: exact match / provider-stripped / `"*"` only),
so the per-model policy is implemented in the `reasoning_clamp.py` callback
via **per-request** `litellm_params.fallbacks`, which override the config
list (`router.py`: `kwargs.get("fallbacks", self.fallbacks)`).
The config catch-all `"*": ["small-model"]` was **removed** (2026-10-25); the
callback sets an explicit `fallbacks` list (often empty) on every request.

## Hybrid model (`hybrid-model`)

One endpoint that spreads the load over five deployments, all at the same level
except for their tier:

| Member | Context (input) | Vision | Tier |
|---|---|---|---|
| `local-model` | 196608 | yes | 1 |
| `small-model` | 196608 | yes | 1 |
| `synthetic/syn:large:text` | 524288 | yes | 1 |
| `synthetic/syn:small:text` | 196608 | **no** | 1 |
| `gemini/models/gemini-flash-latest` | 262144 | yes | 2 (cloud, only when tier 1 is exhausted) |

**Group name vs public name.** `router_settings.routing_groups` defines a
`hybrid-model-pool` group (`routing_strategy: least-busy`) and
`router_settings.model_group_alias` maps the public name
`hybrid-model → hybrid-model-pool`. The split is required by the listings:
`/model/info` — the endpoint **AiderDesk** reads to enumerate models — only
contains deployments (`registry.all()`) and **alias rows**; a routing group
never shows up there. With the public name on the group, `hybrid-model` existed
in `/v1/models` (OpenWebUI lists it) but was invisible to AiderDesk. A group
whose name is also a deployment or alias is *shadowed* by it (litellm warns
`group_name ... is shadowed ... not callable until renamed`), and a self-alias
emits no rows at all — measured on 1.111.

**Tiers are callback policy, not `routing_strategy: priority`.** The native
priority strategy picks **at random** inside a tier and forbids
`routing_strategy_args`, so the tier is chosen by
`async_filter_deployments` in `reasoning_clamp.py`: while at least one tier-1
member is healthy the filter returns only those and `least-busy` balances them
**by load**; `gemini-flash-latest` enters only when all four are in
cooldown/down (log: `[reasoning_clamp] hybrid-model: N/M membri tier1 sani`).
The same hook drops deployments declaring `supports_vision: false`
(`syn:small:text`) when the request carries an image; the filter receives the
**group** name (litellm resolves the alias before selection).

**Tie-break is rotating.** `least-busy` picks the *first* minimum in
`_least_busy_index()` (`litellm/router_strategy/least_busy.py`), so with every
in-flight count at 0 — i.e. any sequential traffic — every request would hit the
same member (measured: 10/10 on one deployment). The filter therefore rotates
the tier-1 list by one position per request (`_HYBRID_RR`, an
`itertools.count()`), which distributes ties evenly (measured: 20 sequential →
5/5/5/5; 8 concurrent → 2/2/2/2) while real load still decides whenever the
counts differ.

**Group context is pinned to 196608** (`PINNED_CONTEXT`), the smallest real
member window: `/v1/models` reports the *widest* limit of a group (and here it
reports 196608/131072 via `register_model`, the group's real ceiling), while
`/model/info` reports the *narrowest* across the alias rows (196608/131072) —
the same honest number for compaction. Sizing the cascade on 524288 would push prompts past
`syn:small:text`'s window and produce 400s upstream.

**Compaction chain**: `small-model` then `synthetic/syn:small:text`
(`COMPACTOR_OVERRIDE`) — the local worker first, the synthetic one when the PC
is off.

**Free reserves** (`groq/*`, `cloudflare/*`, `openrouter/*:free` — only free
ids) are **not** group members (a wildcard member resolves to the literal
pattern and sends `model="*"` upstream, a 400): `reasoning_clamp` picks up to 3
concrete ids per request (context and vision aware, `openrouter/free` first) and
passes them as **per-request `fallbacks`**, so they only fire once the whole
group is exhausted.

**Cooldowns**: every member carries `allowed_fails_policy` (503/429/timeout/
5xx → 1 allowed fail) and `cooldown_time: 45`, so one infrastructure failure
moves the request to another member instead of burning the global retries on a
dead one.

## `reasoning_effort` handling (cascade)

Rule: default `xhigh` where valid, otherwise fall down (`high`, `medium`, ...).

| Provider | Default | Notes |
|---|---|---|
| `synthetic/*` | `xhigh` **only if the model supports it** (callback) | supported values come from the live listing (`reasoning_parameters.efforts`, `_DYN_EFFORTS`) with the static `EFFORTS_BY_MODEL` fallback. If `xhigh` is not listed: `high`, else `max` (`high`/`max` are treated as equivalent); if neither is supported the parameter is **not sent**. Unknown models keep the historical `xhigh`. Example: `syn:small:text` gets `high`, `syn:large:vision` gets `high`, `nomic-embed-text` gets nothing |
| `openrouter/*`, `ollama-cloud/*` | `xhigh` (in config) | accepted by the endpoint (heterogeneous pools: not validable per-model) |
| `gemini/*` | `high` (callback) | `xhigh` not valid on Gemini; `none/minimal/low/medium` respected; `xhigh/max` → clamped to `high` |
| `groq/*` | `high` (callback) | only for reasoning models (`openai/gpt-oss-*`, `qwen/*`); for the others (`allam-2-7b`, whisper, ...) the parameter is **removed** (they would 400) |
| `cloudflare/*` | per-model (callback), no static default | Workers AI validates **per model** and the 400 is hard (`gpt-oss-120b` rejects `xhigh`, `qwen3.8-27b` rejects `high`): sent **only** if the listing exposes `supported_efforts` and the value is in it, otherwise **removed** and the Cloudflare default applies |

The client-provided value always wins (clamped if invalid). The synthetic
default **cannot** live in `litellm_params`: litellm merges those defaults
*after* the callback hook, so a static `xhigh` overrides the per-model clamp
and makes `/v1/messages` fail (see below).

### `/v1/messages` (Anthropic route)

litellm 1.109 routes `openai/...` deployments through the **Responses API**
(`{api_base}/responses`) for this route, which synthetic and the local gateways
do not serve: the request dies with `API route not found: '/openai/v1/responses'`
(and is silently masked by a fallback chain, e.g. `syn:small:text` answering
from `openrouter/free`). `LITELLM_USE_CHAT_COMPLETIONS_URL_FOR_ANTHROPIC_MESSAGES: "True"`
in the `litellm` environment keeps that route on chat/completions, which every
upstream here supports (verified: `syn:large:text`, `syn:small:text`,
`GLM-5.3`, `groq/openai/gpt-oss-120b`, `openrouter/free`, `gemini/*` all 200,
served by the requested model).

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
(`synthetic/*` has no fallback since 2026-10-28; 196608 for `local-model` and
`small-model` since 2026-11-23, 131072 floor for the synthetic wildcard — real
per-model contexts via `_window_for`).

`reasoning_clamp.py` therefore clamps `max_tokens` (and
`max_completion_tokens`) to `context − prompt` of the **model the request
was addressed to** (its real window via `_window_for`) — 2026-10-06 user
directive: "la catena di fallback deve tenere conto del contesto da cui il
modello è partito". The chain no longer shrinks the budget to its smallest
member: a large prompt aimed at a 512k model keeps its full headroom, and if
the primary fails while the prompt does not fit a fallback target, that
target 400s and the chain advances (accepted behaviour). The pre-call hook runs **once
per client request** (`proxy/utils.py`), not once per fallback attempt, so the
budget computed on the requested model is **inherited by every fallback
target**. The `litellm_params` defaults (`max_tokens`) are merged into every
attempt **after** the hook (invisible to it): this is why the cloud
wildcards carry **no** `max_tokens` default and `synthetic/*` is capped at
32768.
The prompt size is **estimated adaptively**: each message part gets a
`chars / token` ratio derived from its symbol density **of its non-digit
characters** — dense code/JSON 2.8 (down to 2.2 at ≥ 35% symbols), prose 4.6,
CJK 1.5, mixed-with-base64 3.6 (linear interpolation in between) — plus
per-message overhead, **tool schemas**, images, and a **fixed +
proportional safety margin** (1024 + 3% of the estimate). Overestimating is
safe (shorter completion), underestimating produces the upstream 400.
Measured on real ~100k-token transcripts `chars / 2.8` alone lands ~1.4–1.8%
below the upstream count and small tool schemas are worth ~80 tokens each —
the old fixed margin lost there and the clamp handed out a `max_tokens` the
upstream refused (2026-10-06 incident, issue #545 text).

**Digits are counted separately** (2026-11-20). `tiktoken cl100k_base` merges
numbers (`2026` = one token) while the llama.cpp pretokenizer splits them:
measured with llama-server `/tokenize` on `small-model`, numeric logs came out
at 2.16 chars/token and JSON with indices at 2.04, i.e. 29% / 15% **more**
tokens than every chars-based estimate. That gap is exactly what pushed a
"compacted" payload ~90k tokens past the context (estimate 214,439, engine
301,397 → upstream 400). The estimate is therefore `digits × 1` + the rest at
its density ratio, and when `tiktoken` (≥ 150k chars) is used it gets
`+ 0.6 × digits`. Re-measured on six calibrated samples (numeric logs, Python
code, base64, prose, JSON, MCP schemas) the estimator now sits **1.03–1.15×**
the engine count instead of 0.71–1.00× — never below it.

**Tool schemas** (the other 2026-10-06 false 413) are counted with
`tiktoken` whenever available (sha1-cached, identical across requests → O(1)
after the first) plus the same digit correction; the density ratio (3.0) is
only the fallback. The older `min(ratio, tiktoken)` kept the estimate from
over-counting the ~100 big MCP schemas of AiderDesk (which the old `chars /
2.2` inflated ~2×: est 126,805 vs 71,796 real) but silently under-counted
numeric **tool results** by ~30% — the tool path was the biggest single hole.
On the opposite side, the old flat `chars / 2.8` over-counted prose
transcripts ~2× (est 171k vs 87.8k real) and triggered the cascade spuriously
— the density ratio fixes that too. If the prompt alone saturates the context
the clamp is useless — there is no room left for the answer — and the
**overflow cascade** below takes over.

Contexts live in `CONTEXT_BY_PREFIX` in `reasoning_clamp.py` and **must stay
aligned with `model_info.max_input_tokens` in `config.yaml` and with the
`--ctx-size` actually passed to the engine** (2026-11-23: **196608** for both
`local-model` — Strata `--context 196608` on the RX 7800 XT, lowered from
262144 — and `small-model` — 262144 dies in the RTX 3060 load-time CUDA
warm-up, 245760 leaves only 1148 MiB free).
The old 163840 was a stale local cap: the clamp was budgeting 98k tokens less
than the engine really had. **synthetic is
13 explicit entries** (12 chat + the nomic embedding) with the real per-model contexts exposed on `/v1/models`
(`syn:large:text` 524288, `syn:small:text` 196608, `hf:openai/gpt-oss-120b`
131072, …) — litellm's wildcard expansion propagates ids only, never context
fields, so a wildcard cannot expose per-model contexts; a NEW synthetic model
needs a config entry. Wildcards whose upstream does **not** expose
`context_length` in its listing (`ollama/*`) keep a static floor from
`CONTEXT_BY_PREFIX`; for the others their real per-model contexts
(10000–1000000) reach the clamp and the worker
pool dynamically from the provider listings via `_DYN_CTX`/`_window_for` in
`reasoning_clamp.py` but cannot be exposed per-model on `/v1/models`; still
131072 for `groq/openai/gpt-oss-120b` / `-20b`,
whose real Groq context is 131072 even though the `groq/*` wildcard declares
262144 as the ceiling for its largest model; 32768 for `embedding-model`;
131072 for `ollama/*` (Ollama does **not** expose `context_length` in its
listing, so the wildcard value stays static); 262144 for `openrouter/*`,
`groq/*`, `gemini/*`, `ollama-cloud/*`). Specific
entries are matched before generic prefixes.

## Context overflow cascade (`B → C → T → T2 → E → T3 → 413`)

When `prompt + 1024 + margin > context` the request cannot fit **any** model in
the chain, so shrinking `max_tokens` cannot save it: the prompt itself has to
shrink. `async_pre_call_hook` then runs a cascade and, after each level,
re-estimates the prompt and re-applies the clamp.

**Worker = the request's own model** (user request 2026-10-28, supersedes the
2026-10-06 dynamic pool): every level that makes LLM calls — B compaction,
T/T2 tail summarisation, E map/reduce — calls **the very model the client
asked for** (`_worker_pool(..., only=model)` returns a one-element pool, and
the internal call runs with `fallbacks=[]`). A request for
`synthetic/syn:large:text` is therefore compacted by `synthetic/syn:large:text`,
never by a free/local model behind the scenes. Capacity filters still apply
(the level is skipped when that model cannot hold the body via `_window_for`,
or when it is a non-chat/media model), but the *never-expensive worker* price
filter is bypassed for the request's own model — it is already the model the
client chose. Consequence, intended: if the model is down or too small for the
body, B/T2/E have no safety net — C (BM25, local) remains, then 413.

**Dynamic worker pool (legacy path, wildcard request names only)**
(`_worker_pool` without `only`): a request whose model name is a literal
wildcard (`groq/*`) cannot be called upstream, so those requests fall back to
the old behaviour: every level builds its worker list **dynamically from
`llm_router.model_list`** — every configured model is eligible, wildcards
included. **Dynamic `/v1/models` interrogation** (2026-10-06 request: "adapt
the code to our model list DINAMICALLY"): a background daemon re-reads the
proxy's own `http://127.0.0.1:4000/v1/models` every 10 min (Bearer `master_key`
from the running process, unauth retry on 401, proxy-env bypass — a blocking
self-call inside the event loop would deadlock, so the fetch only lives in
that thread). The listing **adds models missing from `model_list`** to the
worker pool and replaces the static context guesses with the **real
`max_input_tokens`/`max_output_tokens`** (the prompt must fit the real input
cap; the window is `in + out` when both are declared); a model added/removed
in `config.yaml` (or via `/model/new`) is picked up **without code changes or
restarts**. Until the first successful listing the pool behaves exactly as
before (`model_list` + static contexts). `/v1/models` carries no prices, so
cost tiers still come from the gateway listings (below).

**Temporary worker blacklist** (2026-10-06 request: "temp blacklist models
(incrementally) that fail too often when used for
compression/summarization/clamping"): failures are counted **directly at the
worker call sites** (level B compactors, level T/E summarisers — both
exceptions *and* empty answers), not by parsing logs. 3 failures inside a
sliding 10-min window ⇒ blacklisted for **15 min**, and every relapse
**doubles** the ban (capped at 24 h). A clean success consumes a fallover and
decays one strike; a success *while banned* reinstates the model immediately
(provider quota recovered). Blacklisted workers are **demoted to the end of
the pool, never excluded** — if everything else fails they are retried
anyway (self-healing), and the pool never goes empty because of the
blacklist. Client traffic never touches these counters: only internal
compaction/summarisation calls do. **Wildcard expansion**: `model_list` holds patterns (`groq/*`,
`openrouter/*`, …), but an internal call with the literal pattern would send
`model="*"` upstream and 400 — so each pattern is expanded into its
**requestable concrete ids** from the gateway `/models` listing (same source
as the config auto-import; direct upstreams get the prefix added here),
cached 10 min; if a listing is unreachable the pattern itself is kept as a
fallback (tolerant gateways accept it). The fixed lists only set the
*preferred order*; models whose mapped context cannot hold
`chunk + output + margin` are filtered out (so a 240k body is never handed to
a 192k model — note `groq/openai/gpt-oss-120b`/`-20b` are 131072 for real),
non-chat models (`embed`, `whisper`, `guard`, `rerank` substrings) are
excluded, and **every fitting worker** is attempted within a per-level **time
budget** (`WORKER_TIME_BUDGET_S = 420`, checked before each attempt) with
**sticky failover** (the first worker that answers keeps the remaining chunks;
an error or empty answer advances to the next — failures are fast, so the
budget only cuts hung workers). Partial results are kept: some chunks
summarised ⇒ the level succeeds; the level fails — and the cascade moves on —
only when *nothing* was produced. B sizes its compaction
body with the **tiktoken** estimate (not the conservative dense ratio): a
marginal compactor that 400s costs one failover round, while over-filtering
would skip B entirely and drop the request to the lossy C tier.

**Cost-tier ordering** (2026-10-06 request: free models first, then cheap,
then the rest — dynamically): the pool is sorted by **(cost tier,
preference)**, so the preference lists decide only the order *inside* a
tier. **Tier 0 — free**: `openrouter/free`,
`ollama-cloud/*`, any id with the OpenRouter `:free` suffix, any model whose
gateway/brand name contains `free`, models whose gateway `/models` listing
declares a zero price (captured live during wildcard expansion — a provider
listing a new free model is picked up without code changes), and the local
hardware models (`small-model`, `local-model` — own GPU, zero cost).
**Tier 1 — cheap**: first the **listing price** when known — prompt price ≤
`CHEAP_PRICE_USD_PER_TOKEN = 5e-7` (≈ $0.50/M prompt tokens; zero ⇒ tier 0,
hidden/negative prices are ignored) — otherwise **29 name patterns**: `gpt-oss`,
`flash-lite`, `flash`, `lite`, `gemma`, `syn:small`, `small`, `qwen`, `qwq`,
`nano`, `-mini`, `haiku`, `deepseek`, `glm`, `kimi`, `moonshot`, `llama`,
`mistral-small`, `ministral`, `pixtral`, `minimax`, `phi-`, `phi3`, `phi4`,
`nemotron`, `falcon`, `tiny`, `grok-3-mini`, `grok-4-fast` (`-mini` keeps the
dash: `gemini` *contains* the string `mini`, which would classify every Gemini
as cheap; the bare `small` is safe because the local models already matched
tier 0). **Tier 2 — everything else** (gemini-2.5-flash/pro at real listing
prices, paid deepseek/claude/gpt-5 tiers, …), ordered by the level's
preference list. Because the classification is price/brand/pattern based
rather than a closed id list, delisted models drop out via the 10-minute
listing cache and newly listed ones join automatically.

**Never-expensive worker filter** (user directive 2026-10-07: *"NEVER use
expensive models during prompt compression / summarization / clamping"*):
a costly model is not demoted — it is **excluded** from the worker pool
entirely, by a three-layer price engine in `reasoning_clamp.py`:

1. **Our channel's listing price** (authoritative — it is what we actually
   pay): `pricing.prompt` harvested from the gateway listings into
   `_TIER_HINTS`. Price parsing accepts OpenRouter's plain strings and
   Synthetic's `"$0.0000006"` format (previously `float()` silently failed on
   the `$` and Synthetic prices never reached the tiers). Price **beats the
   name**: a free `pro` stays allowed, Kimi-K3 at $3/M is excluded although
   `kimi` is a cheap pattern.
2. **Web price index**: the daemon periodically fetches OpenRouter's
   *public* `api/v1/models` (no key, ~450 models) into `_WEB_PRICES` and
   applies those market prices to **other providers** whose gateways hide
   pricing (groq, gemini), matching by model **family** — the last id
   segment, with date/instruct suffixes normalized
   (`kimi-k2-instruct-0905` → `kimi-k2`).
3. **Flagship regexp net** (`EXPENSIVE_WORKER_RES`) when no price is known:
   `pro`, `opus`, `sonnet`, `ultra`, `premium`, `large`, `max`,
   `kimi-k[3-9]`, `deepseek-v[4-9]`, `gpt-[4-9]`, `o[1-9]`, `grok-[2-9]`,
   `claude`, `command-*` — with a cheap-variant veto
   (`WORKER_CHEAP_OVERRIDE_RES`: `flash`, `mini`, `nano`, `lite`, `small`,
   `tiny`, `haiku`, `fast`, `free`, `air`), so `deepseek-v4-pro` is excluded
   while `deepseek-v4.1-flash`, `o3-mini`, `grok-4-fast`, `claude-3-haiku`
   stay eligible.

A model with **no price signal at all** is also excluded (never gamble
money on an unproven cost). Never excluded: local hardware and free
channels (brand `free`, `:free`, `ollama-cloud`) — zero cost by
definition. `WORKER_MAX_PRICE_USD_PER_TOKEN = CHEAP_PRICE_USD_PER_TOKEN`
(≈ $0.50/M prompt). `openrouter/auto` was removed from `COMPACTOR_MODELS`
for the same reason: auto-routing can land on an expensive model, and the
choice is not controllable.

| Level | What it does | Lossy? |
|---|---|---|
| **B** | Compact the **history** (everything before the last `user` message) with the **request's own model** (2026-10-28), 45 s, no retries, no fallbacks. Legacy wildcard-name requests only: the dynamic pool in cost-tier order, `COMPACTOR_MODELS` first inside a tier. | Faithful summary |
| **C** | `litellm.compression.compress()` — BM25 scoring, low-relevance messages replaced by stubs, system/last-user/last-assistant protected. **Zero LLM calls**, target = 55 % of the context. | **Yes** — the `litellm_content_retrieve` tool is *not* injected and the originals are *not* kept: nothing in this stack serves that tool, so a stub is gone for good. That is why C comes after B. |
| **T** | **Split + compress the oversized last `user` message** — the mass often sits in the agent's compiled prompt, which B/C/E keep verbatim. Anchors are preserved verbatim (head 3000 chars / tail 6000 chars, the client's current request lives at the tail) and the middle is chunked (max 8; 90k tokens for cloud workers, up to `COMPACTOR_INPUT_TOKENS` — **163 840** for the local compactors, measured in *estimated tokens* not chars) and summarised by the **request's own model**. A structured note marks the compression. | Yes (middle only) |
| **T2** | **Compress the messages after the last `user` message in place** (tool results, assistant turns) — B/C/E keep the whole tail verbatim and T only touches the last `user` message, so when the mass sits in oversized tool results none of them can help (incident 2026-10-06: 126 640-token tail of which only 842 tokens in the last `user` message). Biggest message first, one at a time, re-estimating after each until the estimate is ≤ **95 % of the context**; only the text `content` is replaced with a summary note — roles, `tool_call_id` and `tool_calls` stay intact so no orphaned tool result 400s. | Yes |
| **E** | Map/reduce over the whole history: ≤ 90k-token chunks (max 8, same local cap as T), each summarised by the **request's own model** (`MAP_TIMEOUT_S = 420 s` each — a 163k-token chunk on the 3060 is ~230 s of prefill + synthesis, measured), partials re-placed before the current request. | Yes |
| **T3** | **Deterministic truncation, no model involved**: keeps head (35 %) and tail of the oversized messages after the last `user` message (and the middle of the last `user` message) and drops the centre, re-estimating after each cut until the estimate is ≤ **95 % of the context**. Last chance before the 413 — it is the only level that *always* gains when T/T2 cannot (short last `user`, compactor down/OOM/quarantined). Incident 2026-11-20 17:23: tail 260 578 tokens, last `user` 536, `small-model` dead → T skipped, T2 `no-gain`, deterministic 413. | Yes (middle of tool results dropped) |
| **F** | Nothing freed space → if the **prompt alone fits** (`est ≤ context`) this is *not* a `prompt_too_large`: `max_tokens` is **forced to the room left** (`context − est − 1024`, flat margin, floor `MIN_FORCED_OUTPUT = 256` — below it the real 413 stands) and the request **is sent**: our 413 was deterministic and killed every client retry, while an upstream 400 is recoverable through the fallback chain and Strata shortens the completion itself (`fit_max_tokens`). Only when the prompt genuinely exceeds the context → **413** with a structured detail (`model`, `context_tokens`, `prompt_tokens_estimated`, `tried`, `hint`). | — |

**Tail-alone shortcut**: B/C/E preserve the tail (last `user` message onward)
verbatim, so if the tail saturates the context *by itself* those levels cannot
help — the cascade tries only T, T2 and T3, then goes straight to F, avoiding
guaranteed useless cloud calls.

**Compaction headroom** (`COMPACTION_HEADROOM_PCT = 0.05`, 2026-11-20):
T/T2/T3 stop at `ctx × 0.95`, not at the very edge of the context. Stopping
"just under" is what let the estimator's residual error decide between a 200
and an upstream `400 prompt leaves no room` (measured: T2 estimated 214,439,
the engine counted 301,397). 5 % of a 262k window is ~13k tokens of cushion;
combined with the digit-corrected estimator the compacted payload now lands at
~95 % estimated ≈ ~93 % real.

**Local compactors summarise with reasoning disabled**: `_brief_call` sends
`extra_body={"chat_template_kwargs": {"enable_thinking": false}}` when the
compactor is one of `COMPACTOR_INPUT_TOKENS` (llama.cpp only; verified to pass
through litellm → llama-gate → llama-swap → llama-server). With reasoning on,
a 163k-token chunk spent the whole budget thinking and returned **empty
content** at both `max_tokens` 4096 and 8192 (`finish=length`), so the chunk
was lost on every request; with it off the same chunk returns `finish=stop`
with a 4.5k-char brief in 294 s.

Static per-model contexts: synthetic models are explicit entries in
`config.yaml`, and `CONTEXT_BY_PREFIX` mirrors their real windows exactly
(`syn:large:*` 524 288, `syn:small:text` 196 608, `syn:small:vision` 262 144,
…), matched by exact id before the group floor — the clamp no longer depends
on a completed listing sweep to know a synthetic model's true context.

What is preserved at every level: the **system** messages verbatim, and the
**tail from the last `user` message onward** verbatim (cutting there keeps the
current tool exchanges intact — an orphaned `tool` result would 400).

The cascade runs **only in the async hook** (B, T and E make LLM calls); the
sync `pre_call_hook` applies policy + clamp only. Internal calls are marked with
litellm's own `INTERNAL_CALL_ORIGIN_METADATA_KEY`, which is the recursion
guard (a compaction call must never be compacted again) and keeps them out of
spend logs, rate limits and cooldowns.

## Semantic caching (opt-in)

An optional **semantic cache** (Redis + vector similarity, similar prompts
share a cached answer) is wired into the proxy. It is **OFF by default** and
that default applies to **every** model, so coding agents — which must always
see a fresh answer — are unaffected unless they ask for it. It turns **ON
per-request**, with no extra keys and no second endpoint, only when the client
sends an explicit header:

```
Cache-Control: cache
```

`store` works too. Tokens are matched **exactly** (comma-separated), so
`no-store`/`no-cache` do **not** enable it. The header is translated by
`reasoning_clamp._apply()` into litellm's per-request `cache` flags
(`no-cache` skips the read, `no-store` the write): litellm on its own only
understands `Cache-Control` for `s-maxage` → `ttl`.

Two policies for the same proxy, same key:

| Client | Header | Semantic cache |
|---|---|---|
| coding agent / normal chat | *(none)* | **OFF** (default) |
| project with repetitive requests | `Cache-Control: cache` | **ON** |

```bash
# OFF (default) -> a perfectly normal, uncached call
curl http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer $LITELLM_KEY" -H "Content-Type: application/json" \
  -d '{"model":"synthetic/syn:small:text","messages":[{"role":"user","content":"hi"}]}'

# ON -> this request reads AND writes the semantic cache
curl http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer $LITELLM_KEY" -H "Content-Type: application/json" \
  -H "Cache-Control: cache" \
  -d '{"model":"synthetic/syn:small:text","messages":[{"role":"user","content":"hi"}]}'
```

With the OpenAI SDK, just add the header to the client's default headers. On a
hit litellm returns `x-litellm-cache-key` and `x-litellm-semantic-similarity`
(a miss reports `0.0`); look for them to confirm the cache is live.

Details / constraints:

- **Backend**: the shared Redis at `/opt/docker/compose/redis` (Redis 8 with
  the `search` module), **DB 0**, index `litellm_semantic_cache`, key prefix
  `litellm_semantic_cache:`, `ttl: 86400`. DB 0 is **mandatory**: RediSearch /
  RedisVL refuses `FT.CREATE` on any `db != 0` and the index is db0-global;
  isolation from other data is by key prefix. The `ttl` is mandatory too — that
  Redis runs with `--maxmemory-policy volatile-lru`, which evicts **only** keys
  that have a TTL.
- **Embedding**: only `embedding-model` (Qwen3-Embedding-0.6b, 1024-dim, local
  via llama-gate), with **no fallback** — a different embedding model lives in
  a different vector space and would yield false hits. If `embedding-model` is
  down the cache simply misses (never a wrong answer).
- **Scope**: `supported_call_types: [acompletion]` → `/chat/completions` only
  (not embeddings/media). `similarity_threshold: 0.85` is strict: identical or
  near-identical prompts hit, looser paraphrases miss.
- Caching applies only to requests litellm is willing to cache at all (e.g.
  ≤ `max_messages`, default 4 messages).
- To disable semantic caching **entirely**, set `cache: false` in
  `config/config.yaml` (then `sudo systemctl restart docker-compose@litellm`).

## Deployment

```bash
cp config/config.example.yaml config/config.yaml   # then fill in the keys
sudo systemctl restart docker-compose@litellm
```

`docker-compose.yml` expects:
- l'immagine **`litellm-headroom:local`** (build di `Dockerfile.litellm`):
  `docker compose build litellm` prima del primo `up`, oppure lascia che
  `up` la costruisca quando manca. `manage-docker` la ricostruisce da sola
  quando cambia il base image o esce una nuova `headroom-ai` su PyPI;
- **llmtrim non e' piu' richiesto** (disattivato il 2026-11-14): il suo CA
  bundle resta montato SOLO nei gateway (`GW_CA`, che `gw.py` richiede
  all'avvio) e non e' piu' usato da litellm;
- llama-swap on `blastpc` (192.168.1.29, port 11435, **plain HTTP** since
  2026-10-02 — `llama-gate` runs with `GATE_TLS=auto`, so it also works if
  llama-swap goes back to `https` self-signed: it terminates the TLS itself;
  litellm always talks plain HTTP to the gate);
- a `.env` in this directory with `POSTGRES_PASSWORD` (gitignored, created
  once: `printf 'POSTGRES_PASSWORD=%s\n' "$(openssl rand -hex 24)" > .env`).
  `docker compose up` fails loudly if it is missing.

### PostgreSQL (`litellm-db`)

`litellm` runs with `DATABASE_URL` pointing at the `litellm-db` service
(`postgres:17-alpine`, **no published port**, named volume `litellm_pgdata`).
Without a DB, spend logs have to be disabled (`disable_spend_logs: true`) or
they pile up in RAM; with a DB they are persisted and virtual keys, users,
teams and `/ui` work.

- Migrations: the proxy applies them itself on startup (`run_server` →
  `prisma migrate deploy` + `prisma generate`). The **first** boot with an
  empty DB is slow (minutes, ~224 migrations) — wait for `healthy`, do not
  restart in the middle. `depends_on: litellm-db: service_healthy` keeps the
  proxy from opening its pool against a DB that is not listening yet.
- Tuning is deliberately lightweight (host is an 8GB RPi5):
  `shared_buffers=32MB`, `max_connections=20`, `work_mem=1MB`,
  `maintenance_work_mem=16MB`, `jit=off`, `mem_limit: 192m`. Observed after
  the first migration: ~20-35MB for the DB (the `shared_buffers` 32MB are
  most of it) and ~+15MB in litellm (the prisma query-engine child, 11MB).
  `fsync`/`full_page_writes`/`synchronous_commit` stay at their defaults on
  purpose (host swap is heavily used — durability is not worth the risk).
- `database_connection_pool_limit: 5` in `general_settings`: every Postgres
  connection costs a backend process plus `work_mem`.

**Retention** is done by litellm's own scheduler — no cron, no extra
container, no partitioning. Two independent rules, whichever matches first
deletes the row:

| setting | value | meaning |
| --- | --- | --- |
| `maximum_spend_logs_retention_period` | `30d` | delete rows older than 30 days (job runs every `maximum_spend_logs_retention_interval`, default `1d`) |
| `maximum_spend_logs_size` | `2GB` | hard cap on `LiteLLM_SpendLogs` real size (heap + indexes + TOAST) |
| `maximum_spend_logs_cleanup_target_size` | `1GB` | when the cap is exceeded, delete oldest rows down to this |
| `maximum_spend_logs_size_check_interval` | `1h` | how often the size rule is evaluated (`pg_total_relation_size`) |
| `maximum_spend_logs_cleanup_batch_size` / `_max_batches` | `1000` / `20` | rows per `DELETE`, max statements per run |
| `maximum_spend_logs_cleanup_run_budget` / `_batch_timeout` | `1m` / `30s` | wall-clock budget and per-batch `statement_timeout`, so cleanup never monopolises the small pool |

`store_model_in_db` stays **false**: `config/config.yaml` remains the single
source of truth for models (a UI-created model would bypass the wildcard
expansion and the clamp in `reasoning_clamp.py`).

Backup: `docker exec litellm-db pg_dump -U litellm litellm | gzip > dump.sql.gz`
(the volume survives `docker compose down`, and dies with `down -v`).

> **2026-11-14 — llmtrim disattivato.** Il testo sotto descrive il setup
> *storico* e vale solo se llmtrim viene riacceso. Oggi litellm e i gateway
> escono **diretti** (nessun `HTTPS_PROXY`, nessun `GW_PROXY`, nessun
> `SSL_CERT_FILE`); la compressione del contesto la fa `headroom_callback.py`
> dentro litellm. Con il proxy spento e `HTTPS_PROXY` ancora impostato ogni
> chiamata cloud falliva con `OpenAIException - Connection error` e litellm
> rispondeva 500 su quasi tutti i modelli.

~~All outbound HTTPS traffic from litellm and the gateways goes through
llmtrim (`HTTPS_PROXY`/`GW_PROXY`), so provider calls are intercepted,
tracked and compressed just like direct client traffic. The local model is
in `NO_PROXY` (LAN traffic).~~

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

The context reported by `llama-server` is **`--ctx-size / -np`**: `-np 1`, so
`small-model` exposes the whole **196608** (2026-11-23; 229376 before, and the
262144 it declared earlier never loaded — CUDA OOM in the load-time warm-up —
and every compaction call failed, which is what turned a big prompt into a
413). `local-model` is Strata, `--context 196608` since 2026-11-23 (was
229376, and 262144 before that). The upstream rejects oversized requests with its
own 400 (see **Context clamp** above), consistent with `model_info` in config.
A context flip must be kept in sync in **four** places: llama-swap's `cmd`
(`--ctx-size`) and its `capabilities.context`, `config.yaml` (`model_info`),
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
cloudflare-gw/gw.py       # Workers AI gateway (REST listing -> /v1/models, filters, cache)
provider-gw/gw.py         # generic gateway (ID_PREFIX / PATH_PRE / cache)
.env                      # POSTGRES_PASSWORD + UI_USERNAME/UI_PASSWORD — NOT committed
```

## Security

- `config/config.yaml` is in `.gitignore`: it contains real keys. Use
  `config/config.example.yaml` as the template.
- `.env` is in `.gitignore` too: it holds `POSTGRES_PASSWORD` and the proxy UI
  login (`UI_USERNAME` / `UI_PASSWORD`). `docker-compose.yml` only references
  them (`${UI_USERNAME:?...}`), so a missing `.env` fails the boot instead of
  starting the UI without a password.
- No keys are present in the code or the compose file.