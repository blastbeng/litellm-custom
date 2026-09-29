# litellm-custom

Router LLM personale basato su **LiteLLM Proxy** con gateway Python leggeri,
pensa-a-tutto: un solo endpoint OpenAI-compatible (`:4000`) che aggrega
modelli locali (llama-swap su PC dedicato) e provider cloud (synthetic.new,
OpenRouter, Groq, Gemini, Ollama Cloud, deepseek4free), con listing
**dinamico** delle wildcard, policy di fallback e gestione del `reasoning_effort`.

```
                        ┌──────────────────────────── host (blast) ───────────────────────┐
  client (AiderDesk     │                                                                 │
  / curl / qualsiasi    │   ┌─────────────┐   CONNECT    ┌──────────────────────────┐     │
  SDK OpenAI) ──────────┼──►│  litellm    │─────────────►│  llmtrim (MITM :43117)   │─────┼──► internet
  localhost:4000        │   │  :4000      │              │  traccia/comprime LLM    │     │   (api.groq.com,
                        │   └──┬───┬───┬──┘              └──────────────────────────┘     │   openrouter.ai,
                        │      │   │   │                                                  │   generativelanguage,
                        │      │   │   └────────────► llama-gate ──► blastpc:11435          │   api.synthetic.new,
                        │      │   │                  (gate TCP, PC spento → 503)      │   ollama.com)
                        │      │   │                                                  │
                        │      │   ├────► openrouter-gw ─┐                           │
                        │      │   ├────► groq-gw ───────┤ (via llmtrim)             │
                        │      │   ├────► gemini-gw ─────┤                           │
                        │      │   └────► deepseek4free-gw ─┴──► 192.168.1.13:18010   │
                        │      │                                                      │
                        └────────────────────────────────────────────────────────────────┘
```

## Componenti

| Servizio | File | Ruolo |
|---|---|---|
| `litellm` | `config/config.yaml` | proxy/router LiteLLM (porta 4000) |
| `llama-gate` | `llama-gate/gate.py` | gate TCP verso llama-swap (PC spento → 503 in ~2s) |
| `openrouter-gw` | `openrouter-gw/gw.py` | listing OpenRouter: path `/v1/*`→`/api/v1/*` + prefisso id `openrouter/` |
| `groq-gw`, `gemini-gw`, `deepseek4free-gw` | `provider-gw/gw.py` (generico) | listing con `ID_PREFIX`, `PATH_PRE`, cache last-good |
| `reasoning_clamp.py` | callback litellm | cascata/clamp `reasoning_effort` per provider + policy fallback |
| `llmtrim` (esterno) | `/opt/docker/compose/llmtrim` | MITM proxy: traccia/comprime il traffico LLM |

## Come funziona il routing wildcard

Ogni provider è esposto con **una sola wildcard** in `model_list`:

```yaml
- model_name: groq/*
  litellm_params:
    model: openai/*          # il capture sostituisce "*" con l'id nativo
    api_base: http://groq-gw/openai/v1
```

- Nome pubblico `groq/openai/gpt-oss-120b` → all'upstream arriva l'id
  nativo `openai/gpt-oss-120b` (il capture rimuove il prefisso).
- Con `litellm_settings.check_provider_endpoint: true`, `/v1/models` e
  `/model/info` espongono la **lista reale** dell'endpoint: se i modelli del
  provider cambiano, litellm è sempre aggiornato (fondamentale per
  `deepseek4free/*`, la cui lista è dinamica).

### Perché i gateway

LiteLLM (v1.104.0) ha tre comportamenti che i gateway aggirano:

1. **Listing sempre su `{origin}/v1/models`** ignorando il path dell'api_base
   → chiavi di path diverse servono `PATH_PRE` (Groq: `/openai`,
   Gemini: `/v1beta/openai`, OpenRouter: `/api`).
2. **Rename degli id il cui primo segmento è un provider noto**
   (`openai/...`, `deepseek/...`, ...) nel listing → il gateway prefixa gli id
   con `ID_PREFIX` (es. `groq/`, `deepseek4free/`): il rename non scatta e il
   capture ripristina l'id nativo a runtime.
3. **Timeout unico** (httpx, connect incluso) → non si può distinguere
   "PC spento" da "PC acceso con modello enorme in caricamento":
   `llama-gate` risolve (503 in ~2s se il PC non risponde al TCP, altrimenti
   tubo trasparente e litellm può aspettare fino a 30 min).

### Cache last-good (provider-gw, `MODELS_CACHE=1`)

Per upstream instabili (deepseek4free è un progetto in sviluppo): se
`/models` fallisce, il gateway serve l'**ultima lista valida** salvata su
disco (header `X-Models-Cache: stale`). I modelli restano sempre visibili su
litellm anche quando l'endpoint è giù.

## Policy di fallback

- **`synthetic/*`** → `fully-uncensored` (locale), solo su errore del
  provider (busy/offline/rate-limit).
- **Tutti gli altri** → nessun fallback: errore diretto al client.
- **`fully-uncensored`** e **`openrouter/free`** non fanno MAI fallback
  (nemmeno su se stessi).

⚠️ Le chiavi dei `fallbacks` in litellm **non supportano wildcard**
(`get_fallback_model_group`: solo match esatto / provider-stripped / `"*"`),
quindi la policy per-modello è implementata nella callback
`reasoning_clamp.py` tramite `litellm_params.fallbacks` **per-request**, che
overridea la lista di config (`router.py`: `kwargs.get("fallbacks", self.fallbacks)`).
In `config.yaml` restano solo le disabilitazioni esplicite.

## Gestione di `reasoning_effort` (cascata)

Regola: default `xhigh` dove valido, altrimenti si scende (`high`, `medium`, ...).

| Provider | Default | Note |
|---|---|---|
| `synthetic/*`, `openrouter/*`, `ollama-cloud/*` | `xhigh` (in config) | accettato |
| `gemini/*` | `high` (callback) | `xhigh` non valido su Gemini; `none/minimal/low/medium` rispettati; `xhigh/max` → clamp a `high` |
| `groq/*` | `high` (callback) | solo per i modelli reasoning (`openai/gpt-oss-*`, `qwen/*`); per gli altri (`allam-2-7b`, whisper, ...) il parametro viene **rimosso** (risponderebbero 400) |
| `deepseek4free/*` | nessuno | endpoint non validabile per-modello |

Il valore passato dal client ha sempre precedenza (con clamp se non valido).

## Deploy

```bash
cp config/config.example.yaml config/config.yaml   # poi inserire le chiavi
sudo systemctl restart docker-compose@litellm
```

`docker-compose.yml` si aspetta:
- `llmtrim` in ascolto su `:43117` con CA in
  `/opt/docker/compose/llmtrim/data/.llmtrim/ca-bundle.pem`
  (il bundle è root-CA di sistema + CA llmtrim, sync via `llmtrim-ca-sync`);
- llama-swap su `blastpc` (192.168.1.29, porta 11435) per il modello locale;
- il servizio `deepseek4free` su 192.168.1.13:18010 (facoltativo).

Tutto il traffico HTTPS in uscita da litellm e dai gateway passa per
llmtrim (`HTTPS_PROXY`/`GW_PROXY`), così le chiamate ai provider sono
intercettate, tracciate e compresse come quelle dei client diretti.
Il modello locale è in `NO_PROXY` (traffico LAN).

### Nota su llama-server (modello locale)

Il context dichiarato da `llama-server` è **`--ctx-size / -np`**: con
`-np 2` e `--ctx-size 262144` ogni slot espone 131072 token e litellm
rifiuta richieste grandi con `ContextWindowExceededError`. Con `-np 1`
si ha il context pieno (262144), coerente con `model_info` in config.

## File

```
config/config.yaml        # config REALE (chiavi) — NON committata
config/config.example.yaml
docker-compose.yml
reasoning_clamp.py        # callback litellm (reasoning + fallback policy)
llama-gate/gate.py
openrouter-gw/gw.py
provider-gw/gw.py         # gateway generico (ID_PREFIX / PATH_PRE / cache)
```

## Sicurezza

- `config/config.yaml` è in `.gitignore`: contiene chiavi reali. Usare
  `config/config.example.yaml` come template.
- Nessuna chiave è presente nel codice o nel compose.