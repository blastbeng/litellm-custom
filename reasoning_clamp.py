"""Callback litellm: gestione/cascata di reasoning_effort per provider.

Regola dell'utente: default "xhigh" dove supportato, ma se non valido ->
"high"; se anche high non fosse valido -> "medium"; ecc. Il valore che
passa il client (es. da AiderDesk) ha SEMPRE precedenza, con clamp:
xhigh/max vengono ridotti al massimo valore supportato dal provider.

- groq/*   : i modelli reasoning (openai/gpt-oss-*, qwen/*) accettano solo
  low/medium/high -> xhigh/max clamped a "high", default "high".
  Gli ALTRI modelli Groq (allam-2-7b, whisper-*, prompt-guard) rispondono
  400 a QUALSIASI reasoning_effort -> parametro rimosso.
- gemini/* : valori validi high/medium/low/minimal/none ("xhigh" non
  esiste) -> xhigh/max clamped a "high"; default "high" se il client non
  passa nulla (gemini e gemma lo accettano tutti).
- Gli altri modelli (synthetic/*, openrouter/*, ollama-cloud/*, locale)
  non vengono toccati: il loro default xhigh sta gia' in config.yaml.

POLICY FALLBACK per-modello (richiesta utente 2026-10-02, wildcard-safe):
le chiavi dei fallback in config.yaml NON supportano wildcard (litellm
get_fallback_model_group: solo esatto/stripped-provider/"*"), quindi la
policy e' applicata QUI per-request tramite litellm_params.fallbacks, che
OVERRIDE la lista di config (router.py: kwargs.get("fallbacks",
self.fallbacks)):

  REGOLA: small-model e' l'ULTIMO fallback di TUTTI i modelli, raggiunto
  SOLO dopo che tutti gli altri fallback sono miseramente falliti
  (run_async_fallback walka la catena in ordine e attempted_targets
  impedisce ripetizioni e loop).

- synthetic/*            -> ["local-model", "small-model"]
  (stack: llama-swap locale PRIMO, in FINALE small-model, il secondo
  modello llama-swap)
- synthetic/syn:small:text e synthetic/syn:small:vision -> ["small-model"]
  (richiesta utente 2026-10-04: ALLINEATI, UN SOLO target che e'
  small-model - niente local-model, niente Ollama locale (rimossa
  dappertutto). Piccolo->piccolo: small-model e' la capacita' piu' vicina
  a syn:small:*. syn:small:VISION su small-model (text-only su llama-swap)
  puo' rispondere 400 sull'input vision: scelta esplicita dell'utente,
  la catena resta questa)
- synthetic/hf:nomic-ai/nomic-embed-text-v1.5 -> ["embedding-model"]
  (richiesta utente 2026-10-04: modello EMBEDDING su synthetic.new, endpoint
  /v1/embeddings; quando non disponibile cade sul modello omonimo di
  llama-swap, Qwen3-Embedding-0.6b. UN solo target: gli altri modelli
  llama-swap sono CHAT e su una richiesta embedding risponderebbero 400)
- local-model       -> ["small-model"] (small-model come ultima spiaggia;
  l'Ollama locale e' stata rimossa dappertutto, richiesta utente 2026-10-04)
- small-model       -> [] (nessun auto-fallback: e' lui l'ultimo ricorso
  di tutti gli altri, riprovarlo da solo non avrebbe senso)
- embedding-model   -> [] (nessun auto-fallback: e' l'unico modello
  embedding del fallback chain, un target CHAT su una richiesta embedding
  risponderebbe 400 - stesso motivo di small-model)
- TUTTI gli altri (groq/*, gemini/*, openrouter/*, openrouter/free,
  ollama-cloud/*, inference4free/* e QUALSIASI prefisso futuro)
  -> ["small-model"]: nessun gradino intermedio, small-model e' l'ultimo
  ricorso diretto. La lista di config.yaml mantiene la catch-all
  "*": ["small-model"] come rete di sicurezza (ombreggiata da questo
  override per-request su ogni modello).

Il provider e' dedotto dal nome pubblico del modello (prefisso "groq/",
"gemini/"); si copre anche la forma nativa ("models/gemini-...") per
robustezza rispetto all'ordine hook/routing.

CLAMP DI max_tokens (contesto): il limite LO IMPONE L'UPSTREAM, non litellm.
i modelli grossi locali girano su STRATA (dietro llama-swap), che conta i
token del prompt col tokenizer REALE e risponde 400
("prompt (N tokens) + max tokens (M) exceeds the context (C); requests are
never truncated" - il testo NON e' di litellm, che con
enable_pre_call_checks: false non fa alcun controllo), e litellm incapsula il
400 come BadRequestError: cosi' una richiesta con prompt grande uccide la
CATENA DI FALLBACK intera (synthetic/* -> local-model -> small-model).
Strata puo' anche accorciare max_tokens DA SOLO (esatto, senza stima):
"fit_max_tokens": true in strata-<model>.json sul PC (o About -> Model
settings nella sua web UI) - restano da clampare i cloud (synthetic/* ecc.).
Qui max_tokens viene ridotto a "contesto - prompt" usando
il CONTESTO MINIMO dell'intera catena di fallback: l'hook del proxy gira
UNA sola volta per richiesta client (litellm unisce i litellm_params del
deployment a ogni tentativo, fallback inclusi, SENZA ripassare dall'hook -
verificato su router.py/fallback_event_handlers.py v1.104.0), quindi il
tetto deve valere per ogni hop della catena. Stima CONSERVATIVA dei token
del prompt: ratio caratteri/token ADATTIVO alla densita' del testo (2.8 per
codice/JSON/CJK - sovrastimare e' sicuro, sottostimare produce il 400
dell'upstream - fino a 4.6 per la prosa, dove il fisso 2.8 sovrastimava ~2x
e metteva in cascata richieste che stavano nel contesto) con secondo parere
tiktoken (gia' dentro litellm) sui soli prompt grandi.

CASCATA OVERFLOW (richiesta utente 2026-10-02): quando il prompt da solo
satura il contesto (prompt + MIN_OUTPUT_TOKENS + margine > contesto) il
clamp NON puo' aiutare - non c'e' spazio per la risposta - e la catena di
fallback muore come nel log. In quel caso l'async hook riduce il PROMPT con
una cascata a livelli, e dopo ogni livello riusa la stima per il clamp:

  B  compattazione della STORIA (tutto cio' che precede l'ultimo user, che
     resta verbatim) con un modello a contesto grande. Chiamata interna
     via llm_router, senza fallback e senza retry; si passa al compattore
     successivo se uno fallisce/tiempo scaduto o se il suo contesto non
     contiene il corpo.
  C  compressione DETERMINISTICA con litellm.compression.compress()
     (BM25: sostituisce i messaggi a bassa rilevanza con stub, protegge
     system/ultimo user/ultimo assistant, 0 chiamate LLM).
     NOTA: il tool di retrieval (litellm_content_retrieve) NON viene
     iniettato e la cache degli originali NON viene trattenuta: in questo
     stack non c'e' un agentic loop che lo serva, quindi lo stub e'
     PERSO (compressione lossy). E' il motivo per cui C viene DOPO B.
  T  SPLIT+COMPRESS DELLA CODA (richiesta utente 2026-10-06: "quando il
     contesto supera, splittalolo e comprimilo con un LLM"): quando la
     massa sta nell'ULTIMO messaggio user (il prompt compilato
     dall'agente), B/C/E non possono liberarla perche' la coda resta
     verbatim. T tiene verbatim solo le ANCORE (inizio ~3000 char, fine
     ~6000 char, dove sta la richiesta corrente) e riassume il mezzo a
     chunk. Se la coda satura DA SOLA, B/C/E vengono saltati del tutto.
  E  map/reduce: l'intera storia viene tagliata in chunk e riassunta, poi
     le sintesi vengono ricomposte davanti alla richiesta corrente (che
     resta fedele). Ultima spiaggia perche' e' lossy e costa N chiamate.
  F  se nessun livello libera spazio: 413 con dettaglio strutturato
     (modello, contesto, stima, livelli provati, hint) invece di far
     partire la richiesta e farla morire sulla catena di fallback.

POOL OPERARI DINAMICO (richiesta utente 2026-10-06: "cascade model
selection, exploit ANY possible litellm configured model, even the ones
hosted by inference4free"): T, E e i compattori di B non usano piu' liste
chiuse. _worker_pool enumera OGNI modello configurato su litellm (model_list
del router: wildcard inclusi, es. inference4free/*), ordina per preferenza
(piccoli/veloci prima per T/E, contesto grande prima per B), esclude i
non-chat (embedding/whisper/guard/rerank: risponderebbero 400) e quelli il
cui contesto non contiene il chunk. Failover per-chunk: il primo operario
che riesce resta "appiccicoso" per i chunk successivi, ogni fallimento/
risposta vuota avanza al successivo, finche' non si esaurisce il budget di
tempo WORKER_TIME_BUDGET_S (niente piu' tetto fisso di operari: si prova
TUTTO il pool, richiesta utente 2026-10-06).

La cascata gira SOLO nell'async hook (B, T ed E fanno chiamate LLM); il
pre_call_hook sync applica solo policy/clamp, senza rete. Le chiamate
interne sono marcate con il metadata interno di litellm
(INTERNAL_CALL_ORIGIN_METADATA_KEY): serve da guardia anti-ricorsione
(l'hook del proxy gira una sola volta per richiesta client, ma le
chiamate interne non devono essere ri-compactate) e le esclude da
spend logs / rate limit / cooldown.

Registrato in config.yaml come: litellm_settings.callbacks -> "reasoning_clamp.rclamp"
(montato in /app/reasoning_clamp.py, vedi docker-compose.yml).
"""
import asyncio
import hashlib
import json
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

from fastapi import HTTPException
from litellm.integrations.custom_logger import CustomLogger

GROQ_REASONING = ("openai/gpt-oss", "qwen/")

# Ultimo ricorso di TUTTI i modelli (richiesta utente 2026-10-02): piccolo,
# locale, 503 in ~2s col PC spento. small-model non cade MAI su se stesso
# (ramo esatto in _apply): riprovarlo da solo non avrebbe senso.
FINAL_FALLBACK = "small-model"

# synthetic/syn:small:text e synthetic/syn:small:vision (richiesta utente
# 2026-10-04: ALLINEATI): UN SOLO fallback, small-model - niente local-model,
# niente Ollama locale (rimossa dappertutto). Piccolo->piccolo: small-model e'
# la capacita' piu' vicina a syn:small:* (contesto 131072, stessa gate di
# local-model -> 503 in ~2s col PC spento). syn:small:VISION su small-model
# (text-only su llama-swap) puo' rispondere 400 sull'input vision: scelta
# esplicita dell'utente, la catena resta questa minima.
SYN_SMALL_TEXT = "synthetic/syn:small:text"
SYN_SMALL_VISION = "synthetic/syn:small:vision"
SYN_SMALL_FALLBACKS = ["small-model"]

# synthetic/hf:nomic-ai/nomic-embed-text-v1.5 (richiesta utente 2026-10-04):
# modello EMBEDDING su synthetic.new, servito dall'endpoint /v1/embeddings
# ("api.synthetic.new/openai/v1/embeddings", non compare nel listing chat
# di synthetic). Quando non disponibile cade su llama-swap: "embedding-model"
# (Qwen3-Embedding-0.6b su CPU, contesto 16384). Catena a UN solo target:
# gli altri modelli llama-swap sono CHAT e su una richiesta embedding
# risponderebbero 400 - nessun gradino dopo.
SYN_EMBED = "synthetic/hf:nomic-ai/nomic-embed-text-v1.5"
SYN_EMBED_FALLBACKS = ["embedding-model"]

# --- contesti (model_info.max_input_tokens di config.yaml) ---
# Serve la mappa QUI perche' i modelli sono wildcard/custom:
# litellm.get_max_tokens("local-model") fallisce ("isn't mapped yet") e il
# limite lo impone l'upstream. Valori ALLINEATI a config(.example).yaml:
# un contesto cambiato li' va cambiato anche qui.
CONTEXT_BY_PREFIX = (
    ("local-model", 131072),
    ("small-model", 131072),
    ("embedding-model", 16384),  # llama-swap: Qwen3-Embedding-0.6b (CPU)
    ("synthetic/", 131072),
    ("inference4free/", 131072),
    ("ollama-cloud/", 262144),
    ("openrouter/", 262144),
    # NOTA ordine: gli entry specifici PRIMA del prefisso generico "groq/"
    # (la ricerca si ferma al primo match). Su Groq i gpt-oss hanno ctx
    # 131072 (console.groq.com/docs/model), NON 262144 come il tetto del
    # wildcard (kimi-k2-instruct-0905 e' l'unico 262k): senza l'entry
    # specifico il pool dei compattori (B) manderebbe un corpo da 240k
    # token a un modello che lo rifiuterebbe a runtime.
    ("groq/openai/gpt-oss-120b", 131072),
    ("groq/openai/gpt-oss-20b", 131072),
    ("groq/", 262144),
    ("gemini/", 262144),
)
DEFAULT_CONTEXT = 131072  # il piu' piccolo: clamp conservativo per l'ignoto
MIN_OUTPUT_TOKENS = 1024  # sotto questo una risposta reasoning non e' utile
SAFETY_MARGIN = 1024      # margine per l'errore della stima
# F-EVIT: ultima spiaggia prima del 413 - se il PROMPT da solo ci sta
# (est <= ctx) ma la stanza residua e' piccola, si forza un max_tokens
# ridotto e la richiesta PARTE: l'upstream accorcia (fit_max_tokens su
# Strata, llama-swap auto-limita). Sotto questa soglia di output non ha
# senso partire -> 413 vero.
MIN_FORCED_OUTPUT = 256
CHARS_PER_TOKEN = 2.8     # conservativo per llama (codice, CJK, JSON)
PER_MESSAGE_OVERHEAD = 8  # role/framing per messaggio (ChatML)
# SCHEMI TOOL: NON sono nel conteggio characters/dei messaggi e l'upstream li
# mette nel prompt: misurato su Strata, 18 schemi OpenAI = +1403 token REALI
# (~80/schema). Un client agentico (AiderDesk) ne manda decine: non contarli
# e' una sottostima sistematica di migliaia di token -> il clamp lascia un
# max_tokens che l'upstream rifiuta. CONTEGGIO: tiktoken (cl100k, via
# _tools_tiktoken, CACHE sha1 - gli schemi sono identici a ogni richiesta)
# SE e SOLO SE viene BASSO del ratio; il ratio resta solo fallback (tiktoken
# assente) e floor del framing per-tool. INCIDENTE 2026-10-06: ~100 schemi
# MCP (~250k char) stimati /2.2 = ~110-120k token contro ~70-85k REALI
# (tokenizer litellm su C: 71796) -> la cascata B/E non muoveva piu' la
# stima (gli schemi restano verbatim) e il 413 partiva con il prompt che
# INVECE ci stava (est 126805 <= 131072): i 2.2 erano calibrati su 18
# piccoli schemi Strata e sovrastimano ~2x sui grandi schemi MCP.
TOOLS_CHARS_PER_TOKEN = 3.0
TOOL_FIXED_TOKENS = 32
# IMMAGINI: content list con image_url/image - costo fisso conservativo
# (le immagini multi-tile di llama.cpp possono valere piu' di 1000 token)
IMAGE_TOKEN_COST = 1500
# ERRORE RESIDUO della stima: PROPORZIONALE, non fisso. Misurato: su
# trascrizioni reali ~100k token (codice+JSON, chat template) chars/2.8 resta
# ~1.4-1.8% SOTTO il conteggio reale dell'upstream, quindi il vecchio margine
# fisso 1024 non bastava e il clamp produceva un max_tokens oltre la stanza
# rimasta (400 "exceeds the context" + catena fallback morta). 3% copre
# l'errore osservato con margine 2x.
PROMPT_ERR_PCT = 0.03
MAX_TOKEN_KEYS = ("max_tokens", "max_completion_tokens")

# --- cascata overflow: B -> C -> T -> E -> F ---
# B: ordine PREFERITO dei compattori (modelli a contesto grande). Da
# ottobre 2026 la lista non e' piu' chiusa: se nessun preferito e'
# disponibile/contiene il corpo, _worker_pool apre il pool a OGNI altro
# modello configurato su litellm che possa contenere il corpo (richiesta
# utente 2026-10-06: "exploit any possible litellm configured model").
COMPACTOR_MODELS = (
    "groq/openai/gpt-oss-120b",
    "gemini/models/gemini-2.5-flash-lite",
    "openrouter/auto",
)
COMPACTOR_TIMEOUT_S = 45
COMPACT_OUTPUT_TOKENS = 4096   # tetto della sintesi prodotta dal compattatore
COMPACT_MIN_CHARS = 400        # sotto questo il corpo non merita una chiamata
MIN_BRIEF_CHARS = 200          # sintesi piu' corta accettata (sotto = fallita)
COMPACT_ORIGIN = "context_compaction"
COMPRESS_TARGET_RATIO = 0.55   # C: target = 55% del contesto (margine risposta)

# T/E: ordine PREFERITO degli operari di riassunto (piccoli/veloci prima).
# Il pool REALE e' dinamico (_worker_pool): OGNI modello configurato su
# litellm (wildcard inclusi, es. inference4free/*) entra nel pool se il suo
# contesto contiene il chunk; l'ordine PRIMARIO e' il tier di costo
# (gratis -> economico -> costoso, vedi _model_tier) e i preferiti decidono
# SOLO l'ordine DENTRO ogni tier.
SUMMARIZER_PREFERRED = (
    "small-model",
    "groq/openai/gpt-oss-20b",
    "gemini/models/gemini-2.5-flash-lite",
    "synthetic/syn:small:text",
    "openrouter/free",
    "inference4free/",
    "groq/openai/gpt-oss-120b",
    "gemini/",
    "synthetic/",
    "ollama-cloud/",
    "openrouter/",
)
# non-chat: a una richiesta di riassunto questi modelli risponderebbero
# 400 (embedding/whisper/guard/rerank non sono completations testuali)
NON_CHAT_WORKER_SUBSTRINGS = ("embed", "whisper", "guard", "rerank")
# BUDGET DI TEMPO (richiesta utente 2026-10-06: "provare TUTTI i modelli
# possibili con un limite di tempo, cosi' lo user non aspetta troppo; errore
# o livello successivo SOLO quando proprio non riusciamo a fare nulla"): non
# esiste piu' un tetto fisso di operari - ogni livello che chiama LLM prova
# l'intero pool (gratis -> economici -> costosi) finche' c'e' budget. I
# fallimenti rapidi (connessione rifiutata, 400) consumano pochi secondi; il
# budget taglia solo i worker APPESI. B senza sintesi e T/E con ALMENO un
# chunk riassunto non buttano il lavoro fatto: si tengono i parziali.
WORKER_TIME_BUDGET_S = 420     # secondi TOTALI per livello di cascata
CHEAP_PRICE_USD_PER_TOKEN = 5e-7  # soglia "economico" dal listing (~$0.50/M
                               # di prompt): sotto = TIER_CHEAP, sopra = paid

# TIER DI COSTO per l'ordine del pool (richiesta utente 2026-10-06: prima i
# GRATIS, poi gli ECONOMICI, per ultimi i costosi). Classificazione DINAMICA:
# i provider delistano/listano modelli continuamente, quindi niente elenchi
# chiusi di id - il tier si deduce dal PREZZO del listing del gateway quando
# c'e' (hint), dalla convenzione ':free' e dal brand del provider, e dal nome
# per gli economici noti. _worker_pool ordina per (tier, preferenza): le
# liste preferred di B e T/E decidono solo l'ordine DENTRO ogni tier.
TIER_FREE = 0    # gratis: provider con "free" nel brand (inference4free/*),
                 # openrouter/free, id ":free", listing a prezzo zero,
                 # ollama-cloud (quota gratuita), hardware locale
TIER_CHEAP = 1   # economici: gpt-oss, flash/flash-lite, gemma, qwen, llama,
                 # deepseek, haiku, glm, kimi, small/mini/nano, phi...
TIER_PAID = 2    # tutto il resto (gemini-pro, claude sonnet/opus, gpt-5...)
# NB "-mini" col trattino: "gemini" CONTIENE "mini" e classificherebbe
# tutti i Gemini come economici
CHEAP_PATTERNS = (
    "gpt-oss", "flash-lite", "flash", "lite", "gemma", "syn:small", "small",
    "qwen", "qwq", "nano", "-mini", "haiku", "deepseek", "glm", "kimi",
    "moonshot", "llama", "mistral-small", "ministral", "pixtral", "minimax",
    "phi-", "phi3", "phi4", "nemotron", "falcon", "tiny",
    "grok-3-mini", "grok-4-fast",
)
_TIER_HINTS = {}  # id dal listing -> prezzo USD/token del PROMPT (scritto da
                  # _fetch_pattern_models): 0 = gratis, sotto la soglia
                  # CHEAP_PRICE_USD_PER_TOKEN = economico, sopra = costoso


def _model_tier(n):
    """Tier di costo di un modello: 0 gratis, 1 economico, 2 costoso.
    Segnali in ordine: PREZZO dal listing del gateway (dinamico: 0 = gratis,
    sotto CHEAP_PRICE_USD_PER_TOKEN = economico, sopra = costoso - un modello
    nuovo listato dal provider viene classificato senza toccare il codice),
    convenzione ':free' di OpenRouter, brand del provider ("free" nel nome,
    ollama-cloud a quota gratuita), hardware locale, pattern di nome per gli
    economici noti. Tutto il resto finisce nel tier costoso."""
    s = str(n)
    hint = _TIER_HINTS.get(s)
    if hint is not None and hint >= 0:   # prezzo negativo = nascosto: ignora
        if hint == 0:
            return TIER_FREE
        if hint <= CHEAP_PRICE_USD_PER_TOKEN:
            return TIER_CHEAP
        return TIER_PAID
    if ":free" in s:
        return TIER_FREE
    brand = s.split("/", 1)[0].lower()
    if ("free" in brand or brand == "ollama-cloud"
            or s == "openrouter/free"):
        return TIER_FREE
    if s in ("small-model", "local-model"):
        return TIER_FREE
    low = s.lower()
    if any(p in low for p in CHEAP_PATTERNS):
        return TIER_CHEAP
    return TIER_PAID

# ESPANSIONE WILDCARD -> nomi concreti (listing dei gateway, cache TTL).
# Il model_list di litellm contiene pattern ("groq/*"): per una chiamata
# interna serve un NOME CONCRETO ("groq/openai/gpt-oss-120b"), altrimenti
# l'upstream riceve model="*" e risponde 400/404. Il listing di ogni
# gateway (openrouter-gw, groq-gw, inference4free-gw prefigissano gia' gli
# id; synthetic/ollama sono upstream diretti -> prefisso aggiunto qui) e'
# la stessa fonte usata dall'import automatico dei modelli in config.
WILDCARD_LIST_TIMEOUT_S = 8
WILDCARD_CACHE_TTL_S = 600
WILDCARD_MAX_MODELS = 40       # tetto per-pattern (listing da centinaia)
_WILDCARD_CACHE = {}           # "groq/*" -> (monotonic, [nomi concreti])

# ------------- INTERROGAZIONE DINAMICA del listing /v1/models -------------
# Richiesta utente 2026-10-06: "Have you correctly interrogated litellm
# models list (http://...:4000/v1/models)? adapt the code to our model
# list DINAMICALLY. that interrogation must be DINAMIC".
# Fino ad oggi NO: il pool leggeva solo llm_router.model_list (+ i listing
# dei gateway per espandere i wildcard). ORA il listing del PROXY STESSO
# (/v1/models, la stessa fonte interrogata da fuori su 192.168.1.13:4000)
# viene interrogato e usato per: (a) l'universo dei modelli del pool di
# operari, (b) i contesti REALI (max_input_tokens/max_output_tokens) al
# posto delle stime statiche. DINAMICO: un thread daemon di fondo rilegge
# il listing ogni PROXY_MODELS_TTL_S -> un modello aggiunto/tolto in
# config.yaml (o via /model/new) viene visto senza toccare il codice ne'
# riavviare il proxy. Il tier di costo resta deciso da _model_tier (prezzi
# sui listing dei GATEWAY: /v1/models non espone prezzi, verificato).
# DEADLOCK: una GET bloccante a se stessi DENTRO l'event loop del proxy
# blocca tutto (il loop serve proprio quella richiesta) -> la fetch gira
# SOLO nel thread daemon di fondo; chi legge (event loop o thread C) usa
# solo la cache gia' pronta. Listing non ancora pronto -> si usa solo
# model_list: comportamento identico a ieri, mai un blocco.
PROXY_MODELS_URL = os.environ.get(
    "REASONING_CLAMP_MODELS_URL", "http://127.0.0.1:4000/v1/models")
PROXY_MODELS_TIMEOUT_S = 5
PROXY_MODELS_TTL_S = 600
_PROXY_MODELS_CACHE = {"ts": 0.0, "models": {}}  # id -> (max_in|None, max_out|None)
_PROXY_MODELS_LOCK = threading.Lock()
_PROXY_MODELS_STARTED = [False]


def _fetch_proxy_models():
    """GET PROXY_MODELS_URL (BLOCCANTE: chiamare SOLO dal thread daemon).
    Ritorna {id: (max_input_tokens, max_output_tokens)} con None sui campi
    assenti; {} su qualsiasi errore (chi legge resta sull'ultima lista).
    Auth: master_key gia' caricato nel processo proxy (il listing e' 401
    senza) - MAI loggata; 401 -> riprova senza header. Bypass SEMPRE dei
    proxy env: HTTPS_PROXY=llmtrim non deve intercettare 127.0.0.1."""
    try:
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}))
        req = urllib.request.Request(PROXY_MODELS_URL)
        try:
            from litellm.proxy import proxy_server
            mk = getattr(proxy_server, "master_key", None)
            if mk:
                req.add_header("Authorization", "Bearer " + str(mk))
        except Exception:
            pass
        try:
            with opener.open(req, timeout=PROXY_MODELS_TIMEOUT_S) as resp:
                data = json.loads(resp.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as e:
            if e.code != 401:
                raise
            # master_key non (ancora) caricato: il listing locale senza auth
            req = urllib.request.Request(PROXY_MODELS_URL)
            with opener.open(req, timeout=PROXY_MODELS_TIMEOUT_S) as resp:
                data = json.loads(resp.read().decode("utf-8", "replace"))

        def _int(v):
            try:
                return int(v) if v is not None else None
            except (TypeError, ValueError):
                return None
        models = {}
        for m in (data.get("data") or []):
            if not isinstance(m, dict):
                continue
            mid = str(m.get("id") or "").strip()
            if not mid:
                continue
            models[mid] = (_int(m.get("max_input_tokens")),
                           _int(m.get("max_output_tokens")))
        return models
    except Exception:
        return {}


def _refresher_proxy_models():
    """Thread daemon: rilegge /v1/models ogni PROXY_MODELS_TTL_S (dinamico:
    config.yaml cambiata o /model/new vengono visti senza riavviare)."""
    fails = 0
    while True:
        got = _fetch_proxy_models()
        with _PROXY_MODELS_LOCK:
            if got:
                if fails or not _PROXY_MODELS_CACHE["models"]:
                    print(f"[reasoning_clamp] listing /v1/models: "
                          f"{len(got)} modelli (contesti reali; i PREZZI/"
                          f"tier restano sui listing dei gateway)", flush=True)
                _PROXY_MODELS_CACHE["models"] = got
                _PROXY_MODELS_CACHE["ts"] = time.monotonic()
                fails = 0
            else:
                fails += 1
                if fails == 1 or fails % 10 == 0:
                    print(f"[reasoning_clamp] /v1/models non disponibile "
                          f"(tentativo {fails}); tengo l'ultima lista",
                          flush=True)
        # primo aggancio mancato (proxy in avvio): ritento presto, non tra 10'
        time.sleep(PROXY_MODELS_TTL_S
                   if (got or _PROXY_MODELS_CACHE["models"]) else 30)


def _ensure_proxy_models_refresher():
    """Avvia lazy il thread daemon (una sola volta per processo)."""
    if _PROXY_MODELS_STARTED[0]:
        return
    with _PROXY_MODELS_LOCK:
        if _PROXY_MODELS_STARTED[0]:
            return
        _PROXY_MODELS_STARTED[0] = True
    threading.Thread(target=_refresher_proxy_models, daemon=True,
                     name="reasoning-clamp-models").start()


def _fetch_pattern_models(name):
    """Listing {api_base}/models per un pattern 'X/*': ritorna i nomi
    PUBBLICI richiedibili ('groq/openai/gpt-oss-120b', 'gemini/models/
    gemini-2.5-flash', ...). [] su qualsiasi errore (gateway giu', api_base
    assente, listing non-JSON): il chiamante tiene allora il wildcard."""
    prefix = name[:-1]  # "groq/*" -> "groq/"
    try:
        from litellm.proxy.proxy_server import llm_router
        entry = None
        for dep in (getattr(llm_router, "model_list", None) or []):
            n = (dep.get("model_name") if isinstance(dep, dict)
                 else getattr(dep, "model_name", None))
            if n == name:
                entry = dep
                break
        lp = None
        if isinstance(entry, dict):
            lp = entry.get("litellm_params")
        else:
            lp = getattr(entry, "litellm_params", None)
        api_base = (lp or {}).get("api_base")
        api_key = (lp or {}).get("api_key")
        if not api_base:
            return []
        url = str(api_base).rstrip("/") + "/models"
        req = urllib.request.Request(url)
        if api_key:
            req.add_header("Authorization", "Bearer " + str(api_key))
        # host locale (nomi docker senza punto): bypass di QUALSIASI proxy
        # env (llmtrim) - i gateway sono reachabili solo sulla rete interna
        host = urllib.parse.urlparse(url).hostname or ""
        if "." in host:
            opener = urllib.request.build_opener()
        else:
            opener = urllib.request.build_opener(
                urllib.request.ProxyHandler({}))
        with opener.open(req, timeout=WILDCARD_LIST_TIMEOUT_S) as resp:
            data = json.loads(resp.read().decode("utf-8", "replace"))
        out = []
        for m in (data.get("data") or []):
            mid = str(m.get("id") or "").strip()
            if not mid:
                continue
            full = mid if mid.startswith(prefix) else prefix + mid
            out.append(full)
            # hint di tier dal PREZZO del listing (formato OpenRouter:
            # pricing.prompt in USD/token, "0" = gratis, "-1"/assente =
            # prezzo nascosto -> ignorato): il prezzo DEL PROMPT decide il
            # tier (0 = gratis, sotto CHEAP_PRICE_USD_PER_TOKEN = economico):
            # un modello nuovo listato dal provider viene classificato senza
            # toccare il codice (classificazione dinamica)
            pr = m.get("pricing") if isinstance(m, dict) else None
            if isinstance(pr, dict) and pr.get("prompt") is not None:
                try:
                    p = float(pr.get("prompt"))
                except (TypeError, ValueError):
                    p = -1.0
                if p >= 0:
                    _TIER_HINTS[full] = p
            if len(out) >= WILDCARD_MAX_MODELS:
                break
        return out
    except Exception:
        return []


def _expand_wildcards(names):
    """Sostituisce ogni 'X/*' con i nomi concreti dal listing (cache TTL
    WILDCARD_CACHE_TTL_S); se il listing fallisce resta 'X/*' (i gateway
    tolleranti lo accettano, gli altri 400-eranno ma il failover avanza)."""
    out = []
    now = time.monotonic()
    for n in names:
        if not n.endswith("/*"):
            out.append(n)
            continue
        hit = _WILDCARD_CACHE.get(n)
        if hit and now - hit[0] < WILDCARD_CACHE_TTL_S:
            conc = hit[1]
        else:
            conc = _fetch_pattern_models(n)
            if conc:
                _WILDCARD_CACHE[n] = (now, conc)
            else:
                conc = [n]
        out.extend(conc)
    seen = set()
    res = []
    for n in out:
        if n not in seen:
            seen.add(n)
            res.append(n)
    return res
MAP_OUTPUT_TOKENS = 1024
MAP_CHUNK_INPUT_TOKENS = 90000  # + output < contesto dei piu' piccoli (131072)
MAP_MAX_CHUNKS = 8
MAP_TIMEOUT_S = 180
MAP_ORIGIN = "context_map_reduce"

# T (split+compress della coda): la massa spesso e' nell'ultimo messaggio
# user (il prompt compilato dall'agente), che B/C/E tengono verbatim
TAIL_SPLIT_MIN_TOKENS = 32000  # sotto: la coda non e' il problema, skip
ANCHOR_HEAD_CHARS = 3000       # inizio dell'ultimo user: VERBATIM
ANCHOR_TAIL_CHARS = 6000       # fine dell'ultimo user: VERBATIM (qui sta
                               # la richiesta corrente del client)
TAIL_ORIGIN = "context_tail_split"
TAIL_HEADER = (
    "[NOTA: la parte CENTRALE di questo messaggio (lungo) e' stata "
    "compressa: inizio e fine sono VERBATIM, il mezzo e' riassunto qui "
    "sotto in modo fedele]\n"
)

COMPACT_SYSTEM_PROMPT = (
    "You compress conversation context for a coding agent. Rewrite the "
    "transcript as a dense, faithful brief: keep VERBATIM every file path, "
    "symbol, identifier, number, command, error message and decision; keep "
    "the goal, the constraints, what was tried and what failed. Do not "
    "invent facts, do not answer the request, do not add advice. Output "
    "ONLY the brief."
)
COMPACTED_HEADER = "CONTESTO PRECEDENTE COMPATTATO (fedele, compresso):\n"
MAP_SYSTEM_PROMPT = (
    "You compress one slice of a coding-agent transcript. Keep VERBATIM "
    "file paths, symbols, numbers, commands, error messages and decisions. "
    "Output ONLY the compressed slice, no preamble."
)


def _resolve_internal_call_key() -> str:
    """Chiave metadata che marca le chiamate interne di litellm.

    Usata da litellm stesso (llm_as_a_judge) e consumata da spend logs,
    rate limiter e cooldowns: se un giorno cambiasse nome, la cascata deve
    continuare a funzionare con una chiave propria.
    """
    try:
        from litellm.constants import INTERNAL_CALL_ORIGIN_METADATA_KEY
        return str(INTERNAL_CALL_ORIGIN_METADATA_KEY)
    except Exception:
        return "x-litellm-internal-call-origin"


INTERNAL_CALL_ORIGIN_KEY = _resolve_internal_call_key()


# --- stima token adattiva (densita' + tiktoken) ------------------------------
# Il fisso 2.8 chars/token e' giusto per codice/JSON ma SOVRASTIMA la prosa
# ~2x: su un transcript reale da ~479k char di prosa la stima era 171k token
# contro ~88k reali -> falsa saturazione -> cascata sprecata e 413 su una
# richiesta che stava nel contesto (incidente 2026-10-05). Ratio ADATTIVO
# alla densita' dei simboli + secondo parere tiktoken (gia' dentro litellm)
# sui soli prompt grandi.
NON_WORD_RE = re.compile(r"[^\w\s]|_")
LONG_ALNUM_RE = re.compile(r"[A-Za-z0-9_]{32,}")  # base64/hex/sha: testo denso
CJK_RE = re.compile(
    "[\u1100-\u11ff\u2e80-\ua4cf\ua960-\ua97f\uac00-\ud7ff"
    "\uf900-\ufaff\uff00-\uffef\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]"
)
PROSE_CHARS_PER_TOKEN = 4.6   # inglese col tokenizer llama (misurato)
MIXED_CHARS_PER_TOKEN = 3.6
CJK_CHARS_PER_TOKEN = 1.5     # peggior caso: ~1 token ogni 1.5 char
DENSE_DENSITY = 0.20          # >= 20% simboli/non-parole: codice/JSON
MIXED_DENSITY = 0.10
TIKTOKEN_MIN_CHARS = 150000   # tiktoken solo sui prompt grossi (~30k+ token)
_TIKTOKEN_ENC = None
_TIKTOKEN_FAILED = False


def _chars_per_token(text):
    """Ratio caratteri/token di UN pezzo di testo, dalla densita' dei
    simboli: codice/JSON (denso) -> 2.8, conservativo come prima; prosa
    pura -> 4.6; CJK -> 1.5. Il ratio piu' basso (piu' token stimati) vale
    per il testo denso: si sovrastima la prosa, non si sottostima mai il
    codice."""
    n = len(text)
    if not n:
        return PROSE_CHARS_PER_TOKEN
    if len(CJK_RE.findall(text)) / n > 0.15:
        return CJK_CHARS_PER_TOKEN
    density = len(NON_WORD_RE.findall(text)) / n
    if density >= DENSE_DENSITY:
        return CHARS_PER_TOKEN
    if density <= MIXED_DENSITY:
        # prosa con base64/hash/sha dentro: resta prudente
        return (PROSE_CHARS_PER_TOKEN
                if len(LONG_ALNUM_RE.findall(text)) < 3
                else MIXED_CHARS_PER_TOKEN)
    # misto: interpolazione lineare tra prosa e denso
    t = (density - MIXED_DENSITY) / (DENSE_DENSITY - MIXED_DENSITY)
    return MIXED_CHARS_PER_TOKEN + t * (CHARS_PER_TOKEN - MIXED_CHARS_PER_TOKEN)


def _tiktoken_estimate(texts):
    """Secondo parere tiktoken (cl100k_base, NESSUN moltiplicatore: su
    trascrizioni reali cl100k conta ~1.2x i token llama sulla prosa e
    grossomodo gli stessi sul codice, quindi aggiungerne un altro qui
    ricreerebbe il falso positivo della cascata; il margine 3% di
    _margin basta). None se tiktoken non e' disponibile."""
    global _TIKTOKEN_ENC, _TIKTOKEN_FAILED
    if _TIKTOKEN_FAILED:
        return None
    try:
        if _TIKTOKEN_ENC is None:
            import tiktoken
            _TIKTOKEN_ENC = tiktoken.get_encoding("cl100k_base")
        n = 0
        for t in texts:
            if t:
                n += len(_TIKTOKEN_ENC.encode(t, disallowed_special=()))
        return n
    except Exception:
        _TIKTOKEN_FAILED = True
        return None


# cache dei conteggi tiktoken degli SCHEMI TOOLS: chiave = sha1 del testo
# unito. I client agentici rimandano GLI STESSI schemi a ogni richiesta
# (decine/centinaia di kB rivalutati a ogni chiamata): dopo la prima il
# conteggio e' O(1). Poche entrate (una per set di schemi), svuotata se
# cresce troppo.
_TOOLS_CACHE = {}


def _tools_tiktoken(tool_texts):
    """Conteggio tiktoken dei SOLI schemi tools (cl100k, nessun
    moltiplicatore: su JSON denso cl100k conta grossomodo quanto il
    tokenizer llama, spesso MENO - e' il candidato basso del min col
    ratio, vedi TOOLS_CHARS_PER_TOKEN). CACHE sha1: schemi identici a
    ogni richiesta -> conteggio O(1) dopo la prima. None se tiktoken non
    e' disponibile (resta il solo ratio)."""
    if not tool_texts:
        return 0
    global _TIKTOKEN_ENC, _TIKTOKEN_FAILED
    if _TIKTOKEN_FAILED:
        return None
    try:
        if _TIKTOKEN_ENC is None:
            import tiktoken
            _TIKTOKEN_ENC = tiktoken.get_encoding("cl100k_base")
        key = hashlib.sha1(
            "\n".join(tool_texts).encode("utf-8", "replace")).hexdigest()
        n = _TOOLS_CACHE.get(key)
        if n is None:
            n = sum(len(_TIKTOKEN_ENC.encode(t, disallowed_special=()))
                    for t in tool_texts)
            if len(_TOOLS_CACHE) > 8:
                _TOOLS_CACHE.clear()
            _TOOLS_CACHE[key] = n
        return n
    except Exception:
        _TIKTOKEN_FAILED = True
        return None


class ReasoningClamp(CustomLogger):

    def _context_for(self, model):
        for pref, ctx in CONTEXT_BY_PREFIX:
            if model == pref.rstrip("/") or model.startswith(pref):
                return ctx
        return DEFAULT_CONTEXT

    def _estimate_prompt_tokens(self, data):
        """Stima CONSERVATIVA (mai sottostimare: l'upstream conta col
        tokenizer reale e un max_tokens oltre la stanza e' un 400 che
        uccide la catena di fallback).

        Ratio adattivo per pezzo (densita', vedi _chars_per_token) e max()
        con tiktoken sui soli prompt grandi. Conta ANCHE gli schemi
        tools/functions (tiktoken cachato sha1 quando viene piu' BASSO del
        ratio: il vecchio /2.2 sovrastimava ~2x i ~100 schemi MCP di
        AiderDesk -> falsi 413, incidente 2026-10-06) e le immagini."""
        msgs = data.get("messages") or []
        # tools/functions: MIN fra tiktoken (cachato sha1, spesso piu'
        # basso sul JSON denso) e il ratio; mai il contrario - non si
        # sottostima mai, ma il vecchio ratio /2.2 SOVRASTIMAVA ~2x i
        # grandi schemi MCP (incidente 2026-10-06, vedi la costante)
        schemas = data.get("tools") or data.get("functions") or []
        tool_texts = []
        for t in schemas:
            try:
                j = t if isinstance(t, str) else json.dumps(t)
            except Exception:
                j = str(t)
            tool_texts.append(j)
        fixed = TOOL_FIXED_TOKENS * len(tool_texts)
        tool_tokens = (sum(int(len(j) / TOOLS_CHARS_PER_TOKEN)
                           for j in tool_texts) + fixed)
        tk = _tools_tiktoken(tool_texts)
        if tk is not None and tk + fixed < tool_tokens:
            tool_tokens = tk + fixed
        texts = []
        images = 0
        for m in msgs:
            if not isinstance(m, dict):
                continue
            c = m.get("content")
            if isinstance(c, str):
                texts.append(c)
            elif isinstance(c, list):
                for part in c:
                    if isinstance(part, dict):
                        if part.get("type") in ("image_url", "image"):
                            images += 1
                            continue
                        for k in ("text", "content"):
                            v = part.get(k)
                            if isinstance(v, str):
                                texts.append(v)
                    elif isinstance(part, str):
                        texts.append(part)
            for k in ("reasoning", "reasoning_content", "name"):
                v = m.get(k)
                if isinstance(v, str):
                    texts.append(v)
            if m.get("tool_calls"):
                try:
                    texts.append(json.dumps(m["tool_calls"], ensure_ascii=False))
                except Exception:
                    texts.append(str(m["tool_calls"]))
        est = sum(int(len(t) / _chars_per_token(t)) for t in texts)
        chars = sum(len(t) for t in texts)
        if chars >= TIKTOKEN_MIN_CHARS:
            tk = _tiktoken_estimate(texts)
            if tk and tk > est:
                est = tk
        return (est + PER_MESSAGE_OVERHEAD * len(msgs)
                + tool_tokens + images * IMAGE_TOKEN_COST)

    def _estimate_subset(self, msgs, data):
        """Stima di SOLO alcuni messaggi (con gli stessi tools): serve a
        decidere se la coda (dall'ultimo user in poi) satura da sola - in
        quel caso B/C/E, che la tengono verbatim, non possono aiutare."""
        sub = {"messages": list(msgs)}
        for k in ("tools", "functions"):
            if data.get(k):
                sub[k] = data[k]
        return self._estimate_prompt_tokens(sub)

    def _margin(self, est):
        """Margine di sicurezza della stima: FISSO + PROPORZIONALE.

        L'errore residuo della stima cresce col prompt (chat template,
        ratio caratteri/token variabile: ~1.4-1.8% sotto su trascrizioni
        reali da ~100k token), quindi un margine solo fisso perde lì."""
        return SAFETY_MARGIN + int(est * PROMPT_ERR_PCT)

    def _saturated(self, est, ctx):
        """Il prompt da solo lascia meno di MIN_OUTPUT_TOKENS: il clamp non
        puo' aiutare, serve ridurre il PROMPT (cascata)."""
        return est + MIN_OUTPUT_TOKENS + self._margin(est) > ctx

    def _serialize(self, msgs):
        """Trascrizione testuale dei messaggi, per compattazione/riassunto."""
        out = []
        for m in msgs:
            if not isinstance(m, dict):
                continue
            role = str(m.get("role") or "user")
            c = m.get("content")
            if isinstance(c, list):
                parts = []
                for part in c:
                    if isinstance(part, dict):
                        for k in ("text", "content"):
                            v = part.get(k)
                            if isinstance(v, str):
                                parts.append(v)
                    elif isinstance(part, str):
                        parts.append(part)
                c = "\n".join(parts)
            elif not isinstance(c, str):
                c = "" if c is None else str(c)
            line = f"[{role}] {c}"
            if m.get("tool_calls"):
                line += f"\n[tool_calls] {m['tool_calls']}"
            out.append(line)
        return "\n\n".join(out)

    def _internal(self, data):
        md = data.get("metadata")
        return isinstance(md, dict) and md.get(INTERNAL_CALL_ORIGIN_KEY)

    def _clamp_output(self, data, model, est=None):
        """max_tokens <= contesto - prompt. Il contesto e' il MINIMO della
        catena di fallback (l'hook del proxy NON viene rieseguito sui target
        di fallback: litellm riusa gli stessi kwargs a ogni hop, quindi il
        tetto scelto qui deve starci dentro ovunque). Nessun clamp se il
        client non passa un tetto: in quel caso llama-swap
        auto-limita il completamento."""
        keys = [k for k in MAX_TOKEN_KEYS
                if isinstance(data.get(k), (int, float)) and data[k] > 0]
        if not keys:
            return
        ctx = self._context_for(model)
        chain = data.get("fallbacks")
        if isinstance(chain, list) and chain:
            # la richiesta deve starci ANCHE se cade sull'ultimo target
            # (small-model, 131072): limita al contesto piu' piccolo della
            # catena (i default litellm_params.max_tokens sono uniti da
            # litellm DOPO l'hook e non sono visibili qui - vedi config)
            ctx = min([ctx] + [self._context_for(str(t)) for t in chain])
        if est is None:
            est = self._estimate_prompt_tokens(data)
        budget = ctx - est - self._margin(est)
        if budget < MIN_OUTPUT_TOKENS:
            # prompt gia' quasi saturo: nessun tetto sensato, lascia fare
            # all'upstream (400 suo, come prima di questo clamp)
            return
        for k in keys:
            cur = int(data[k])
            if cur > budget:
                data[k] = int(budget)
                print(f"[reasoning_clamp] {model}: {k} {cur} -> {int(budget)} "
                      f"(ctx {ctx}, prompt ~{est} tok stimati)", flush=True)

    def _force_output(self, data, model, est):
        """F-EVIT: max_tokens forzato alla stanza residua. Vale SOLO quando
        il PROMPT da solo ci sta (est <= ctx, controllo nel chiamante). Il
        vecchio _clamp_output si rifiutava sotto MIN_OUTPUT_TOKENS e la
        richiesta restava senza NESSUN percorso (incidente 2026-10-06:
        est 126805 <= 131072 ma budget 4267 - marginale - < 1024 -> 413
        assurdo mentre l'upstream Strata con fit_max_tokens avrebbe
        accettato e accorciato il completamento da solo). Qui NIENTE
        margine percentuale (e' proprio il caso in cui la stima e' oltre
        la stanza): solo SAFETY_MARGIN piatto. False se nemmeno
        MIN_FORCED_OUTPUT ci sta -> al chiamante il 413 vero."""
        keys = [k for k in MAX_TOKEN_KEYS
                if isinstance(data.get(k), (int, float)) and data[k] > 0]
        ctx = self._context_for(model)
        chain = data.get("fallbacks")
        if isinstance(chain, list) and chain:
            # stessa regola di _clamp_output: deve starci ANCHE sull'ultimo
            # target della catena di fallback
            ctx = min([ctx] + [self._context_for(str(t)) for t in chain])
        room = ctx - est - SAFETY_MARGIN
        if room < MIN_FORCED_OUTPUT:
            return False
        if not keys:
            # niente tetto dal client: l'upstream auto-limita (llama-swap)
            return True
        for k in keys:
            cur = int(data[k])
            if cur > room:
                data[k] = int(room)
                print(f"[reasoning_clamp] {model}: {k} {cur} -> {int(room)} "
                      f"(FORZATO, ctx {ctx}, prompt ~{est} tok stimati)",
                      flush=True)
        return True

    # ---------------- B: compattazione con un modello da 262k ----------------

    def _split_tail(self, msgs):
        """Corpo da compattare = tutto cio' che precede l'ultimo messaggio
        'user'; la coda (dall'ultimo user in poi) resta FEDELE.

        Tagliare sull'ultimo user mantiene interi gli scambi tool
        (assistant tool_calls + tool result) della richiesta corrente:
        un tool result orfano sarebbe un 400 dal provider.
        """
        for i in range(len(msgs) - 1, -1, -1):
            m = msgs[i]
            if isinstance(m, dict) and m.get("role") == "user":
                return msgs[:i], msgs[i:]
        return [], msgs

    def _apply_compacted(self, data, brief, tail):
        msgs = data.get("messages") or []
        systems = [m for m in msgs
                   if isinstance(m, dict) and m.get("role") == "system"]
        new_msgs = list(systems)
        new_msgs.append({"role": "user",
                         "content": COMPACTED_HEADER + brief})
        new_msgs.extend(tail)
        data["messages"] = new_msgs
        return data

    async def _compact_with_provider(self, data, model, est):
        """Livello B: sintetizza la STORIA (tutto cio' che precede l'ultimo
        user, che resta verbatim) con un modello a contesto grande. I
        compattori preferiti (COMPACTOR_MODELS) vengono provati per primi;
        se nessuno e' disponibile o contiene il corpo, il pool si apre a
        OGNI modello configurato su litellm che possa contenerlo
        (_worker_pool, richiesta utente 2026-10-06)."""
        msgs = [m for m in (data.get("messages") or []) if isinstance(m, dict)]
        body_src, tail = self._split_tail(msgs)
        if not body_src:
            return None, "nothing-to-compact"
        body = self._serialize(body_src)
        if len(body) < COMPACT_MIN_CHARS:
            return None, "body-too-small"
        from litellm.proxy.proxy_server import llm_router
        if llm_router is None:
            return None, "no-router"
        # stima del corpo per il FILTRO del pool: qui serve ACCURATEZZA, non
        # prudenza - il ratio denso (2.8) sovrastima la prosa ~40-60% e
        # escluderebbe compattori capaci, buttando la cascata sul livello
        # lossy C. tiktoken (seconda opinione gia' wired) se disponibile:
        # un compattatore marginale che risponde 400 costa un giro di
        # failover, escluderli TUTTI costa la compressione lossy.
        body_est = (_tiktoken_estimate([body])
                    or int(len(body) / CHARS_PER_TOKEN))
        pool = self._worker_pool(body_est, COMPACTOR_MODELS,
                                 output_tokens=COMPACT_OUTPUT_TOKENS)
        if not pool:
            return None, f"no-compactor-fits:{body_est}tok"
        failures = []
        # budget di tempo: si prova TUTTO il pool (gratis -> economico ->
        # costoso) finche' c'e' budget; niente tetto fisso di compattori
        deadline = time.monotonic() + WORKER_TIME_BUDGET_S
        for comp in pool:
            if time.monotonic() >= deadline:
                failures.append(f"time-budget:{WORKER_TIME_BUDGET_S}s")
                break
            try:
                resp = await asyncio.wait_for(
                    llm_router.acompletion(
                        model=comp,
                        messages=[{"role": "system",
                                   "content": COMPACT_SYSTEM_PROMPT},
                                  {"role": "user", "content": body}],
                        max_tokens=COMPACT_OUTPUT_TOKENS,
                        num_retries=0,
                        fallbacks=[],
                        drop_params=True,
                        metadata={INTERNAL_CALL_ORIGIN_KEY: COMPACT_ORIGIN},
                    ),
                    timeout=COMPACTOR_TIMEOUT_S,
                )
                brief = (resp.choices[0].message.content or "").strip()
            except Exception as e:
                failures.append(f"{comp}:{type(e).__name__}:{str(e)[:120]}")
                continue
            if len(brief) < MIN_BRIEF_CHARS:
                failures.append(f"{comp}:empty-brief")
                continue
            print(f"[reasoning_clamp] B {model}: compattato da {comp} "
                  f"({len(body)} char -> {len(brief)} char di sintesi)",
                  flush=True)
            return self._apply_compacted(data, brief, tail), comp
        return None, "|".join(failures) or "no-compactor"

    # ---------------- C: compressione deterministica -------------------------

    def _compress_locally(self, data, model, est):
        """Livello C: litellm.compression.compress() - BM25, nessuna chiamata
        LLM. Gli stub NON sono recuperabili (nessun agentic loop serve
        litellm_content_retrieve in questo stack), quindi e' lossy: per
        questo viene dopo B."""
        from litellm.compression import compress
        from litellm.types.utils import CallTypes
        msgs = [m for m in (data.get("messages") or []) if isinstance(m, dict)]
        if len(msgs) < 3:
            return None, "too-few-messages"
        ctx = self._context_for(model)
        target = max(MIN_OUTPUT_TOKENS * 2, int(ctx * COMPRESS_TARGET_RATIO))
        res = compress(messages=list(msgs), model=model,
                       call_type=CallTypes.completion,
                       compression_trigger=1, compression_target=target)
        after = int(res.get("compressed_tokens") or 0)
        if res.get("compression_skipped_reason"):
            return None, f"skipped:{res['compression_skipped_reason']}"
        if not after or after >= est:
            # nessun guadagno reale: non ha senso spedire stubs inutili
            return None, "no-gain"
        data["messages"] = res["messages"]
        print(f"[reasoning_clamp] C {model}: compressione deterministica "
              f"{res.get('original_tokens')} -> {after} tok "
              f"(ratio {res.get('compression_ratio')}, target {target})",
              flush=True)
        return data, f"compress:{after}"

    # ---------------- operari di riassunto (pool dinamico) ------------------

    def _worker_pool(self, needed_tokens, preferred, output_tokens=MAP_OUTPUT_TOKENS):
        """OGNI modello configurato su litellm che puo' contenere
        `needed_tokens` di input (+ output + margine), ordinato per
        (tier di costo, preferenza): prima i modelli GRATIS
        (inference4free, openrouter/:free, listing a prezzo zero, locali),
        poi gli ECONOMICI (gpt-oss, flash-lite, gemma, syn:small...),
        per ultimi i costosi - richiesta utente 2026-10-06. La lista
        `preferred` ordina solo DENTRO il tier. Richiesta utente
        2026-10-06: "cascade model selection, exploit ANY possible
        litellm configured model, even the ones hosted by
        inference4free" - niente piu' liste chiude di operari.
        I WILDCARD della model_list (groq/*, gemini/*, ...) vengono ESPASI
        nei nomi concreti richiedibili via listing del gateway (cache TTL):
        chiamare acompletion col nome letterale "groq/*" manderebbe
        model="*" all'upstream -> 400/404 garantiti (visto in produzione).
        Se il listing non risponde resta il nome wildcard: i gateway
        tolleranti (inference4free, synthetic) lo accettano comunque.
        L'INTERROGAZIONE DINAMICA di /v1/models (thread daemon, cache) unisce
        al pool i modelli presenti nel listing ma non in model_list e usa i
        contesti REALI (max_input_tokens/max_output_tokens) invece della
        stima statica: aggiungere/togliere un modello da litellm viene visto
        senza toccare il codice (richiesta utente 2026-10-06)."""
        names = []
        try:
            from litellm.proxy.proxy_server import llm_router
            for dep in (getattr(llm_router, "model_list", None) or []):
                n = (dep.get("model_name") if isinstance(dep, dict)
                     else getattr(dep, "model_name", None))
                if n and str(n) not in names:
                    names.append(str(n))
        except Exception:
            names = []
        names = _expand_wildcards(names)

        # listing dinamico /v1/models (cache del thread daemon): gli id non
        # in model_list entrano nel pool; i contesti reali sostituiscono la
        # stima. Listing non ancora pronto -> dyn vuoto: solo model_list.
        _ensure_proxy_models_refresher()
        dyn = {}
        with _PROXY_MODELS_LOCK:
            dyn.update(_PROXY_MODELS_CACHE["models"])
        for mid in dyn:
            if mid not in names:
                names.append(mid)

        def rank(n):
            for i, p in enumerate(preferred):
                if n == p or n.startswith(p):
                    return i
            return len(preferred)

        budget = needed_tokens + output_tokens + self._margin(needed_tokens)

        def fits(n):
            # non-chat esclusi ANCHE se arrivano solo dal listing
            if any(b in n.lower() for b in NON_CHAT_WORKER_SUBSTRINGS):
                return False
            static = self._context_for(n)
            mit, mot = dyn.get(n, (None, None))
            # finestra: mit+mot quando il listing li da' entrambi; con solo
            # mit: mit e' per definizione <= finestra reale, quindi
            # max(static, mit) non e' MAI peggiore della stima statica
            window = (mit + mot) if (mit is not None and mot) else (
                static if mit is None else max(static, mit))
            if window < budget:
                return False
            # il PROMPT deve stare nel cap di INPUT reale (mit): e' la regola
            # con cui l'upstream rifiuta; il margine (>= output) copre la
            # risposta nella finestra
            if mit is not None and needed_tokens + self._margin(
                    needed_tokens) > mit:
                return False
            return True

        # tier di costo PRIMARIO, preferenza del livello secondaria: il pool
        # e' gratis -> economico -> costoso a prescindere dalle liste preferred
        return [n for n in sorted(
                    names, key=lambda n: (_model_tier(n), rank(n)))
                if fits(n)]

    async def _summarize_chunks(self, chunks, origin, model):
        """Riassume i chunk con FAILOVER sull'INTERO pool dinamico entro il
        budget di tempo WORKER_TIME_BUDGET_S (richiesta utente 2026-10-06:
        "provare tutti i possibili modelli, con un limite di tempo; errore
        solo quando proprio non riusciamo a fare nulla"): operario
        "appiccicoso" (il primo che riesce resta per i chunk successivi),
        ogni fallimento/risposta vuota avanza al successivo finche' c'e'
        budget. Se il budget (o il pool) finisce a meta', si tengono i
        PARZIALI: almeno un chunk riassunto = compressione utile, il livello
        successivo della cascata completa se serve. Ritorna (brief, info)
        con brief=None solo su fallimento TOTALE."""
        from litellm.proxy.proxy_server import llm_router
        if llm_router is None:
            return None, "no-router"
        # chunk misurato col ratio denso: la stima del CHUNK deve stare
        # bassa (non sovrastimare) per non escludere operari validi; il
        # margine del pool copre il residuo
        chunk_est = max(int(len(c) / CHARS_PER_TOKEN) for c in chunks)
        pool = self._worker_pool(chunk_est, SUMMARIZER_PREFERRED)
        if not pool:
            return None, f"no-worker-fits:{chunk_est}tok"
        deadline = time.monotonic() + WORKER_TIME_BUDGET_S
        failures = []
        wi = 0
        used = None
        summaries = []
        for i, chunk in enumerate(chunks):
            s = None
            while wi < len(pool):
                if time.monotonic() >= deadline:
                    failures.append(f"time-budget:{WORKER_TIME_BUDGET_S}s")
                    break
                w = pool[wi]
                try:
                    resp = await asyncio.wait_for(
                        llm_router.acompletion(
                            model=w,
                            messages=[{"role": "system",
                                       "content": MAP_SYSTEM_PROMPT},
                                      {"role": "user", "content": chunk}],
                            max_tokens=MAP_OUTPUT_TOKENS,
                            num_retries=0,
                            fallbacks=[],
                            drop_params=True,
                            metadata={INTERNAL_CALL_ORIGIN_KEY: origin},
                        ),
                        timeout=MAP_TIMEOUT_S,
                    )
                    s = (resp.choices[0].message.content or "").strip()
                except Exception as e:
                    failures.append(f"{w}:{type(e).__name__}")
                    wi += 1
                    continue
                if not s:
                    failures.append(f"{w}:empty")
                    wi += 1
                    continue
                used = w
                break
            if not s:
                break   # budget esaurito o pool esaurito: usa i parziali
            summaries.append(f"--- parte {i + 1}/{len(chunks)} ---\n{s}")
        if not summaries:
            return None, "|".join(failures[-4:]) or "no-worker"
        if len(summaries) < len(chunks):
            print(f"[reasoning_clamp] riassunti PARZIALI {len(summaries)}/"
                  f"{len(chunks)} chunk su '{used}' (budget/pool esauriti; "
                  f"falliti: {'; '.join(failures[-6:])})", flush=True)
        else:
            print(f"[reasoning_clamp] riassunti {len(chunks)} chunk su "
                  f"'{used}' (falliti prima: "
                  f"{'; '.join(failures[-4:]) if failures else 'nessuno'})",
                  flush=True)
        return "\n\n".join(summaries), used or "?"

    # ---------------- T: split+compress della coda --------------------------

    async def _split_compress_tail(self, data, model, est):
        """Livello T (richiesta utente 2026-10-06: "quando il contesto
        supera lo splitiamo e lo compriamiamo con un qualche llm"): quando
        la massa e' nell'ULTIMO messaggio user (il prompt compilato
        dall'agente), B/C/E non possono liberarla perche' la coda resta
        verbatim. T tiene verbatim solo le ANCORE (inizio + fine, dove sta
        la richiesta corrente) e riassume il mezzo con il pool dinamico."""
        msgs = [m for m in (data.get("messages") or []) if isinstance(m, dict)]
        idx = None
        for i in range(len(msgs) - 1, -1, -1):
            m = msgs[i]
            if isinstance(m, dict) and m.get("role") == "user":
                idx = i
                break
        if idx is None:
            return None, "no-user-msg"
        c = msgs[idx].get("content")
        if not isinstance(c, str) or not c:
            # content list (vision/parts): non riscriverlo qui
            return None, "tail-not-text"
        tail_est = int(len(c) / _chars_per_token(c))
        if tail_est < TAIL_SPLIT_MIN_TOKENS:
            return None, f"tail-too-small:{tail_est}"
        if len(c) <= ANCHOR_HEAD_CHARS + ANCHOR_TAIL_CHARS:
            return None, "no-middle"
        head = c[:ANCHOR_HEAD_CHARS]
        anchor = c[len(c) - ANCHOR_TAIL_CHARS:]
        middle = c[ANCHOR_HEAD_CHARS:len(c) - ANCHOR_TAIL_CHARS]
        if len(middle) < COMPACT_MIN_CHARS:
            return None, "middle-too-small"
        chunks = self._chunk_for_map(middle)
        if not chunks:
            return None, "no-chunks"
        brief, info = await self._summarize_chunks(chunks, TAIL_ORIGIN, model)
        if brief is None:
            return None, info
        new_c = head + "\n" + TAIL_HEADER + brief + "\n" + anchor
        new_msgs = list(msgs)
        new_msgs[idx] = dict(msgs[idx], content=new_c)
        data["messages"] = new_msgs
        print(f"[reasoning_clamp] T {model}: ultimo user {len(c)} char -> "
              f"{len(new_c)} char (mezzo in {len(chunks)} chunk, {info})",
              flush=True)
        return data, f"tail:{len(chunks)}"

    # ---------------- E: map/reduce sul pool dinamico -----------------------

    def _chunk_for_map(self, body):
        """Taglia la trascrizione in chunk di ~MAP_CHUNK_INPUT_TOKENS token;
        un messaggio piu' grande del chunk viene tagliato a meta'."""
        limit = int(MAP_CHUNK_INPUT_TOKENS * CHARS_PER_TOKEN)
        chunks, cur = [], []
        cur_len = 0
        for block in body.split("\n\n"):
            if len(block) > limit:
                for i in range(0, len(block), limit):
                    chunks.append(block[i:i + limit])
                continue
            if cur and cur_len + len(block) > limit:
                chunks.append("\n\n".join(cur))
                cur, cur_len = [], 0
            cur.append(block)
            cur_len += len(block) + 2
        if cur:
            chunks.append("\n\n".join(cur))
        return chunks[:MAP_MAX_CHUNKS]

    async def _map_reduce(self, data, model, est):
        """Livello E: riassunto a chunk dell'intera storia (l'ultimo user
        resta verbatim come in B/C) + ricomposizione davanti alla coda."""
        msgs = [m for m in (data.get("messages") or []) if isinstance(m, dict)]
        body_src, tail = self._split_tail(msgs)
        if not body_src:
            return None, "nothing-to-compact"
        body = self._serialize(body_src)
        if len(body) < COMPACT_MIN_CHARS:
            return None, "body-too-small"
        chunks = self._chunk_for_map(body)
        if not chunks:
            return None, "no-chunks"
        brief, info = await self._summarize_chunks(chunks, MAP_ORIGIN, model)
        if brief is None:
            return None, info
        print(f"[reasoning_clamp] E {model}: map/reduce "
              f"({len(chunks)} chunk, {len(body)} char -> {len(brief)} char)",
              flush=True)
        return self._apply_compacted(data, brief, tail), f"map:{len(chunks)}"

    # ---------------- F: errore strutturato ---------------------------------

    def _overflow(self, data, model, est, tried):
        ctx = self._context_for(model)
        print(f"[reasoning_clamp] F {model}: prompt ~{est} tok > contesto "
              f"{ctx}; cascata esaurita ({'|'.join(tried)})", flush=True)
        raise HTTPException(
            status_code=413,
            detail={
                "error": "prompt_too_large",
                "model": model,
                "context_tokens": ctx,
                "prompt_tokens_estimated": est,
                "tried": tried,
                "hint": ("il prompt supera il contesto del modello e la "
                         "cascata di compattazione (B compattori a contesto "
                         "grande -> C compressione BM25 -> T split+compress "
                         "della coda -> E map/reduce, tutti con failover sul "
                         "pool dinamico dei modelli configurati) non ha "
                         "liberato spazio: riduci il contesto del client o "
                         "usa un modello con contesto maggiore"),
            },
        )

    # ---------------- cascata ----------------------------------------------

    async def _cascade(self, data):
        model = str(data.get("model") or "")
        if self._internal(data):
            # chiamata interna della cascata stessa: mai ri-compactare
            return data
        if not (data.get("messages") or []):
            return data
        ctx = self._context_for(model)
        est = self._estimate_prompt_tokens(data)
        if not self._saturated(est, ctx):
            return data
        print(f"[reasoning_clamp] overflow {model}: ~{est} tok + "
              f"{MIN_OUTPUT_TOKENS} tok > contesto {ctx} -> cascata B/C/T/E",
              flush=True)
        # La coda (dall'ultimo user in poi) resta verbatim in B/C/E: se
        # satura DA SOLA quei livelli non possono liberare spazio - solo T
        # (che comprime il mezzo dell'ultimo messaggio user) puo' aiutare.
        # Cosi' si evitano 3 chiamate cloud prima di un 413 assicurato.
        msgs = [m for m in (data.get("messages") or []) if isinstance(m, dict)]
        body_src, tail = self._split_tail(msgs)
        if tail:
            tail_est = self._estimate_subset(tail, data)
            if self._saturated(tail_est, ctx):
                tried = [f"tail-alone:{tail_est}"]
                try:
                    new, info = await self._split_compress_tail(data, model, est)
                except Exception as e:
                    new, info = None, f"exc:{type(e).__name__}:{str(e)[:160]}"
                tried.append(f"T:{info}")
                if new is not None:
                    data = new
                    est = self._estimate_prompt_tokens(data)
                    self._clamp_output(data, model, est)
                    if not self._saturated(est, ctx):
                        print(f"[reasoning_clamp] {model}: cascata risolta "
                              f"da T (~{est} tok, contesto {ctx})", flush=True)
                        return data
                # la coda resta verbatim anche per B/C/E: nessun aiuto
                return self._overflow(data, model, est, tried)
        tried = []
        for name, fn in (("B", self._compact_with_provider),
                         ("C", self._compress_locally),
                         ("T", self._split_compress_tail),
                         ("E", self._map_reduce)):
            try:
                if name == "C":
                    # compressione BM25: CPU-heavy, fuori dal loop degli eventi
                    # (asyncio.to_thread: niente dipendenze dagli helper interni
                    # di litellm, che su 'asyncify' cambiano posizione)
                    new, info = await asyncio.to_thread(fn, data, model, est)
                else:
                    new, info = await fn(data, model, est)
            except Exception as e:
                new, info = None, f"exc:{type(e).__name__}:{str(e)[:160]}"
            tried.append(f"{name}:{info}")
            if new is None:
                # visibilita': un livello che salta non deve sparire nei log
                print(f"[reasoning_clamp] {name} {model}: nessun guadagno "
                      f"({info})", flush=True)
                continue
            data = new
            est = self._estimate_prompt_tokens(data)
            self._clamp_output(data, model, est)
            if not self._saturated(est, ctx):
                print(f"[reasoning_clamp] {model}: cascata risolta da {name} "
                      f"(~{est} tok, contesto {ctx})", flush=True)
                return data
        # F: nessun livello ha liberato spazio. Se il PROMPT da solo ci sta
        # (est <= ctx) NON e' un prompt_too_large: e' solo la stanza residua
        # che e' piccola. Il 413 e' deterministico e uccide OGNI retry del
        # client; un 400 dell'upstream invece cade nella catena di fallback
        # (e Strata ha fit_max_tokens). Si forza un max_tokens ridotto e la
        # richiesta PARTE (incidente 2026-10-06: est 126805 <= 131072 con
        # ~72k reali secondo il tokenizer di litellm -> 413 falso, il
        # prompt c'entrava con ~54k di margine).
        if est <= ctx and self._force_output(data, model, est):
            print(f"[reasoning_clamp] {model}: F evitato - prompt ~{est} tok "
                  f"ci sta nel contesto {ctx}: max_tokens alla stanza "
                  f"residua, richiesta parte ({'|'.join(tried)})", flush=True)
            return data
        return self._overflow(data, model, est, tried)

    def _apply(self, data):
        model = str(data.get("model") or "")
        # --- policy fallback per-modello (override per-request) ---
        # Regola (richiesta utente 2026-10-02): small-model e' l'ULTIMO
        # fallback di tutti i modelli, dopo che tutti gli altri sono falliti.
        if model == "small-model":
            # niente auto-fallback: e' lui l'ultimo ricorso di tutti gli
            # altri; litellm skipperebbe comunque un target gia' tentato
            data["fallbacks"] = []
        elif model == "embedding-model":
            # niente auto-fallback: l'unico modello che potrebbe servire una
            # richiesta embedding e' lui stesso, gli altri target (CHAT)
            # risponderebbero 400 (stesso motivo di small-model)
            data["fallbacks"] = []
        elif model == "local-model":
            # small-model in ultima spiaggia (Ollama locale rimossa
            # dappertutto, richiesta utente 2026-10-04)
            data["fallbacks"] = [FINAL_FALLBACK]
        elif model in (SYN_SMALL_TEXT, SYN_SMALL_VISION):
            # richiesta utente 2026-10-04: syn:small:TEXT e syn:small:VISION
            # ALLINEATI - UN SOLO fallback, small-model (piccolo->piccolo).
            # syn:small:VISION puo' rispondere 400 su small-model (text-only
            # su llama-swap): scelta esplicita dell'utente, catena minima.
            # Ramo esatto PRIMA del ramo generico synthetic/*: la callback
            # sovrascrive i fallback di OGNI richiesta synthetic/*.
            data["fallbacks"] = list(SYN_SMALL_FALLBACKS)
        elif model == SYN_EMBED:
            # richiesta utente 2026-10-04: il modello EMBEDDING di synthetic
            # cade su llama-swap "embedding-model" (richiesta embedding ->
            # /v1/embeddings; i modelli CHAT di llama-swap risponderebbero
            # 400 su input embedding, quindi nessun altro gradino).
            # Ramo esatto PRIMA del ramo generico synthetic/* (come
            # syn:small:text): la callback sovrascrive i fallback di OGNI
            # richiesta synthetic/*, un ramo generico renderebbe morta la
            # catena embedding.
            data["fallbacks"] = list(SYN_EMBED_FALLBACKS)
        elif model.startswith("synthetic/"):
            # catena fallback (richiesta utente): local-model PRIMO, poi
            # small-model come ultimo ricorso.
            # ECCEZIONI: syn:small:text, syn:small:vision e l'embedding
            # nomic, gestiti dai rami esatti qui sopra
            data["fallbacks"] = ["local-model", FINAL_FALLBACK]
        else:
            # TUTTI gli altri (groq/*, gemini/*, openrouter/*,
            # openrouter/free, ollama-cloud/*, inference4free/* e
            # qualunque prefisso futuro): nessun gradino intermedio,
            # small-model e' l'ultimo ricorso diretto (richiesta utente
            # 2026-10-02). La catch-all "*": ["small-model"] in config.yaml
            # resta come rete di sicurezza ma e' ombreggiata da questo
            # override per-request
            data["fallbacks"] = [FINAL_FALLBACK]
        # --- reasoning_effort per-provider ---
        eff = data.get("reasoning_effort")
        if model.startswith("groq/"):
            native = model[len("groq/"):]
            if eff in ("xhigh", "max"):
                eff = "high"
            if any(native.startswith(p) for p in GROQ_REASONING):
                if not eff:
                    eff = "high"
            else:
                # modelli Groq non-reasoning: qualunque valore e' un 400
                eff = None
            if eff:
                data["reasoning_effort"] = eff
            else:
                data.pop("reasoning_effort", None)
        elif model.startswith("gemini/") or model.startswith("models/"):
            if eff in ("xhigh", "max"):
                eff = "high"
            if not eff:
                eff = "high"
            data["reasoning_effort"] = eff
        # --- clamp max_tokens sul contesto del modello di QUESTO tentativo ---
        self._clamp_output(data, model)
        return data

    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        data = self._apply(data)
        # cascata SOLO nel percorso async: B ed E fanno chiamate LLM
        return await self._cascade(data)

    def pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        # sync: policy + clamp, nessuna rete (la cascata richiede await)
        return self._apply(data)


rclamp = ReasoningClamp()
