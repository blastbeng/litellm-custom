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
- cloudflare/* (Workers AI) : la validazione e' PER-MODELLO e il 400 e'
  secco -> il parametro si manda SOLO se il listing (cloudflare-gw) espone
  supported_efforts e il valore e' tra quelli (xhigh/max scivolano su
  high/max); senza lista pubblica viene RIMOSSO e vale il default
  Cloudflare (verificato 2026-11-18: qwen3.8-27b rifiuta "high",
  gpt-oss-120b rifiuta "xhigh", glm-4.7-flash rifiuta none/max/xhigh).
- Gli altri modelli (synthetic/*, openrouter/*, ollama/*, ollama-cloud/*,
  locale) non vengono toccati: il loro default xhigh sta gia' in config.yaml.

POLICY FALLBACK per-modello (richiesta utente 2026-10-02, wildcard-safe):
le chiavi dei fallback in config.yaml NON supportano wildcard (litellm
get_fallback_model_group: solo esatto/stripped-provider/"*"), quindi la
policy e' applicata QUI per-request tramite litellm_params.fallbacks, che
OVERRIDE la lista di config (router.py: kwargs.get("fallbacks",
self.fallbacks)):

  REGOLA (richiesta utente 2026-10-25): small-model NON e' piu' un fallback
  di NESSUN modello. Resta un modello autonomo, invocabile direttamente, ma
  nessuna catena lo contiene piu': quando una catena si esaurisce l'errore
  dell'upstream arriva al client.

  REGOLA (richiesta utente 2026-10-28): le chiamate INTERNE della cascata
  (B/T/T2/E, riconosciute dal metadata interno) girano con fallbacks=[] -
  il clamp usa lo STESSO modello della richiesta e non hopa su altri modelli.
  Nello stesso giro local-model e' stato rimosso dalla catena synthetic/*.

- synthetic/*            -> [] (NESSUN fallback: local-model rimosso il
  2026-10-28; restano solo i rami esatti qui sotto - syn:small:text,
  syn:small:vision, embedding nomic)
- synthetic/syn:small:text -> ["ollama/small-model","openrouter/free"]
  (richiesta utente 2026-10-12: ollama/small-model (Ollama sullo STESSO PC:
  server INDIPENDENTE da llama-swap, regge dove llama-swap da
  429/serializza), poi il cloud GRATIS come rete di sicurezza col PC spento.
  ollama/small-model e openrouter/free sono nomi concreti risolti dalle
  wildcard ollama/* e openrouter/* del model_list - vedi la definizione di
  SYN_SMALL_TEXT_FALLBACKS)
- synthetic/syn:small:vision -> IDENTICA a syn:small:text (richiesta utente
  2026-10-14): stessi gradini, stesso ordine. syn:small:VISION su
  ollama/small-model (text-only) puo' rispondere 400 sull'input vision: hop
  sprecato, la catena prosegue - vedi SYN_SMALL_VISION_FALLBACKS)
- synthetic/hf:nomic-ai/nomic-embed-text-v1.5 -> ["embedding-model"]
  (richiesta utente 2026-10-04: modello EMBEDDING su synthetic.new, endpoint
  /v1/embeddings; quando non disponibile cade sul modello omonimo di
  llama-swap, Qwen3-Embedding-0.6b. UN solo target: gli altri modelli
  llama-swap sono CHAT e su una richiesta embedding risponderebbero 400)
- local-model       -> [] (nessun fallback: small-model non e' piu' un
  gradino della catena - richiesta utente 2026-10-25)
- small-model       -> [] (modello autonomo, mai fallback - richiesta
  utente 2026-10-25)
- embedding-model   -> [] (nessun auto-fallback: e' l'unico modello
  embedding del fallback chain, un target CHAT su una richiesta embedding
  risponderebbe 400)
- llama-swap/* (img-*, vid-*, tts-*, music-*) -> [] (richiesta utente 2026-10-16:
  i MEDIA di llama-swap - image gen, video gen, TTS, musica - NON hanno
  fallback. Un target CHAT risponderebbe 400 su una richiesta
  immagine/audio e non esiste un "equivalente" media: l'errore dell'upstream
  va al client cosi' com'e'. Il prefisso pubblico "llama-swap/" copre
  TUTTE e quattro le wildcard parziali di config.yaml - e anche i media che
  llama-swap aggiungera' in futuro senza toccare la config - mentre i
  modelli CHAT di llama-swap hanno nomi senza prefisso e restano nelle
  catene qui sopra. Vedi LLAMA_SWAP_MEDIA_PREFIX)
- TUTTI gli altri (groq/*, gemini/*, openrouter/*, openrouter/free,
  ollama/*, ollama-cloud/* e QUALSIASI prefisso futuro)
  -> []: nessun fallback (richiesta utente 2026-10-25). La catch-all
  "*": ["small-model"] e' stata RIMOSSA da config.yaml; questo override
  per-request rende comunque esplicito il vuoto su ogni modello.

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
CATENA DI FALLBACK intera (synthetic/*; local-model era l'unico gradino fino
al 2026-10-28, ora il ramo generico synthetic/* non ha fallback).
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

POOL OPERARI = MODELLO DELLA RICHIESTA (richiesta utente 2026-10-28, che
sostituisce il pool dinamico del 2026-10-06): B, T, T2 ed E chiamano SEMPRE
lo stesso modello della richiesta. _worker_pool riceve only=<model> e
restituisce un pool di UN solo elemento (niente wildcard, niente listing,
niente ordine per tier) e la chiamata interna gira con fallbacks=[]: una
richiesta a synthetic/syn:large:text viene compattata da
synthetic/syn:large:text, una a gemini/... da gemini/..., e cosi' via.
Restano i filtri di CAPACITA': il livello si salta se il modello non
contiene il corpo (contesto reale via _window_for), se e' non-chat
(embedding/whisper/guard/rerank: 400) o un media di llama-swap. Il filtro di
PREZZO (mai modelli costosi come operari) NON si applica al modello della
richiesta: e' gia' quello scelto dal client. Conseguenza voluta: col modello
giu' o troppo piccolo per il corpo, B/T2/E non hanno piu' una rete di
sicurezza su altri modelli - resta C (BM25, locale) e poi F (413).

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
import itertools
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

# Testo aggiunto quando lo stream upstream viene troncato dopo che il client
# ha gia' ricevuto contenuto (vedi async_post_call_streaming_iterator_hook):
# meglio una risposta parziale con un avviso esplicito di un 500 che la perde.
TRUNCATED_STREAM_MARKER = (
    "\n\n[risposta interrotta: il provider ha chiuso lo stream in anticipo]"
)

# Testo che sostituisce un'immagine il cui URL il provider non puo' risolvere
# (path relativo /api/v1/files/... rimasto tale perche' OpenWebUI non e'
# riuscita a inlinearla in base64: file mancante o permessi). Vedi
# _sanitize_unresolvable_images: senza questa sostituzione synthetic risponde
# 400 "Could not parse image as valid URL" e la richiesta non parte.
UNRESOLVABLE_IMAGE_NOTE = (
    "[immagine non disponibile: file locale di OpenWebUI non risolvibile dal provider]"
)

# Avviso di THROTTLING servito come CONTENUTO di una risposta 200.
# Caso reale 2026-11-02: un provider free (Yqcloud) che risponde
# "该ip请求过多已被暂时限流 ... (目前限制了每小时60次)" come messaggio
# assistant. Non c'e' nessun 429: litellm lo tratta come un successo (niente
# retry, niente fallback) e il client riceve quel testo al posto della
# risposta. Riconosciuto da async_post_call_success_hook (-> 429 esplicito)
# e da async_post_call_streaming_iterator_hook (-> chunk esplicito).
UPSTREAM_NOTICE_MARKERS = (
    "该ip请求过多",
    "请求过多已被暂时限流",
    "chat19.aichatos.xyz",
)
# Si guarda solo l'INIZIO del contenuto: l'avviso arriva come PRIMA frase,
# mentre una risposta legittima che cita questi caratteri non va toccata.
UPSTREAM_NOTICE_SCAN_CHARS = 300
# Testo che sostituisce lo stream quando l'avviso arriva mentre si legge:
# le header SSE (200) sono gia' partite, l'unica cosa utile e' dirlo.
UPSTREAM_NOTICE_STREAM_TEXT = (
    "[provider free temporaneamente limitato (quota per IP): la risposta non "
    "e' stata generata. Riprova tra qualche minuto oppure usa un modello "
    "di un altro provider.]"
)


def _content_text(content):
    """Testo da un `content` OpenAI: stringa, oppure lista di parti."""
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        out = []
        for part in content:
            if isinstance(part, str):
                out.append(part)
                continue
            text = getattr(part, "text", None)
            if text is None and isinstance(part, dict):
                text = part.get("text")
            if isinstance(text, str):
                out.append(text)
        return " ".join(out)
    return ""


def _response_text(response):
    """Testo del contenuto di una risposta NON-streaming.

    Copre chat (choices[].message.content, anche a parti), completions
    testuali (choices[].text) e Responses API (output[].content[].text).
    None se non c'e' testo (embedding, stream, ...): chi chiama ignora.
    """
    try:
        choices = getattr(response, "choices", None)
        if choices:
            first = choices[0]
            message = getattr(first, "message", None)
            text = None
            if message is not None:
                text = _content_text(getattr(message, "content", None))
            if not text:
                text = _content_text(getattr(first, "text", None))
            return text
        output = getattr(response, "output", None)
        if output:
            parts = [_content_text(getattr(item, "content", None)) for item in output]
            text = " ".join(p for p in parts if p)
            return text or None
    except Exception:
        return None
    return None


def _chunk_text(chunk):
    """Testo del delta di un chunk SSE ("" se il chunk non porta contenuto)."""
    try:
        choices = getattr(chunk, "choices", None)
        if not choices:
            return ""
        delta = getattr(choices[0], "delta", None)
        if delta is None:
            return ""
        return _content_text(getattr(delta, "content", None))
    except Exception:
        return ""


def _upstream_notice(text):
    """True se `text` e' l'avviso di throttling servito come contenuto 200."""
    if not text:
        return False
    head = text[:UPSTREAM_NOTICE_SCAN_CHARS]
    return any(marker in head for marker in UPSTREAM_NOTICE_MARKERS)

# NOTA (richiesta utente 2026-10-25): small-model NON e' piu' un fallback di
# NESSUN modello - nessuna costante FINAL_FALLBACK. Resta solo un modello
# autonomo, invocabile direttamente; quando una catena si esaurisce l'errore
# dell'upstream arriva al client.

# synthetic/syn:small:text (richiesta utente 2026-10-12): la catena parte dal
# gemello Ollama e prosegue sui cloud GRATIS. La divergenza da
# syn:small:vision e' DURATA fino al 2026-10-14: da allora le due catene sono
# di nuovo IDENTICHE (richiesta utente 2026-10-14), vedi
# SYN_SMALL_VISION_FALLBACKS.
#   1) ollama/small-model (Ollama sullo STESSO PC 192.168.1.29:11434, via
#      ollama-gw + ollama-gate -> 503 in ~2s col PC spento: server INDIPENDENTE
#      da llama-swap, quindi regge quando llama-swap e' occupo o da 429 "Too
#      many requests" - la sua serializzazione one-request-at-a-time e' il
#      motivo principale di questo gradino. Contesto statico della wildcard
#      131072: un prompt che lo supera puo' 400 qui, la catena prosegue (hop
#      sprecato, non fatale). Il nome DEVE esistere in Ollama sul PC: se manca
#      l'upstream da 404, anche quello e' un fallimento d'hop e la catena
#      prosegue)
#   2) openrouter/free (cloud, GRATIS)
# Il nome cloud e' CONCRETO, risolto dalla wildcard "openrouter/*" del
# model_list (litellm: get_model_list -> _get_wildcard_deployments ->
# PatternMatchRouter.route: "openrouter/free" matcha "openrouter/.*" e
# instrada verso il gateway openrouter-gw) - verificato sul listing del
# proxy, NON un prefisso fittizio.
# "ollama/small-model" segue la STESSA meccanica con la wildcard "ollama/*"
# (il capture toglie "ollama/" e a Ollama arriva il nome nativo "small-model"):
# la wildcard instrada ANCHE col PC spento perche' PatternMatchRouter matcha il
# PATTERN della model_list, non il listing, che per ollama-gw e' dinamico.
# ORDINE: il gradino locale PRIMA (unico posto con i dati dell'utente; 503 in
# ~2s col PC spento, quindi i cloud restano raggiungibili subito), i cloud
# gratis come rete di sicurezza.
SYN_SMALL_TEXT = "synthetic/syn:small:text"
SYN_SMALL_TEXT_FALLBACKS = ["ollama/small-model", "openrouter/free"]

# synthetic/syn:small:vision (richiesta utente 2026-10-14): catena IDENTICA a
# syn:small:text - stessi gradini, stesso ordine. Definiti per ALIAS su
# SYN_SMALL_TEXT_FALLBACKS cosi' le due catene non possono piu' divergere
# (il ramo _apply resta separato solo per chiarezza dei commenti).
# syn:small:VISION su ollama/small-model (text-only) puo' rispondere 400
# sull'input vision: hop sprecato, la catena prosegue.
SYN_SMALL_VISION = "synthetic/syn:small:vision"
SYN_SMALL_VISION_FALLBACKS = list(SYN_SMALL_TEXT_FALLBACKS)

# synthetic/hf:nomic-ai/nomic-embed-text-v1.5 (richiesta utente 2026-10-04):
# modello EMBEDDING su synthetic.new, servito dall'endpoint /v1/embeddings
# ("api.synthetic.new/openai/v1/embeddings", non compare nel listing chat
# di synthetic). Quando non disponibile cade su llama-swap: "embedding-model"
# (Qwen3-Embedding-0.6b su CPU, contesto 16384). Catena a UN solo target:
# gli altri modelli llama-swap sono CHAT e su una richiesta embedding
# risponderebbero 400 - nessun gradino dopo.
SYN_EMBED = "synthetic/hf:nomic-ai/nomic-embed-text-v1.5"
SYN_EMBED_FALLBACKS = ["embedding-model"]

# Modelli MEDIA di llama-swap (richiesta utente 2026-10-16): image gen, video
# gen, TTS e musica del PC 192.168.1.29, esposti dalle QUATTRO wildcard
# parziali "llama-swap/img-*", "llama-swap/vid-*", "llama-swap/tts-*" e
# "llama-swap/music-*" di config.yaml (il listing di llama-swap cambia
# spesso: le wildcard lo seguono DAL VIVO via check_provider_endpoint, quindi
# un prefisso basta a coprire anche i modelli che verranno).
# I modelli CHAT di llama-swap NON stanno sotto questo prefisso: restano
# local-model / small-model / embedding-model, con le loro entry esplicite e
# i loro contesti. Quindi "llama-swap/" identifica UNIVOCAMENTE un media.
# NESSUN FALLBACK (richiesta utente 2026-10-16): un target CHAT
# su una richiesta immagine/audio risponderebbe 400 - hop inutile, e non
# esiste un "equivalente" media su cui ripiegare. L'errore deve arrivare al
# client cosi' com'e'.
LLAMA_SWAP_MEDIA_PREFIX = "llama-swap/"

# --- contesti (model_info.max_input_tokens di config.yaml) ---
# Serve la mappa QUI perche' i modelli sono wildcard/custom:
# litellm.get_max_tokens("local-model") fallisce ("isn't mapped yet") e il
# limite lo impone l'upstream. Valori ALLINEATI a config(.example).yaml:
# un contesto cambiato li' va cambiato anche qui.
CONTEXT_BY_PREFIX = (
    # GRUPPO hybrid-model: 196608 = il MINIMO vero tra i membri
    # (syn:small:text). Duplicato in PINNED_CONTEXT (unica fonte usata da
    # _window_for, che consulta questo valore solo come ultima spiaggia).
    ("hybrid-model", 196608),
    # locali: 196608 dal 2026-11-23, ALLINEATI a
    # /home/blast/.config/llama-swap/config.yaml su 192.168.1.29. local-model
    # (Strata, RX 7800 XT): --context 196608 --kv k8v4
    # --vram-reserve-mib 2048 (262144 fino al 2026-11-22, 229376 fino al
    # 2026-11-23). small-model (llama-server, RTX 3060 12GB): --ctx-size
    # 196608 --cache-type-k/v q4_0 -ngl all --fit off - MISURATO: 262144 NON
    # carica (CUDA OOM nel warm-up di load), 245760 lascia 1148 MiB.
    # Stesso numero per entrambi i locali: 196608 = anche il MINIMO del
    # gruppo hybrid-model. Il valore 163840 usato in passato era solo una
    # riduzione per RAM/VRAM rimasta qui a mano: il clamp lavorava con 98k di
    # stanza in meno. synthetic: i
    # 13 modelli sono entry ESPLICITE in config
    # dal 2026-10-06 -> qui sotto i contesti REALI per-modello (misurati sui
    # listing di api.synthetic.new). INCIDENTE 2026-10-06: GLM-5.3-Flash
    # valeva 131072 (floor di gruppo) mentre il reale e' 524288 -> cascata
    # di compattazione AVVIATA E INUTILE e 413 su un prompt che C'ENTRAVA
    # (est 148443 < 524288, _DYN_CTX vuoto al momento dell'incidente): la
    # tabella statica ora e' vera per-modello e non dipende dal daemon.
    ("local-model", 196608),
    ("small-model", 196608),
    ("embedding-model", 16384),  # llama-swap: Qwen3-Embedding-0.6b (CPU)
    # --- synthetic, contesti REALI per-modello. Match ESATTO via
    # _context_for (due passate: l'uguaglianza vince su startswith, altrimenti
    # "GLM-5.3" catturerebbe anche "-Flash" a prescindere dall'ordine).
    # Allineati a config(.example).yaml: cambiati li' -> cambiare qui.
    ("synthetic/syn:large:text", 524288),
    ("synthetic/syn:large:vision", 524288),
    ("synthetic/syn:small:text", 196608),
    ("synthetic/syn:small:vision", 262144),
    ("synthetic/hf:zai-org/GLM-5.3", 524288),
    ("synthetic/hf:zai-org/GLM-5.3-Flash", 524288),
    ("synthetic/hf:zai-org/GLM-4.7-Flash", 196608),
    ("synthetic/hf:deepseek-ai/DeepSeek-V4.1-Flash", 524288),
    ("synthetic/hf:moonshotai/Kimi-K3", 524288),
    ("synthetic/hf:Qwen/Qwen3.8-27B", 262144),
    ("synthetic/hf:openai/gpt-oss-120b", 131072),
    ("synthetic/hf:nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4", 262144),
    ("synthetic/hf:nomic-ai/nomic-embed-text-v1.5", 8192),
    ("synthetic/", 131072),  # floor per id synthetic NON in tabella
    ("ollama/", 131072),
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
    # Cloudflare Workers AI: floor = MINIMO vero del gruppo LISTATO dopo i
    # filtri di cloudflare-gw (llama-3.3-70b-instruct-fp8-fast = 24000;
    # clef-flash 24576, eurollm 32000, ...). Il contesto REALE per-modello
    # arriva dal listing del gateway (context_length -> _DYN_CTX) e ha la
    # precedenza: questo valore conta solo finche' il listing non e' stato
    # letto (startup/gateway irraggiungibile), dove un valore generoso
    # (262144, il tetto del gruppo) manderebbe un prompt enorme a un modello
    # da 24k -> 400. Allineato a model_info.max_input_tokens di config.yaml
    # (quello e' il TETTO, questo il pavimento).
    ("cloudflare/", 24000),
)

# --- reasoning_effort SUPPORTATI per-modello (synthetic) ---
# Direttiva utente 2026-11-14: default "xhigh" SOLO se il modello lo supporta;
# altrimenti si ripiega su "high" (o "max": i due sono equivalenti, direttiva
# utente); se nessuno dei due e' supportato il parametro NON viene inviato.
# Prima di questa tabella xhigh era un default STATICO in litellm_params su
# TUTTI i modelli synthetic, anche su quelli che non lo elencano (syn:small:
# text, syn:large:vision, GLM-5.3*, Kimi-K3, ...): comportamento sbagliato e,
# sulla rotta /v1/messages (bridge OpenAI Responses), 400 secco perche' li'
# litellm valida `reasoning.effort` contro il cost map (drop_params: false).
# Valori dal listing api.synthetic.new (campo reasoning_parameters.efforts,
# verificato il 2026-11-14); _DYN_EFFORTS li rilegge DAL VIVO e ha la
# precedenza, questa tabella e' il fallback quando il listing non risponde.
EFFORTS_BY_MODEL = (
    ("synthetic/syn:large:text", ("none", "low", "high", "xhigh", "max")),
    ("synthetic/syn:small:text", ("none", "low", "medium", "high")),
    ("synthetic/syn:large:vision", ("low", "high", "max")),
    ("synthetic/syn:small:vision", ("none", "low", "high", "xhigh", "max")),
    ("synthetic/hf:openai/gpt-oss-120b", ("none", "low", "medium", "high")),
    ("synthetic/hf:zai-org/GLM-5.3", ("low", "high", "max")),
    ("synthetic/hf:zai-org/GLM-5.3-Flash", ("low", "high", "max")),
    ("synthetic/hf:deepseek-ai/DeepSeek-V4.1-Flash",
     ("none", "low", "high", "xhigh", "max")),
    ("synthetic/hf:moonshotai/Kimi-K3", ("low", "high", "max")),
    ("synthetic/hf:Qwen/Qwen3.8-27B", ("low", "medium", "xhigh")),
    ("synthetic/hf:zai-org/GLM-4.7-Flash", ("none", "low", "medium", "high")),
    # embedding: nessun reasoning_effort (lista VUOTA = parametro sempre
    # rimosso). Prima aveva un xhigh copiato per inerzia dagli entry chat.
    ("synthetic/hf:nomic-ai/nomic-embed-text-v1.5", ()),
    # NVIDIA Nemotron: non compare nel listing chat -> SCONOSCIUTO; il ramo
    # synthetic/ lascia allora il default storico (xhigh).
)
# Direttive equivalenti, in ordine di preferenza (direttiva utente 2026-11-14).
EFFORT_HIGH_FAMILY = ("high", "max")
DEFAULT_CONTEXT = 131072  # il piu' piccolo: clamp conservativo per l'ignoto

# --- hybrid-model: gruppo least-busy con priorita' --------------------------
# I membri stanno in config.yaml (router_settings.routing_groups); qui vive la
# POLITICA. Tier 1 = i modelli "nostri" (locali + synthetic): hanno SEMPRE la
# precedenza. Tier 2 = il cloud a consumo (gemini-flash-latest): entra solo a
# tier 1 esaurito (tutti in cooldown/down). Il tier lo decide
# async_filter_deployments, NON routing_strategy "priority": quella, entro il
# tier, sceglie A CASO e vieta routing_strategy_args (misurato su 1.111),
# mentre qui least-busy continua a bilanciare per carico dentro il tier.
# Nome INTERNO del routing_group. Il nome PUBBLICO ("hybrid-model") e' un
# model_group_alias verso questo: /model/info - l'endpoint che AiderDesk legge
# per elencare i modelli - contiene SOLO deployment e righe ALIAS, mai i
# routing_group (misurato: con il nome pubblico sul gruppo hybrid-model
# esisteva in /v1/models ma NON in /model/info). Il filtro sotto riceve il
# nome del GRUPPO (litellm risolve l'alias prima della selezione), mentre il
# pre-call hook vede il nome mandato dal client: servono entrambi.
HYBRID_GROUP = "hybrid-model-pool"
HYBRID_PUBLIC = "hybrid-model"
HYBRID_TIER1 = ("local-model", "small-model",
                "synthetic/syn:large:text", "synthetic/syn:small:text")
# Contesto del GRUPPO = MINIMO vero tra i membri (syn:small:text = 196608).
# Va pinnato: _window_for preferisce il listing /v1/models, che per un GRUPPO
# riporta il MASSIMO tra i membri (524288). Dimensionare la cascata su 524288
# manderebbe prompt oltre la finestra di syn:small:text (400 upstream) e
# produrrebbe max_tokens che i membri cloud rifiutano.
PINNED_CONTEXT = {HYBRID_PUBLIC: 196608}
# Riserve di ULTIMA RISORSA (free): le wildcard NON possono essere membri di
# un routing_group - il pattern verrebbe scelto alla lettera e all'upstream
# arriverebbe model="*" (400, misurato). Si espandono per-request dai listing
# in cache (nomi CONCRETI) e si passano come fallbacks per-request: partono
# solo a gruppo esaurito. openrouter: SOLO id gratuiti.
HYBRID_RESERVE_PREFIXES = ("openrouter/", "cloudflare/", "groq/")
HYBRID_RESERVE_MAX = 3


def _register_hybrid_listing():
    """Pubblica i limiti di hybrid-model in /v1/models.

    Un routing_group NON ha deployment, quindi il listing_info del registro e'
    None e /v1/models pubblicava il nome SENZA max_input_tokens/
    max_output_tokens (OpenWebUI lo vedeva, ma senza limiti). Il listing pesca
    i limiti dal cost map quando il gruppo non ne ha
    (create_model_info_response -> _resolve_listing_model_info), quindi si
    registra qui la entry col nome PUBBLICO con il contesto VERO del gruppo
    (196608 input / 131072 output, come PINNED_CONTEXT). Nota: AiderDesk NON
    usa /v1/models ma
    /model/info, dove il gruppo compare grazie alle righe ALIAS di
    model_group_alias (li' i limiti sono il MINIMO tra i membri, 196608).

    Le capability sono quelle VERE del gruppo: vision e' supportata da tutti i
    membri che restano eleggibili su una richiesta con immagine
    (syn:small:text viene filtrato), quindi dichiararla e' corretto. Senza
    queste chiavi litellm le tratterebbe come False e potrebbe scartare
    parametri per una richiesta che i membri sanno gestire.
    """
    try:
        import litellm as _litellm

        # DUE nomi: /v1/models risolve l'ALIAS prima di cercare i limiti
        # (create_model_info_response -> resolve_model_group_alias ->
        # registry().group(target).listing_info, che per un routing_group e'
        # None), quindi i limiti vanno sul nome del GRUPPO; sul nome pubblico
        # servono se l'alias venisse rimosso dalla config.
        # NIENTE "mode": quella chiave finisce in /v1/models e nelle UI che
        # filtrano sul mode; i limiti bastano (richiesta utente 2026-11-23).
        _caps = {
            "max_input_tokens": 196608,
            "max_output_tokens": 131072,
            "max_tokens": 131072,
            "litellm_provider": "openai",
            "supports_vision": True,
            "supports_function_calling": True,
            "supports_parallel_function_calling": True,
            "supports_response_schema": True,
            "supports_reasoning": True,
        }
        _litellm.register_model({
            HYBRID_PUBLIC: dict(_caps),
            HYBRID_GROUP: dict(_caps),
        })
    except Exception as _err:      # mai bloccare l'import della callback
        print(f"[reasoning_clamp] register_model({HYBRID_PUBLIC}) fallita: "
              f"{type(_err).__name__}: {_err}", flush=True)


_register_hybrid_listing()
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
# MARGINE DI COMPATTAZIONE: T/T2/T3 non si fermano al PRIMO token utile, si
# fermano a ctx*(1-headroom). Il motivo e' misurato: la stima puo' restare
# qualche % sotto il conteggio del motore (pretokenizer del modello), e un
# payload compattato "a filo" del contesto (es. 251309 su 262144) resta
# sopra il vero limite e l'upstream risponde 400 "prompt ... leaves no room"
# anche se il 413 e' stato evitato (incidente 2026-11-20: T2 -> stima 214439,
# motore 301397). 5% di un contesto da 262144 = ~13k token di cuscinetto:
# costo trascurabile, e copre l'errore residuo dello stimatore.
COMPACTION_HEADROOM_PCT = 0.05
MAX_TOKEN_KEYS = ("max_tokens", "max_completion_tokens")

# --- cascata overflow: B -> C -> T -> E -> F ---
# B: compattatore = MODELLO DELLA RICHIESTA (richiesta utente 2026-10-28).
# COMPACTOR_MODELS e' solo l'ordine preferito quando il nome della richiesta
# e' un WILDCARD letterale (unico caso in cui _worker_pool resta dinamico):
# il percorso normale non sceglie piu' un compattatore lato server.
COMPACTOR_MODELS = (
    "groq/openai/gpt-oss-120b",
    # gemini-3.5-flash-lite: gemini-2.5-flash-lite e' stato RITIRATO da
    # Google (404 "no longer available to new users", verificato il
    # 2026-11-14) -> voce morta nella pool dei compattatori/riassuntori.
    "gemini/models/gemini-3.5-flash-lite",
    # "openrouter/auto" RIMOSSO (utente 2026-10-07: mai modelli costosi tra
    # gli operari): auto instrada a MODELLI QUALSIASI, scelta non controllabile
    # - puo' atterrare su un flagship costoso. "openrouter/free" (solo gratis)
    # resta nel pool via listing.
)
# Modelli che NON compattano con se stessi (richiesta utente 2026-11-02):
# local-model gira su llama-swap con UN SOLO slot di concorrenza (la seconda
# richiesta prende subito 429 "concurrency_limit"), quindi chiedergli di
# riassumere la propria storia mentre la richiesta vera attende il turno
# significa mettersi in coda dietro se stessi -> TimeoutError, 3 falli/10m,
# quarantena 15m e modello nascosto dalle UI (OpenWebUI/AiderDesk "model not
# found"). La compattazione passa quindi a small-model: stesso llama-swap ma
# processo/slot SEPARATO, quindi non aspetta lo slot di local-model.
# Chiave = modello della richiesta, valore = compattatore che esegue B/T/T2/E.
COMPACTOR_OVERRIDE = {
    "local-model": "small-model",
    # hybrid-model: il compattatore e' una CATENA ORDINATA. small-model e'
    # quello dimensionato per questo (COMPACTOR_INPUT_TOKENS), ma e' locale:
    # se e' offline la compattazione deve continuare con syn:small:text
    # invece di fallire (richiesta utente 2026-11-23). Il primo che regge il
    # corpo viene usato; il resto della cascata resta invariato.
    "hybrid-model": ("small-model", "synthetic/syn:small:text"),
}
COMPACTOR_TIMEOUT_S = 45
# Timeout per un compattatore DELEGATO (COMPACTOR_OVERRIDE): il locale che
# eredita il lavoro puo' dover essere CARICATO da llama-swap (cold start) e il
# corpo da sintetizzare e' enorme (141k token: decine di secondi di prompt
# eval anche a modello caldo). 45s farebbero fallire la compattazione proprio
# nel caso che l'override deve risolvere, e ogni TimeoutError e' un fallo in
# piu' verso la quarantena del delegato. Resta dentro WORKER_TIME_BUDGET_S.
COMPACTOR_OVERRIDE_TIMEOUT_S = 300
COMPACT_OUTPUT_TOKENS = 4096   # tetto della sintesi prodotta dal compattatore
COMPACT_MIN_CHARS = 400        # sotto questo il corpo non merita una chiamata
MIN_BRIEF_CHARS = 200          # sintesi piu' corta accettata (sotto = fallita)
COMPACT_ORIGIN = "context_compaction"
COMPRESS_TARGET_RATIO = 0.55   # C: target = 55% del contesto (margine risposta)

# T/E/T2: operario = MODELLO DELLA RICHIESTA (richiesta utente 2026-10-28).
# SUMMARIZER_PREFERRED vale solo se il nome della richiesta e' un wildcard
# letterale, quando _worker_pool resta dinamico; nel percorso normale non c'e'
# scelta ne' ordine di tier: c'e' un solo modello, quello chiamato dal client.
SUMMARIZER_PREFERRED = (
    "small-model",
    "groq/openai/gpt-oss-20b",
    # gemini-3.5-flash-lite: gemini-2.5-flash-lite e' stato RITIRATO da
    # Google (404 "no longer available to new users", verificato il
    # 2026-11-14) -> voce morta nella pool dei compattatori/riassuntori.
    "gemini/models/gemini-3.5-flash-lite",
    "synthetic/syn:small:text",
    "openrouter/free",
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
TIER_FREE = 0    # gratis: openrouter/free, id ":free", listing a prezzo
                 # zero, ollama-cloud (quota gratuita), hardware locale
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


# ---------- FILTRO COSTO degli operari (utente 2026-10-07: "NEVER use
# expensive models during prompt compression/summarization/clamping - deepseek
# v4 pro, kimi k3, ecc.") ----------
# Un operario (B compattori, T/T2/E riassunti) viene ESCLUSO dal pool (non
# solo demotato) se e' COSTOSO. Tre strati, in ordine di autorevolezza:
#   1. PREZZO DEL NOSTRO CANALE (_TIER_HINTS dai listing dei gateway:
#      synthetic e openrouter espongono pricing.prompt) - e' quello che
#      paghiamo DAVVERO. Sopra WORKER_MAX_PRICE -> MAI. VINCE sul nome: un
#      "pro" gratis resta ammesso, un "flash" a $3/M (Kimi-K3) e' escluso.
#   2. INDICE PREZZI WEB (_WEB_PRICES: listing PUBBLICO di openrouter -
#      api/v1/models senza key, ~450 modelli - fetch periodica del thread
#      daemon): copre i gateway che NON espongono prezzi (groq, gemini)
#      applicando il prezzo di mercato openrouter al modello equivalente di
#      altro provider (richiesta utente 2026-10-07), match per FAMIGLIA
#      (ultimo segmento dell'id, varianti -data/-instrutto normalizzate).
#   3. REGEXP dei FLAGSHIP (EXPENSIVE_WORKER_RES: "pro", "opus", "sonnet",
#      "large", "max", "kimi-k3", "deepseek-v4", "gpt-4/5", "o1/o3",
#      "grok-4", "claude", ...) - rete di sicurezza quando NESSUN prezzo e'
#      disponibile. WORKER_CHEAP_OVERRIDE_RES vetoa i match per le varianti
#      economiche by-design (flash/mini/nano/lite/small/haiku/fast/air).
# Mai esclusi: hardware locale e canali GRATIS (brand "free", ":free",
# ollama-cloud) - costo zero per definizione, a prescindere dal nome.
# Nessun segnale di costo -> ESCLUSO (conservativo: non si rischiano soldi
# su un costo non provato).
WORKER_MAX_PRICE_USD_PER_TOKEN = CHEAP_PRICE_USD_PER_TOKEN  # ~$0.50/M prompt
EXPENSIVE_WORKER_RES = tuple(re.compile(p) for p in (
    # suffissi/titoli premium (boundari: "minimax" NON matcha "max")
    r"(?:^|[^a-z0-9])(?:pro|opus|sonnet|ultra|premium|large|max)(?:$|[^a-z0-9])",
    r"kimi-?k[3-9]",          # flagship moonshot (k2/k2-instruct resta)
    r"deepseek-?v[4-9]",      # v4+ (chat/reasoner/v3 restano; -flash vetoato)
    r"gpt-[4-9]",             # gpt-4/5+ (gpt-oss NO: "o" non e' cifra)
    r"(?:^|[^a-z0-9])o[1-9](?:$|[^a-z0-9])",  # o1/o3/o4 reasoning (non "omni")
    r"grok-[2-9]",            # grok-4 (grok-3-mini/grok-4-fast vetoati)
    r"claude",                # qualsiasi claude (haiku vetoato)
    r"command-(?:a|r|x|l)",   # cohere premium
))
WORKER_CHEAP_OVERRIDE_RES = tuple(re.compile(p) for p in (
    r"flash", r"mini", r"nano", r"lite", r"small", r"tiny", r"haiku",
    r"fast", r"free", r"air",
))

WEB_PRICES_URL = os.environ.get(
    "REASONING_CLAMP_WEB_PRICES_URL", "https://openrouter.ai/api/v1/models")
WEB_PRICES_TTL_S = 6 * 3600    # i prezzi di mercato cambiano raramente
WEB_PRICES_TIMEOUT_S = 25
_WEB_PRICES = {"ts": 0.0, "fam": {}, "id": {}}  # famiglia/id -> USD/token
_WEB_PRICES_LOCK = threading.Lock()
_WEB_PRICES_SEEN = [False, -1]  # [log gia' dato, ultimo conteggio famiglie]


def _web_family_variants(fam):
    """Varianti normalizzate di una famiglia ('kimi-k2-instruct-0905' ->
    ['kimi-k2-instruct-0905', 'kimi-k2-instruct', 'kimi-k2']): le marche di
    data (-0905, -0423) e '-instruct' che i provider mettono nell'id non
    esistono su openrouter. Versioni vere (glm-5.3) NON vengono toccate."""
    out = []
    cur = fam
    for _ in range(4):
        if cur in out:
            break
        out.append(cur)
        nxt = re.sub(r"-\d{3,5}$", "", cur)
        if nxt == cur:
            nxt = re.sub(r"-instruct$", "", cur)
        if nxt == cur:
            break
        cur = nxt
    return out


def _ingest_web_prices(data):
    """Listing pubblico openrouter -> indice prezzi. SOLO thread daemon
    (bloccante). pricing.prompt e' USD/token (stringa, anche con $)."""
    by_id, by_fam = {}, {}
    for m in (data.get("data") or []):
        if not isinstance(m, dict):
            continue
        mid = str(m.get("id") or "").strip().lower()
        pr = m.get("pricing")
        p = _price_usd_per_token(pr.get("prompt")) if isinstance(pr, dict) \
            else None
        if not mid or p is None:
            continue
        by_id[mid] = p
        fam = mid.rsplit("/", 1)[-1]
        for v in _web_family_variants(fam):
            cur = by_fam.get(v)
            if cur is None or p < cur:
                by_fam[v] = p
    return by_id, by_fam


def _maybe_refresh_web_prices():
    """Fetch dell'indice prezzi web se stale. SOLO thread daemon (bloccante):
    stessa disciplina del listing /v1/models - mai fetch bloccanti nell'event
    loop. Host con punto = default opener (via proxy env -> llmtrim, come i
    gateway). Fallimento: si tiene l'indice vecchio, ritento al giro dopo."""
    with _WEB_PRICES_LOCK:
        stale = (time.monotonic() - _WEB_PRICES["ts"]) >= WEB_PRICES_TTL_S
    if not stale:
        return
    try:
        req = urllib.request.Request(WEB_PRICES_URL)
        with urllib.request.build_opener().open(
                req, timeout=WEB_PRICES_TIMEOUT_S) as resp:
            data = json.loads(resp.read().decode("utf-8", "replace"))
        by_id, by_fam = _ingest_web_prices(data)
        if not by_fam:
            raise ValueError("indice prezzi vuoto")
        with _WEB_PRICES_LOCK:
            _WEB_PRICES["id"] = by_id
            _WEB_PRICES["fam"] = by_fam
            _WEB_PRICES["ts"] = time.monotonic()
            n = len(by_fam)
        if not _WEB_PRICES_SEEN[0] or n != _WEB_PRICES_SEEN[1]:
            print(f"[reasoning_clamp] prezzi web (openrouter pubblico): "
                  f"{len(by_id)} modelli, {n} famiglie - operari sopra "
                  f"${WORKER_MAX_PRICE_USD_PER_TOKEN * 1e6:.2f}/M ESCLUSI",
                  flush=True)
            _WEB_PRICES_SEEN[0] = True
            _WEB_PRICES_SEEN[1] = n
    except Exception as exc:
        if not _WEB_PRICES_SEEN[0]:
            print(f"[reasoning_clamp] prezzi web non disponibili: "
                  f"{type(exc).__name__}: {str(exc)[:120]}", flush=True)


def _web_price_for(model):
    """Prezzo di mercato USD/token (PROMPT) dall'indice web per un modello
    nostro, o None se sconosciuto. Match per id SENZA il nostro prefisso
    ('synthetic/hf:moonshotai/Kimi-K3' -> 'moonshotai/kimi-k3'), poi per
    FAMIGLIA con varianti normalizzate."""
    with _WEB_PRICES_LOCK:
        by_id = _WEB_PRICES["id"]
        by_fam = _WEB_PRICES["fam"]
        if not by_id and not by_fam:
            return None
    # i dict sono RIMPIAZZATI (mai mutati) dal daemon: il riferimento letto
    # sotto lock resta valido e immutabile
    s = str(model).lower()
    core = s.split("/", 1)[-1]           # via il prefisso provider nostro
    if core.startswith("hf:"):
        core = core[3:]
    p = by_id.get(core)
    if p is not None:
        return p
    fam = core.rsplit("/", 1)[-1]
    for v in _web_family_variants(fam):
        p = by_fam.get(v)
        if p is not None:
            return p
    return None


def _worker_too_expensive(model):
    """TRUE = modello ESCLUSO dal pool degli operari di compressione/riassunto
    (utente 2026-10-07: 'NEVER use expensive models - deepseek v4 pro, kimi
    k3, ecc.'). Prezzo noto VINCE sul nome: un 'pro' gratis resta ammesso, un
    'flash' a $3/M e' escluso; nessun segnale di costo -> escluso."""
    s = str(model)
    if s in ("small-model", "local-model", "embedding-model"):
        return False                      # hardware locale: costo zero
    hint = _TIER_HINTS.get(s)
    if hint is not None and hint >= 0:    # prezzo del NOSTRO canale: autorevole
        return hint > WORKER_MAX_PRICE_USD_PER_TOKEN
    if ":free" in s:
        return False
    brand = s.split("/", 1)[0].lower()
    if "free" in brand or brand == "ollama-cloud" or s == "openrouter/free":
        return False                      # canale gratis: costo zero
    p = _web_price_for(s)                 # mercato (openrouter pubblico)
    if p is not None:
        return p > WORKER_MAX_PRICE_USD_PER_TOKEN
    low = s.lower()
    if any(rx.search(low) for rx in EXPENSIVE_WORKER_RES):
        return not any(rx.search(low) for rx in WORKER_CHEAP_OVERRIDE_RES)
    if any(pat in low for pat in CHEAP_PATTERNS):
        return False
    return True


# ESPANSIONE WILDCARD -> nomi concreti (listing dei gateway, cache TTL).
# Il model_list di litellm contiene pattern ("groq/*"): per una chiamata
# interna serve un NOME CONCRETO ("groq/openai/gpt-oss-120b"), altrimenti
# l'upstream riceve model="*" e risponde 400/404. Il listing di ogni
# gateway (openrouter-gw, groq-gw prefigissano gia' gli id;
# synthetic/ollama sono upstream diretti -> prefisso aggiunto qui) e'
# la stessa fonte usata dall'import automatico dei modelli in config.
WILDCARD_LIST_TIMEOUT_S = 8
WILDCARD_CACHE_TTL_S = 600
# tetto per-pattern: il vecchio 40 TRONCAVA i listing da ~50 id (caso storico
# del proxy inference4free, ora spento: gli id ai margini della lista
# perdevano il contesto reale 10000 -> floor statico 131072 -> prompt
# oversize spediti e fallback a monte). 300 resta un tetto per i listing
# enormi (openrouter: centinaia); nessun provider attuale lo raggiunge.
WILDCARD_MAX_MODELS = 300
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
# contesto REALE per-modello dai listing gateway/upstream (scritto da
# _fetch_pattern_models: context_length/max_model_len/context_window =
# finestra TOTALE, max_input_tokens = cap input, conservativo). Serve perche'
# i gruppi wildcard ETEROGENEI non possono dichiarare un unico contesto vero
# in config.yaml (synthetic: 131k..512k; groq/gemini: valori diversi per
# modello) e il
# listing litellm /v1/models NON porta i campi contesto per questi provider
# (get_valid_models espande solo gli id): il clamp e il pool leggono QUI.
_DYN_CTX = {}                  # id pubblico -> finestra (int)
_DYN_EFFORTS = {}              # id pubblico -> tuple(esforzi supportati)
# id pubblico -> accetta IMMAGINI in input (True/False). Solo i provider che
# lo dichiarano nel listing finiscono qui (input_modalities /
# architecture.input_modalities / supports_vision): CHI MANCA E' IGNOTO, non
# text-only. Serve alla scelta delle riserve free di hybrid-model, dove un
# modello text-only su una richiesta con immagine sarebbe un 400.
_DYN_VISION = {}
# ultimo stato (n tier1 sani, n totale) pubblicato nel log dal filtro dei
# deployment di hybrid-model: si stampa solo quando CAMBIA, perche' l'hook
# gira a ogni richiesta.
_HYBRID_TIER_STATE = [None]
# TIE-BREAK ROTANTE del filtro tier di hybrid-model. least-busy sceglie
# _least_busy_index() = il PRIMO indice con conteggio minimo
# (litellm/router_strategy/least_busy.py): a carico zero i conteggi sono tutti
# 0 e vince sempre il primo deployment dell'ordine ricevuto, quindi richieste
# sequenziali finiscono tutte sullo stesso membro (misurato: 10/10 su
# local-model). Il filtro ruota la lista a ogni richiesta e i pareggi si
# distribuiscono a rotazione; quando i conteggi differiscono (carico reale)
# decide ancora least-busy. itertools.count().next e' una singola chiamata C:
# atomica rispetto al GIL, nessun lock.
_HYBRID_RR = itertools.count()
_SWEEP_SEEN = [False]          # primo sweep contesti riuscito (log una volta)


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


def _fetch_explicit_contexts(deployments):
    """Listing di ogni api_base di deployment ESPLICITI (non wildcard) per
    contesto/prezzo per-modello: i gruppi senza wildcard (synthetic dal
    2026-10-06) non passano da _fetch_pattern_models. Per ogni id upstream U
    del listing riceve _DYN_CTX (context_length/max_model_len/context_window/
    max_input_tokens) e _TIER_HINTS (pricing.prompt) il deployment il cui
    model_name == U o finisce con "/"+U. BLOCCANTE: SOLO dal thread daemon.
    api_base raggruppati: UNA sola fetch per upstream. Stesso bypass proxy
    di _fetch_pattern_models (host con punto = default opener, altrimenti
    no-proxy)."""
    groups = {}
    for dep in deployments:
        lp = (dep.get("litellm_params") if isinstance(dep, dict)
              else getattr(dep, "litellm_params", None)) or {}
        base = lp.get("api_base")
        if not base:
            continue
        key = lp.get("api_key")
        name = (dep.get("model_name") if isinstance(dep, dict)
                else getattr(dep, "model_name", "")) or ""
        groups.setdefault((str(base), str(key) if key else ""), []).append(str(name))
    for (base, key), names in groups.items():
        try:
            url = str(base).rstrip("/") + "/models"
            req = urllib.request.Request(url)
            if key and key != "dummy":
                req.add_header("Authorization", "Bearer " + key)
            host = urllib.parse.urlparse(url).hostname or ""
            opener = (urllib.request.build_opener() if "." in host
                      else urllib.request.build_opener(
                          urllib.request.ProxyHandler({})))
            with opener.open(req, timeout=WILDCARD_LIST_TIMEOUT_S) as resp:
                data = json.loads(resp.read().decode("utf-8", "replace"))
        except Exception as exc:
            # gap osservabilita' incidente 2026-10-06: il fallimento del
            # listing era MUTO e i contesti restavano sui valori statici
            print(f"[reasoning_clamp] listing {base} non disponibile "
                  f"({type(exc).__name__}: {str(exc)[:120]})", flush=True)
            continue
        for m in (data.get("data") or []):
            if not isinstance(m, dict):
                continue
            uid = str(m.get("id") or "").strip()
            if not uid:
                continue
            ctx = None
            for f in ("context_length", "max_model_len", "context_window",
                      "max_input_tokens"):
                v = m.get(f)
                if isinstance(v, (int, float)) and v > 0:
                    ctx = int(v)
                    break
            price = None
            pr = m.get("pricing")
            if isinstance(pr, dict):
                price = _price_usd_per_token(pr.get("prompt"))
            # reasoning_parameters.efforts: i valori di reasoning_effort che
            # il modello accetta (synthetic li espone; altri provider no).
            efforts = None
            rp = m.get("reasoning_parameters")
            if isinstance(rp, dict):
                raw = rp.get("efforts")
                if isinstance(raw, (list, tuple)):
                    vals = tuple(str(x).strip().lower() for x in raw
                                 if isinstance(x, str) and str(x).strip())
                    if vals:
                        efforts = vals
            if ctx is None and price is None and efforts is None:
                continue
            for name in names:
                if name == uid or name.endswith("/" + uid):
                    if ctx is not None:
                        _DYN_CTX[name] = ctx
                    if price is not None:
                        _TIER_HINTS[name] = price
                    if efforts is not None:
                        _DYN_EFFORTS[name] = efforts


def _efforts_for(model):
    """reasoning_effort supportati per ``model``, o None se sconosciuto.

    Prima il listing DAL VIVO (_DYN_EFFORTS, vedi _fetch_explicit_contexts),
    poi la tabella statica EFFORTS_BY_MODEL: match ESATTO prima, poi prefisso
    (stessa regola di _context_for: "synthetic/hf:zai-org/GLM-5.3" non deve
    catturare "...GLM-5.3-Flash" per startswith).
    """
    found = _DYN_EFFORTS.get(model)
    if found:
        return found
    for name, effs in EFFORTS_BY_MODEL:
        if model == name:
            return effs
    for name, effs in EFFORTS_BY_MODEL:
        if model.startswith(name):
            return effs
    return None


def _pick_effort(requested, supported):
    """Applica la direttiva utente 2026-11-14 sul reasoning_effort.

    - default (client silenzioso): xhigh se supportato, altrimenti high/max,
      altrimenti NIENTE parametro;
    - client esplicito: il valore resta se supportato; xhigh/max scivolano su
      high/max disponibili, altrimenti il parametro viene DROPPATO (meglio
      nessun reasoning che un 400);
    - high e max sono equivalenti: se manca uno dei due vale l'altro.
    """
    if not requested:
        if "xhigh" in supported:
            return "xhigh"
        for e in EFFORT_HIGH_FAMILY:
            if e in supported:
                return e
        return None
    eff = str(requested).strip().lower()
    if eff in supported:
        return eff
    if eff in ("xhigh", "max"):
        for e in EFFORT_HIGH_FAMILY:
            if e in supported:
                return e
        return None
    if eff == "high":
        for e in EFFORT_HIGH_FAMILY:
            if e in supported:
                return e
    return eff


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
        # contesti per-modello dai listing gateway/upstream: i WILDCARD del
        # model_list via _fetch_pattern_models, le entry ESPLICITE via
        # _fetch_explicit_contexts (synthetic dal 2026-10-06). Alimenta
        # _DYN_CTX per clamp/pool: anche il modello MAIN ha cosi' il suo
        # contesto REALE dal vivo senza aspettare la prima cascata
        # sweep VISIBILE (gap osservabilita' incidente 2026-10-06: il sweep
        # dei contesti era muto). Si stampa solo al primo successo e quando
        # il numero di contesti cambia - il daemon gira ogni TTL, niente spam
        try:
            from litellm.proxy.proxy_server import llm_router
            wildcards = []
            explicit = []
            for dep in (getattr(llm_router, "model_list", None) or []):
                n = (dep.get("model_name") if isinstance(dep, dict)
                     else getattr(dep, "model_name", None))
                n = str(n) if n else ""
                if n.endswith("/*"):
                    wildcards.append(n)
                elif n:
                    explicit.append(dep)
            n_ctx = len(_DYN_CTX)
            for w in wildcards:
                _fetch_pattern_models(w)
            if explicit:
                _fetch_explicit_contexts(explicit)
            if len(_DYN_CTX) != n_ctx or not _SWEEP_SEEN[0]:
                print(f"[reasoning_clamp] sweep contesti: "
                      f"{len(_DYN_CTX)} voci in _DYN_CTX "
                      f"({len(wildcards)} wildcard, {len(explicit)} "
                      f"deployment espliciti)", flush=True)
                _SWEEP_SEEN[0] = True
        except Exception as exc:
            if not _SWEEP_SEEN[0]:
                print(f"[reasoning_clamp] sweep contesti non riuscito: "
                      f"{type(exc).__name__}", flush=True)
        # INDICE PREZZI WEB (openrouter pubblico, nessuna key): classifica
        # come COSTOSI anche i modelli dei gateway che NON espongono prezzi
        # (groq, gemini) - utente 2026-10-07: mai operari costosi
        _maybe_refresh_web_prices()
        # primo aggancio mancato (proxy in avvio): ritento presto, non tra 10'
        time.sleep(PROXY_MODELS_TTL_S
                   if (got or _PROXY_MODELS_CACHE["models"]) else 30)


def _ensure_proxy_models_refresher():
    """Avvia il thread daemon (una sola volta per processo)."""
    if _PROXY_MODELS_STARTED[0]:
        return
    with _PROXY_MODELS_LOCK:
        if _PROXY_MODELS_STARTED[0]:
            return
        _PROXY_MODELS_STARTED[0] = True
    threading.Thread(target=_refresher_proxy_models, daemon=True,
                     name="reasoning-clamp-models").start()


# avvio EAGER all'import: il listing e' gia' pronto al PRIMO overflow dopo un
# restart (il lazy partirebbe solo alla prima cascata, che leggerebbe la
# cache ancora vuota) e i cambi di config vengono seguiti da subito; se il
# proxy non ascolta ancora (fase di avvio) il thread ritenta ogni 30s
_ensure_proxy_models_refresher()

# ---------------- QUARANTENA dei modelli che si spaccano -------------------
# Due richieste utente che qui sono UNA sola cosa:
#  - 2026-10-06: blacklist temporanea (incrementale) degli operari che
#    fallivano troppo spesso in compressione/riassunto;
#  - 2026-11-02: "escludere da litellm con un TTL i modelli che chiamiamo che
#    si spaccano troppo spesso o vanno in ratelimit, cosi' un provider
#    esterno non li 'vede' momentaneamente".
# Stesso stato, stesso contatore. L'id e' il NOME PUBBLICO del modello
# (es. groq/openai/gpt-oss-120b), alimentato da tre sorgenti:
#  - i tentativi interni della cascata (eccezione o risposta vuota),
#  - i fallimenti delle richieste dei CLIENT (async_post_call_failure_hook:
#    429, 5xx, timeout, errore di connessione; NON 400/401/404/context),
#  - l'avviso di quota travestito da 200 (-> 429) e lo stream troncato.
# Finestra scivolante: QUAR_FAIL_THRESHOLD falli entro QUAR_FAIL_WINDOW_S ->
# modello in quarantena per QUAR_BASE_S; ogni recidiva RADDOPPIA (tetto
# QUAR_MAX_S). Un successo azzera i falli e fa decadere uno strike; un
# successo DURANTE la quarantena la chiude subito (il provider e' tornato).
# Effetti della quarantena:
#  - il modello SPARISCE dalla listing (async_filter_listed_models ->
#    /v1/models, /model/info, /model_group/info; /v1/models/{id} risponde
#    404) finche' il TTL non scade: nessuna config toccata, la wildcard
#    resta e il nome torna da solo;
#  - NON viene scelto come operaio dalla cascata (fits() in _worker_pool),
#    cosi' una compressione non spreca il primo tentativo;
#  - resta CHIAMABILE se un client lo chiede esplicitamente (nascondere non
#    e' bloccare): cosi' un modello tornato sano si riabilita al primo
#    successo, senza aspettare il TTL.
# MAI in quarantena i router automatici (`auto`, `*/auto`): sono il ripiego
# per definizione, nasconderli svuoterebbe la lista dei ripieghi. Restano
# contati? No: non si contano nemmeno i loro falli.
QUAR_FAIL_WINDOW_S = 600        # finestra dei falli (10 min)
QUAR_FAIL_THRESHOLD = 3         # falli nella finestra -> quarantena
QUAR_BASE_S = 900               # prima quarantena: 15 min
QUAR_MAX_S = 21600              # tetto escalation: 6h (era 24h quando valeva
                                # solo per gli operari: per la listing e' troppo)
QUAR_PROTECTED_SUFFIX = "/auto"  # oltre al nome esatto "auto"
_QUAR_FAILS = {}                # id -> [timestamp dei falli nella finestra]
_QUAR_BANS = {}                 # id -> [until_monotonic, strikes]
_QUAR_LOCK = threading.Lock()
_QUAR_LAST_HIDDEN = [set()]     # ultimo insieme loggato (solo per il log)


def _quar_protected(w):
    """True per i modelli che non vanno MAI in quarantena (router automatici)."""
    if not w:
        return False
    s = str(w)
    return s == "auto" or s.endswith(QUAR_PROTECTED_SUFFIX)


def _quarantined(w):
    """True se `w` e' in quarantena ADESSO.

    Lettura lock-free: get atomico sotto GIL; la scadenza e' verificata sul
    monotonic corrente, cosi' la listing smette di filtrare da sola allo
    scadere del TTL (nessun thread di pulizia).
    """
    if not w or _quar_protected(w):
        return False
    ban = _QUAR_BANS.get(w)
    return bool(ban and ban[0] > time.monotonic())


def _quar_note_fail(w, err=""):
    """Conta un fallo di `w` (eccezione, risposta vuota, quota travestita);
    alla soglia mette il modello in quarantena, con log e durata raddoppiata
    a ogni recidiva. I protetti (*/auto) non vengono nemmeno contati."""
    if not w or _quar_protected(w):
        return
    now = time.monotonic()
    with _QUAR_LOCK:
        fl = [t for t in _QUAR_FAILS.get(w, [])
              if now - t < QUAR_FAIL_WINDOW_S]
        fl.append(now)
        _QUAR_FAILS[w] = fl
        if len(fl) < QUAR_FAIL_THRESHOLD:
            return
        ban = _QUAR_BANS.get(w)
        if ban and ban[0] > now:
            return                      # gia' in quarantena: nessun nuovo log
        strikes = (ban[1] if ban else 0) + 1
        dur = min(QUAR_MAX_S, QUAR_BASE_S * (2 ** (strikes - 1)))
        _QUAR_BANS[w] = [now + dur, strikes]
        _QUAR_FAILS[w] = []
        print(f"[reasoning_clamp] quarantena {w} per {dur // 60}m "
              f"(recidiva {strikes}, {len(fl)} falli/"
              f"{QUAR_FAIL_WINDOW_S // 60}m"
              f"{': ' + str(err)[:80] if err else ''}) -> fuori dalla listing",
              flush=True)


def _quar_note_ok(w):
    """Successo: azzera i falli e fa decadere uno strike; se il modello era
    in quarantena la chiude subito (ha risposto di nuovo)."""
    if not w:
        return
    now = time.monotonic()
    with _QUAR_LOCK:
        _QUAR_FAILS.pop(w, None)
        ban = _QUAR_BANS.get(w)
        if not ban:
            return
        was_banned = ban[0] > now
        strikes = ban[1] - 1
        if strikes <= 0:
            _QUAR_BANS.pop(w, None)
        else:
            _QUAR_BANS[w] = [0, strikes]   # strike decaduto, non dimenticato
        if was_banned:
            print(f"[reasoning_clamp] quarantena: {w} riabilitato (ha risposto "
                  f"di nuovo) -> di nuovo nella listing", flush=True)


def _quar_failure_kind(exc):
    """Etichetta del guasto se e' colpa del MODELLO, altrimenti None.

    Contano solo i guasti del provider: rate limit, 5xx, timeout, errore di
    connessione. Un 400/401/404 o un context overflow sono errori NOSTRI
    (parametri, chiave, prompt troppo lungo): mettere in quarantena il modello
    per quelli nasconderebbe modelli sani.
    """
    code = getattr(exc, "status_code", None)
    if code == 429:
        return "rate-limit"
    if isinstance(code, int) and 500 <= code <= 599:
        return f"http-{code}"
    name = type(exc).__name__
    if "Timeout" in name:
        return "timeout"
    if "Connection" in name:
        return "connection"
    if "RateLimit" in name:
        return "rate-limit"
    if ("InternalServer" in name or "ServiceUnavailable" in name
            or "BadGateway" in name):
        return "server"
    return None


def _price_usd_per_token(v):
    """pricing.prompt dei listing -> USD/token (float) o None (nascosto/
    assente/non numerico). Formati visti: openrouter '0.00000069' (stringa),
    synthetic '$0.0000006' (stringa CON $ - il vecchio float() lanciava
    ValueError e i prezzi synthetic non arrivavano MAI ai tier/filter),
    numeri gia' float."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v) if v >= 0 else None
    s = str(v).strip().lstrip("$").replace(",", "").strip()
    if not s:
        return None
    try:
        p = float(s)
    except ValueError:
        return None
    return p if p >= 0 else None


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
            if isinstance(pr, dict):
                p = _price_usd_per_token(pr.get("prompt"))
                if p is not None:
                    _TIER_HINTS[full] = p
            # contesto REALE per-modello (vedi _DYN_CTX): campo per provider -
            # synthetic/openrouter "context_length", groq "context_window",
            # llama.cpp "max_model_len"; "max_input_tokens"
            # (cap input) solo come ultima scelta, conservativo
            for f in ("context_length", "max_model_len", "context_window",
                      "max_input_tokens"):
                v = m.get(f) if isinstance(m, dict) else None
                if isinstance(v, (int, float)) and v > 0:
                    _DYN_CTX[full] = int(v)
                    break
            # capacita' VISION del modello (vedi _DYN_VISION): campo diverso
            # per provider, quindi si provano le forme note e si lascia
            # IGNOTO chi non dichiara nulla.
            if isinstance(m, dict):
                mods = m.get("input_modalities")
                if mods is None and isinstance(m.get("architecture"), dict):
                    mods = m["architecture"].get("input_modalities")
                if isinstance(mods, (list, tuple)) and mods:
                    _DYN_VISION[full] = any(
                        str(x).strip().lower() in ("image", "vision")
                        for x in mods)
                elif isinstance(m.get("supports_vision"), bool):
                    _DYN_VISION[full] = m["supports_vision"]
            # reasoning_effort supportati dal modello, quando il GATEWAY li
            # espone nel listing (cloudflare-gw li ricava da supported_efforts
            # di Workers AI). Prima questo campo veniva letto solo per le
            # entry ESPLICITE (_fetch_explicit_contexts): cosi' anche i
            # modelli di una WILDCARD hanno la lista autorevole e il ramo
            # per-provider puo' validare invece di indovinare.
            rp = m.get("reasoning_parameters") if isinstance(m, dict) else None
            if isinstance(rp, dict):
                raw_eff = rp.get("efforts")
                if isinstance(raw_eff, (list, tuple)):
                    vals = tuple(str(x).strip().lower() for x in raw_eff
                                 if isinstance(x, str) and str(x).strip())
                    if vals:
                        _DYN_EFFORTS[full] = vals
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
# Budget della SINTESI (chunk di T/T2/E). Non e' solo "quanto puo' essere
# lungo il brief": su un modello con --reasoning attivo il ragionamento esce
# dallo STESSO budget, e un chunk grande lo consuma tutto prima di scrivere
# una riga di risposta. MISURATO 2026-11-20 su small-model: chunk da 48k token
# con max_tokens 1024 -> finish=length, 962 token di reasoning e brief
# TRONCATO; chunk piu' grandi -> content VUOTO ("empty"), che vale uno strike
# di quarantena e uccide la cascata (incidente: 471k token di prompt ->
# "T:small-model:empty" -> 413). MISURATO: il brief reale di un chunk da 136k
# token e' ~1025 token TOTALI (reasoning + testo) con finish=stop, ma su un
# chunk LARGO (185k) il ragionamento da solo supera i 2048 e il content torna
# VUOTO - misurato: con 2048 il primo chunk e' andato in "empty" e il retry a
# 8192 ha bruciato tutto il budget del livello (413 con T2/T3 "no-post-user").
# 4096 e' il primo tentativo che regge anche i chunk larghi (la generazione
# reale resta ~1000 token, quindi non costa tempo: ~40 s a 23 tok/s).
MAP_OUTPUT_TOKENS = 4096
BRIEF_RETRY_TOKENS = 8192
MAP_CHUNK_INPUT_TOKENS = 90000  # + output < contesto dei piu' piccoli (131072)
MAP_MAX_CHUNKS = 8
# Timeout di UNA chiamata di riassunto. Con i chunk larghi di un compattatore
# LOCALE (fino a COMPACTOR_INPUT_TOKENS) il prefill da solo costa: misurato
# 2026-11-20 su small-model (RTX 3060, ~700-1178 tok/s su codice denso) -> un
# chunk da 163840 tok impiega ~230-290 s di prefill + sintesi: con i vecchi
# 180 s andrebbe SEMPRE in timeout (strike di quarantena -> cascata senza
# operaio).
# Il tetto per livello resta WORKER_TIME_BUDGET_S: qui basta non tagliare una
# singola chiamata legittima.
MAP_TIMEOUT_S = 420
MAP_ORIGIN = "context_map_reduce"

# TETTO di INPUT per un compattatore LOCALE. Il contesto DICHIARATO non e'
# quello che il motore riesce a TENERE: un chunk da 90k token e' dimensionato
# sui contesti cloud, non su una scheda da 12GB.
#
# MISURATO 2026-11-20 (llama-swap su 192.168.1.29, config.yaml aggiornato):
#   small-model (RTX 3060 12GB, Ornith-1.5-9B Q4_K_M + mmproj Q8, KV q4_0,
#   --fit off): --ctx-size 262144 NON CARICA (CUDA OOM dentro
#   server_context_impl::load_model -> il warm-up decodifica l'intero
#   contesto), 245760 carica con 1148 MiB liberi. Dal 2026-11-23 il
#   --ctx-size e' 196608 su ENTRAMBI i locali: lascia margine vero a un
#   browser/Xorg che cresce (ogni 16k di ctx costa ~215 MiB).
#   Prefill: 1178 tok/s su codice denso -> 196k tok in ~167 s.
#   local-model (RX 7800 XT 16GB, Strata, --vram-reserve-mib 2048): 262144
#   reggeva (15.0 GiB usati) ma dal 2026-11-23 e' sceso a 196608, allineato
#   a small-model, cosi' un solo numero vale per entrambi i locali.
# Conseguenza del valore sbagliato: small-model moriva a OGNI richiesta
# (84 crash CUDA in un giorno), la cascata restava senza operaio e finiva in
# 413 "T2:no-gain" (incidente 2026-11-20 17:23, prompt 265182 tok).
#
# Il tetto NON e' il contesto del motore (la KV e' preallocata al load: un
# chunk piu' corto non risparmia VRAM) e va scelto sul COMPORTAMENTO.
#
# Storia delle misure (llama-swap, 2026-11-20):
#   - 185k token con thinking ON: il ragionamento mangia 4096 token, content
#     VUOTO, e anche il retry a 8192 resta VUOTO (misurato: finish=length).
#     E' il motivo per cui il tetto era stato abbassato a 131072.
#   - da allora _brief_call spegne il thinking sui compattatori locali
#     (chat_template_kwargs.enable_thinking=false): stessa macchina, chunk da
#     163307 token -> finish=stop, brief da 4533 char, 294 s TOTALI (prefill
#     ~700-900 tok/s + 1187 token di sintesi a ~23 tok/s). Il tetto a 131072
#     non serve piu'.
#   - 163840 = 20 x 8192: il piu' largo che resta dentro UNA chiamata
#     (MAP_TIMEOUT_S 420 s) con margine; il contesto del motore (196608)
#     lascia ~33k token per output + template, e la richiesta FINALE (non il
#     chunk) resta li'.
# Oltre: un chunk da 196k costa ~350 s di solo prefill e sfonda il budget di
# livello (WORKER_TIME_BUDGET_S) al primo chunk -> riassunti parziali.
COMPACTOR_INPUT_TOKENS = {
    "small-model": 163840,
    "local-model": 163840,
}

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
# T2: compressione dei messaggi DOPO l'ultimo user (risultati tool,
# assistant). B/C/E tengono la coda verbatim e T tocca SOLO il mezzo
# dell'ultimo user: INCIDENTE 2026-10-06 coda 126640 tok ma ultimo user
# 842 -> T "tail-too-small", B/C/E fermi, 413 assicurato.
T2_MAX_MESSAGES = 12    # tetto di messaggi compressi in una passata
T2_TIME_BUDGET_S = 420  # budget TOTALE di wall-time della passata (TTL)
T2_HEADER = (
    "[NOTA: questo messaggio (lungo, tipicamente un risultato tool) e' "
    "stato COMPRESSO dal proxy per rientrare nel contesto: qui sotto un "
    "riassunto fedele del contenuto originale]\n"
)
# T3: TRONCAMENTO deterministico (nessun modello). Ultimo livello prima del
# 413: quando T/T2 non possono lavorare (operaio di compattazione giu'/in
# quarantena/OOM o troppo lento per il budget) serve un livello che GUADAGNA
# SEMPRE. Incidente 2026-11-20 17:23: coda 260578 tok, ultimo user 536,
# small-model morto -> T skip, T2 "no-gain", 413. Taglia testa+coda (inizio e
# fine sono la parte utile di un tool result) dei messaggi DOPO l'ultimo user
# e, per ultimo, il MEZZO dell'ultimo user lasciando le ancore verbatim (li'
# nessun altro livello deterministico arriva: T dipende dall'operaio).
T3_HEADER = (
    "[NOTA: contenuto TRONCATO dal proxy per rientrare nel contesto: "
    "restano inizio e fine VERBATIM, il centro e' stato rimosso]\n"
)
T3_MIN_KEEP_CHARS = 2000   # sotto questo tagliare non ha piu' senso: si passa
                           # al messaggio successivo (o al 413)
T3_TRIM_SLACK_TOKENS = 512  # arrotondamento token->char: senza slack il
                            # taglio resta 1-2 token sopra il target
T3_HEAD_RATIO = 0.35       # quota di testa del testo che resta (il resto coda)

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
# JSON/numeri: sopra EXTREME_DENSITY il ratio scende ancora. MISURATO
# 2026-11-20 (llama-server, /tokenize, small-model) su ~50k numeri dentro
# JSON denso: 2.04 char/token reali contro 2.41 stimati con il floor 2.8.
EXTREME_DENSITY = 0.35
EXTREME_CHARS_PER_TOKEN = 2.2
DENSE_DENSITY = 0.20          # >= 20% simboli/non-parole: codice/JSON
MIXED_DENSITY = 0.10
DIGITS = "0123456789"
# Correzione delle cifre sul conteggio tiktoken (vedi _tiktoken_corrected):
# cl100k le fonde, il pretokenizer llama le spezza. MISURATO 2026-11-20.
DIGIT_TOKEN_EXTRA = 0.6
TIKTOKEN_MIN_CHARS = 150000   # tiktoken solo sui prompt grossi (~30k+ token)
_TIKTOKEN_ENC = None
_TIKTOKEN_FAILED = False


def _ratio_from_density(density, cjk, long_alnum):
    """caratteri/token attesi da una densita' di simboli."""
    if cjk:
        return CJK_CHARS_PER_TOKEN
    if density >= EXTREME_DENSITY:
        # numeri+punteggiatura (JSON compatto, log, tabelle): interpolazione
        # fino a EXTREME_CHARS_PER_TOKEN invece di fermarsi a 2.8
        t = min(1.0, (density - DENSE_DENSITY)
                / (EXTREME_DENSITY - DENSE_DENSITY))
        return CHARS_PER_TOKEN - t * (CHARS_PER_TOKEN - EXTREME_CHARS_PER_TOKEN)
    if density >= DENSE_DENSITY:
        return CHARS_PER_TOKEN
    if density <= MIXED_DENSITY:
        # prosa con base64/hash/sha dentro: resta prudente
        return (PROSE_CHARS_PER_TOKEN if long_alnum < 3
                else MIXED_CHARS_PER_TOKEN)
    # misto: interpolazione lineare tra prosa e denso
    t = (density - MIXED_DENSITY) / (DENSE_DENSITY - MIXED_DENSITY)
    return MIXED_CHARS_PER_TOKEN + t * (CHARS_PER_TOKEN - MIXED_CHARS_PER_TOKEN)


def _text_stats(text):
    """(char/token, cifre) di UN testo, in un solo giro.

    Il ratio e' calcolato sui caratteri NON cifra: il pretokenizer di
    llama.cpp spezza i numeri (~1 token per cifra), quindi le cifre non
    vanno diluite nella densita' di simboli - vanno contate a parte.
    MISURATO 2026-11-20 con /tokenize su small-model (Ornith-1.5-9b):
      log con date e contatori  213200 char, 24% cifre -> 98800 tok (2.16)
      JSON con indici           358836 char, 14% cifre -> 176171 tok (2.04)
      codice Python             200000 char,  2% cifre ->  57316 tok (3.49)
      prosa                     200000 char,  0% cifre ->  37989 tok (5.26)
    La vecchia stima dava 70200 e 149001: 29% e 15% SOTTO il motore, cioe'
    esattamente il buco che faceva finire la compattazione ~90k token oltre
    il contesto (incidente 2026-11-20: stima 214439, motore 301397 -> 400).
    """
    n = len(text)
    if not n:
        return PROSE_CHARS_PER_TOKEN, 0
    digits = 0
    for d in DIGITS:
        digits += text.count(d)
    nd = n - digits
    if not nd:
        return PROSE_CHARS_PER_TOKEN, digits
    cjk = len(CJK_RE.findall(text)) / n > 0.15
    density = len(NON_WORD_RE.findall(text)) / nd
    long_alnum = len(LONG_ALNUM_RE.findall(text))
    return _ratio_from_density(density, cjk, long_alnum), digits


def _chars_per_token(text):
    """Ratio caratteri/token dell'intero testo (calcolato sui caratteri
    NON cifra, vedi _text_stats). Resta per i chiamanti che vogliono solo
    un ordine di grandezza; per le STIME usare _ratio_estimate."""
    return _text_stats(text)[0]


def _ratio_estimate(text):
    """Token stimati di un testo: cifre a 1 token l'una (peggior caso del
    pretokenizer llama) + resto al ratio della sua classe di densita'."""
    n = len(text)
    if not n:
        return 0
    ratio, digits = _text_stats(text)
    nd = n - digits
    return digits + (int(nd / ratio) if nd else 0)


def _digit_count(text):
    """Quante CIFRE contiene il testo (scan in C, una per decina)."""
    n = 0
    for d in DIGITS:
        n += text.count(d)
    return n


def _tiktoken_corrected(texts):
    """tiktoken + correzione per le CIFRE.

    cl100k fonde i numeri (un "2026" e' un token, "123456" magari due),
    il pretokenizer di llama.cpp li spezza: misurato 2026-11-20 su
    small-model, la differenza e' 0.5-0.55 token per cifra su log/JSON
    numerici. 0.6 e' la stessa misura con un pelo di margine. Sui testi
    senza cifre (prosa, codice, base64) non cambia nulla.
    """
    tk = _tiktoken_estimate(texts)
    if tk is None:
        return None
    extra = int(sum(_digit_count(t) for t in texts if t) * DIGIT_TOKEN_EXTRA)
    return tk + extra


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


def _request_has_image(data):
    """True se `data` contiene un'IMMAGINE in input.

    Accetta il body completo della richiesta oppure la sola lista `messages`
    e copre le due forme che attraversano questo stack: chat
    (`{"type": "image_url", "image_url": {...}}`) e Responses
    (`{"type": "input_image", ...}`, anche dentro function_call_output).
    Nessuna scansione del testo: si guardano solo chiavi e tipi, con un tetto
    di profondita' per non esplorare strutture patologiche."""
    def scan(obj, depth=0):
        if depth > 6:
            return False
        if isinstance(obj, dict):
            if obj.get("type") in ("image_url", "input_image", "image"):
                return True
            if obj.get("image_url") is not None \
                    or obj.get("input_image") is not None:
                return True
            return any(scan(v, depth + 1) for v in obj.values())
        if isinstance(obj, (list, tuple)):
            return any(scan(v, depth + 1) for v in obj)
        return False

    if data is None:
        return False
    try:
        return scan(data)
    except Exception:
        return False


class ReasoningClamp(CustomLogger):

    def _context_for(self, model):
        """STATICo: mappa allineata a config(.example).yaml - per i gruppi
        wildcard senza tabella per-modello e' il valore MINIMO vero del
        gruppo. Per il contesto REALE usare _window_for. DUE passate:
        prima l'uguaglianza ESATTA (entry per-modello), poi il prefisso -
        con un matcher singolo "synthetic/hf:zai-org/GLM-5.3" catturerebbe
        anche "...GLM-5.3-Flash" via startswith, a prescindere dall'ordine
        della tabella."""
        for pref, ctx in CONTEXT_BY_PREFIX:
            if model == pref.rstrip("/"):
                return ctx
        for pref, ctx in CONTEXT_BY_PREFIX:
            if model.startswith(pref):
                return ctx
        return DEFAULT_CONTEXT

    def _window_for(self, model):
        """Finestra TOTALE utilizzabile (prompt + output) di `model`,
        PER-MODELLO quando possibile: 1) listing gateway/upstream (_DYN_CTX:
        context_length/max_model_len/context_window - synthetic, groq,
        openrouter...), 2) listing litellm /v1/models (max_input_tokens
        del registro: openrouter e simili; usato da solo perche' per i
        locali e' la finestra intera e nel registro e' <= finestra reale),
        3) mappa statica (minimo vero del gruppo). Cosi' syn:large:text
        usa davvero 512k e hf:openai/gpt-oss-120b resta a 131k senza 400."""
        # PINNED PRIMA DI TUTTO: per un GRUPPO il listing litellm riporta il
        # MASSIMO tra i membri (hybrid-model -> 524288), che e' il valore
        # SBAGLIATO per dimensionare clamp e compattazione: il minimo vero e'
        # 196608 (syn:small:text). Vedi PINNED_CONTEXT.
        pinned = PINNED_CONTEXT.get(model)
        if pinned:
            return pinned
        ctx = _DYN_CTX.get(model)
        if ctx:
            return ctx
        with _PROXY_MODELS_LOCK:
            mit = _PROXY_MODELS_CACHE["models"].get(model, (None, None))[0]
        if mit is not None:
            return mit
        return self._context_for(model)

    @staticmethod
    def _estimate_text_tokens(text):
        """Token stimati di UN testo: stessa regola di
        _estimate_prompt_tokens (ratio per densita' e, per i testi grandi,
        tiktoken se piu' alto). Serve a T3 per convertire in caratteri il
        deficit di token con il ratio REALE di quel testo."""
        if not isinstance(text, str) or not text:
            return 0
        est = _ratio_estimate(text)
        if len(text) >= TIKTOKEN_MIN_CHARS:
            tk = _tiktoken_corrected([text])
            if tk and tk > est:
                est = tk
        return est

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
        tool_tokens = sum(_ratio_estimate(j) for j in tool_texts) + fixed
        # tiktoken (cachato sha1) quando disponibile: e' il conteggio piu'
        # fedele anche sui grandi schemi MCP (misurato 2026-11-20: 20251 su
        # 20891 reali contro 30256 del modello a densita'), corretto per le
        # cifre come i messaggi. Il ratio resta solo fallback: prima qui
        # c'era un MIN(tiktoken, ratio) che sottostimava del 30% i tool
        # result numerici (log, JSON con indici: 3 x 98800 reali contati
        # 70200 -> payload compattato 90k token OLTRE il contesto).
        tk = _tools_tiktoken(tool_texts)
        if tk is not None:
            tk += int(sum(_digit_count(j) for j in tool_texts)
                      * DIGIT_TOKEN_EXTRA)
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
        est = sum(_ratio_estimate(t) for t in texts)
        chars = sum(len(t) for t in texts)
        if chars >= TIKTOKEN_MIN_CHARS:
            tk = _tiktoken_corrected(texts)
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

    @staticmethod
    def _settled(est, ctx):
        """True quando un livello di compattazione puo' SMETTERE: la stima
        sta sotto ctx*(1-COMPACTION_HEADROOM_PCT). Piu' stretto di
        _saturated: la compattazione deve lasciare il cuscinetto per
        l'errore residuo dello stimatore (vedi la costante)."""
        return est <= int(ctx * (1.0 - COMPACTION_HEADROOM_PCT))

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
        # DIRETTIVA UTENTE 2026-10-06: la catena di fallback deve tenere
        # conto del contesto DA CUI il modello e' partito -> il budget e'
        # quello del modello RICHIESTO (finestra reale via _window_for), NON
        # il minimo della catena: un prompt grande destinato a un modello da
        # 512k non viene piu' striminzito per farcelo stare anche nei
        # fallback da 196k. Se il primario cade e il prompt non entra nel
        # target di fallback, il target risponde 400 e la catena avanza
        # (comportamento accettato esplicitamente).
        ctx = self._window_for(model)
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
        # stessa regola di _clamp_output (direttiva utente 2026-10-06): il
        # budget e' quello del modello DI PARTENZA, non il minimo della catena
        ctx = self._window_for(model)
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

    def _compactor_for(self, model):
        """Modello che ESEGUE la compattazione (B/T/T2/E) per `model`.

        Percorso normale: e' il modello della richiesta stesso (richiesta
        utente 2026-10-28). Eccezione COMPACTOR_OVERRIDE: i modelli single-slot
        di llama-swap (local-model) delegano a un ALTRO modello locale, perche'
        compattare con se stessi su uno slot unico significa mettersi in coda
        dietro la richiesta e andare in TimeoutError (richiesta utente
        2026-11-02). Un override verso se stesso o vuoto vale come "nessun
        override".

        Puo' restituire una TUPLA (catena ordinata di compattatori: e' il caso
        di hybrid-model, vedi COMPACTOR_OVERRIDE). Chi deve solo NOMINARLI usa
        _compactor_chain; chi li passa a _worker_pool puo' passare la tupla
        cosi' com'e'."""
        m = str(model or "")
        ov = COMPACTOR_OVERRIDE.get(m)
        if ov and ov != m:
            return ov
        return model

    def _compactor_chain(self, model):
        """Lista ORDINATA dei compattatori da provare per `model`.

        Esiste perche' _compactor_for puo' restituire una stringa (caso
        normale) o una tupla (catena): chi deve iterare o comporre un
        messaggio usa questa, senza duplicare il controllo di tipo."""
        c = self._compactor_for(model)
        if isinstance(c, (list, tuple)):
            return [str(n) for n in c if n]
        return [str(c)] if c else []

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
        # SOLO il modello della richiesta (richiesta utente 2026-10-28): la
        # sintesi della storia usa lo stesso modello che il client ha
        # chiamato, non un compattatore altrui. Eccezione: i single-slot di
        # llama-swap delegano a un altro modello locale (COMPACTOR_OVERRIDE,
        # richiesta utente 2026-11-02).
        pool = self._worker_pool(body_est, COMPACTOR_MODELS,
                                 output_tokens=COMPACT_OUTPUT_TOKENS,
                                 only=self._compactor_for(model))
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
            brief, err = await self._brief_call(
                comp, COMPACT_SYSTEM_PROMPT, body, COMPACT_OUTPUT_TOKENS,
                (COMPACTOR_OVERRIDE_TIMEOUT_S if comp != model
                 else COMPACTOR_TIMEOUT_S), COMPACT_ORIGIN, deadline=deadline)
            if err:
                failures.append(f"{comp}:{err}")
                if not err.startswith("level-budget"):
                    _quar_note_fail(comp, err)
                continue
            if len(brief) < MIN_BRIEF_CHARS:
                failures.append(f"{comp}:brief-too-short:{len(brief)}")
                _quar_note_fail(comp, "brief-too-short")
                continue
            _quar_note_ok(comp)
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
        ctx = self._window_for(model)
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

    # ------------- operari di riassunto (modello della richiesta) ------------

    def _worker_pool(self, needed_tokens, preferred,
                     output_tokens=MAP_OUTPUT_TOKENS, only=None):
        """Pool di operari di un livello della cascata. PERCORSO NORMALE
        (`only` = modello della richiesta, richiesta utente 2026-10-28): UN
        solo candidato, nessuna scelta lato server - vedi l'ultimo paragrafo.
        PERCORSO DINAMICO (solo con nome wildcard letterale): OGNI modello
        configurato su litellm che puo' contenere
        `needed_tokens` di input (+ output + margine), ordinato per
        (tier di costo, preferenza): prima i modelli GRATIS
        (openrouter/:free, listing a prezzo zero, locali),
        poi gli ECONOMICI (gpt-oss, flash-lite, gemma, syn:small...),
        per ultimi i costosi - richiesta utente 2026-10-06. La lista
        `preferred` ordina solo DENTRO il tier. Richiesta utente
        2026-10-06: "cascade model selection, exploit ANY possible
        litellm configured model" - niente piu' liste chiuse di operari.
        I WILDCARD della model_list (groq/*, gemini/*, ...) vengono ESPASI
        nei nomi concreti richiedibili via listing del gateway (cache TTL):
        chiamare acompletion col nome letterale "groq/*" manderebbe
        model="*" all'upstream -> 400/404 garantiti (visto in produzione).
        Se il listing non risponde resta il nome wildcard: un gateway
        tollerante lo accetta comunque, altrove e' un hop fallito (model="*"
        sconosciuto) e la catena prosegue.
        L'INTERROGAZIONE DINAMICA di /v1/models (thread daemon, cache) unisce
        al pool i modelli presenti nel listing ma non in model_list e usa i
        contesti REALI (max_input_tokens/max_output_tokens) invece della
        stima statica: aggiungere/togliere un modello da litellm viene visto
        senza toccare il codice (richiesta utente 2026-10-06).

        MODELLO DELLA RICHIESTA (richiesta utente 2026-10-28): con `only` il
        pool e' composto da QUEL SOLO modello - la cascata (B/T/T2/E) usa lo
        stesso modello della richiesta, non un pool di modelli altrui. Con
        `only` non c'e' espansione wildcard ne' merge del listing ne' ordine
        per tier: un solo candidato, valutato solo sul fatto che contenga il
        corpo. Un nome wildcard letterale resta sul pool dinamico perche'
        richiederebbe model="*" all'upstream (400/404 visto in produzione)."""
        # solo = modello/i della richiesta da usare in ESCLUSIVA (vedi
        # docstring). `only` e' di norma una STRINGA; e' una SEQUENZA ORDINATA
        # per le catene di COMPACTOR_OVERRIDE (hybrid-model: prima small-model
        # e, se offline, syn:small:text): in quel caso conta l'ORDINE dato, non
        # il tier - il primo che regge il corpo deve essere davvero il primo.
        if isinstance(only, (list, tuple)):
            only_seq = [str(n) for n in only if n]
        elif only and "*" not in str(only):
            only_seq = [str(only)]
        else:
            only_seq = []
        only_set = set(only_seq)
        names = []
        dyn = {}
        _ensure_proxy_models_refresher()
        with _PROXY_MODELS_LOCK:
            dyn.update(_PROXY_MODELS_CACHE["models"])
        if only_seq:
            # niente espansione/dedup: la catena e' gia' di nomi concreti
            names = [n for n in only_seq if "*" not in n]
        else:
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

            # listing dinamico /v1/models (cache del thread daemon): gli id
            # non in model_list entrano nel pool; i contesti reali
            # sostituiscono la stima. Listing non ancora pronto -> dyn vuoto:
            # solo model_list.
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
            # QUARANTENA (2026-11-02): chi ha appena fallito troppo spesso
            # non deve essere scelto come operaio. I protetti ("auto",
            # "*/auto") non sono MAI in quarantena, quindi non li tocca.
            if _quarantined(n):
                return False
            # MEDIA di llama-swap (img-*/vid-*/tts-*/music-*): MAI operari
            # della cascata. Entrano nel pool dal modello wildcard
            # ("llama-swap/img-*") e dal listing di /v1/models, e _model_tier
            # li classificherebbe GRATIS (hardware locale): senza questo
            # filtro finirebbero in TESTA alla lista e ogni compressione
            # sprecherebbe il primo giro su un modello che non fa chat.
            # Il prefisso "llama-swap/" li identifica tutti (i CHAT di
            # llama-swap hanno nomi senza prefisso). Vedi
            # LLAMA_SWAP_MEDIA_PREFIX.
            if n.startswith(LLAMA_SWAP_MEDIA_PREFIX):
                return False
            # MAI un modello costoso come operario (utente 2026-10-07: il
            # prezzo comprime, non il flagship): escluso, non demotato.
            # ECCEZIONE: il modello della richiesta (solo) - e' gia' quello
            # scelto dal client, escluderlo toglierebbe del tutto B/T/T2/E
            # (richiesta utente 2026-10-28)
            if (not only_seq or n not in only_set) \
                    and _worker_too_expensive(n):
                return False
            # finestra REALE per-modello (gateway _DYN_CTX -> listing
            # litellm -> statico di gruppo): stessa fonte del clamp
            if self._window_for(n) < budget:
                return False
            # il PROMPT deve stare nel cap di INPUT reale (max_input_tokens
            # del listing litellm, se noto): e' la regola con cui l'upstream
            # rifiuta; il margine (>= output) copre la risposta in finestra
            mit = dyn.get(n, (None, None))[0]
            if mit is not None and needed_tokens + self._margin(
                    needed_tokens) > mit:
                return False
            return True

        # tier di costo PRIMARIO, preferenza del livello secondaria: il pool
        # e' gratis -> economico -> costoso a prescindere dalle liste preferred.
        # ECCEZIONE catena (`only` sequenza): l'ordine e' quello DATO dal
        # chiamante e viene rispettato alla lettera.
        if only_seq:
            ordered = [n for n in names if fits(n)]
        else:
            ordered = [n for n in sorted(
                        names, key=lambda n: (_model_tier(n), rank(n)))
                    if fits(n)]
        # I modelli in QUARANTENA sono gia' esclusi da fits(), non demoti:
        # la cascata non deve sprecare un tentativo su chi ha appena fallito
        # (richiesta utente 2026-11-02). Se il pool resta vuoto si risale di
        # livello (vedi _cascade), non si ripesca il quarantenato.
        return ordered

    async def _brief_call(self, worker, system_prompt, text, max_tokens,
                          timeout, origin, deadline=None):
        """UNA sintesi dal worker, con UN retry a budget maggiore quando il
        modello spende tutto il budget nel ragionamento e restituisce content
        vuoto (vedi MAP_OUTPUT_TOKENS: misurato su small-model il 2026-11-20).
        Un "empty" non retry-ato era uno strike di quarantena e uccideva la
        cascata. Ritorna (testo, errore): testo None = worker inutilizzabile.

        `deadline` (monotonic) limita il TEMPO TOTALE dei due tentativi al
        budget del livello, cosi' un retry non raddoppia l'attesa del client."""
        from litellm.proxy.proxy_server import llm_router
        if llm_router is None:
            return None, "no-router"
        # RAGIONAMENTO SPENTO sui compattatori LOCALI (chat_template_kwargs
        # passa intatto: litellm -> llama-gate -> llama-swap -> llama-server,
        # verificato 2026-11-20). MISURATO sullo stesso chunk da ~163k token:
        # thinking ON -> content VUOTO a 4096 E anche a 8192 (finish=length,
        # il ragionamento mangia tutto il budget); thinking OFF -> finish=stop
        # con brief da 4533 char in 294 s. Il retry a 8192 non salvava il caso
        # largo, quindi senza questo un chunk > ~150k andava perso a ogni
        # richiesta. Solo per i modelli locali: `chat_template_kwargs` e' un
        # campo di llama.cpp, i provider cloud lo rifiuterebbero.
        extra = {}
        if worker in COMPACTOR_INPUT_TOKENS:
            extra["extra_body"] = {"chat_template_kwargs":
                                   {"enable_thinking": False}}
        last = None
        for mt in (max_tokens, max(max_tokens, BRIEF_RETRY_TOKENS)):
            t0 = time.monotonic()
            wait = timeout
            if deadline is not None:
                wait = min(wait, deadline - time.monotonic())
                if wait <= 0:
                    return None, last or f"time-budget:{WORKER_TIME_BUDGET_S}s"
            try:
                resp = await asyncio.wait_for(
                    llm_router.acompletion(
                        model=worker,
                        messages=[{"role": "system", "content": system_prompt},
                                  {"role": "user", "content": text}],
                        max_tokens=mt,
                        num_retries=0,
                        fallbacks=[],
                        drop_params=True,
                        metadata={INTERNAL_CALL_ORIGIN_KEY: origin},
                        **extra,
                    ),
                    timeout=wait,
                )
            except Exception as e:
                if deadline is not None and wait < timeout:
                    # Il tempo e' finito per il LIVELLO (deadline), non per
                    # colpa del worker: non e' una sua failure -> niente
                    # strike di quarantena (altrimenti un prompt enorme
                    # disabilita small-model per 15 minuti).
                    return None, f"level-budget:{type(e).__name__}"
                return None, f"{type(e).__name__}: {str(e)[:120]}"
            s = (resp.choices[0].message.content or "").strip()
            if s:
                return s, None
            fr = resp.choices[0].finish_reason
            last = f"empty-brief@{mt}(finish={fr})"
            if mt >= BRIEF_RETRY_TOKENS:
                break
            # Il retry raddoppia l'attesa di QUESTO chunk: si fa solo se il
            # budget del livello puo' permetterselo, altrimenti il tempo
            # residuo serve agli altri chunk (meglio riassunti PARZIALI che
            # nessuno, e T3 copre comunque il buco).
            if deadline is not None and \
                    (deadline - time.monotonic()) < (time.monotonic() - t0):
                return None, f"{last};retry-skipped:no-budget"
            print(f"[reasoning_clamp] {worker}: brief VUOTO a max_tokens={mt} "
                  f"(finish={fr}) - retry a {max(max_tokens, BRIEF_RETRY_TOKENS)}",
                  flush=True)
        return None, last

    async def _summarize_chunks(self, chunks, origin, model):
        """Riassume i chunk con l'operario = MODELLO DELLA RICHIESTA
        (richiesta utente 2026-10-28: niente piu' pool dinamico di modelli
        altrui - un chunk viene riassunto dallo stesso modello che il client
        ha chiamato), salvo COMPACTOR_OVERRIDE per i single-slot di llama-swap
        (local-model -> small-model, richiesta utente 2026-11-02). Resta il
        budget di tempo WORKER_TIME_BUDGET_S: pool di
        UN elemento, quindi un solo tentativo per chunk; se fallisce si
        tengono i PARZIALI gia' prodotti (almeno un chunk riassunto =
        compressione utile, il livello successivo della cascata completa se
        serve). Ritorna (brief, info) con brief=None solo su fallimento
        TOTALE."""
        from litellm.proxy.proxy_server import llm_router
        if llm_router is None:
            return None, "no-router"
        # Chunk misurato con la STESSA stima del taglio (_estimate_text_tokens,
        # ratio per densita' + tiktoken sui grossi): deve stare bassa - non
        # sovrastimare - per non escludere operari validi, ma con il ratio
        # denso fisso (2.8 char/token) un chunk di prosa a 196k token vero
        # risultava ~323k e il pool lo rifiutava ("no-worker-fits") proprio
        # quando il tetto largo serviva. Il margine del pool copre il residuo.
        chunk_est = max(self._estimate_text_tokens(c) for c in chunks)
        # SOLO il modello della richiesta (richiesta utente 2026-10-28): i
        # livelli T/T2/E riassumono col modello che il client ha chiamato;
        # per i single-slot di llama-swap vale COMPACTOR_OVERRIDE (small-model)
        pool = self._worker_pool(chunk_est, SUMMARIZER_PREFERRED,
                                 only=self._compactor_for(model))
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
                s, err = await self._brief_call(
                    w, MAP_SYSTEM_PROMPT, chunk, MAP_OUTPUT_TOKENS,
                    MAP_TIMEOUT_S, origin, deadline=deadline)
                if err:
                    failures.append(f"{w}:{err}")
                    if not err.startswith("level-budget"):
                        _quar_note_fail(w, err)
                    wi += 1
                    continue
                _quar_note_ok(w)
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
        la richiesta corrente) e riassume il mezzo col modello della
        richiesta."""
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
        tail_est = self._estimate_text_tokens(c)
        if tail_est < TAIL_SPLIT_MIN_TOKENS:
            return None, f"tail-too-small:{tail_est}"
        if len(c) <= ANCHOR_HEAD_CHARS + ANCHOR_TAIL_CHARS:
            return None, "no-middle"
        head = c[:ANCHOR_HEAD_CHARS]
        anchor = c[len(c) - ANCHOR_TAIL_CHARS:]
        middle = c[ANCHOR_HEAD_CHARS:len(c) - ANCHOR_TAIL_CHARS]
        if len(middle) < COMPACT_MIN_CHARS:
            return None, "middle-too-small"
        chunks = self._chunk_for_map(middle, model)
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

    # ---------------- T2: compressione dei messaggi DOPO l'ultimo user -----

    async def _compress_tail_messages(self, data, model, est):
        """Livello T2 (incidente 2026-10-06: coda 126640 tok ma ultimo user
        842 - T esce "tail-too-small", B/C/E tengono la coda verbatim, F
        413). Comprime IN PLACE i messaggi DOPO l'ultimo messaggio user
        (risultati tool / assistant), il piu' grande prima, finche' la
        stima rientra nel contesto. Viene sostituito SOLO il 'content'
        testuale (T2_HEADER + riassunto del modello della richiesta): ruolo,
        tool_call_id e tool_calls restano - un tool result orfanizzato
        sarebbe un 400 del provider. TTL: T2_TIME_BUDGET_S totale per la
        passata + budget per-chunk dentro _summarize_chunks; i guadagni
        parziali restano (come i riassunti parziali del pool).
        Ritorna (data, info) o (None, motivo)."""
        msgs = [m for m in (data.get("messages") or []) if isinstance(m, dict)]
        idx = None
        for i in range(len(msgs) - 1, -1, -1):
            if msgs[i].get("role") == "user":
                idx = i
                break
        if idx is None:
            return None, "no-user-msg"
        post = msgs[idx + 1:]
        if not post:
            return None, "no-post-user"
        # candidati: content testuale abbastanza lungo da valere una chiamata
        cand = []
        for i, m in enumerate(post):
            c = m.get("content")
            if not isinstance(c, str) or len(c) < COMPACT_MIN_CHARS:
                continue  # content list (vision) / vuoto / solo tool_calls
            cand.append((self._estimate_subset([m], data), i))
        if not cand:
            return None, "nothing-compressible"
        cand.sort(key=lambda t: (-t[0], t[1]))  # il piu' grande prima
        ctx = self._window_for(model)
        cur = self._estimate_prompt_tokens(data)
        if not self._saturated(cur, ctx):
            return None, "already-fits"
        deadline = time.monotonic() + T2_TIME_BUDGET_S
        new_msgs = list(msgs)
        base = idx + 1
        done = 0
        reasons = []
        for _, i in cand:
            if (self._settled(cur, ctx)
                    or done >= T2_MAX_MESSAGES
                    or time.monotonic() >= deadline):
                break
            m = post[i]
            chunks = self._chunk_for_map(m["content"], model)
            if not chunks:
                reasons.append("no-chunks")
                continue
            brief, info = await self._summarize_chunks(chunks, TAIL_ORIGIN, model)
            if brief is None:
                # Il motivo NON va perso: "no-gain" secco rende il 413
                # indiagnosticabile (incidente 2026-11-20: l'operaio small-model
                # rispondeva 500 "upstream command exited prematurely" per OOM e
                # il 413 diceva solo "no-gain").
                reasons.append(str(info))
                # FAIL-FAST: l'operaio e' uno solo, il prossimo messaggio
                # fallirebbe allo stesso modo - e ogni fallimento e' uno strike
                # di quarantena che poi disabilita il compattatore per 15m.
                break
            new_msgs[base + i] = dict(m, content=T2_HEADER + brief)
            data["messages"] = new_msgs  # progressivo: i guadagni restano
            cur = self._estimate_prompt_tokens(data)
            done += 1
        if not done:
            return None, ("no-gain[" + "|".join(reasons[-3:]) + "]"
                          if reasons else "no-gain")
        print(f"[reasoning_clamp] T2 {model}: {done} messaggi post-user "
              f"compressi, stima ~{cur} tok (contesto {ctx})", flush=True)
        return data, f"t2:{done}"

    # ---------------- T3: troncamento deterministico (nessun modello) -------

    async def _truncate_tail_messages(self, data, model, est):
        """Livello T3: tronca IN PLACE i messaggi dopo l'ultimo user, il piu'
        grande prima, finche' la stima rientra nel contesto.

        Non chiama NESSUN modello: e' l'ultimo livello prima del 413 e deve
        guadagnare sempre. Serve esattamente al caso in cui T (mezzo
        dell'ultimo user) e T2 (riassunto dei messaggi post-user) non possono
        lavorare - operaio di compattazione giu', in quarantena, in OOM o
        troppo lento per il budget del livello (incidente 2026-11-20 17:23:
        coda 260578 tok, ultimo user 536, small-model morto -> T
        "tail-too-small", T2 "no-gain", 413 deterministico). Per questo, dopo
        i messaggi post-user, taglia anche il MEZZO dell'ultimo user
        conservando le ancore (ANCHOR_HEAD_CHARS / ANCHOR_TAIL_CHARS) verbatim.
        Si conservano inizio e fine del contenuto (li' stanno intestazione e
        conclusione di un tool result / log) e si rimuove il centro, con T3_HEADER
        a dirlo esplicitamente al modello. Ruolo, tool_call_id e tool_calls
        restano: un tool result orfano sarebbe un 400 del provider."""
        msgs = [m for m in (data.get("messages") or []) if isinstance(m, dict)]
        idx = None
        for i in range(len(msgs) - 1, -1, -1):
            if msgs[i].get("role") == "user":
                idx = i
                break
        if idx is None:
            return None, "no-user-msg"
        post = msgs[idx + 1:]
        ctx = self._window_for(model)
        cur = self._estimate_prompt_tokens(data)
        if not self._saturated(cur, ctx):
            return None, "already-fits"
        # Candidati: i messaggi DOPO l'ultimo user, il piu' grande prima, e per
        # ULTIMO il mezzo dell'ultimo user. Quello non lo tocca nessun altro
        # livello deterministico - T lo affida all'operaio, che puo' essere
        # morto/in quarantena/OOM o non finire nel budget (misurato 2026-11-20:
        # 1.3 MB nell'ultimo user, small-model in timeout -> "T:level-budget"
        # -> 413 con T2/T3 "no-post-user"). Le ANCORE (inizio e fine, dove sta
        # la richiesta corrente) restano verbatim.
        cand = []   # (char tagliabili, indice in msgs, pavimento da tenere)
        for i, m in enumerate(post):
            c = m.get("content")
            if isinstance(c, str) and len(c) >= COMPACT_MIN_CHARS:
                cand.append((len(c), idx + 1 + i, T3_MIN_KEEP_CHARS))
        # content list (vision) / vuoto / solo tool_calls: niente da tagliare
        last_c = msgs[idx].get("content")
        if isinstance(last_c, str):
            floor = ANCHOR_HEAD_CHARS + ANCHOR_TAIL_CHARS
            if len(last_c) > floor + COMPACT_MIN_CHARS:
                cand.append((len(last_c) - floor, idx, floor))
        if not cand:
            return None, "nothing-truncatable"
        cand.sort(key=lambda t: -t[0])  # il piu' grande prima
        new_msgs = list(msgs)
        done = 0
        # piu' giri: dopo un taglio il margine (3%) cambia, e un taglio
        # stimato con il ratio sbagliato puo' non bastare. Ogni giro riparte
        # dal piu' grande ANCORA troppo lungo; ci si ferma quando si rientra
        # o quando nessun messaggio puo' piu' scendere (T3_MIN_KEEP_CHARS).
        for _round in range(4):
            if self._settled(cur, ctx):
                break
            progressed = False
            for _, i, floor in cand:
                if self._settled(cur, ctx):
                    break
                slot = new_msgs[i]
                c = slot.get("content")
                if not isinstance(c, str) or len(c) <= floor:
                    continue
                # il deficit si misura sul TARGET (ctx - headroom), non sul
                # bordo del contesto: un taglio "a filo" lascia il payload
                # dentro la stima e fuori dal motore. + SLACK: la conversione
                # token->caratteri arrotonda, e senza slack si resta 1-2 token
                # SOPRA il target (misurato: 249038 contro 249036) e il
                # livello non risulta "settled".
                deficit = (cur - int(ctx * (1.0 - COMPACTION_HEADROOM_PCT))
                           + T3_TRIM_SLACK_TOKENS)
                if deficit <= 0:
                    break
                # char/token EFFETTIVO di QUESTO testo (tiktoken sui testi
                # grandi): CHARS_PER_TOKEN fisso sottostima i run ripetuti e
                # il taglio resterebbe corto
                per_tok = len(c) / max(self._estimate_text_tokens(c), 1)
                keep = len(c) - int(deficit * per_tok)
                if keep < floor:
                    keep = floor
                head = int(keep * T3_HEAD_RATIO)
                tail = keep - head
                header = "" if c.startswith(T3_HEADER) else T3_HEADER
                slot["content"] = (header + c[:head] + "\n[...]\n"
                                   + (c[len(c) - tail:] if tail > 0 else ""))
                data["messages"] = new_msgs  # progressivo: i guadagni restano
                cur = self._estimate_prompt_tokens(data)
                done += 1
                progressed = True
            if not progressed:
                break
        if not done:
            return None, "no-gain"
        print(f"[reasoning_clamp] T3 {model}: {done} messaggi troncati "
              f"(coda e/o mezzo dell'ultimo user), stima ~{cur} tok "
              f"(contesto {ctx})", flush=True)
        return data, f"t3:{done}"

    # ---------------- E: map/reduce (modello della richiesta) ---------------

    def _chunk_for_map(self, body, model=None):
        """Taglia la trascrizione in chunk di ~limit token (MAP_CHUNK_INPUT_TOKENS
        oppure il tetto misurato del compattatore locale); quello che avanza
        da un blocco troppo grande viene spezzato in parti proporzionali.

        `model` = modello della RICHIESTA: il taglio scende al tetto del suo
        COMPATTATORE quando e' un modello locale (COMPACTOR_INPUT_TOKENS).
        Senza questo un chunk da 90k token finisce su una scheda da 12GB e la
        uccide (CUDA OOM -> llama-swap 500 -> cascata senza operaio).

        Il taglio e' misurato in TOKEN STIMATI, non in caratteri: il vecchio
        `limit_tokens * CHARS_PER_TOKEN` assumeva 2.8 char/token, ma su
        codice/JSON densi (2.0) o CJK (1.5) un chunk "da 196k" pesava davvero
        270-360k -> 400 dell'upstream, strike di quarantena e cascata di nuovo
        senza operaio. Con il tetto in token, COMPACTOR_INPUT_TOKENS e' quello
        che arriva davvero al modello."""
        limit_tokens = MAP_CHUNK_INPUT_TOKENS
        if model is not None:
            # Il tetto del compattatore LOCALE SOSTITUISCE quello generico: e'
            # misurato su quel motore, mentre MAP_CHUNK_INPUT_TOKENS e'
            # dimensionato sui contesti cloud piu' piccoli. Prima lo abbassava
            # soltanto, quindi il tetto locale non entrava mai in gioco e si
            # restava a 90000: il "piu' largo possibile" non aveva effetto.
            # Con una CATENA di compattatori (hybrid-model) vale il tetto piu'
            # LARGO della catena: il chunk deve stare nel primo che risponde,
            # e il locale (small-model) e' comunque il piu' stretto - se e'
            # offline il chunk resta valido anche per syn:small:text.
            caps = [COMPACTOR_INPUT_TOKENS[n]
                    for n in self._compactor_chain(model)
                    if n in COMPACTOR_INPUT_TOKENS]
            if caps:
                limit_tokens = max(caps)

        def pieces(block):
            """(testo, token stimati): un blocco oltre il tetto viene spezzato
            in parti proporzionali e RI-misurato - la densita' delle parti puo'
            differire da quella del blocco intero."""
            n = self._estimate_text_tokens(block)
            if n <= limit_tokens or not block:
                return [(block, n)]
            parts = -(-n // limit_tokens)
            step = max(1, -(-len(block) // parts))
            out = []
            for i in range(0, len(block), step):
                piece = block[i:i + step]
                out.append((piece, self._estimate_text_tokens(piece)))
            return out

        chunks, cur, cur_tok = [], [], 0
        for block in body.split("\n\n"):
            for piece, tok in pieces(block):
                if cur and cur_tok + tok > limit_tokens:
                    chunks.append("\n\n".join(cur))
                    cur, cur_tok = [], 0
                cur.append(piece)
                cur_tok += tok
            if len(chunks) >= MAP_MAX_CHUNKS:
                break
        if cur and len(chunks) < MAP_MAX_CHUNKS:
            chunks.append("\n\n".join(cur))

        def split_over(chunk):
            """Il conto per-parte non e' quello del testo UNITO: la densita'
            del chunk intero puo' essere diversa (e sopra 150k char entra
            tiktoken, che sulle parti piccole non veniva consultato).
            MISURATO: 800k char di codice -> chunk "da 163840" che da unito
            ne stima 170670, +4%. Si ri-divide finche' il tetto e' vero."""
            out = [chunk]
            for _ in range(3):
                nxt = []
                for c in out:
                    n = self._estimate_text_tokens(c)
                    if n <= limit_tokens or len(c) < 2:
                        nxt.append(c)
                        continue
                    k = max(2, -(-n // limit_tokens))
                    step = max(1, -(-len(c) // k))
                    nxt.extend(c[i:i + step] for i in range(0, len(c), step))
                out = nxt
            return out

        final = []
        for c in chunks:
            final.extend(split_over(c))
        return final[:MAP_MAX_CHUNKS]

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
        chunks = self._chunk_for_map(body, model)
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
        ctx = self._window_for(model)
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
                         "della coda -> T2 compressione dei messaggi dopo "
                         "l'ultimo user -> E map/reduce, eseguiti con "
                         f"{', '.join(self._compactor_chain(model))} -> "
                         "T3 troncamento "
                         "deterministico) non ha "
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
        ctx = self._window_for(model)
        est = self._estimate_prompt_tokens(data)
        if not self._saturated(est, ctx):
            return data
        print(f"[reasoning_clamp] overflow {model}: ~{est} tok + "
              f"{MIN_OUTPUT_TOKENS} tok > contesto {ctx} -> cascata "
              f"B/C/T/T2/E/T3", flush=True)
        if model not in _DYN_CTX:
            # il contesto NON viene dal listing dinamico: o tabella statica
            # per-modello (vera, ok) o floor di gruppo. Gap osservabilita'
            # incidente 2026-10-06 (contesti statici usati senza saperlo)
            with _PROXY_MODELS_LOCK:
                mit = _PROXY_MODELS_CACHE["models"].get(model,
                                                        (None, None))[0]
            if mit is None:
                print(f"[reasoning_clamp] ATTENZIONE {model}: contesto "
                      f"{ctx} dalla TABELLA STATICA (listing dinamico non "
                      f"ancora disponibile: sweep in corso o gateway "
                      f"irraggiungibile)", flush=True)
        # La coda (dall'ultimo user in poi) resta verbatim in B/C/E: se
        # satura DA SOLA quei livelli non possono liberare spazio - solo T
        # (mezzo dell'ultimo messaggio user) e T2 (messaggi DOPO l'ultimo
        # user: INCIDENTE 2026-10-06, coda 126640 tok di cui 842 nell'ultimo
        # user) possono aiutare. Cosi' si evitano 3 chiamate cloud prima di
        # un 413 assicurato.
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
                    # _settled (non "appena sotto il bordo"): un payload a
                    # 99% del contesto STIMATO e' spesso oltre quello REALE
                    if self._settled(est, ctx):
                        print(f"[reasoning_clamp] {model}: cascata risolta "
                              f"da T (~{est} tok, contesto {ctx})", flush=True)
                        return data
                # T non ha bastato (tipico: la massa e' nei tool result DOPO
                # l'ultimo user, che T non tocca): T2 li comprime in place
                try:
                    new, info = await self._compress_tail_messages(
                        data, model, est)
                except Exception as e:
                    new, info = None, f"exc:{type(e).__name__}:{str(e)[:160]}"
                tried.append(f"T2:{info}")
                if new is not None:
                    est = self._estimate_prompt_tokens(data)
                    self._clamp_output(data, model, est)
                    if self._settled(est, ctx):
                        print(f"[reasoning_clamp] {model}: cascata risolta "
                              f"da T2 (~{est} tok, contesto {ctx})",
                              flush=True)
                        return data
                # la coda resta verbatim anche per B/C/E: nessun aiuto.
                # T3 tronca i messaggi post-user SENZA modello: e' l'ultimo
                # livello utile quando l'operaio di compattazione e' morto,
                # in OOM o in quarantena (incidente 2026-11-20 17:23).
                try:
                    new, info = await self._truncate_tail_messages(
                        data, model, est)
                except Exception as e:
                    new, info = None, f"exc:{type(e).__name__}:{str(e)[:160]}"
                tried.append(f"T3:{info}")
                if new is not None:
                    est = self._estimate_prompt_tokens(data)
                    self._clamp_output(data, model, est)
                    if self._settled(est, ctx):
                        print(f"[reasoning_clamp] {model}: cascata risolta "
                              f"da T3 (~{est} tok, contesto {ctx})",
                              flush=True)
                        return data
                # F-EVIT vale anche qui: se il prompt da solo ci sta, un 413
                # sarebbe deterministico e inutile (l'upstream, se proprio,
                # risponde 400 e la catena di fallback prosegue).
                if est <= ctx and self._force_output(data, model, est):
                    print(f"[reasoning_clamp] {model}: F evitato - prompt "
                          f"~{est} tok ci sta nel contesto {ctx}: max_tokens "
                          f"alla stanza residua, richiesta parte "
                          f"({'|'.join(tried)})", flush=True)
                    return data
                return self._overflow(data, model, est, tried)
        tried = []
        for name, fn in (("B", self._compact_with_provider),
                         ("C", self._compress_locally),
                         ("T", self._split_compress_tail),
                         ("T2", self._compress_tail_messages),
                         ("E", self._map_reduce),
                         ("T3", self._truncate_tail_messages)):
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

    @staticmethod
    def _image_part_url(part):
        """URL di una parte immagine OpenAI/Responses (dict o stringa)."""
        if not isinstance(part, dict):
            return None
        if part.get("type") not in ("image_url", "input_image"):
            return None
        value = part.get("image_url")
        if isinstance(value, dict):
            return value.get("url")
        return value if isinstance(value, str) else None

    @staticmethod
    def _unresolvable_image_url(url):
        """True se il PROVIDER remoto non puo' scaricare quell'immagine.

        Copre i path relativi di OpenWebUI (/api/v1/files/<id>/content, ossia
        la conversione in base64 fallita: file assente o permessi) e gli host
        non raggiungibili da fuori (loopback, reti private, .local). I data:
        URL (base64) e gli http(s) pubblici restano intatti.
        """
        if not isinstance(url, str) or not url:
            return False
        u = url.strip()
        if u.startswith("data:"):
            return False
        if u.startswith("/") or u.startswith("file:"):
            return True
        if not u.lower().startswith(("http://", "https://")):
            return True
        try:
            host = (urllib.parse.urlsplit(u).hostname or "").lower()
        except ValueError:
            return True
        if not host:
            return True
        if host in ("localhost", "127.0.0.1", "0.0.0.0", "::1",
                    "host.docker.internal"):
            return True
        if host.endswith(".local"):
            return True
        return bool(re.match(r"^(10\.|192\.168\.|172\.(1[6-9]|2\d|3[01])\.|169\.254\.)",
                             host))

    def _sanitize_unresolvable_images(self, data, model):
        """Sostituisce le immagini non scaricabili con una nota testuale.

        Vale sia per il body chat (messages[].content) sia per il body
        Responses (input[] con input_image / function_call_output / message).
        """
        note = UNRESOLVABLE_IMAGE_NOTE
        replaced = 0

        def fix_parts(parts, text_type):
            nonlocal replaced
            out = []
            for part in parts:
                url = self._image_part_url(part)
                if url is not None and self._unresolvable_image_url(url):
                    out.append({"type": text_type, "text": note})
                    replaced += 1
                else:
                    out.append(part)
            return out

        for msg in data.get("messages") or []:
            if isinstance(msg, dict) and isinstance(msg.get("content"), list):
                msg["content"] = fix_parts(msg["content"], "text")

        inputs = data.get("input")
        if isinstance(inputs, list):
            for idx, item in enumerate(inputs):
                if not isinstance(item, dict):
                    continue
                kind = item.get("type")
                if kind == "input_image":
                    url = self._image_part_url(item)
                    if url is not None and self._unresolvable_image_url(url):
                        inputs[idx] = {"type": "input_text", "text": note}
                        replaced += 1
                elif kind == "message" and isinstance(item.get("content"), list):
                    item["content"] = fix_parts(item["content"], "input_text")
                elif kind == "function_call_output" and isinstance(item.get("output"), list):
                    item["output"] = fix_parts(item["output"], "input_text")

        if replaced:
            print(f"[reasoning_clamp] {model}: {replaced} immagine/i con URL non "
                  f"risolvibile dal provider -> nota testuale", flush=True)
        return data

    def _apply(self, data):
        model = str(data.get("model") or "")
        # --- toggle semantic cache per-request (DOPPIA POLITICA) ------------
        # DEFAULT = cache SPENTA: senza header la richiesta NON legge e NON
        # scrive la semantic cache (gli agent coding non mandano nulla -> OFF).
        # Si ACCENDE solo con header esplicito "Cache-Control: cache" oppure
        # "Cache-Control: store" (token ESATTI, split sulla virgola: "no-store"
        # NON accende). Unico punto di controllo REALE: litellm legge l'header
        # standard "Cache-Control" SOLO per s-maxage -> ttl
        # (litellm_pre_call_utils.py), quindi qui lo traduciamo nelle chiavi
        # che litellm riconosce a livello per-request (caching_handler:
        # kwargs["cache"]["no-cache"] salta la LETTURA, ["no-store"] salta la
        # SCRITTURA). Resta un semplice header, senza key dedicate ne' DB.
        #   agent coding -> nessun header              -> OFF (default)
        #   altri client -> "Cache-Control: cache"     -> ON
        # (richiesta utente 2026-11-06). Gli header del client sono in
        # proxy_server_request/http metadata (dict di header "logging safe").
        opt_in = False
        hdrs = ((data.get("proxy_server_request") or {}).get("headers")
                or (data.get("metadata") or {}).get("headers")
                or (data.get("litellm_metadata") or {}).get("headers")
                or {})
        if isinstance(hdrs, dict):
            cc = ""
            for hk, hv in hdrs.items():
                if isinstance(hk, str) and hk.lower() == "cache-control" and hv:
                    cc = str(hv).lower()
                    break
            tokens = {t.strip() for t in cc.split(",") if t.strip()}
            opt_in = "cache" in tokens or "store" in tokens
        if opt_in:
            data["cache"] = {"no-cache": False, "no-store": False}
        else:
            data["cache"] = {"no-cache": True, "no-store": True}
        # --- immagini con URL che il PROVIDER non puo' risolvere ---
        # OpenWebUI manda le immagini come data: URL (base64) e le converte
        # solo se il file esiste. Quando la conversione fallisce (file
        # sparito, permessi) resta il path relativo /api/v1/files/<id>/content
        # o un host locale: il provider remoto risponde 400 "Could not parse
        # image as valid URL" (incidente 2026-11-02: upload su tmpfs azzerati
        # al reboot). Qui la parte immagine viene sostituita con una nota
        # testuale: la richiesta parte e il modello sa che c'era un'immagine.
        self._sanitize_unresolvable_images(data, model)
        # --- policy fallback per-modello (override per-request) ---
        # Regola (richiesta utente 2026-10-25): small-model NON e' piu' un
        # fallback di nessun modello. Resta un modello autonomo, invocabile
        # direttamente; qui garantiamo che nessuna catena lo contenga.
        if self._internal(data):
            # chiamata INTERNA della cascata (B/T/T2/E): stesso modello della
            # richiesta e NESSUN fallback - un hop su un altro modello
            # tradirebbe la regola "il clamp usa il modello della richiesta"
            # (richiesta utente 2026-10-28)
            data["fallbacks"] = []
        elif model == "small-model":
            # modello autonomo, mai fallback (richiesta utente 2026-10-25)
            data["fallbacks"] = []
        elif model == "embedding-model":
            # niente auto-fallback: l'unico modello che potrebbe servire una
            # richiesta embedding e' lui stesso, gli altri target (CHAT)
            # risponderebbero 400
            data["fallbacks"] = []
        elif model == "local-model":
            # nessun fallback (richiesta utente 2026-10-25: small-model non
            # e' piu' un gradino della catena)
            data["fallbacks"] = []
        elif model == SYN_SMALL_TEXT:
            # richiesta utente 2026-10-12: syn:small:TEXT parte da
            # ollama/small-model (gemello Ollama sullo STESSO PC, server
            # indipendente: regge quando llama-swap e' occupo o da 429), poi il
            # cloud GRATIS openrouter/free. Dal 2026-10-14
            # syn:small:vision ha la stessa identica catena (ramo seguente,
            # alias sul medesimo costante). Ramo esatto PRIMA del ramo generico
            # synthetic/*: la callback sovrascrive i fallback di OGNI richiesta
            # synthetic/*.
            data["fallbacks"] = list(SYN_SMALL_TEXT_FALLBACKS)
        elif model == SYN_SMALL_VISION:
            # richiesta utente 2026-10-14: syn:small:VISION ha la STESSA catena
            # di syn:small:text (alias su SYN_SMALL_TEXT_FALLBACKS) - prima
            # ollama/small-model (gemello Ollama sullo STESSO PC, server
            # indipendente), poi il cloud GRATIS openrouter/free. Il target
            # locale e' text-only: una richiesta vision puo' 400, hop
            # sprecato, la catena prosegue.
            # Ramo esatto PRIMA del ramo generico synthetic/*.
            data["fallbacks"] = list(SYN_SMALL_VISION_FALLBACKS)
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
            # NESSUN fallback (richiesta utente 2026-10-28: local-model
            # rimosso da ogni catena synthetic/*, dopo small-model il
            # 2026-10-25). I rami esatti qui sopra (syn:small:text,
            # syn:small:vision, embedding nomic) restano le uniche catene
            # synthetic/* esistenti.
            data["fallbacks"] = []
        elif model.startswith(LLAMA_SWAP_MEDIA_PREFIX):
            # richiesta utente 2026-10-16: i MEDIA di llama-swap (img-*,
            # vid-*, tts-*, music-*) NON hanno MAI un fallback. Il ramo sta
            # PRIMA della catch-all finale: senza di esso i media
            # erediterebbero la catch-all di config.yaml, cioe' un modello
            # CHAT su una richiesta /v1/images/generations o /v1/audio/speech
            # -> 400 garantito e cliente confuso da un secondo errore che non
            # spiega nulla. I modelli CHAT di llama-swap hanno nomi SENZA
            # prefisso (local-model/small-model/embedding-model) e restano
            # nelle catene qui sopra. Vedi LLAMA_SWAP_MEDIA_PREFIX.
            data["fallbacks"] = []
        elif model == HYBRID_PUBLIC:
            # Riserve free di ULTIMA RISORSA (vedi HYBRID_RESERVE_PREFIXES):
            # scelte ORA, per QUESTA richiesta (contesto e vision), e passate
            # come fallbacks per-request -> partono SOLO se tutto il gruppo e'
            # esaurito (i 5 membri in cooldown/down). Nel corpo della richiesta
            # servono id CONCRETI: un nome wildcard qui manderebbe model="*"
            # all'upstream (400), ed e' il motivo per cui la lista si calcola
            # a ogni richiesta dai listing in cache.
            data["fallbacks"] = self._hybrid_reserves(
                self._estimate_prompt_tokens(data), _request_has_image(data))
        else:
            # TUTTI gli altri (groq/*, gemini/*, openrouter/*,
            # openrouter/free, ollama/*, ollama-cloud/* e qualunque prefisso
            # futuro): NESSUN fallback (richiesta utente
            # 2026-10-25: la catch-all "*": ["small-model"] e' stata rimossa).
            data["fallbacks"] = []
        # --- reasoning_effort per-provider ---
        eff = data.get("reasoning_effort")
        if model == HYBRID_PUBLIC:
            # Il gruppo mescola provider con vocabolari diversi e il membro lo
            # si scopre solo DOPO la selezione, non qui: "xhigh"/"max" NON sono
            # validi su Gemini (che accetta high/medium/low/minimal/none),
            # quindi si normalizza a "high" - valore accettato da tutti i
            # membri. Nessuna INIEZIONE quando il client non manda nulla:
            # resta il default del membro scelto (richiesta utente 2026-11-23).
            if eff in ("xhigh", "max"):
                data["reasoning_effort"] = "high"
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
        elif model.startswith("synthetic/"):
            # Direttiva utente 2026-11-14: xhigh di default SOLO dove il
            # modello lo supporta, altrimenti high/max, altrimenti niente
            # parametro. La lista arriva dal listing (reasoning_parameters.
            # efforts) con fallback statico; se il modello e' SCONOSCIUTO
            # (nomic/nemotron, o listing non ancora letto) resta il default
            # storico xhigh. Il default sta QUI e non piu' in litellm_params
            # perche' i default di litellm_params vengono uniti DOPO l'hook:
            # uno statico xhigh avrebbe sovrascritto questo clamp e mandato
            # xhigh anche a syn:small:text / syn:large:vision che non lo
            # elencano (e sulla rotta /v1/messages faceva 400).
            supported = _efforts_for(model)
            if supported is None:
                # sconosciuto: comportamento storico, solo xhigh se il client
                # non ha chiesto nulla
                if not eff:
                    data["reasoning_effort"] = "xhigh"
            else:
                eff = _pick_effort(eff, supported)
                if eff:
                    data["reasoning_effort"] = eff
                else:
                    data.pop("reasoning_effort", None)
        elif model.startswith("cloudflare/"):
            # Workers AI valida reasoning_effort PER-MODELLO e risponde 400
            # secco (nessun default silenzioso come Groq/Gemini): il
            # parametro si manda SOLO se il listing dichiara la lista
            # (supported_efforts -> _DYN_EFFORTS via _fetch_pattern_models) e
            # il valore e' tra quelli. Verificato 2026-11-18: qwen3.8-27b
            # accetta xhigh/medium/low e RIFIUTA high; gpt-oss-120b accetta
            # high/medium/low e RIFIUTA xhigh; glm-4.7-flash (lista NON
            # pubblicata) accetta low/medium/high e RIFIUTA none/max/xhigh.
            # Senza lista pubblica il parametro viene RIMOSSO (vale il
            # default Cloudflare, default_enabled:true): un default nostro
            # sarebbe una scommessa, e la direttiva utente 2026-11-14 dice
            # "default SOLO dove il modello lo supporta".
            supported = _efforts_for(model)
            if supported is None:
                data.pop("reasoning_effort", None)
            else:
                eff = _pick_effort(eff, supported)
                if eff and eff in supported:
                    data["reasoning_effort"] = eff
                else:
                    data.pop("reasoning_effort", None)
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

    async def async_post_call_failure_hook(
        self, request_data, original_exception, user_api_key_dict,
        traceback_str=None,
    ):
        """Fallimento di una richiesta CLIENT -> conta per la quarantena.

        Arriva quando litellm ha esaurito i retry del router (num_retries 3),
        quindi un 429 transitorio isolato non basta: servono
        QUAR_FAIL_THRESHOLD falli in QUAR_FAIL_WINDOW_S. Contano solo i guasti
        del modello (429/5xx/timeout/connessione, vedi _quar_failure_kind):
        400/401/404 e context overflow sono errori nostri. Non solleva nulla:
        l'errore al client resta esattamente quello originale.
        """
        kind = _quar_failure_kind(original_exception)
        if kind:
            _quar_note_fail(
                str((request_data or {}).get("model") or ""), kind
            )
        return None

    def _hybrid_reserves(self, needed_tokens, wants_image):
        """Fino a HYBRID_RESERVE_MAX id CONCRETI di riserva free, per UNA
        richiesta a hybrid-model.

        Le wildcard (groq/*, cloudflare/*, openrouter/*) non possono essere
        membri di un routing_group, quindi la riserva si risolve qui, per
        singola richiesta, sui listing gia' in cache. Si scarta:
          - chi non e' sotto HYBRID_RESERVE_PREFIXES;
          - i NON-CHAT del listing (embed/whisper/guard/rerank) e i media;
          - i modelli openrouter a PAGAMENTO (solo ":free"/"openrouter/free");
          - i quarantenati;
          - chi non contiene prompt + margine (contesto reale dal listing);
          - chi dichiara di NON accettare immagini, se la richiesta ne ha.
        Ordine: prima chi accetta immagini (quando servono), poi contesto piu'
        largo, poi tier di costo, poi openrouter (endpoint free piu' stabile).
        Non solleva mai: senza listing torna [] e il gruppo resta senza
        riserve (comportamento identico a prima)."""
        try:
            _ensure_proxy_models_refresher()
            with _PROXY_MODELS_LOCK:
                known = dict(_PROXY_MODELS_CACHE["models"])
            names = list(known)
            for n in _DYN_CTX:
                if n not in known:
                    names.append(n)
            budget = needed_tokens + self._margin(needed_tokens)
            out = []
            for n in names:
                if not any(n.startswith(p) for p in HYBRID_RESERVE_PREFIXES):
                    continue
                if any(b in n.lower() for b in NON_CHAT_WORKER_SUBSTRINGS):
                    continue
                if n.startswith(LLAMA_SWAP_MEDIA_PREFIX):
                    continue
                if n.startswith("openrouter/") and not (
                        n == "openrouter/free" or n.endswith(":free")):
                    continue
                if _quarantined(n):
                    continue
                win = _DYN_CTX.get(n) or known.get(n, (None, None))[0]
                if win is not None and win < budget:
                    continue
                # VISION: si accetta solo chi dichiara ESPLICITAMENTE di
                # accettare immagini (True nel listing). Il silenzio NON e' una
                # promessa: cloudflare/groq non espongono le modalita', e un
                # text-only su una richiesta con immagine e' un 400 certo.
                if wants_image and _DYN_VISION.get(n) is not True:
                    continue
                out.append(n)
            # "openrouter/free" per primo: e' il meta-router gratuito che
            # sceglie da solo il miglior modello free disponibile lato
            # OpenRouter (e' la richiesta esplicita dell'utente), poi contesto.
            out.sort(key=lambda n: (
                0 if n == "openrouter/free" else 1,
                -(_DYN_CTX.get(n) or known.get(n, (0, None))[0] or 0),
                _model_tier(n),
                0 if n.startswith("openrouter/") else 1,
            ))
            return out[:HYBRID_RESERVE_MAX]
        except Exception:
            return []

    async def async_filter_deployments(self, model, healthy_deployments,
                                       messages=None, request_kwargs=None,
                                       parent_otel_span=None):
        """TIER di hybrid-model: i modelli "nostri" prima del cloud.

        litellm invoca questo hook PRIMA della strategia del gruppo
        (callback_filters stage, verificato su 1.111) e il risultato VINCOLA
        la scelta: restituire i soli tier 1 fa si' che least-busy bilanci per
        CARICO tra local-model/small-model/syn:large:text/syn:small:text e che
        gemini-flash-latest entri SOLO quando nessuno di loro e' sano (tutti in
        cooldown/down). La strategia nativa "priority" non sapeva farlo: entro
        il tier sceglie A CASO e vieta routing_strategy_args.

        La lista restituita viene RUOTATA di una posizione a ogni richiesta
        (_HYBRID_RR): least-busy sceglie il primo indice a conteggio minimo,
        quindi a conteggi pari (traffico sequenziale) senza rotazione
        vincerebbe sempre lo stesso membro.

        Replica inoltre il pre-filtro vision nativo per i membri che
        dichiarano supports_vision: False (syn:small:text): una richiesta con
        immagine non deve mai finirci.

        Fail-open su qualunque errore o per qualunque modello diverso: un
        filtro non deve MAI rompere una richiesta."""
        try:
            # litellm risolve il model_group_alias PRIMA della selezione: qui
            # arriva il nome del GRUPPO. Si accetta anche il nome pubblico per
            # robustezza (se un giorno l'alias sparisce dalla config).
            if str(model or "") not in (HYBRID_GROUP, HYBRID_PUBLIC):
                return healthy_deployments
            deps = [d for d in (healthy_deployments or [])
                    if isinstance(d, dict)]
            if not deps:
                return healthy_deployments
            wants_image = _request_has_image(
                messages if messages is not None else request_kwargs)
            eligible = [
                d for d in deps
                if not (wants_image
                        and (d.get("model_info") or {}).get(
                            "supports_vision") is False)
            ]
            if not eligible:
                return deps
            tier1 = [d for d in eligible
                     if str(d.get("model_name") or "") in HYBRID_TIER1]
            chosen = tier1 or eligible
            # Rotazione: vedi _HYBRID_RR. Sposta in testa, a turno, un membro
            # diverso cosi' i pareggi di least-busy non cadono sempre sul primo.
            if len(chosen) > 1:
                shift = next(_HYBRID_RR) % len(chosen)
                chosen = chosen[shift:] + chosen[:shift]
            state = (len(tier1), len(eligible))
            if state != _HYBRID_TIER_STATE[0]:
                _HYBRID_TIER_STATE[0] = state
                print(f"[reasoning_clamp] {HYBRID_PUBLIC}: "
                      f"{len(tier1)}/{len(eligible)} membri tier1 sani -> "
                      + ("cloud in attesa" if tier1
                         else "tier1 ESAURITO, si passa al cloud"),
                      flush=True)
            return chosen
        except Exception:
            return healthy_deployments

    async def async_filter_listed_models(self, user_api_key_dict, model_names):
        """Nasconde i modelli in quarantena dalle listing.

        Gira su /v1/models, /v1/models/{id}, /model/info e
        /model_group/info: e' l'unico punto in cui litellm permette di
        stringere la lista (mai di allargarla). Il nome ricompare da solo
        allo scadere del TTL (nessun thread di pulizia: _quarantined guarda
        l'orologio) e chi ha bisogno di quel modello puo' ancora chiederlo
        esplicitamente: qui si toglie solo dalla VETRINA.
        """
        # NOTA: NON nascondere HYBRID_GROUP qui. litellm espande la richiesta
        # anche agli ALIAS il cui target e' nascosto
        # (_names_hidden_by_listing_callbacks), quindi nascondere il gruppo
        # faceva sparire hybrid-model da /v1/models (misurato): il nome
        # interno resta visibile, e' solo una entry in piu'.
        hidden = {n for n in model_names if _quarantined(n)}
        if not hidden:
            return model_names
        # log solo quando l'insieme CAMBIA: /v1/models viene interrogato di
        # continuo dalle UI e un print per chiamata sarebbe rumore
        if hidden != _QUAR_LAST_HIDDEN[0]:
            _QUAR_LAST_HIDDEN[0] = hidden
            shown = ", ".join(sorted(hidden)[:5])
            more = "..." if len(hidden) > 5 else ""
            print(f"[reasoning_clamp] listing: {len(hidden)} modelli in "
                  f"quarantena nascosti ({shown}{more})", flush=True)
        return [n for n in model_names if n not in hidden]

    async def async_post_call_success_hook(
        self, data, user_api_key_dict, response
    ):
        """Avviso di throttling travestito da risposta 200 -> errore 429.

        Un provider free puo' rispondere 200 mettendo il messaggio di quota
        nel CONTENUTO (Yqcloud, richiesta utente 2026-11-02): per
        litellm e' un successo, quindi niente retry/fallback e il client
        riceve testo inutilizzabile al posto della risposta. Qui la risposta
        viene ispezionata PRIMA di uscire e trasformata in un 429 esplicito,
        cosi' l'errore e' visibile (OpenWebUI lo mostra) invece di sembrare
        una risposta. Il resto delle risposte passa invariato.
        """
        text = _response_text(response)
        model = str((data or {}).get("model") or "")
        if text is None:
            # stream (o embedding): lo stato lo decide l'iterator hook
            return response
        if not _upstream_notice(text):
            _quar_note_ok(model)
            return response
        _quar_note_fail(model, "quota-200")
        print(
            f"[reasoning_clamp] avviso di throttling servito come 200 "
            f"(model={model}) -> 429 esplicito",
            flush=True,
        )
        from fastapi import HTTPException

        raise HTTPException(
            status_code=429,
            detail=(
                "provider free temporaneamente limitato (quota per IP) per "
                f"'{model}': l'upstream ha restituito l'avviso come contenuto "
                "di un 200. Riprova tra qualche minuto oppure usa un modello "
                "di un altro provider."
            ),
        )

    async def async_post_call_streaming_iterator_hook(
        self, user_api_key_dict, response, request_data
    ):
        """Chiude in modo PULITO uno stream che l'upstream tronca a meta'.

        Caso reale (2026-10-30, syn:large:text 95k token): synthetic.new -
        nginx davanti a vLLM - chiude il body chunked SENZA il chunk finale
        quando il backend resta muto ~60s (visto anche un 502 nginx a 63s
        sulla stessa richiesta). aiohttp lo riporta come
        "TransferEncodingError: Not enough data to satisfy transfer length
        header"; litellm, avendo gia' mandato chunk al client, NON puo' fare
        fallback (router.stream_with_fallbacks rilancia l'originale) e
        OpenWebUI mostra un 500 perdendo la risposta parziale.

        Qui, se l'eccezione arriva DOPO che il client ha ricevuto contenuto,
        lo stream viene chiuso con un chunk finale esplicito: il client tiene
        il testo parziale + il marcatore. Se non e' stato emesso NULLA
        l'eccezione risale, cosi' retry/fallback di litellm restano invariati.
        Client disconnesso (CancelledError/GeneratorExit) resta un errore
        del client, non dell'upstream: rilanciato.

        In piu' (2026-11-02) guarda i PRIMI UPSTREAM_NOTICE_SCAN_CHARS
        caratteri di testo: se l'upstream ha spedito l'avviso di throttling
        come contenuto di un 200 (Yqcloud), il testo-spazzatura non
        arriva al client: al suo posto un chunk esplicito. Le header SSE sono
        gia' partite (200), quindi non si puo' piu' cambiare lo status.
        """
        emitted = False
        last = None
        probe = ""
        probing = True
        try:
            async for chunk in response:
                if probing:
                    probe += _chunk_text(chunk)
                    if _upstream_notice(probe):
                        _quar_note_fail(
                            str((request_data or {}).get("model") or ""),
                            "quota-200",
                        )
                        print(
                            "[reasoning_clamp] avviso di throttling nello "
                            f"stream (model={str((request_data or {}).get('model') or '')})"
                            " -> sostituito con messaggio esplicito",
                            flush=True,
                        )
                        marker = self._truncated_stream_marker(
                            last, request_data, UPSTREAM_NOTICE_STREAM_TEXT
                        )
                        if marker is not None:
                            yield marker
                        return
                    if len(probe) >= UPSTREAM_NOTICE_SCAN_CHARS:
                        probing = False
                emitted = True
                last = chunk
                yield chunk
        except (asyncio.CancelledError, GeneratorExit):
            raise
        except Exception as e:
            if not emitted:
                raise
            print(
                f"[reasoning_clamp] stream upstream troncato dopo l'output "
                f"({type(e).__name__}: {e}) -> chiusura pulita con marcatore",
                flush=True,
            )
            _quar_note_fail(
                str((request_data or {}).get("model") or ""),
                f"stream {type(e).__name__}",
            )
            marker = self._truncated_stream_marker(last, request_data)
            if marker is not None:
                yield marker
        else:
            # stream finito senza errori: il modello ha risposto davvero
            _quar_note_ok(str((request_data or {}).get("model") or ""))

    @staticmethod
    def _truncated_stream_marker(last, request_data, text=TRUNCATED_STREAM_MARKER):
        """Chunk finale (finish_reason "stop") col testo del marcatore.

        Ricostruito dall'ultimo chunk per conservare id/created/model del
        provider: OpenWebUI aggancia la fine dello stream a quel messaggio.
        Se la costruzione fallisce (cambio di tipi litellm) si chiude e
        basta, senza rimettere in piedi l'errore.
        `text` e' il testo da mettere come contenuto: il default e' il
        marcatore di stream troncato, l'avviso di throttling passa il suo.
        """
        try:
            from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices

            return ModelResponseStream(
                id=getattr(last, "id", None) or "chatcmpl-stream-truncated",
                created=getattr(last, "created", None) or int(time.time()),
                model=getattr(last, "model", None)
                or str((request_data or {}).get("model") or ""),
                choices=[
                    StreamingChoices(
                        index=0,
                        delta=Delta(content=text),
                        finish_reason="stop",
                    )
                ],
            )
        except Exception as e:
            print(
                f"[reasoning_clamp] marcatore stream non costruito: "
                f"{type(e).__name__}: {e}",
                flush=True,
            )
            return None


rclamp = ReasoningClamp()
