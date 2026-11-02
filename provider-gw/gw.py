"""Gateway generico per il LISTING wildcard di litellm (groq-gw, gemini-gw, ...).

Stesso problema di openrouter-gw/gw.py:
la logica di listing di litellm (v1.104.0) rinomina gli id il cui primo
segmento coincide con un provider noto (openai/..., anthropic/...,
deepseek/...), corrompendoli. Prefixando gli id con ID_PREFIX nella risposta
di /models il rename non scatta e la wildcard (es. groq/*) instrada 1:1: il
capture della wildcard rimuove il prefisso prima di chiamare l'upstream.

Differenza con openrouter-gw: nessuna riscrittura di path opzionale gestita
via PATH_PRE (litellm chiama SEMPRE {origin}/v1 per il listing, ma per le
chiamate chat usa il path completo dell'api_base).

CACHE LISTING (opt-in, MODELS_CACHE=1): per upstream instabili
(deepseek4free e' un progetto in sviluppo) la richiesta /models di litellm
puo' fallire -> litellm mostra ZERO modelli per la wildcard. Con la cache,
l'ultima lista valida viene salvata su disco e servita quando l'upstream
fallisce (header X-Models-Cache: stale), cosi' i modelli restano SEMPRE
visibili su litellm. La cache e' scritta DOPO il prefixing (gli id cache
sono gia' nel formato pubblico).

Il traffico in uscita verso l'upstream passa per llmtrim (proxy CONNECT,
GW_PROXY), cosi' anche queste chiamate sono intercettate/tracciate.

Config via env:
  UPSTREAM   - base URL upstream (es. https://api.groq.com)
  PATH_PRE   - prefisso di path da anteporre a /v1/... (es. "/openai" per
               Groq: /v1/models -> /openai/v1/models; "/v1beta/openai" per
               Gemini)
  ID_PREFIX  - prefisso da aggiungere agli id in /models (es. "groq/")
  PORT       - porta di ascolto (default 80)
  GW_PROXY   - proxy HTTPS per l'upstream (llmtrim)
  GW_CA      - bundle CA (root + llmtrim)
  MODELS_CACHE - "1" = abilita la cache last-good del listing
  MODELS_CACHE_FILE - path del file cache (default /cache/models.json)
  CONTEXT_FROM - path di un endpoint NATIVO che espone il context_length
               per-modello quando il listing OpenAI-compatible NON lo
               include (Ollama: "/api/tags", campo details.context_length).
               Il valore viene iniettato in /v1/models come "context_length"
               cosi' reasoning_clamp (_DYN_CTX) usa il contesto REALE
               per-modello, come per gli altri provider. Best-effort: se
               l'endpoint non risponde o il campo manca, il listing resta
               com'e' (vale il valore statico di config.yaml).
  UPSTREAM_KEY - chiave PROPRIA dell'upstream (opt-in). Senza questa il
               gateway inoltra l'Authorization del client (e' cosi' che
               funzionano groq/gemini/ollama: la chiave e' quella che litellm
               manda). Con questa la chiave del client viene SOSTITUITA:
               serve quando litellm ha solo un segnaposto (gpt4free: la
               chiave vera vive nell'.env del progetto upstream e NON deve
               essere copiata nel repo litellm).
  UPSTREAM_KEY_FILE / UPSTREAM_KEY_NAME - alternativa a UPSTREAM_KEY: la
               chiave si legge da un file (es. l'.env del progetto upstream,
               montato read-only) e UPSTREAM_KEY_NAME e' la variabile da
               estrarre (es. "G4F_API_KEY"); senza NAME vale il file intero.
               Rilettura con TTL 60s: una chiave ruotata viene presa senza
               riavviare il gateway.
  UPSTREAM_AUTH_HEADER - header usato per UPSTREAM_KEY (default
               "Authorization", inviato come "Bearer <key>"; con un nome
               diverso la chiave va nuda).
"""
import json
import os
import time

import httpx
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM = os.environ["UPSTREAM"]
PATH_PRE = os.environ.get("PATH_PRE", "")
ID_PREFIX = os.environ.get("ID_PREFIX", "")
PORT = int(os.environ.get("PORT", "80"))
PROXY = os.environ.get("GW_PROXY")
CA = os.environ.get("GW_CA", "/data/ca-bundle.pem")
STRIP_LATEST = os.environ.get("STRIP_LATEST", "") == "1"
MODELS_CACHE = os.environ.get("MODELS_CACHE", "") == "1"
CACHE_FILE = os.environ.get("MODELS_CACHE_FILE", "/cache/models.json")
CONTEXT_FROM = os.environ.get("CONTEXT_FROM", "")
UPSTREAM_KEY = os.environ.get("UPSTREAM_KEY", "")
UPSTREAM_KEY_FILE = os.environ.get("UPSTREAM_KEY_FILE", "")
UPSTREAM_KEY_NAME = os.environ.get("UPSTREAM_KEY_NAME", "")
UPSTREAM_AUTH_HEADER = os.environ.get("UPSTREAM_AUTH_HEADER", "Authorization")

_client_kwargs = {"timeout": httpx.Timeout(600.0, connect=15.0)}
if PROXY:
    _client_kwargs["proxy"] = PROXY
_client_kwargs["verify"] = CA
client = httpx.Client(**_client_kwargs)

HOP_HEADERS = {"host", "content-length", "transfer-encoding", "connection",
               "accept-encoding", "keep-alive"}
# header di autenticazione del CLIENT: vanno rimossi quando il gateway usa una
# chiave propria (litellm manda un segnaposto, l'upstream pretende la sua)
AUTH_HEADERS = {"authorization", "g4f-api-key", "x-api-key"}

_KEY_CACHE = {"ts": 0.0, "key": None}
_KEY_TTL = 60.0


def _upstream_key():
    """Chiave PROPRIA dell'upstream: None = non configurata (passthrough
    dell'Authorization del client), stringa (anche vuota) = configurata.
    Vedi UPSTREAM_KEY* nel docstring del modulo. Memo TTL 60s: la rilettura
    del file non costa un open per ogni richiesta e una rotazione della
    chiave nell'.env upstream viene vista senza riavviare il gateway."""
    if not UPSTREAM_KEY and not UPSTREAM_KEY_FILE:
        return None
    now = time.time()
    if _KEY_CACHE["key"] is not None and now - _KEY_CACHE["ts"] < _KEY_TTL:
        return _KEY_CACHE["key"]
    key = UPSTREAM_KEY
    if not key and UPSTREAM_KEY_FILE:
        try:
            with open(UPSTREAM_KEY_FILE) as f:
                txt = f.read()
        except OSError as e:
            print(f"gw: upstream key file skip: {type(e).__name__}: {e}", flush=True)
            key = _KEY_CACHE["key"] or ""
        else:
            key = txt.strip()
            if UPSTREAM_KEY_NAME:
                key = ""
                for line in txt.splitlines():
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    if k.strip() == UPSTREAM_KEY_NAME:
                        key = v.strip().strip('"').strip("'")
                        break
    _KEY_CACHE.update(ts=now, key=key)
    return key


_CTX_CACHE = (0.0, {})
_CTX_TTL = 60.0


def _native_contexts():
    """Mappa {nome_nativo: context_length} dall'endpoint CONTEXT_FROM.

    Ollama NON espone context_length nel listing OpenAI-compatible
    (/v1/models), ma lo dichiara nell'endpoint nativo /api/tags
    (details.context_length): senza questo, reasoning_clamp userebbe il
    valore statico della wildcard per TUTTI i modelli ollama/*. Best-effort
    con memo TTL 60s: {} se l'upstream non risponde o il campo manca.
    """
    global _CTX_CACHE
    if not CONTEXT_FROM:
        return {}
    now = time.time()
    ts, cached = _CTX_CACHE
    if cached and now - ts < _CTX_TTL:
        return cached
    try:
        r = client.get(UPSTREAM + CONTEXT_FROM)
        out = {}
        if r.status_code == 200:
            for m in r.json().get("models", []):
                name = str(m.get("name") or m.get("model") or "")
                ctx = (m.get("details") or {}).get("context_length")
                if name and isinstance(ctx, (int, float)) and ctx > 0:
                    out[name] = int(ctx)
        _CTX_CACHE = (now, out)
        return out
    except Exception as e:
        print(f"gw: context fetch skip: {type(e).__name__}: {e}", flush=True)
        return {}


def _load_cache():
    try:
        with open(CACHE_FILE) as f:
            j = json.load(f)
        return json.dumps(j).encode()
    except Exception:
        return None


def _save_cache(data: bytes):
    try:
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        with open(CACHE_FILE, "w") as f:
            f.write(data.decode())
    except Exception as e:
        print(f"gw: cache save skip: {e}", flush=True)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        print("gw: " + fmt % args, flush=True)

    def _upstream_headers(self):
        headers = {k: v for k, v in self.headers.items()
                   if k.lower() not in HOP_HEADERS}
        key = _upstream_key()
        if key is not None:
            # auth PROPRIA dell'upstream (UPSTREAM_KEY*): quella del client e'
            # un segnaposto di litellm e va SOSTITUITA, non affiancata -
            # altrimenti l'upstream riceve due credenziali e risponde 401
            headers = {k: v for k, v in headers.items()
                       if k.lower() not in AUTH_HEADERS}
            if key:
                headers[UPSTREAM_AUTH_HEADER] = (
                    f"Bearer {key}"
                    if UPSTREAM_AUTH_HEADER.lower() == "authorization" else key)
        headers["Accept-Encoding"] = "identity"
        headers["Connection"] = "close"
        return headers

    def do_GET(self):
        self._forward("GET")

    def do_POST(self):
        self._forward("POST")

    def do_DELETE(self):
        self._forward("DELETE")

    def do_PUT(self):
        self._forward("PUT")

    def do_PATCH(self):
        self._forward("PATCH")

    def _send(self, status, headers, data):
        self.send_response(status)
        for k, v in headers.items():
            if k.lower() not in ("content-length", "transfer-encoding", "connection"):
                self.send_header(k, v)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _rewrite_models(self, data: bytes) -> bytes:
        """Prefixa gli id con ID_PREFIX (solo data[].id).
        Con STRIP_LATEST=1 rimuove prima il suffisso ":latest" (Ollama lo
        aggiunge ai modelli senza tag esplicito): il nome pubblico diventa
        "<PREFIX><model>" invece di "<PREFIX><model>:latest" (Ollama risolve
        comunque il nome senza tag). ATTENZIONE: se esistessero sia "x" che
        "x:latest" gli id colliderebbero.
        Con CONTEXT_FROM il context_length per-modello viene iniettato
        nell'entry (mappato per nome NATIVO, prima dello strip/prefix)."""
        try:
            j = json.loads(data)
            ctxs = _native_contexts()
            for m in j.get("data", []):
                if "id" not in m:
                    continue
                mid = str(m["id"])
                if mid in ctxs and "context_length" not in m:
                    m["context_length"] = ctxs[mid]
                if STRIP_LATEST and mid.endswith(":latest"):
                    mid = mid[: -len(":latest")]
                if not mid.startswith(ID_PREFIX):
                    mid = ID_PREFIX + mid
                m["id"] = mid
            return json.dumps(j).encode()
        except Exception as e:
            print(f"gw: models rewrite skip: {e}", flush=True)
            return data

    def _serve_cached_models(self):
        cached = _load_cache()
        if cached is not None:
            print("gw: models upstream FAILED -> serving last-good cache", flush=True)
            self._send(200, {"Content-Type": "application/json",
                             "X-Models-Cache": "stale"}, cached)
            return True
        return False

    def _forward(self, method):
        # PATH_PRE solo per le chiamate "/v1/..." (il listing di litellm
        # va sempre su {origin}/v1/models); per le chiamate chat litellm
        # usa gia' il path completo dell'api_base (/openai/v1/...,
        # /v1beta/openai/...) e il prepend andrebbe fatto 2 volte.
        pre = PATH_PRE if (self.path == "/v1" or self.path.startswith("/v1/")) else ""
        target = UPSTREAM + pre + self.path
        body = None
        clen = int(self.headers.get("Content-Length") or 0)
        if clen:
            body = self.rfile.read(clen)
        headers = self._upstream_headers()
        try:
            if self.path.rstrip("/").endswith("/models"):
                try:
                    with client.stream(method, target, content=body, headers=headers) as r:
                        r.read()
                        status, raw = r.status_code, r.content
                except Exception as e:
                    print(f"gw: models upstream error: {type(e).__name__}: {e}", flush=True)
                    status, raw = None, None
                if status == 200 and raw:
                    data = self._rewrite_models(raw)
                    if MODELS_CACHE:
                        _save_cache(data)
                    self._send(200, dict([]), data)
                    return
                # upstream failed / errore non-200: serve la cache last-good
                if MODELS_CACHE and self._serve_cached_models():
                    return
                if status is not None:
                    self._send(status, {"Content-Type": "application/json"}, raw or b"{}")
                    return
                msg = json.dumps({"error": {"message": f"provider-gw: models upstream unavailable"}}).encode()
                self._send(502, {"Content-Type": "application/json"}, msg)
                return
            # passthrough con streaming (SSE compreso): body delimitato da close
            with client.stream(method, target, content=body, headers=headers) as r:
                self.send_response(r.status_code)
                for k, v in r.headers.items():
                    if k.lower() not in ("content-length", "transfer-encoding", "connection"):
                        self.send_header(k, v)
                self.send_header("Connection", "close")
                self.end_headers()
                for chunk in r.iter_raw():
                    if chunk:
                        self.wfile.write(chunk)
        except Exception as e:
            msg = json.dumps({"error": {"message": f"provider-gw: {type(e).__name__}: {e}"}}).encode()
            self._send(502, {"Content-Type": "application/json"}, msg)


ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()