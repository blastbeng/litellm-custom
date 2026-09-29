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
"""
import json
import os

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

_client_kwargs = {"timeout": httpx.Timeout(600.0, connect=15.0)}
if PROXY:
    _client_kwargs["proxy"] = PROXY
_client_kwargs["verify"] = CA
client = httpx.Client(**_client_kwargs)

HOP_HEADERS = {"host", "content-length", "transfer-encoding", "connection",
               "accept-encoding", "keep-alive"}


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
        return {k: v for k, v in self.headers.items()
                if k.lower() not in HOP_HEADERS} | {"Accept-Encoding": "identity",
                                                    "Connection": "close"}

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
        "x:latest" gli id colliderebbero."""
        try:
            j = json.loads(data)
            for m in j.get("data", []):
                if "id" not in m:
                    continue
                mid = str(m["id"])
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