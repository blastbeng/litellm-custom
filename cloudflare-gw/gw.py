"""Gateway per Cloudflare Workers AI (cloudflare-gw).

Stessa ragione di groq-gw/openrouter-gw, con un vincolo in piu': litellm
espande le wildcard chiamando SEMPRE {origin}/v1/models (ignora il path
dell'api_base) e Workers AI NON ha un listing OpenAI-compatible
(GET /ai/v1/models -> 7001 "GET not supported for requested URI"). Il
catalogo vive solo su /ai/models/search: REST, paginato, con i metadati
dentro properties[] (context_window, price, reasoning_effort, ...).

Questo gateway quindi:
  1. traduce /v1/models (o /models) nel listing REST, FILTRA i modelli non
     usabili su questo account e mappa i metadati nei campi che litellm e
     reasoning_clamp leggono gia' dagli altri provider (context_length,
     pricing, reasoning_parameters.efforts);
  2. inoltra tutto il resto - chat/completions con streaming SSE compreso -
     in passthrough: nessuna riscrittura del body, perche' l'id arriva GIA'
     nativo ("@cf/vendor/model"). La capture della wildcard toglie
     "cloudflare/" e litellm toglie il prefisso provider "openai/"; gli id
     Cloudflare iniziano con "@cf/" e non collidono con i provider noti di
     litellm (nessun rename corrotto, a differenza di OpenRouter);
  3. tiene una cache last-good del listing (scritta DOPO il mapping: gli id
     sono gia' nel formato pubblico): se Cloudflare e' irraggiungibile i
     modelli restano visibili su litellm invece di sparire.

FILTRI (li decide l'ACCOUNT, non il codice): questo account e' su Workers
FREE, quindi i modelli con require_workers_paid=true rispondono 403 ("not
available on the Workers Free plan"); quelli con planned_deprecation_date
passata rispondono 5028 ("Model has been deprecated"); i deployment che
finiscono in "-lora" sono basi LoRA, non modelli general purpose. Con un
account pagato basta DROP_PAID=0.

Config via env:
  UPSTREAM   - base AI dell'account,
               es. https://api.cloudflare.com/client/v4/accounts/<id>/ai
  ID_PREFIX  - prefisso degli id nel listing (default "cloudflare/")
  PORT       - porta di ascolto (default 80)
  CF_TASK    - task del listing (default "Text Generation"; le altre task,
               embeddings/ASR/TTS, non passano dalla rotta chat di litellm)
  DROP_PAID / DROP_DEPRECATED / DROP_LORA - "0" disattiva il filtro
               (default: tutti attivi)
  MODELS_CACHE - "0" disattiva la cache last-good (default attiva)
  MODELS_CACHE_FILE - path del file cache (default /cache/models.json)
  GW_PROXY   - proxy HTTPS per l'upstream (opzionale)
  GW_CA      - bundle CA, usato SOLO se il file esiste (come provider-gw)
"""
import json
import os
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx

UPSTREAM = os.environ["UPSTREAM"].rstrip("/")
ID_PREFIX = os.environ.get("ID_PREFIX", "cloudflare/")
PORT = int(os.environ.get("PORT", "80"))
CF_TASK = os.environ.get("CF_TASK", "Text Generation")
DROP_PAID = os.environ.get("DROP_PAID", "1") == "1"
DROP_DEPRECATED = os.environ.get("DROP_DEPRECATED", "1") == "1"
DROP_LORA = os.environ.get("DROP_LORA", "1") == "1"
MODELS_CACHE = os.environ.get("MODELS_CACHE", "1") == "1"
CACHE_FILE = os.environ.get("MODELS_CACHE_FILE", "/cache/models.json")
PROXY = os.environ.get("GW_PROXY")
CA = os.environ.get("GW_CA", "/etc/ssl/certs/ca-certificates.crt")

# /ai/models/search ignora il parametro page quando il filtro task= e'
# presente (verificato: per_page=50 -> 40 risultati, page=2 -> 0), ma il
# loop resta per quando il catalogo crescera' oltre una pagina.
PER_PAGE = 100
MAX_PAGES = 10
LIST_TIMEOUT = httpx.Timeout(60.0, connect=15.0)

_client_kwargs = {"timeout": httpx.Timeout(600.0, connect=15.0)}
if PROXY:
    _client_kwargs["proxy"] = PROXY
# GW_CA SOLO se il file esiste davvero (stessa regola di provider-gw: un
# path mancante faceva crashare il gateway all'avvio).
if CA and os.path.exists(CA):
    _client_kwargs["verify"] = CA
client = httpx.Client(**_client_kwargs)
# client separato per il listing: un catalogo bloccato non deve tenere in
# ostaggio una richiesta chat (e viceversa)
list_client = httpx.Client(**{**_client_kwargs, "timeout": LIST_TIMEOUT})

HOP_HEADERS = {"host", "content-length", "transfer-encoding", "connection",
               "accept-encoding", "keep-alive"}


def _expired(value):
    """True se planned_deprecation_date e' una data ISO gia' passata."""
    if not value:
        return False
    today = time.strftime("%Y-%m-%d", time.gmtime())
    return str(value)[:10] <= today


def _pricing(price):
    """properties.price (USD per MILIONE di token) -> pricing OpenAI
    (USD per token, come OpenRouter/synthetic: reasoning_clamp lo legge con
    _price_usd_per_token per classificare gratis/economico/costoso)."""
    if not isinstance(price, list):
        return None
    inp = out = None
    for p in price:
        if not isinstance(p, dict):
            continue
        unit = str(p.get("unit") or "").lower()
        try:
            v = float(p.get("price"))
        except (TypeError, ValueError):
            continue
        if "input" in unit:
            inp = v
        elif "output" in unit:
            out = v
    d = {}
    if inp is not None:
        d["prompt"] = inp / 1e6
    if out is not None:
        d["completion"] = out / 1e6
    return d or None


def _efforts(re_prop):
    """properties.reasoning_effort -> reasoning_parameters.efforts.

    SOLO supported_efforts: e' l'elenco AUTOREVOLE dei valori accettati
    (verificato: qwen3.8-27b rifiuta "high", gpt-oss-120b rifiuta "xhigh").
    I modelli reasoning che NON lo dichiarano (glm-4.7-flash, gemma-4) non
    hanno una lista pubblica: meglio nessun elenco (reasoning_clamp rimuove
    allora il parametro e vale il default Cloudflare) che una lista inventata.
    """
    if not isinstance(re_prop, dict):
        return None
    raw = re_prop.get("supported_efforts")
    if not isinstance(raw, (list, tuple)):
        return None
    vals = [str(x).strip().lower() for x in raw
            if isinstance(x, str) and str(x).strip()]
    return vals or None


def _to_openai(raw):
    """Catalogo REST -> {"object": "list", "data": [...]} per litellm."""
    data = []
    for m in raw:
        if not isinstance(m, dict):
            continue
        name = str(m.get("name") or "").strip()
        if not name:
            continue
        props = {}
        for p in (m.get("properties") or []):
            if isinstance(p, dict) and p.get("property_id"):
                props[p["property_id"]] = p.get("value")
        if DROP_PAID and str(props.get("require_workers_paid")).lower() == "true":
            continue
        if DROP_DEPRECATED and _expired(props.get("planned_deprecation_date")):
            continue
        if DROP_LORA and name.endswith("-lora"):
            continue
        entry = {"id": ID_PREFIX + name if not name.startswith(ID_PREFIX)
                 else name,
                 "object": "model", "owned_by": "cloudflare"}
        ctx = props.get("context_window")
        try:
            ctx = int(float(ctx))
        except (TypeError, ValueError):
            ctx = 0
        if ctx > 0:
            entry["context_length"] = ctx
        pr = _pricing(props.get("price"))
        if pr:
            entry["pricing"] = pr
        effs = _efforts(props.get("reasoning_effort"))
        if effs:
            entry["reasoning_parameters"] = {"efforts": effs}
        data.append(entry)
    return {"object": "list", "data": data}


def _fetch_listing(headers):
    """Catalogo REST completo (tutte le pagine del task CF_TASK)."""
    out = []
    for page in range(1, MAX_PAGES + 1):
        url = (f"{UPSTREAM}/models/search?task="
               f"{urllib.parse.quote(CF_TASK)}&per_page={PER_PAGE}&page={page}")
        r = list_client.get(url, headers=headers)
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
        j = r.json()
        if not j.get("success", False):
            raise RuntimeError(f"errors={j.get('errors')}")
        batch = j.get("result") or []
        if not batch:
            break
        out.extend(batch)
        if len(batch) < PER_PAGE:
            break
    return out


def _load_cache():
    try:
        with open(CACHE_FILE) as f:
            return json.dumps(json.load(f)).encode()
    except Exception:
        return None


def _save_cache(data):
    try:
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        tmp = CACHE_FILE + ".tmp"
        with open(tmp, "w") as f:
            f.write(data.decode())
        os.replace(tmp, CACHE_FILE)
    except Exception as e:
        print(f"gw: cache save skip: {e}", flush=True)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        print("gw: " + fmt % args, flush=True)

    def _headers_for(self, headers=None):
        src = headers if headers is not None else self.headers
        out = {k: v for k, v in src.items() if k.lower() not in HOP_HEADERS}
        out["Accept-Encoding"] = "identity"
        out["Connection"] = "close"
        return out

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
            if k.lower() not in ("content-length", "transfer-encoding",
                                 "connection"):
                self.send_header(k, v)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _serve_models(self, headers):
        try:
            raw = _fetch_listing(headers)
        except Exception as e:
            print(f"gw: listing upstream error: {type(e).__name__}: {e}",
                  flush=True)
            cached = _load_cache() if MODELS_CACHE else None
            if cached is not None:
                print("gw: listing FAILED -> serving last-good cache",
                      flush=True)
                self._send(200, {"Content-Type": "application/json",
                                 "X-Models-Cache": "stale"}, cached)
                return
            msg = json.dumps({"error": {
                "message": f"cloudflare-gw: listing unavailable: "
                           f"{type(e).__name__}: {e}"}}).encode()
            self._send(502, {"Content-Type": "application/json"}, msg)
            return
        data = json.dumps(_to_openai(raw)).encode()
        if MODELS_CACHE:
            _save_cache(data)
        self._send(200, {"Content-Type": "application/json"}, data)

    def _forward(self, method):
        path = urllib.parse.urlparse(self.path).path
        body = None
        clen = int(self.headers.get("Content-Length") or 0)
        if clen:
            body = self.rfile.read(clen)
        headers = self._headers_for()
        if path.rstrip("/").endswith("/models") and method == "GET":
            self._serve_models(headers)
            return
        target = UPSTREAM + self.path
        try:
            with client.stream(method, target, content=body,
                               headers=headers) as r:
                self.send_response(r.status_code)
                for k, v in r.headers.items():
                    if k.lower() not in ("content-length", "transfer-encoding",
                                         "connection"):
                        self.send_header(k, v)
                self.send_header("Connection", "close")
                self.end_headers()
                for chunk in r.iter_raw():
                    if chunk:
                        self.wfile.write(chunk)
        except Exception as e:
            msg = json.dumps({"error": {
                "message": f"cloudflare-gw: {type(e).__name__}: {e}"}}).encode()
            self._send(502, {"Content-Type": "application/json"}, msg)


ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
