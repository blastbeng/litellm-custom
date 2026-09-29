"""Gateway OpenRouter per litellm (vedi docker-compose.yml).

Perche' esiste:
1. L'espansione wildcard di litellm (v1.104.0) con provider "openai" chiama
   sempre {host}/v1/models, ma l'API OpenRouter vive su /api/v1/models:
   su openrouter.ai/v1/models il sito risponde HTML -> JSONDecodeError.
2. La logica di listing di litellm rinomina gli id il cui primo segmento
   coincide con un provider noto (openai/..., anthropic/..., deepseek/...),
   corrompendo ~185/458 modelli OpenRouter. Prefixando gli id con
   "openrouter/" nella risposta di /models il rename non scatta e la
   wildcard openrouter/* instrada 1:1 (il capture rimuove il prefisso).

Il traffico in uscita verso openrouter.ai passa per llmtrim (proxy CONNECT,
come litellm): cosi' le chiamate openrouter di litellm sono intercettate,
comprese e tracciate da llmtrim come quelle di AiderDesk diretto.
"""
import json
import os

import httpx
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM = "https://openrouter.ai"
PROXY = os.environ.get("GW_PROXY")          # http://host.docker.internal:43117 (llmtrim)
CA = os.environ.get("GW_CA", "/data/ca-bundle.pem")

_client_kwargs = {"timeout": httpx.Timeout(600.0, connect=15.0)}
if PROXY:
    _client_kwargs["proxy"] = PROXY
_client_kwargs["verify"] = CA
client = httpx.Client(**_client_kwargs)

HOP_HEADERS = {"host", "content-length", "transfer-encoding", "connection",
               "accept-encoding", "keep-alive"}


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

    def _models(self, r):
        """Listing: prefixa gli id con "openrouter/" (solo data[].id)."""
        data = r.content
        try:
            j = json.loads(data)
            for m in j.get("data", []):
                if "id" in m and not str(m["id"]).startswith("openrouter/"):
                    m["id"] = "openrouter/" + str(m["id"])
            data = json.dumps(j).encode()
        except Exception as e:
            print(f"gw: models rewrite skip: {e}", flush=True)
        self._send(r.status_code, dict(r.headers), data)

    def _forward(self, method):
        # /v1/... -> /api/v1/...
        target = UPSTREAM + "/api" + self.path
        body = None
        clen = int(self.headers.get("Content-Length") or 0)
        if clen:
            body = self.rfile.read(clen)
            # Il modello nativo "openrouter/free" (Free Models Router) ha il
            # prefisso openrouter/ DENTRO l'id: la capture della wildcard
            # openrouter/* di litellm lo riduce a "free" -> 404 "No endpoints
            # available". Qui lo si ripristina (unico id nativo con prefisso).
            try:
                j = json.loads(body)
                if isinstance(j, dict) and j.get("model") == "free":
                    j["model"] = "openrouter/free"
                    body = json.dumps(j).encode()
            except Exception:
                pass
        headers = self._upstream_headers()
        try:
            if self.path.rstrip("/") == "/v1/models":
                with client.stream(method, target, content=body, headers=headers) as r:
                    r.read()
                    self._models(r)
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
            msg = json.dumps({"error": {"message": f"openrouter-gw: {type(e).__name__}: {e}"}}).encode()
            self._send(502, {"Content-Type": "application/json"}, msg)


ThreadingHTTPServer(("0.0.0.0", 80), Handler).serve_forever()
