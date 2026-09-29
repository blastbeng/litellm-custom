"""Gate TCP per llama-swap su blastpc (192.168.1.29:11435).

Perche' esiste: litellm puo' impostare un SOLO timeout per tutta la richiesta
(httpx non distingue connect/read tramite litellm_params). Serve però:
- PC SPENTO  -> fallire SUBITO (non aspettare il timeout lungo)
- PC ACCESO -> aspettare anche 10-20 min (caricamento/inferenza modelli enormi)

Soluzione: litellm punta qui (api_base http://llama-gate/v1) con timeout lungo.
Se il TCP verso llama-swap non si connette entro HEALTH_TIMEOUT, la gate
risponde 503 in fretta (il PC e' spento/non raggiungibile). Se si connette,
fa da tubo trasparente bidirezionale (HTTP e SSE passano invariati).

Solo stdlib: nessuna dipendenza da installare.
"""
import os
import socket
import threading
import time

UP_HOST = os.environ.get("GATE_UP_HOST", "192.168.1.29")
UP_PORT = int(os.environ.get("GATE_UP_PORT", "11435"))
HEALTH_TIMEOUT = float(os.environ.get("GATE_HEALTH_TIMEOUT", "2"))
LISTEN_PORT = int(os.environ.get("GATE_PORT", "80"))

_503 = (b"HTTP/1.1 503 Service Unavailable\r\n"
        b"Content-Type: application/json\r\n"
        b"Content-Length: 94\r\n"
        b"Connection: close\r\n\r\n"
        b'{"error":{"message":"blastpc offline (gate: connect timeout) - pc spento/non raggiungibile"}}\n')


def _pipe(src: socket.socket, dst: socket.socket):
    try:
        while True:
            data = src.recv(65536)
            if not data:
                break
            dst.sendall(data)
    except OSError:
        pass
    finally:
        try:
            dst.shutdown(socket.SHUT_WR)
        except OSError:
            pass


class Handler(threading.Thread):
    def __init__(self, conn: socket.socket):
        super().__init__(daemon=True)
        self.conn = conn

    def run(self):
        try:
            t = time.time()
            up = socket.create_connection((UP_HOST, UP_PORT), timeout=HEALTH_TIMEOUT)
            # disattiva il timeout di health-check per il tubo (attese lunghe ok)
            up.settimeout(None)
            self.conn.settimeout(None)
            threading.Thread(target=_pipe, args=(up, self.conn), daemon=True).start()
            _pipe(self.conn, up)
        except OSError as e:
            # PC spento / non raggiungibile: 503 subito
            print(f"gate: upstream connect failed in {time.time()-t:.2f}s: {e}", flush=True)
            try:
                self.conn.sendall(_503)
            except OSError:
                pass
        finally:
            try:
                self.conn.close()
            except OSError:
                pass


def main():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", LISTEN_PORT))
    srv.listen(64)
    print(f"gate: listening :{LISTEN_PORT} -> {UP_HOST}:{UP_PORT} (health {HEALTH_TIMEOUT}s)", flush=True)
    while True:
        conn, _ = srv.accept()
        Handler(conn).start()


if __name__ == "__main__":
    main()
