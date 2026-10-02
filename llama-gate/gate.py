"""Gate TCP per llama-swap su blastpc (192.168.1.29:11435).

Perche' esiste: litellm puo' impostare un SOLO timeout per tutta la richiesta
(httpx non distingue connect/read tramite litellm_params). Serve però:
- PC SPENTO  -> fallire SUBITO (non aspettare il timeout lungo)
- PC ACCESO -> aspettare anche 10-20 min (caricamento/inferenza modelli enormi)

Soluzione: litellm punta qui (api_base http://llama-gate/v1) con timeout lungo.
Se il TCP/TLS verso llama-swap non si stabilisce entro HEALTH_TIMEOUT, la gate
risponde 503 in fretta (il PC e' spento/non raggiungibile). Se si stabilisce,
fa da tubo bidirezionale (HTTP e SSE passano invariati).

UPSTREAM TLS o HTTP - GATE_TLS=auto (default): la gate PROVA il TLS verso
l'upstream; se l'handshake fallisce perche' l'upstream parla HTTP in chiaro
(SSL: WRONG_VERSION_NUMBER / EOF), riparte su una connessione TCP NUOVA in
chiaro. Funziona quindi con llama-swap SU QUALSIASI schema:
  - https + certificato self-signed (CN/SAN = IP, senza CA): la gate termina
    il TLS e serve a litellm HTTP in chiaro - litellm NON deve fidarsi del
    certificato (verifica disattivata: endpoint LAN self-signed, non MITM);
  - http senza ssl (llama-swap SENZA ssl, dal 2026-10-02): tubo in chiaro.
GATE_TLS=1 forza il TLS (se il TLS non si stabilisce -> 503), GATE_TLS=0
forza il chiaro (nessun handshake). L'handshake TLS avviene DENTRO il budget
HEALTH_TIMEOUT, quindi un upstream che accetta il TCP ma non completa ne' TLS
ne' HTTP produce comunque il 503 veloce.

Solo stdlib: nessuna dipendenza da installare.
"""
import os
import socket
import ssl
import threading
import time

UP_HOST = os.environ.get("GATE_UP_HOST", "192.168.1.29")
UP_PORT = int(os.environ.get("GATE_UP_PORT", "11435"))
HEALTH_TIMEOUT = float(os.environ.get("GATE_HEALTH_TIMEOUT", "2"))
LISTEN_PORT = int(os.environ.get("GATE_PORT", "80"))

# auto (default): prova TLS, poi chiaro | 1: solo TLS | 0: solo chiaro
_TLS_RAW = os.environ.get("GATE_TLS", "auto").strip().lower()
if _TLS_RAW in ("1", "true", "yes", "tls", "https"):
    GATE_TLS = "1"
elif _TLS_RAW in ("0", "false", "no", "plain", "http"):
    GATE_TLS = "0"
else:
    GATE_TLS = "auto"

# Self-signed LAN: niente CA da validare, niente hostname da verificare.
_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE

# Il downgrade TLS->chiaro viene annunciato UNA SOLA volta (e' per-connessione:
# stamparlo ogni richiesta renderebbe il log inutilmente rumoroso).
_plain_announced = False

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

    @staticmethod
    def _connect(t0: float):
        """TCP + (se serve) TLS verso l'upstream, entro il budget di health.

        Ritorna (socket_pronto_per_il_tubo, modalita').
        """
        up = socket.create_connection((UP_HOST, UP_PORT), timeout=HEALTH_TIMEOUT)
        if GATE_TLS == "0":
            return up, "plain"
        try:
            return _SSL_CTX.wrap_socket(up, server_hostname=UP_HOST), "tls"
        except ssl.SSLError as e:
            # L'upstream NON e' TLS (llama-swap senza ssl = HTTP in chiaro).
            # Dopo un handshake fallito il socket e' in stato indefinito: si
            # riparte con una connessione TCP NUOVA. La richiesta del client
            # non si perde: resta nel buffer di kernel del socket in ingresso,
            # quindi il tubo riparte dal primo byte.
            up.close()
            if GATE_TLS == "1":
                raise  # TLS forzato: nessun downgrade -> 503
            global _plain_announced
            if not _plain_announced:
                _plain_announced = True
                print(f"gate: upstream {UP_HOST}:{UP_PORT} e' HTTP in chiaro "
                      f"(TLS handshake fallito: {e}) - tubo in chiaro", flush=True)
            remaining = max(HEALTH_TIMEOUT - (time.time() - t0), 0.5)
            return socket.create_connection((UP_HOST, UP_PORT),
                                            timeout=remaining), "plain"

    def run(self):
        try:
            t = time.time()
            up, _mode = self._connect(t)
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
    print(f"gate: listening :{LISTEN_PORT} -> {UP_HOST}:{UP_PORT}"
          f" (tls:{GATE_TLS}) (health {HEALTH_TIMEOUT}s)", flush=True)
    while True:
        conn, _ = srv.accept()
        Handler(conn).start()


if __name__ == "__main__":
    main()
