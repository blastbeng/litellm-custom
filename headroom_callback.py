"""Istanza del callback Headroom per il proxy litellm.

PERCHE' UN MODULO E NON IL PATH ALLA CLASSE
litellm 1.109 valida ogni voce di ``litellm_settings.callbacks``: un dotted
path che risolve a una CLASSE viene rifiutato all'avvio (verificato il
2026-11-14 sul container di test):

    ValueError: callbacks entry '...HeadroomCallback' resolved to the class
    ..., which is neither a CustomLogger instance nor a callable ... Point it
    at an instance instead

Quindi si espone l'ISTANZA (stessa forma di reasoning_clamp.rclamp) e la
config.yaml punta a ``headroom_callback.hr_compress``.

COMPORTAMENTO (headroom-ai 0.41.0, vedi Dockerfile.litellm):
- agisce SOLO su call_type completion/acompletion/anthropic_messages: media
  (immagini/audio), embeddings e modelli locali non vengono toccati;
- FAIL-OPEN: qualunque eccezione in compressione viene loggata come warning
  ("Headroom compression failed, using original messages") e la richiesta
  prosegue con i messaggi ORIGINALI. Nessun processo esterno, nessun hop di
  rete: headroom e' una libreria dentro questo processo.
- comprime tool output e messaggi utente strutturati (JSON/CSV tabellari); il
  testo in prosa, il codice e i dump di log restano invariati. Kompress (il
  compressore ML) NON e' installato: niente AI locale, niente torch/onnxruntime
  nell'immagine.

PERCHE' UNA SOTTOCLASSE E HeadroomUserClamp.async_pre_call_hook
HeadroomCallback non espone ``compress_user_messages`` (default upstream
False = "coding agent": i turni utente restano intatti). Qui si sovrascrive
la SOLA compressione locale per passare l'opzione a headroom.compress():
stesso percorso, stesso fail-open, in piu' i messaggi utente vengono
compattati. Misure sul contenuto utente (2026-11-14, payload sintetici):
- JSON 300 record: 3908->1144 token (-71%), tutte le 300 righe preservate in
  forma tabellare, valori identici (verificato riga per riga);
- log 402 righe: TENUTI INTATTI dal veto ProtectUserLogs (vedi sotto);
- prosa e codice: invariati (nessun compressore ML installato).
La compressione dei log e' lossy ("[N lines omitted]" + hash CCR che un client
chat non puo' risolvere): un dump incollato dall'utente verrebbe svuotato e la
domanda diventerebbe irrispondibile, quindi i messaggi ``user`` che sembrano
log non vengono toccati. I ``tool`` output restano comprimibili come prima.

Il metodo ``async_pre_call_hook`` qui sotto e' un override ESPLICITO anche se
identico all'upstream: litellm 1.109 (proxy/utils.py, ProxyLogging.pre_call_hook)
esegue l'hook SOLO se compare in ``vars(type(callback))``, cioe' se e'
definito nella classe concreta e non ereditato. Senza questo override la
sottoclasse viene saltata in silenzio e il proxy non comprime NULLA
(verificato: 6256 prompt token con e senza la compressione attesa).
"""

import re

from headroom.hooks import CompressionHooks
from headroom.integrations.litellm_callback import HeadroomCallback

# Riga di log: timestamp ISO/epoch-ish in testa, oppure prefisso syslog
# ("Nov 14 17:19:29"), oppure livello subito dopo un prefisso arbitrario.
_LOG_LINE_RE = re.compile(
    r"^\s*(?:\[?\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}"
    r"|\[?(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}"
    r"|\d{2}:\d{2}:\d{2}[.,]\d{3}"
    r"|\S{0,80}?\s(?i:INFO|WARN|WARNING|ERROR|DEBUG|TRACE|FATAL|CRITICAL)\b)"
)

# Sotto questi numeri la compressione dei log non e' mai stata aggressiva
# (headroom lascia intatte le sezioni corte): la soglia evita falsi positivi
# su frasi che iniziano con un orario.
_LOG_MIN_LINES = 20
_LOG_MIN_CHARS = 2000
_LOG_LINE_RATIO = 0.5


def _text_of(message):
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") in (None, "text")
        )
    return ""


def _looks_like_log(message):
    """True se il messaggio e' un dump di log (riga per riga)."""
    text = _text_of(message)
    if len(text) < _LOG_MIN_CHARS:
        return False
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < _LOG_MIN_LINES:
        return False
    hits = sum(1 for ln in lines if _LOG_LINE_RE.match(ln))
    return hits / len(lines) >= _LOG_LINE_RATIO


class ProtectUserLogs(CompressionHooks):
    """Veto per i dump di log nei messaggi UTENTE.

    Il compressore dei log di headroom e' LOSSY: su 402 righe INFO identiche
    lascia al modello "[402 lines omitted: 400 INFO]" piu' un hash CCR che un
    client chat non puo' risolvere (misurato via proxy: prompt 8049 -> 546,
    risposta sbagliata e 28s di reasoning per un numero che con il log
    originale arrivava in 4.6s). Un messaggio incollato dall'utente e' il testo
    su cui l'utente fa domande: qui resta INTATTO.

    Solo i messaggi ``user``: i ``tool`` output continuano a passare dal
    compressore come prima (sono rumore macchina, non una domanda), e i
    payload strutturati dell'utente (JSON/CSV) restano comprimibili.
    """

    def protect_messages(self, messages, ctx):
        return {i for i, msg in enumerate(messages)
                if isinstance(msg, dict) and msg.get("role") == "user" and _looks_like_log(msg)}


class HeadroomUserClamp(HeadroomCallback):
    """HeadroomCallback + compressione dei messaggi utente.

    Override di ``_local_compress`` (unico punto in cui il callback upstream
    chiama headroom.compress()) per aggiungere ``compress_user_messages=True``.
    Tutto il resto - gating su call_type, try/except fail-open, log - resta
    quello di headroom: se compress() solleva, la richiesta passa originale.
    """

    async def async_pre_call_hook(
        self,
        user_api_key_dict=None,
        cache=None,
        data=None,
        call_type="",
        *args,
        **kwargs,
    ):
        # Override richiesto da litellm (vedi docstring del modulo): senza
        # questo metodo la classe non entra nel giro degli hook.
        return await super().async_pre_call_hook(
            user_api_key_dict, cache, data, call_type, *args, **kwargs
        )

    def _local_compress(self, messages, model):
        from headroom.compress import compress

        result = compress(
            messages=messages,
            model=model or "claude-sonnet-4-5-20250929",
            model_limit=self._model_limit,
            hooks=self._hooks,
            compress_user_messages=True,
            # Nessun compressore ML: senza transformers/onnxruntime installati
            # ogni tentativo finiva in "Compression failed, returning original
            # messages: No module named 'transformers'" (warning + percorso di
            # eccezione a ogni richiesta con testo). "disabled" rende esplicito
            # che qui girano solo le trasformazioni deterministiche (tabellari,
            # log, JSON), senza AI locale.
            kompress_model="disabled",
        )
        return {
            "messages": result.messages,
            "tokens_before": result.tokens_before,
            "tokens_after": result.tokens_after,
            "tokens_saved": result.tokens_saved,
            "compression_ratio": result.compression_ratio,
        }


# min_tokens=500: default upstream. In headroom-ai 0.41.0 il valore e' solo
# memorizzato (nessun gate): la soglia reale per-messaggio e'
# CompressConfig.min_tokens_to_compress=250 token.
# hooks=ProtectUserLogs(): i log incollati dall'utente restano verbatim.
hr_compress = HeadroomUserClamp(min_tokens=500, hooks=ProtectUserLogs())
