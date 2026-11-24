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
  testo in prosa e il codice restano invariati. I dump di log dei tool output
  SONO compressi, e in modo LOSSY (vedi "TRE STRATI" sotto). Kompress (il
  compressore ML) NON e' installato: niente AI locale, niente
  torch/onnxruntime nell'immagine.

PERCHE' UNA SOTTOCLASSE E HeadroomUserClamp.async_pre_call_hook
HeadroomCallback non espone ``compress_user_messages`` (default upstream
False = "coding agent": i turni utente restano intatti). Qui si sovrascrive
la SOLA compressione locale per passare l'opzione a headroom.compress():
stesso percorso, stesso fail-open, in piu' i messaggi utente vengono
compattati. Misure sul contenuto utente (2026-11-14, payload sintetici):
- JSON 300 record: 3908->1144 token (-71%), tutte le 300 righe preservate in
  forma tabellare, valori identici (verificato riga per riga);
- log 402 righe: TENUTI INTATTI dal veto ProtectRecentAndLogs (vedi sotto);
- prosa e codice: invariati (nessun compressore ML installato).
La compressione dei log e' lossy ("[N lines omitted]" + hash CCR che un client
chat non puo' risolvere): un dump incollato dall'utente verrebbe svuotato e la
domanda diventerebbe irrispondibile, quindi i messaggi ``user`` che sembrano
log non vengono toccati. I ``tool`` output restano comprimibili come prima.

TRE STRATI CONTRO LA PERDITA DEI TOOL OUTPUT (2026-11-19)
Il compressore dei log/search/diff di headroom e' LOSSY: su un log da 400 righe
lascia al modello "[400 lines omitted: ...]" piu' un hash CCR che un client chat
non puo' riscattare (headroom_retrieve non e' esposto: lo store CCR vive in
questo processo). In un loop agentico come AiderDesk il tool output fresco
diventa quindi inutilizzabile: l'agente non lo trova, riscrive l'output su file e
lo rilegge -> turni sprecati che costano piu' dei token risparmiati.

La compressione resta quella di prima (e' da li' che viene il risparmio: -93% su
un log da 400 righe, -62% su 300 record JSON), con tre protezioni sopra:

1. FINESTRA RECENTE (``ProtectRecentAndLogs.protect_messages``): gli ultimi
   ``HR_PROTECT_LAST`` messaggi (default 6) non vengono toccati, qualunque sia
   il content type. Cosi' il tool output che l'agente sta leggendo ORA e' sempre
   integro. Costo misurato (2026-11-19, payload sintetici): su 20 messaggi/49k
   token il risparmio passa da -64% a -54%; su 50 messaggi/116k token da -76% a
   -72%. Il costo e' O(finestra), non O(contenuto): piu' la sessione e' lunga,
   meno incide. ``HR_PROTECT_LAST=0`` disabilita lo strato.
2. TOOL DI LETTURA ESCLUSI dal path lossy (``_extend_excluded_tools``):
   headroom esclude gia' ``Read``/``read_file``/``view``/``Grep`` per nome
   ESATTO, ma AiderDesk manda i nomi qualificati (``power---file_read``,
   ``power---grep``, ...) che non matchano -> i file letti finivano nel
   compressore lossy. Si estendono ``DEFAULT_EXCLUDE_TOOLS`` e
   ``DEFAULT_VERBATIM_EXCLUDE_TOOLS`` con i pattern glob dei tool di lettura:
   il loro output resta byte-exact (headroom applica al massimo la compattazione
   lossless), che e' il requisito per gli edit old_string. Override con
   ``HR_EXCLUDE_TOOLS="pat1,pat2"``.
3. VETO SUI LOG INCOLLATI DALL'UTENTE (``_looks_like_log``): invariato.

Il resto della storia (log/diff/search piu' vecchi della finestra) resta lossy:
se l'agente torna su un output di 4+ turni prima, quel contenuto non c'e' piu'.
Se serve piu' margine: ``HR_PROTECT_LAST=10`` costa ~10 punti di risparmio in
piu'.

QUARTO STRATO: REGIME PER RICHIESTA (2026-11-24)
La finestra da sola non bastava. Riprodotto in container su un trascritto
agentico di 12 turni (tool output = dump di log, 34.993 token): il regime
upstream riduceva a 3.742 token (-89%) sostituendo 11 tool output su 12 con
``[120 lines omitted ... hash=...]``, e i tre strati sopra coprivano solo gli
ultimi 6 messaggi (~1,5 turni). Un agente che torna su un output di due turni
prima trova un hash che nessun client puo' riscattare: e' cosi' che nasce
"my tool-output channel is broken in this session" (l'agente riscrive l'output
su file e lo rilegge: turni sprecati che costano piu' dei token risparmiati).

Ora il regime e' scelto PER RICHIESTA (``_mode_for``):
- **agentico** (body con ``tools``/``functions``, oppure storia con ruolo
  ``tool``/``function``, ``tool_calls``, ``function_call``, o ``input[]`` con
  ``function_call_output``) -> ``mode="agent"`` di headroom (lossless
  verificato: nessun marcatore, nessuna rimozione). Misurato in questo stack:
  log/codice/diff restano intatti, quindi il regime lossless costa ~0; headroom
  comprimeva solo dump di log (-98%) e array JSON (-26%), cioe' proprio i
  contenuti su cui l'agente costruisce gli edit.
- **non agentico** (chat, log analysis, RAG, traduzione) -> regime lossy di
  prima, con ``compress_user_messages=True``: li' il risparmio e' reale
  (-93% su un log, -26% su JSON) e un marcatore non rompe nessun loop.
- ``off`` -> nessuna compressione.

Override, in ordine di precedenza: header ``x-headroom-mode:
off|lossless|lossy`` (per singolo client), ``HR_MODE`` (forzatura globale),
``HR_AGENT_MODE`` (default ``lossless``) per le richieste agentiche. Ogni
richiesta lascia una riga ``[headroom] <mode> <model>: N -> M tok`` nel log:
prima non c'era NESSUNA riga headroom, quindi non si poteva distinguere "non
comprime" da "sta fallendo in silenzio".

Il metodo ``async_pre_call_hook`` qui sotto e' un override ESPLICITO anche se
identico all'upstream: litellm 1.109 (proxy/utils.py, ProxyLogging.pre_call_hook)
esegue l'hook SOLO se compare in ``vars(type(callback))``, cioe' se e'
definito nella classe concreta e non ereditato. Senza questo override la
sottoclasse viene saltata in silenzio e il proxy non comprime NULLA
(verificato: 6256 prompt token con e senza la compressione attesa).
"""

import contextvars
import os
import re

from headroom.config import DEFAULT_EXCLUDE_TOOLS, DEFAULT_VERBATIM_EXCLUDE_TOOLS
from headroom.hooks import CompressionHooks
from headroom.integrations.litellm_callback import HeadroomCallback
from headroom.transforms import content_router

# Regime scelto per la RICHIESTA in corso. Il callback e' un singleton
# condiviso da tutte le richieste concorrenti: un attributo su ``self``
# verrebbe sovrascritto da un'altra richiesta nel mezzo di un await, mentre
# il ContextVar segue il task asyncio (stessa tecnica di contextvars nei
# server ASGI). Il default None = "lossy" (comportamento upstream).
_MODE = contextvars.ContextVar("headroom_mode", default=None)

# "off"      nessuna compressione (passa tutto originale)
# "lossless" headroom mode="agent": densificazione deterministica verificata,
#            nessun marcatore, nessuna rimozione (vedi mode_for sotto)
# "lossy"    pipeline upstream (log/JSON -> stub CCR, non recuperabili)
_MODES = ("off", "lossless", "lossy")

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

# Tool di LETTURA di AiderDesk (nomi qualificati dal gruppo: "power---file_read",
# "gateway---..."). Glob perche' is_tool_excluded() usa fnmatch sui pattern.
_READ_TOOL_PATTERNS = (
    "*file_read*",
    "*read_file*",
    "*file_view*",
    "*grep*",
    "*glob*",
    "*list_dir*",
)

# Quanti messaggi finali restano intoccati (strato 1, vedi docstring modulo).
_DEFAULT_PROTECT_LAST = 6


def _protect_last():
    raw = os.environ.get("HR_PROTECT_LAST", "").strip()
    if not raw:
        return _DEFAULT_PROTECT_LAST
    try:
        return max(0, int(raw))
    except ValueError:
        return _DEFAULT_PROTECT_LAST


def _headers_of(data):
    """Header HTTP della richiesta, minuscoli, dal posto in cui litellm li
    mette: ``proxy_server_request``/``metadata``/``litellm_metadata`` (stesso
    percorso usato da reasoning_clamp per ``Cache-Control``). Vuoto se il
    chiamante non e' il proxy (es. uso della libreria)."""
    if not isinstance(data, dict):
        return {}
    for src in ("proxy_server_request", "metadata", "litellm_metadata"):
        box = data.get(src)
        hdrs = box.get("headers") if isinstance(box, dict) else None
        if isinstance(hdrs, dict) and hdrs:
            return {str(k).lower(): v for k, v in hdrs.items()}
    return {}


def _is_agentic(data):
    """True quando la richiesta e' un turno di un loop agentico (tool calling).

    Due segnali, entrambi necessari:
    - ``tools``/``functions`` non vuoti: il client ha dichiarato dei tool,
      quindi il modello puo' emettere tool_calls e il prossimo turno avra'
      bisogno del RISULTATO INTEGRO (un hash CCR non e' riscattabile);
    - un messaggio ``role: tool``/``function`` o con ``tool_calls`` dentro la
      storia: il loop e' gia' in corso (vale anche per il body Responses,
      dove i risultati arrivano come ``input[].type ==
      "function_call_output"``).

    Una chat normale (log analysis, RAG, traduzione) non ha nessuno dei due:
    li' la compressione lossy resta attiva e resta conveniente.
    """
    if not isinstance(data, dict):
        return False
    if data.get("tools") or data.get("functions"):
        return True
    for msg in data.get("messages") or []:
        if not isinstance(msg, dict):
            continue
        if (msg.get("role") in ("tool", "function")
                or msg.get("tool_calls") or msg.get("function_call")):
            return True
    for item in data.get("input") or []:
        if isinstance(item, dict) and item.get("type") in (
            "function_call", "function_call_output"):
            return True
    return False


def _mode_for(data):
    """Regime di compressione della richiesta: off | lossless | lossy.

    Precedenza:
    1. header esplicito ``x-headroom-mode`` (vale per un singolo client:
       AiderDesk o un'app possono scegliere senza toccare il proxy);
    2. ``HR_MODE`` (forzatura globale dell'operatore);
    3. ``HR_AGENT_MODE`` (default ``lossless``) quando la richiesta e'
       agentica, ``lossy`` altrimenti.
    """
    hdr = _headers_of(data).get("x-headroom-mode")
    if isinstance(hdr, str) and hdr.strip().lower() in _MODES:
        return hdr.strip().lower()
    forced = os.environ.get("HR_MODE", "").strip().lower()
    if forced in _MODES:
        return forced
    if _is_agentic(data):
        agent = os.environ.get("HR_AGENT_MODE", "lossless").strip().lower()
        return agent if agent in _MODES else "lossless"
    return "lossy"


def _extend_excluded_tools():
    """Aggiunge i tool di lettura alla blacklist lossy di headroom.

    ``DEFAULT_EXCLUDE_TOOLS`` (lossy) e ``DEFAULT_VERBATIM_EXCLUDE_TOOLS``
    (fold/byte-exact) sono risolti DENTRO ``ContentRouter.apply()`` leggendo il
    global del modulo ``headroom.transforms.content_router``, quindi basta
    riassegnare l'attributo: ``compress()`` non accetta ``exclude_tools`` e la
    pipeline di libreria non ha altre vie per configurarla.

    I due set sono necessari entrambi: il primo impedisce la compressione lossy
    (log/search/JSON), il secondo anche il fold "lossless" (grep -> heading,
    JSON minify), che cambierebbe i byte su cui l'agente costruisce gli edit
    old_string (stessa motivazione di headroom per ``view``/``read``).

    Fail-open: se l'attributo non esiste piu' in una versione futura si resta
    al comportamento upstream (nomi esatti) senza far fallire il proxy.
    """
    raw = os.environ.get("HR_EXCLUDE_TOOLS", "").strip()
    patterns = (
        frozenset(p.strip() for p in raw.split(",") if p.strip())
        if raw
        else frozenset(_READ_TOOL_PATTERNS)
    )
    if not patterns:
        return
    try:
        content_router.DEFAULT_EXCLUDE_TOOLS = DEFAULT_EXCLUDE_TOOLS | patterns
        content_router.DEFAULT_VERBATIM_EXCLUDE_TOOLS = (
            DEFAULT_VERBATIM_EXCLUDE_TOOLS | patterns
        )
    except Exception:
        pass


def _force_estimation_without_transformers():
    """Usa la stima dei token quando transformers non e' installato.

    headroom 0.41.0 importa transformers FUORI dal try in
    ``tokenizers/huggingface.py::_load_tokenizer``. Per i modelli che mappano
    al backend HuggingFace (qwen, deepseek, llama, ...) l'ImportError risale
    fino a ``compress()`` e la richiesta NON viene compressa affatto:

        Compression failed, returning original messages: No module named
        'transformers'

    (verificato il 2026-11-14: DeepSeek/Qwen/llama -> before=0 after=0, mentre
    un modello che usa la stima, es. synthetic/syn:large:text, comprime
    3209 -> 1154 token). Il backend HF prevede gia' il fallback alla stima
    quando il tokenizer non si carica: qui lo si registra a monte via API
    pubblica ``TokenizerRegistry.register_backend``, senza toccare il pacchetto
    ne' installare torch/transformers (nessuna AI locale).

    Silenzioso e non-fatale: se l'API interna cambia in un aggiornamento di
    headroom, il proxy parte comunque (si torna al comportamento upstream).
    """
    try:
        from headroom.tokenizers import huggingface
        from headroom.tokenizers.estimator import EstimatingTokenCounter
        from headroom.tokenizers.registry import TokenizerRegistry

        if huggingface.HuggingFaceTokenizer.is_available():
            return
        TokenizerRegistry.register_backend(
            "huggingface", lambda model: EstimatingTokenCounter()
        )
    except Exception:
        pass


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


class ProtectRecentAndLogs(CompressionHooks):
    """Veto su (1) la finestra recente e (2) i dump di log incollati dall'utente.

    ``CompressionHooks.protect_messages`` e' un veto DURO: i messaggi restituiti
    passano intatti, qualunque sia il content type o il ruolo.

    Finestra recente
        Gli ultimi ``protect_last`` messaggi restano integri. E' lo strato che
        risolve il sintomo di AiderDesk: il tool output che l'agente sta leggendo
        adesso e' sempre byte-exact, quindi non deve piu' riscriverlo su file
        per rileggerlo. Il costo e' O(finestra), non O(contenuto): su una
        sessione lunga pesa sempre meno (misurato: 20 msg/49k token -64% ->
        -54%, 50 msg/116k token -76% -> -72%). ``protect_last=0`` disabilita lo
        strato e torna al comportamento a sola compressione.

    Log incollati dall'utente
        Il compressore dei log di headroom e' LOSSY: su 402 righe INFO identiche
        lascia "[402 lines omitted: 400 INFO]" piu' un hash CCR che un client
        chat non puo' risolvere (misurato via proxy: prompt 8049 -> 546,
        risposta sbagliata e 28s di reasoning per un numero che con il log
        originale arrivava in 4.6s). Un dump incollato dall'utente e' il testo
        su cui l'utente fa domande: qui resta INTATTO.

    I ``tool`` output fuori finestra continuano a passare dal compressore come
    prima (sono rumore macchina, non una domanda) e i payload strutturati
    dell'utente (JSON/CSV) restano comprimibili.
    """

    def __init__(self, protect_last=_DEFAULT_PROTECT_LAST):
        self._protect_last = max(0, int(protect_last))

    def protect_messages(self, messages, ctx):
        protected = set()
        if self._protect_last and messages:
            protected.update(
                range(max(0, len(messages) - self._protect_last), len(messages))
            )
        protected.update(
            i for i, msg in enumerate(messages)
            if isinstance(msg, dict)
            and msg.get("role") == "user"
            and _looks_like_log(msg)
        )
        return protected


class HeadroomUserClamp(HeadroomCallback):
    """HeadroomCallback con regime di compressione PER RICHIESTA.

    Override di ``_local_compress`` (unico punto in cui il callback upstream
    chiama headroom.compress()) per:
    - scegliere il regime (``_mode_for``: lossless per i loop agentici,
      lossy per tutto il resto, ``off`` su richiesta esplicita);
    - aggiungere ``compress_user_messages=True`` nel solo regime lossy;
    - lasciare una riga di log per richiesta, cosi' il regime scelto e' visibile
      (prima non c'era NESSUNA riga headroom nei log: impossibile capire se
      stesse comprimendo o fallendo in silenzio).

    Tutto il resto - gating su call_type, try/except fail-open - resta quello
    di headroom: se compress() solleva, la richiesta passa originale.
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
        mode = _mode_for(data) if isinstance(data, dict) else None
        token = _MODE.set(mode)
        try:
            return await super().async_pre_call_hook(
                user_api_key_dict, cache, data, call_type, *args, **kwargs
            )
        finally:
            # il ContextVar non deve sopravvivere alla richiesta: il task
            # asyncio puo' essere riusato per il turno successivo
            _MODE.reset(token)

    def _local_compress(self, messages, model):
        from headroom.compress import compress

        mode = _MODE.get() or "lossy"
        if mode == "off":
            # Nessuna compressione: si restituisce la lista originale e
            # tokens_saved=0, cosi' l'upstream non sostituisce nulla.
            return {
                "messages": messages,
                "tokens_before": 0,
                "tokens_after": 0,
                "tokens_saved": 0,
                "compression_ratio": 0.0,
            }

        # lossless = mode="agent" di headroom: densificazione deterministica
        # con verify_lossless forzato, nessun marcatore CCR e nessuna rimozione.
        # Misurato in questo stack (2026-11-24, trascritto agentico di 12 turni):
        #   lossy    34.993 -> 3.742 tok (11 tool output su 12 ridotti a
        #            "[120 lines omitted ... hash=...]", non riscattabile);
        #   lossless 34.993 -> 34.993 tok, nessun messaggio riscritto.
        # Il costo del regime lossless e' quindi ~0 su log/codice/diff: qui
        # headroom comprime SOLO dump di log (-98%) e array JSON (-26%), che
        # sono esattamente i contenuti su cui un agente costruisce gli edit.
        lossless = mode == "lossless"
        kwargs = {"mode": "agent"} if lossless else {}
        result = compress(
            messages=messages,
            model=model or "claude-sonnet-4-5-20250929",
            model_limit=self._model_limit,
            hooks=self._hooks,
            # Nel regime lossy i messaggi utente strutturati (JSON/CSV) si
            # comprimono: e' li' che sta il risparmio per log analysis e RAG.
            # Nel regime lossless l'opzione e' irrilevante (niente rimozioni) e
            # resta False per esplicitezza.
            compress_user_messages=not lossless,
            # Nessun compressore ML: senza transformers/onnxruntime installati
            # ogni tentativo finiva in "Compression failed, returning original
            # messages: No module named 'transformers'" (warning + percorso di
            # eccezione a ogni richiesta con testo). "disabled" rende esplicito
            # che qui girano solo le trasformazioni deterministiche (tabellari,
            # log, JSON), senza AI locale.
            kompress_model="disabled",
            **kwargs,
        )
        print(f"[headroom] {mode} {model or '?'}: "
              f"{result.tokens_before} -> {result.tokens_after} tok "
              f"(saved {result.tokens_saved})", flush=True)
        return {
            "messages": result.messages,
            "tokens_before": result.tokens_before,
            "tokens_after": result.tokens_after,
            "tokens_saved": result.tokens_saved,
            "compression_ratio": result.compression_ratio,
        }


_force_estimation_without_transformers()
# Strati 1+2 (vedi docstring modulo): tool di lettura fuori dal path lossy.
_extend_excluded_tools()

# min_tokens=500: default upstream. In headroom-ai 0.41.0 il valore e' solo
# memorizzato (nessun gate): la soglia reale per-messaggio e'
# CompressConfig.min_tokens_to_compress=250 token.
# hooks=ProtectRecentAndLogs(): ultimi HR_PROTECT_LAST messaggi + log utente
# restano verbatim.
hr_compress = HeadroomUserClamp(
    min_tokens=500, hooks=ProtectRecentAndLogs(_protect_last())
)
