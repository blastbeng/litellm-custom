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

POLICY FALLBACK per-modello (richiesta utente, wildcard-safe):
le chiavi dei fallback in config.yaml NON supportano wildcard (litellm
get_fallback_model_group: solo esatto/stripped-provider/"*"), quindi la
policy e' applicata QUI per-request tramite litellm_params.fallbacks, che
OVERRIDE la lista di config (router.py: kwargs.get("fallbacks",
self.fallbacks)):
- synthetic/*            -> ["local-model", "ollama/local-model"]
  (stack: llama-swap locale PRIMO, poi il modello Ollama omonimo via
  wildcard dinamica ollama/*, disponibile solo quando il PC Windows e'
  online)
- local-model       -> ["ollama/local-model"] (SOLO il modello Ollama
  omonimo: small-model NON e' piu' fallback di nessuno - richiesta utente
  2026-10-01; resta un modello autonomo invocabile direttamente)
- openrouter/free, ollama/* -> [] (mai fallback, nemmeno su se stessi)
- tutto il resto (groq/*, gemini/*, ollama-cloud/*, openrouter/*,
  inference4free/* e QUALSIASI prefisso futuro) -> nessun override: vale la
  lista di config.yaml, che AL MOMENTO non ha catch-all (richiesta utente:
  nessun modello deve piu' cadere su openrouter/free; la catch-all
  "*": ["openrouter/free"] e' stata rimossa)

Il provider e' dedotto dal nome pubblico del modello (prefisso "groq/",
"gemini/"); si copre anche la forma nativa ("models/gemini-...") per
robustezza rispetto all'ordine hook/routing.

CLAMP DI max_tokens (contesto): il limite LO IMPONE L'UPSTREAM, non litellm.
llama-swap conta i token del prompt col tokenizer REALE e risponde 400
("prompt (N tokens) + max tokens (M) exceeds the context (C); requests are
never truncated" - il testo NON e' di litellm, che con
enable_pre_call_checks: false non fa alcun controllo), e litellm incapsula il
400 come BadRequestError: cosi' una richiesta con prompt grande uccide la
CATENA DI FALLBACK intera (synthetic/* -> local-model -> ollama/local-model,
tutti 131072). Qui max_tokens viene ridotto a "contesto - prompt" su
OGNI tentativo (l'hook gira anche per ogni target di fallback), con stima
CONSERVATIVA dei token del prompt (caratteri/2.8: sovrastimare e' sicuro,
sottostimare produce il 400 dell'upstream).

CASCATA OVERFLOW (richiesta utente 2026-10-02): quando il prompt da solo
satura il contesto (prompt + MIN_OUTPUT_TOKENS + margine > contesto) il
clamp NON puo' aiutare - non c'e' spazio per la risposta - e la catena di
fallback muore come nel log. In quel caso l'async hook riduce il PROMPT con
una cascata a livelli, e dopo ogni livello riusa la stima per il clamp:

  B  compattazione con un modello da 262k DISPONIBILE (i 131072 locali non
     possono fare la compattazione: il corpo da compattare ci sta giusto a
     malapena). Chiamata interna via llm_router, senza fallback e senza
     retry; si passa al compattore successivo se uno fallisce/tiempo scaduto
     o se il suo contesto non contiene il corpo.
  C  compressione DETERMINISTICA con litellm.compression.compress()
     (BM25: sostituisce i messaggi a bassa rilevanza con stub, protegge
     system/ultimo user/ultimo assistant, 0 chiamate LLM).
     NOTA: il tool di retrieval (litellm_content_retrieve) NON viene
     iniettato e la cache degli originali NON viene trattenuta: in questo
     stack non c'e' un agentic loop che lo serva, quindi lo stub e'
     PERSO (compressione lossy). E' il motivo per cui C viene DOPO B.
  E  map/reduce: il corpo viene tagliato in chunk e riassunto da
     small-model, poi le sintesi vengono ricomposte davanti alla richiesta
     corrente (che resta fedele). Ultima spiaggia perche' e' lossy e
     costa N chiamate sul stack locale gia' sotto pressione.
  F  se nessun livello libera spazio: 413 con dettaglio strutturato
     (modello, contesto, stima, livelli provati, hint) invece di far
     partire la richiesta e farla morire sulla catena di fallback.

La cascata gira SOLO nell'async hook (B ed E fanno chiamate LLM); il
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

from fastapi import HTTPException
from litellm.integrations.custom_logger import CustomLogger

GROQ_REASONING = ("openai/gpt-oss", "qwen/")

# "ollama" copre la wildcard dinamica ollama/* (modelli Ollama locali:
# mai fallback); "ollama-cloud" e' un provider cloud separato.
# local-model NON e' qui: ha il suo fallback (solo l'ollama omonimo),
# gestito dal ramo dedicato in _apply.
# small-model NON e' qui e non compare in nessuna catena di fallback: e'
# un modello autonomo, invocabile solo direttamente (richiesta utente).
NO_FALLBACK_MODELS = ("openrouter/free", "ollama")

# --- contesti (model_info.max_input_tokens di config.yaml) ---
# Serve la mappa QUI perche' i modelli sono wildcard/custom:
# litellm.get_max_tokens("local-model") fallisce ("isn't mapped yet") e il
# limite lo impone l'upstream. Valori ALLINEATI a config(.example).yaml:
# un contesto cambiato li' va cambiato anche qui.
CONTEXT_BY_PREFIX = (
    ("local-model", 131072),
    ("small-model", 131072),
    ("ollama/", 131072),
    ("synthetic/", 131072),
    ("inference4free/", 131072),
    ("ollama-cloud/", 262144),
    ("openrouter/", 262144),
    ("groq/", 262144),
    ("gemini/", 262144),
)
DEFAULT_CONTEXT = 131072  # il piu' piccolo: clamp conservativo per l'ignoto
MIN_OUTPUT_TOKENS = 1024  # sotto questo una risposta reasoning non e' utile
SAFETY_MARGIN = 1024      # margine per l'errore della stima
CHARS_PER_TOKEN = 2.8     # conservativo per llama/ollama (codice, CJK, JSON)
PER_MESSAGE_OVERHEAD = 8  # role/framing per messaggio (ChatML)
MAX_TOKEN_KEYS = ("max_tokens", "max_completion_tokens")

# --- cascata overflow: B -> C -> E -> F ---
# Compattatori: modelli con contesto >= 262144 (i 131072 locali non possono
# contenere corpo + sintesi). Sono modelli di ROUTING (wildcard), quindi
# possono non essere disponibili: la cascata li prova in ordine e passa
# oltre su errore/timeout.
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

# E: map/reduce sul modello locale autonomo (non e' fallback di nessuno,
# qui viene usato ESPLICITAMENTE come operaio di compattazione)
MAP_MODEL = "small-model"
MAP_OUTPUT_TOKENS = 1024
MAP_CHUNK_INPUT_TOKENS = 90000  # + output < 131072 di small-model
MAP_MAX_CHUNKS = 8
MAP_TIMEOUT_S = 180
MAP_ORIGIN = "context_map_reduce"

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


class ReasoningClamp(CustomLogger):

    def _context_for(self, model):
        for pref, ctx in CONTEXT_BY_PREFIX:
            if model == pref.rstrip("/") or model.startswith(pref):
                return ctx
        return DEFAULT_CONTEXT

    def _estimate_prompt_tokens(self, data):
        """Stima CONSERVATIVA (sovrastima): l'upstream conta col tokenizer
        reale, quindi sottostimare significa prendere il 400 dall'upstream."""
        msgs = data.get("messages") or []
        chars = 0
        for m in msgs:
            if not isinstance(m, dict):
                continue
            c = m.get("content")
            if isinstance(c, str):
                chars += len(c)
            elif isinstance(c, list):
                for part in c:
                    if isinstance(part, dict):
                        for k in ("text", "content"):
                            v = part.get(k)
                            if isinstance(v, str):
                                chars += len(v)
            for k in ("reasoning", "reasoning_content", "name"):
                v = m.get(k)
                if isinstance(v, str):
                    chars += len(v)
            if m.get("tool_calls"):
                chars += len(str(m["tool_calls"]))
        return int(chars / CHARS_PER_TOKEN) + PER_MESSAGE_OVERHEAD * len(msgs)

    def _saturated(self, est, ctx):
        """Il prompt da solo lascia meno di MIN_OUTPUT_TOKENS: il clamp non
        puo' aiutare, serve ridurre il PROMPT (cascata)."""
        return est + MIN_OUTPUT_TOKENS + SAFETY_MARGIN > ctx

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
        """max_tokens <= contesto - prompt, per OGNI tentativo (hook gira anche
        sui target di fallback). Nessun clamp se il client non passa un tetto:
        in quel caso llama-swap/Ollama auto-limitano il completamento."""
        keys = [k for k in MAX_TOKEN_KEYS
                if isinstance(data.get(k), (int, float)) and data[k] > 0]
        if not keys:
            return
        ctx = self._context_for(model)
        if est is None:
            est = self._estimate_prompt_tokens(data)
        budget = ctx - est - SAFETY_MARGIN
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
        """Livello B: sintetizza il corpo con un modello da 262k."""
        msgs = [m for m in (data.get("messages") or []) if isinstance(m, dict)]
        body_src, tail = self._split_tail(msgs)
        if not body_src:
            return None, "nothing-to-compact"
        body = self._serialize(body_src)
        if len(body) < COMPACT_MIN_CHARS:
            return None, "body-too-small"
        from litellm.proxy.proxy_server import llm_router
        failures = []
        for comp in COMPACTOR_MODELS:
            cctx = self._context_for(comp)
            body_est = int(len(body) / CHARS_PER_TOKEN)
            if self._saturated(body_est, cctx):
                # il compattore non contiene il corpo: inutile provarlo
                failures.append(f"{comp}:too-small")
                continue
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

    # ---------------- E: map/reduce sui modelli locali -----------------------

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
        """Livello E: riassunto a chunk su small-model + ricomposizione."""
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
        from litellm.proxy.proxy_server import llm_router
        summaries = []
        for i, chunk in enumerate(chunks):
            try:
                resp = await asyncio.wait_for(
                    llm_router.acompletion(
                        model=MAP_MODEL,
                        messages=[{"role": "system",
                                   "content": MAP_SYSTEM_PROMPT},
                                  {"role": "user", "content": chunk}],
                        max_tokens=MAP_OUTPUT_TOKENS,
                        num_retries=0,
                        fallbacks=[],
                        drop_params=True,
                        metadata={INTERNAL_CALL_ORIGIN_KEY: MAP_ORIGIN},
                    ),
                    timeout=MAP_TIMEOUT_S,
                )
                s = (resp.choices[0].message.content or "").strip()
            except Exception as e:
                return None, f"chunk{i}:{type(e).__name__}:{str(e)[:120]}"
            if not s:
                return None, f"chunk{i}:empty"
            summaries.append(f"--- parte {i + 1}/{len(chunks)} ---\n{s}")
        brief = "\n\n".join(summaries)
        print(f"[reasoning_clamp] E {model}: map/reduce su {MAP_MODEL} "
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
                         "cascata di compattazione (B compattatore 262k -> C "
                         "compressione BM25 -> E map/reduce) non ha liberato "
                         "spazio: riduci il contesto del client o usa un "
                         "modello con contesto maggiore"),
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
              f"{MIN_OUTPUT_TOKENS} tok > contesto {ctx} -> cascata B/C/E",
              flush=True)
        tried = []
        for name, fn in (("B", self._compact_with_provider),
                         ("C", self._compress_locally),
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
                continue
            data = new
            est = self._estimate_prompt_tokens(data)
            self._clamp_output(data, model, est)
            if not self._saturated(est, ctx):
                print(f"[reasoning_clamp] {model}: cascata risolta da {name} "
                      f"(~{est} tok, contesto {ctx})", flush=True)
                return data
        # F: nessun livello ha liberato spazio
        return self._overflow(data, model, est, tried)

    def _apply(self, data):
        model = str(data.get("model") or "")
        # --- policy fallback per-modello (override per-request) ---
        if model == "local-model":
            # fallback (richiesta utente 2026-10-01): SOLO ollama/local-model,
            # il modello Ollama omonimo. small-model e' stato RIMOSSO da ogni
            # catena di fallback (resta modello autonomo, solo su richiesta
            # diretta). PC spento -> la gate 503 in ~2s; il gruppo ollama/*
            # esiste solo col PC online
            data["fallbacks"] = ["ollama/local-model"]
        elif any(model == m or model.startswith(m + "/") for m in NO_FALLBACK_MODELS):
            data["fallbacks"] = []
        elif model.startswith("synthetic/"):
            # catena fallback (richiesta utente): local-model PRIMO,
            # poi il modello Ollama locale omonimo "ollama/local-model"
            # (dinamico via wildcard ollama/*, presente solo quando il PC
            # Windows e' online: se spento la ollama-gate risponde 503 in
            # ~2s e litellm prosegue/termina)
            data["fallbacks"] = ["local-model", "ollama/local-model"]
        else:
            # tutti gli altri: nessun override, vale la lista di config.yaml
            # (senza catch-all = NESSUN fallback; openrouter/free = modello
            # NATIVO OpenRouter "Free Models Router", non piu' usato come
            # fallback su richiesta dell'utente)
            data.pop("fallbacks", None)
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
