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
- synthetic/*            -> ["fully-uncensored", "ollama/fully-uncensored"]
  (stack: llama-swap locale PRIMO, poi il modello Ollama omonimo via
  wildcard dinamica ollama/*, disponibile solo quando il PC Windows e'
  online)
- fully-uncensored       -> ["ollama/fully-uncensored"] (il fallback di
  fully-uncensored E' il modello Ollama omonimo, anche su richiesta diretta)
- openrouter/free, ollama/* -> [] (mai fallback, nemmeno su se stessi)
- tutto il resto (groq/*, gemini/*, ollama-cloud/*, openrouter/*,
  deepseek4free/* e QUALSIASI prefisso futuro) -> nessun override: vale la
  lista di config.yaml, che AL MOMENTO non ha catch-all (richiesta utente:
  nessun modello deve piu' cadere su openrouter/free; la catch-all
  "*": ["openrouter/free"] e' stata rimossa)

Il provider e' dedotto dal nome pubblico del modello (prefisso "groq/",
"gemini/"); si copre anche la forma nativa ("models/gemini-...") per
robustezza rispetto all'ordine hook/routing.

Registrato in config.yaml come: litellm_settings.callbacks -> "reasoning_clamp.rclamp"
(montato in /app/reasoning_clamp.py, vedi docker-compose.yml).
"""
from litellm.integrations.custom_logger import CustomLogger

GROQ_REASONING = ("openai/gpt-oss", "qwen/")

# "ollama" copre la wildcard dinamica ollama/* (modelli Ollama locali:
# mai fallback); "ollama-cloud" e' un provider cloud separato.
# fully-uncensored NON e' qui: ora ha il suo fallback (ollama/omonimo),
# gestito dal ramo dedicato in _apply.
NO_FALLBACK_MODELS = ("openrouter/free", "ollama")


class ReasoningClamp(CustomLogger):

    def _apply(self, data):
        model = str(data.get("model") or "")
        # --- policy fallback per-modello (override per-request) ---
        if model == "fully-uncensored":
            # stack (richiesta utente): fully-uncensored cade a sua volta su
            # ollama/fully-uncensored (PC spento -> llama-gate 503 in ~2s ->
            # fallback immediato; il gruppo esiste solo col PC online)
            data["fallbacks"] = ["ollama/fully-uncensored"]
        elif any(model == m or model.startswith(m + "/") for m in NO_FALLBACK_MODELS):
            data["fallbacks"] = []
        elif model.startswith("synthetic/"):
            # catena fallback (richiesta utente): fully-uncensored PRIMO,
            # poi il modello Ollama locale omonimo "ollama/fully-uncensored"
            # (dinamico via wildcard ollama/*, presente solo quando il PC
            # Windows e' online: se spento la ollama-gate risponde 503 in
            # ~2s e litellm prosegue/termina)
            data["fallbacks"] = ["fully-uncensored", "ollama/fully-uncensored"]
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
        return data

    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        return self._apply(data)

    def pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        return self._apply(data)


rclamp = ReasoningClamp()
