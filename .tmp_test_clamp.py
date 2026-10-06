"""Check OFFLINE dell'ordine a TIER DI COSTO del pool cascata (richiesta
utente 2026-10-06: prima i gratis, poi gli economici, per ultimi i costosi;
classificazione dinamica perche' i provider delistano/listano modelli).
File TEMPORANEO: rimosso prima del commit (igiene repo, come le volte prima).
"""
import sys
import time
import types

# --- stub di litellm/fastapi (il file gira nel container, non sull'host) ---
lit = types.ModuleType("litellm")
integ = types.ModuleType("litellm.integrations")
cl = types.ModuleType("litellm.integrations.custom_logger")


class CustomLogger:
    pass


cl.CustomLogger = CustomLogger
integ.custom_logger = cl
lit.integrations = integ
proxy = types.ModuleType("litellm.proxy")
ps = types.ModuleType("litellm.proxy.proxy_server")
ps.llm_router = None
proxy.proxy_server = ps
lit.proxy = proxy
fa = types.ModuleType("fastapi")
fae = types.ModuleType("fastapi.exceptions")


class HTTPException(Exception):
    def __init__(self, status_code=None, detail=None):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


fae.HTTPException = HTTPException
fa.exceptions = fae
sys.modules.update({
    "litellm": lit, "litellm.integrations": integ,
    "litellm.integrations.custom_logger": cl,
    "litellm.proxy": proxy, "litellm.proxy.proxy_server": ps,
    "fastapi": fa, "fastapi.exceptions": fae,
})

import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location("rc", "reasoning_clamp.py")
rc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rc)

FAIL = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        FAIL.append(name)


# ---------- 1. matrice _model_tier ----------
cases = [
    ("inference4free/meta-llama/llama-3.3-70b-instruct", rc.TIER_FREE),
    ("openrouter/deepseek/deepseek-chat-v3:free", rc.TIER_FREE),
    ("openrouter/free", rc.TIER_FREE),
    ("small-model", rc.TIER_FREE),
    ("local-model", rc.TIER_FREE),
    ("groq/openai/gpt-oss-120b", rc.TIER_CHEAP),
    ("ollama-cloud/gpt-oss:20b", rc.TIER_CHEAP),
    ("gemini/models/gemini-2.5-flash-lite", rc.TIER_CHEAP),
    ("gemini/models/gemma-3-27b-it", rc.TIER_CHEAP),
    ("synthetic/syn:small:text", rc.TIER_CHEAP),
    ("synthetic/syn:small:vision", rc.TIER_CHEAP),
    ("groq/qwen/qwen3.8-27b", rc.TIER_CHEAP),
    ("gemini/models/gemini-2.5-pro", rc.TIER_PAID),
    ("gemini/models/gemini-2.5-flash", rc.TIER_PAID),
    ("synthetic/syn:large:text", rc.TIER_PAID),
    ("openrouter/anthropic/claude-sonnet-4", rc.TIER_PAID),
    ("openrouter/openai/gpt-5", rc.TIER_PAID),
]
for i, (m, want) in enumerate(cases):
    check(f"tier{i:02d}:{m}=={want}", rc._model_tier(m) == want)

# hint DINAMICO dal prezzo del listing del gateway (nuovo modello gratis
# listato dal provider -> tier FREE senza toccare il codice)
rc._TIER_HINTS["openrouter/newco/model-x"] = rc.TIER_FREE
check("tier-hint-listing-prezzo-0",
      rc._model_tier("openrouter/newco/model-x") == rc.TIER_FREE)
rc._TIER_HINTS.pop("openrouter/newco/model-x")
check("tier-senza-hint-torna-paid",
      rc._model_tier("openrouter/newco/model-x") == rc.TIER_PAID)


# ---------- 2. ordine del pool: tier PRIMARIO, preferenza secondaria ----------
class FakeRouter:
    def __init__(self, names):
        self.model_list = [{"model_name": n} for n in names]


NAMES = [
    "small-model",                              # free (hardware locale)
    "inference4free/meta-llama/llama-3.3-70b",  # free (cloud)
    "groq/openai/gpt-oss-120b",                 # cheap
    "synthetic/syn:small:text",                 # cheap
    "gemini/models/gemini-2.5-pro",             # paid
    "openrouter/anthropic/claude-sonnet-4",     # paid
    "embedding-model",                          # NON-chat -> escluso
]
ps.llm_router = FakeRouter(NAMES)
rcm = rc.ReasoningClamp()

pool = rcm._worker_pool(5000, rc.SUMMARIZER_PREFERRED)
want = [
    "small-model",                              # tier0, pref idx 0
    "inference4free/meta-llama/llama-3.3-70b",  # tier0, pref idx 5
    "synthetic/syn:small:text",                 # tier1, pref idx 3
    "groq/openai/gpt-oss-120b",                 # tier1, pref idx 6
    "gemini/models/gemini-2.5-pro",             # tier2, pref "gemini/"
    "openrouter/anthropic/claude-sonnet-4",     # tier2, pref "openrouter/"
]
check("pool-ordine-tier-prima-gratis-poi-economici-poi-costosi", pool == want)
check("pool-esclude-non-chat", all("embed" not in n for n in pool))

# B con corpo 200k: solo i gemini (262k) contengono il corpo; il tier mette
# flash-lite (cheap) davanti a pro (paid)
ps.llm_router = FakeRouter(NAMES + ["gemini/models/gemini-2.5-flash-lite"])
poolB = rcm._worker_pool(200000, rc.COMPACTOR_MODELS,
                         output_tokens=rc.COMPACT_OUTPUT_TOKENS)
check("pool-B-big-flashlite-prima-di-pro",
      poolB == ["gemini/models/gemini-2.5-flash-lite",
                "gemini/models/gemini-2.5-pro"])

# ---------- 3. wildcard espansi rispettano il tier ----------
ps.llm_router = FakeRouter(["small-model", "groq/openai/gpt-oss-120b",
                            "openrouter/*"])
rc._WILDCARD_CACHE["openrouter/*"] = (
    time.monotonic(),
    ["openrouter/anthropic/claude-sonnet-4",
     "openrouter/deepseek/deepseek-chat-v3:free"])
poolW = rcm._worker_pool(5000, rc.SUMMARIZER_PREFERRED)
check("wildcard-free-id-in-tier0",
      poolW[:3] == ["small-model",
                    "openrouter/deepseek/deepseek-chat-v3:free",
                    "groq/openai/gpt-oss-120b"]
      and "openrouter/anthropic/claude-sonnet-4" in poolW[-2:])
rc._WILDCARD_CACHE.pop("openrouter/*")

# ---------- 4. pool vuoto se nessuno contiene il corpo ----------
ps.llm_router = FakeRouter(["small-model"])
check("pool-vuoto-se-nessuno-contiene",
      rcm._worker_pool(300000, rc.SUMMARIZER_PREFERRED) == [])

print()
print(f"CHECKS: {18 + 5}, FAILURES: {FAIL if FAIL else 'NONE'}")
sys.exit(1 if FAIL else 0)
