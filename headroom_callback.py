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
- comprime tool output JSON/codice (misurato: 56.5% su un tool output JSON
  ripetuto, ~9ms a regime). Il testo in prosa resta invariato: il compressore
  Kompress richiede il modello ML, non installato qui (niente download HF).

min_tokens=500 (default upstream): le richieste sotto soglia non pagano
nemmeno il costo di compress().
"""

from headroom.integrations.litellm_callback import HeadroomCallback

hr_compress = HeadroomCallback(min_tokens=500)
