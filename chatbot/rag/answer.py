"""Answer generation with strict source citation (after VaidyaBot), for the Navez letters."""
import re

from .openrouter import chat
from .search import Searcher

ANSWER_MODEL = "anthropic/claude-sonnet-4.5"

SOURCE_LABEL = {"corrected_htr": "gecorrigeerde HTR", "manual": "manuele transcriptie",
                "opus_htr": "ongecorrigeerde HTR"}

SYSTEM = """Je bent een onderzoeksassistent voor de briefwisseling van de Belgische schilder \
François-Joseph Navez (1787-1869), bewaard in de KBR (Ms. II 70). De brieven zijn grotendeels \
in 19de-eeuws Frans geschreven.

STRIKTE REGELS:
1. Baseer je antwoord UITSLUITEND op de meegeleverde brieffragmenten. Gebruik geen externe \
kennis over Navez of zijn kring, ook niet als je het antwoord denkt te weten.
2. Citeer bij ELKE bewering de bron als [#n] (het fragmentnummer). Meerdere bronnen: [#1][#3].
3. Vermeld waar relevant wie aan wie schreef en wanneer. Citeer korte passages in het Frans \
en geef er een vertaling bij.
4. Als de fragmenten het antwoord niet bevatten, zeg dan expliciet: "Dit staat niet in de \
gevonden brieven." Noem eventueel welke fragmenten zijdelings relevant zijn.
5. Je ziet alleen een selectie van fragmenten, niet de hele collectie. Doe dus geen uitspraken \
over aantallen of volledigheid ("Navez schreef nooit...", "dit is de enige brief...").
6. Transcripties gemarkeerd als "ongecorrigeerde HTR" kunnen leesfouten bevatten, vooral in \
namen. Gekoppelde personen zijn suggesties van een automatisch systeem: noem ze voorzichtig.
7. Antwoord in de taal van de vraag, helder en beknopt."""


def _context_block(i: int, r: dict) -> str:
    who = f"{r['sender'] or 'onbekende afzender'} aan {r['recipient'] or 'onbekende ontvanger'}"
    head = f"[#{i}] {r['letter_id']} — {who}, {r['date'] or 'datum onbekend'} ({SOURCE_LABEL.get(r['text_source'], r['text_source'])})"
    ents = []
    if r["persons_linked"]:
        ents.append("Gekoppelde personen (suggestie): " + ", ".join(r["persons_linked"]))
    if r["places"]:
        ents.append("Plaatsen: " + ", ".join(r["places"]))
    return "\n".join([head, *ents, f"Tekst: {r['text']}"])


def answer(query: str, searcher: Searcher, k: int = 8, model: str = ANSWER_MODEL, **filters) -> dict:
    sources = searcher.search(query, k=k, **filters)
    if not sources:
        return {"answer": "Geen relevante brieven gevonden met deze filters.", "sources": [], "cost": 0.0}
    context = "\n\n".join(_context_block(i, r) for i, r in enumerate(sources, 1))
    text, cost = chat([{"role": "system", "content": SYSTEM},
                       {"role": "user", "content": f"BRIEFFRAGMENTEN:\n\n{context}\n\nVRAAG: {query}"}],
                      model=model, max_tokens=4000)
    cited = {int(m) for m in re.findall(r"\[#(\d+)\]", text)}
    for i, r in enumerate(sources, 1):
        r["cited"] = i in cited
    return {"answer": text, "sources": sources, "cost": cost}


def export_rows(sources: list[dict]) -> list[dict]:
    return [{
        "brief": r["letter_id"], "afzender": r["sender"], "ontvanger": r["recipient"], "datum": r["date"],
        "transcriptie": SOURCE_LABEL.get(r["text_source"], r["text_source"]),
        "relevantie": r.get("confidence"), "geciteerd": r.get("cited", False),
        "personen_gekoppeld": "; ".join(r["persons_linked"]), "plaatsen": "; ".join(r["places"]),
        "tekst": r["text"], "kbr_link": r["permalink"],
    } for r in sources]
