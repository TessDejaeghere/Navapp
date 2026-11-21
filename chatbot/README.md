# Navapp-chatbot

Retrieval-augmented chat over the François-Joseph Navez correspondence (KBR, Ms. II 70),
built on the VaidyaBot code (Chainlit UI, hybrid search, OpenRouter, cited answers).

## In Navapp (recommended)

The chatbot is built into Navapp (`navapp.py`, started with `start_navapp.bat`):

- **Pipeline** page, *Step 4 — send to the Navapp chatbot*: adds freshly processed letters
  (text, sender, recipient, year, persons with suggested links, places, institutions,
  artworks, events) to `data/corpus/uploads.jsonl` and rebuilds the index.
- **Chatbot** page: the chat itself (Streamlit version of this app).
- **Chatbot data** page: list and remove uploaded letters, rebuild the index.

Each rebuild writes `data/index/vNNN/` and switches `data/index/CURRENT`; embeddings are
cached in `data/index/embed_cache.sqlite`, so only new or changed fragments cost money.

## Standalone (Chainlit)

Double-click `Start_NavappBot.bat` (port 8766). Until this folder has its own `.venv`,
the launcher borrows VaidyaBot's environment at `%USERPROFILE%\Sanskriet\.venv`.

The OpenRouter key is read from `chatbot/openrouter_key.txt` or else `../config/key_openrouter.txt`.

## Rebuild the index (after new transcriptions, metadata or NER output)

```
python scripts/build_corpus.py     # letters + metadata + entities -> data/corpus/letters.jsonl
python scripts/build_index.py      # SQLite FTS + embeddings (OpenRouter, about $0.20); --local for e5
python scripts/ask.py "vraag" --no-llm    # quick retrieval check from the command line
```

## What goes into the index

| Source (Navez repo) | Used for |
|---|---|
| `NER_pipeline/predict_dalembert_corpus.load_corpus()` | letter text: corrected HTR > manual > raw Claude HTR |
| `data/letters_overview.csv` | sender, recipient, date, KBR permalink |
| `data/ner_corpus/per_mentions_openrefine.csv` | persons mentioned, suggested person links |
| `data/ner_corpus/dalembert_corpus_spans.jsonl` | places, institutions, artworks, events |

Letters are cut into chunks of about 1,600 characters (most letters are one chunk).
Each chunk carries its letter's metadata and the entities found inside it.

## Retrieval

1. BM25 over a trigram full-text index of metadata + entities + text (diacritics folded).
2. Dense search with `text-embedding-3-large` (so Dutch or English questions find French letters).
3. A correspondent channel: when the question names a sender or recipient, their letters get an extra ranking.
4. Reciprocal rank fusion, metadata filters (correspondent, person, years), at most 3 chunks per letter.
5. Reranking with `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (from the local Hugging Face cache).

The answer model (Claude via OpenRouter) may only use the retrieved fragments and must cite them as [#n].

## Known limits

- Questions about counts ("how many letters…") cannot be answered reliably: the model only sees the retrieved fragments.
- Metadata errors carry over (e.g. Ms. II 70/1/184 is listed as Portaels but written by Robert).
- Person links are unverified suggestions from the pipeline (e.g. "Léonard" → du Bus de Gisignies).
- 613 chunks have no year; they are kept in period searches unless you switch "Ongedateerde brieven meenemen" off.
