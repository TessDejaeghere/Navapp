# Navapp: pipeline en chatbotdatabase

Stand van zaken op 21 november 2026 (chatbotindex v006).

Navapp is één Streamlit-app (`navapp.py`) met drie pagina's:

| Pagina | Bestand | Voor wie, waarvoor |
|---|---|---|
| Pipeline | `pipeline_app.py` | Fien: van scan tot transcriptie, entiteiten, clusters, koppelingen en doorsturen naar de chatbot |
| Chatbot | `navapp_pages/chat.py` | vragen stellen over de brieven (inhoud én tellingen) |
| Chatbot data | `navapp_pages/chat_data.py` | toegevoegde brieven bekijken of verwijderen, een nieuwe nodegoat-export laden, de index herbouwen |

```
scans ──► 1 invoer ──► 2 HTR ──► 3 NER ► clustering ► (LLM-check) ► linking ──► mention-CSV (review)
                                         │
                                         └──► 4 naar de chatbot ──► uploads.jsonl ──► nieuwe indexversie
                                                                                      │
nodegoat-export (metadata) ─────────────────────────────────────────────────────────►┤
onderzoekscorpus (letters.jsonl) ────────────────────────────────────────────────────►┘
                                                                         chatbot: zoeken + tellen + antwoorden
```

---

## 1. De pipeline (pagina "Pipeline")

### Stap 1: invoer
Vijf manieren om brieven binnen te brengen: afbeeldingen van één brief, een zip met meerdere brieven
(één map per brief of bestanden `<brief>_<pagina>.jpg`), een map op de computer (voor grote batches),
een IIIF-manifest-URL, of een bestaande transcriptie plakken. Per brief vult Fien een brief-ID in en
eventueel afzender, ontvanger en jaar. De app waarschuwt als een brief al eerder verwerkt werd.

### Stap 2: transcriptie (HTR)
- **Model:** Claude Opus 4.8 via OpenRouter, met dezelfde prompt als de corpusrun
  (`HTR_pipeline/run_htr_opus.py`). Alle pagina's van een brief gaan in één oproep, temperatuur 0.
- **Kostprijs:** ongeveer $0,035 per pagina (gemeten op 3.929 pagina's: $129).
- De transcriptie is daarna bewerkbaar in de app, net als afzender, ontvanger en jaar.

### Stap 3: entiteiten, clusters, koppelingen
1. **NER:** het fijngetunede d'AlemBERT-model (`NER_pipeline/ckpts/dalembert_fold2/checkpoint-450`) draait
   lokaal en herkent zes soorten entiteiten:

   | Label | Betekenis |
   |---|---|
   | `PER` | persoonsnaam |
   | `PER_NOM` | omschrijving van een persoon ("mon frère", "le directeur") |
   | `LOC` | plaats |
   | `ORG` | instelling |
   | `ART` | kunstwerk |
   | `EVT` | gebeurtenis |

   Elke persoonsvermelding wordt een rij, met 25 woorden context links en rechts.
2. **Clustering:** vermeldingen die waarschijnlijk naar dezelfde persoon verwijzen, worden gegroepeerd. De regels:
   - Aanspreekvormen (M., Mme, Monsieur …) worden weggelaten en `PER`-namen worden over het hele corpus
     gegroepeerd (`gs::`).
   - Komt dezelfde naam zowel met mannelijke als vrouwelijke aanspreekvorm voor, dan worden die twee gesplitst.
   - `PER_NOM` en vermeldingen zonder bruikbare naam worden alleen binnen één brief gegroepeerd (`lt::`).
3. **LLM-check (optioneel):** voor familienamen die meerdere personen in de personenlijst delen, beslist een
   taalmodel welke vermeldingen bij elkaar horen. Kost ongeveer $0,03 per twijfelachtige naam.
4. **Linking:** elke cluster wordt gekoppeld aan een persoon uit Fiens personenlijst
   (`CLUSTERING_pipeline/glossary/PERSON_export_2708.csv`, met nodegoat-ID's):
   - fuzzy matching (token_set, drempel 90);
   - bij gelijke scores beslissen levensdata, voorkennis uit het corpus en context;
   - een gendercontrole en een fonetische ondergrens (92) houden foute koppelingen tegen;
   - onder de drempel wordt de cluster "NIL" (geen koppeling).
5. **Uitvoer:** een mention-CSV (`mentions.csv`) om na te kijken, met de voorgestelde cluster, persoon,
   score en alternatieve kandidaten.

### Stap 4: naar de chatbot
- **Wat er meegaat:** de verwerkte brieven met hun tekst, metadata, gekoppelde personen en alle andere
  entiteiten (plaatsen, instellingen, kunstwerken, gebeurtenissen).
- **Ontbrekende metadata:** ontbreken afzender, ontvanger of datum, dan worden die aangevuld uit Fiens
  nodegoat-overzicht (`data/letters_overview.csv`). Daar komen ook het archief en de KBR-link vandaan.
- **Status van de tekst:** met het vinkje "transcriptie nagekeken" krijgt de tekst de status `corrected_htr`
  in plaats van `opus_htr`. Dan waarschuwt de bot niet meer voor mogelijke leesfouten.
- **Opslag:** de brieven komen in `chatbot/data/corpus/uploads.jsonl`. Een brief die opnieuw wordt
  doorgestuurd, vervangt de oude versie.
- **Index:** daarna wordt meteen een nieuwe indexversie gebouwd. Alleen nieuwe fragmenten krijgen een
  embedding; dat kost minder dan een cent per tien brieven.

---

## 2. Wat de chatbotdatabase bevat

De index staat in `chatbot/data/index/vNNN/`. Het bestand `CURRENT` wijst naar de actieve versie, en de
twee recentste versies worden bewaard. Een versie bestaat uit `chunks.db` (SQLite), `embeddings.npy` en
`embeddings_meta.json`.

### 2.1 Brieftekst: fragmenten (tabel `chunks`)

| | |
|---|---|
| Brieven met tekst | **1.987** |
| Fragmenten | **3.494** (gemiddeld 1.206 tekens; max. 1.600 tekens met 250 tekens overlap) |
| Totale tekst | ± 4,2 miljoen tekens |
| Door Fien toegevoegde brieven (`uploads.jsonl`) | 0 (nog niets doorgestuurd) |

**Herkomst van de tekst.** Per brief wordt de beste beschikbare versie gekozen:

| Bron | Brieven | Fragmenten |
|---|---|---|
| Nagekeken HTR (`corrected_htr`) | 327 | 609 |
| Manuele transcriptie (`manual`) | 193 | 460 |
| Ruwe Claude Opus-HTR (`opus_htr`) | 1.467 | 2.425 |

Driekwart van de brieven is dus ongecorrigeerde HTR. De bot vermeldt dat bij citaten, maar neemt
leesfouten in namen wel over (zie het evaluatieverslag).

**Velden per fragment.** Elk fragment draagt deze velden:

| Groep | Veld | Inhoud |
|---|---|---|
| Identificatie | `id`, `letter_id`, `chunk` | bv. "Ms. II 70/1/181#0" |
| Metadata | `sender`, `recipient` | naam met levensdata, zoals in nodegoat |
| | `date`, `year` | datum en jaar van de brief |
| | `archive` | bewaarplaats |
| | `permalink` | link naar de KBR-catalogus |
| | `text_source` | welke transcriptie (zie hierboven) |
| Tekst | `text` | het brieffragment |
| Entiteiten | `persons_linked` | personen uit de personenlijst, via de linking. *Automatische suggestie* |
| | `persons_mentioned` | de namen zoals ze in de tekst staan |
| | `places`, `organisations`, `artworks`, `events` | de overige NER-entiteiten van dat fragment |

**Wat er geëmbed wordt.** Vóór het embedden komt er een kopregel voor de tekst: *Letter from X to Y |
date … | persons: … | places: … | artworks: …*. Zo vindt de zoekmachine een fragment ook via de metadata
en entiteiten, niet alleen via de tekst zelf.

**Dekking van de entiteiten:**

| Veld | Fragmenten |
|---|---|
| Gekoppelde personen | 3.045 |
| Plaatsen | 3.007 |
| Kunstwerken | 2.267 |

**Zoekindexen:**
- `chunks_fts`: een SQLite FTS5-trigramindex op tekst plus kopregel, zonder accenten. Hij dient voor
  zoeken op trefwoorden (BM25), en vangt ook spellingvarianten en HTR-ruis gedeeltelijk op.
- `embeddings.npy`: 3.494 vectoren van `openai/text-embedding-3-large` (3.072 dimensies), opgehaald via
  OpenRouter. Een embeddingcache (`embed_cache.sqlite`) zorgt dat ongewijzigde fragmenten nooit opnieuw
  betaald worden.

### 2.2 Metadata voor tellingen

Deze tabellen komen uit Fiens nodegoat-export (`data/nodegoat_export_2106.csv`, te vervangen via de pagina
"Chatbot data") en de personenlijst. Ze bevatten **alle** brieven, ook die zonder transcriptie.

| Tabel | Rijen | Inhoud |
|---|---|---|
| `letters` | 2.372 | één rij per brief, met afzender en ontvanger (naam + nodegoat-ID), datum, jaar, archief, IDN, of er tekst is en welke |
| `persons` | 1.067 | personenlijst: nodegoat-ID, naam, naam zonder levensdata, geboorte en overlijden (330 personen hebben levensdata) |
| `letter_persons` | 6.178 | welke gekoppelde personen in welke brief vermeld worden (1.870 brieven, 590 verschillende personen) |
| `letter_entities` | 20.450 | plaatsen, instellingen, kunstwerken en gebeurtenissen per brief (zie hieronder) |

**Brieven (`letters`):**

| Kenmerk | Aantal |
|---|---|
| Met transcriptie | 1.987 |
| Zonder transcriptie | 385 |
| Afzender met nodegoat-ID | 2.353 |
| Ontvanger met nodegoat-ID | 2.336 |
| Gedateerd (1803–1870) | 1.913 |
| Ongedateerd | 459 |

**Archieven:**

| Archief | Brieven |
|---|---|
| KBR | 2.001 |
| OCMW Brussel | 111 |
| Académie royale | 90 |
| Neuchâtel | 43 |
| Privécollectie | 25 |
| Overige (o.a. Fondation Custodia, Rijksmuseum, Thorvaldsen) | 26 |
| Geen archief ingevuld | 76 |

**Entiteiten per brief (`letter_entities`):**

| Soort | Vermeldingen | Brieven |
|---|---|---|
| Plaatsen | 7.748 | 1.821 |
| Kunstwerken | 7.078 | 1.443 |
| Instellingen | 3.965 | 1.257 |
| Gebeurtenissen | 1.659 | 821 |

**Twee soorten tellingen.** Wie wie schreef, wanneer en vanuit welk archief: dat telt de bot exact,
op nodegoat-ID. Wie of wat er in een brief vermeld wordt: dat komt uit de automatische NER en linking.
De bot noemt zulke getallen dus een *schatting*.

---

## 3. Hoe de chatbot de database gebruikt

1. **Model:** Claude Sonnet 4.5 (of Opus 4.8, te kiezen in de app), via OpenRouter, met tools.
2. **Inhoudsvragen** (`search_letters`):
   - De bot zoekt hybride: trefwoorden (BM25) en embeddings, samengevoegd met reciprocal rank fusion.
   - Daarna sorteert een lokale reranker de resultaten opnieuw (`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`).
   - Er komen maximaal 3 fragmenten per brief mee, en de bot mag maximaal 2 keer zoeken per vraag.
   - Filters: correspondent, periode, vermelde persoon en ongedateerde brieven aan of uit.
   - Elk antwoord verwijst met [#n] naar de fragmenten; de app toont die met een link naar de KBR.
3. **Telvragen:** daarvoor zijn er aparte tools.

   | Tool | Wat het doet |
   |---|---|
   | `find_person` | zoekt welke persoon bedoeld wordt; bij meerdere kandidaten vraagt de bot het na |
   | `count_letters` | telt brieven |
   | `letters_per_year` | telt brieven per jaar |
   | `top_people` | rangschikt afzenders, ontvangers, correspondenten of vermelde personen |
   | `top_entities` | toont de meest vermelde plaatsen, kunstwerken enz. |
   | `list_letters` | geeft een lijst van brieven |
   | `collection_overview` | geeft een overzicht van de collectie |

   De gegevens achter elk getal verschijnen als tabel of grafiek onder het antwoord.
4. **Kostprijs en snelheid:** ongeveer $0,05 en 16 seconden per vraag (evaluatie van 21/11/2026).

## 4. Bekende beperkingen van de data

**Fouten in de koppelingen.** Die komen in `persons_linked` terecht. Voorbeelden:
- "Robert" wordt soms Louis-Léopold waar Alexandre bedoeld is;
- "Léonard" wordt du Bus de Gisignies;
- "Madame Navez" wordt François-Joseph Navez.

**Scheve ranglijsten van vermelde personen**, door losse "Mr"-vermeldingen gekoppeld aan B. Renard en door
koppelfouten naar Antoinnette Granet.

**Een metadatafout:** Ms. II 70/1/184 staat in nodegoat op naam van Portaels, maar is van Robert.

**Oudere clusterregel in het corpus.** Het onderzoekscorpus gebruikt nog de oudere dedupe-clusters (`dd::`),
niet de regel uit de app. Nieuw doorgestuurde brieven volgen de app-regel.

**HTR-leesfouten.** Ongecorrigeerde HTR (1.467 brieven) bevat leesfouten, vooral in eigennamen.

**Brieven zonder tekst.** 385 brieven tellen mee in de statistieken, maar hun inhoud is niet doorzoekbaar.
