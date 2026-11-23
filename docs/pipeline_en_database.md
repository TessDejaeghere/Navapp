# Navapp: pipeline en chatbotdatabase

Stand van zaken op 23 november 2026 (chatbotindex v007, met Fiens correcties van Batch 1).

Navapp is één Streamlit-app (`navapp.py`) met drie pagina's:

| Pagina | Bestand | Voor wie, waarvoor |
|---|---|---|
| Pipeline | `pipeline_app.py` | Fien: van scan tot transcriptie, entiteiten, clusters, koppelingen en doorsturen naar de chatbot |
| Chatbot | `navapp_pages/chat.py` | vragen stellen over de brieven (inhoud én tellingen); gesprekken worden bewaard |
| Chatbot data | `navapp_pages/chat_data.py` | toegevoegde brieven, OpenRefine-correcties, wat de bot onthoudt, nodegoat-export, index herbouwen |

```
scans ──► 1 invoer ──► 2 HTR ──► 3 NER ► clustering ► (LLM-check) ► linking ──► mention-CSV
                                         │                                          │
                                         │                                OpenRefine (Fien verbetert)
                                         │                                          │
                                         └──► 4 naar de chatbot ──► uploads.jsonl   ▼
                                                                         correcties (person_mentions.jsonl)
onderzoekscorpus (letters.jsonl) ────────────────────────────────────────►┐
nodegoat-export (metadata) ──────────────────────────────────────────────►├──► indexversie vNNN
                                                                          ┘
                     chatbot: zoeken + tellen + antwoorden  ◄──►  geheugen (correcties uit gesprekken)
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
   score en alternatieve kandidaten. Fien verbetert die in OpenRefine (zie 2.3).

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
`embeddings_meta.json`. Bij elke bouw worden achtereenvolgens het onderzoekscorpus, de doorgestuurde
brieven en Fiens OpenRefine-correcties samengevoegd.

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
| | `start` | positie van het fragment in de brief, zodat correcties per vermelding exact geplaatst worden |
| Metadata | `sender`, `recipient` | naam met levensdata; voor gecorrigeerde brieven Fiens `Source_v2` en `Target_v2` |
| | `date`, `year` | datum en jaar van de brief |
| | `archive` | bewaarplaats |
| | `permalink` | link naar de KBR-catalogus |
| | `text_source` | welke transcriptie (zie hierboven) |
| Tekst | `text` | het brieffragment |
| Personen | `persons_checked` | personen **nagekeken door het onderzoeksteam** (uit de OpenRefine-export) |
| | `persons_linked` | automatisch gekoppelde personen, alleen voor brieven die nog niet nagekeken zijn |
| | `persons_mentioned` | de namen zoals ze in de tekst staan |
| Overige entiteiten | `places`, `organisations`, `artworks`, `events` | de overige NER-entiteiten van dat fragment |

**Wat er geëmbed wordt.** Vóór het embedden komt er een kopregel voor de tekst: *Letter from X to Y |
date … | persons: … | places: … | artworks: …*. Zo vindt de zoekmachine een fragment ook via de metadata
en entiteiten, niet alleen via de tekst zelf.

**Dekking van de personen:**

| | Fragmenten | Brieven |
|---|---|---|
| Met nagekeken personen | 2.621 | 1.564 |
| Nog met automatische koppelingen | 365 | |

**Zoekindexen:**
- `chunks_fts`: een SQLite FTS5-trigramindex op tekst plus kopregel, zonder accenten. Hij dient voor
  zoeken op trefwoorden (BM25), en vangt ook spellingvarianten en HTR-ruis gedeeltelijk op.
- `embeddings.npy`: 3.494 vectoren van `openai/text-embedding-3-large` (3.072 dimensies), opgehaald via
  OpenRouter. Een embeddingcache (`embed_cache.sqlite`) zorgt dat ongewijzigde fragmenten nooit opnieuw
  betaald worden. Het verwerken van Batch 1 kostte $0,19 (3.110 fragmenten opnieuw geëmbed).

### 2.2 Metadata voor tellingen

Deze tabellen komen uit Fiens nodegoat-export (`data/nodegoat_export_2106.csv`, te vervangen via de pagina
"Chatbot data") en de personenlijst. Ze bevatten **alle** brieven, ook die zonder transcriptie.

| Tabel | Rijen | Inhoud |
|---|---|---|
| `letters` | 2.372 | één rij per brief, met afzender en ontvanger (naam + nodegoat-ID), datum, jaar, archief, IDN, of er tekst is en welke |
| `persons` | 1.067 | personenlijst: nodegoat-ID, naam, naam zonder levensdata, geboorte en overlijden (330 personen hebben levensdata) |
| `letter_persons` | 6.338 | welke personen in welke brief vermeld worden (1.826 brieven, 727 personen); nagekeken waar gecorrigeerd, anders automatisch |
| `letter_entities` | 18.626 | plaatsen, instellingen, kunstwerken en gebeurtenissen per brief (zie hieronder) |

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
| Plaatsen | 7.349 | 1.812 |
| Kunstwerken | 6.381 | 1.431 |
| Instellingen | 3.394 | 1.238 |
| Gebeurtenissen | 1.502 | 806 |

Deze aantallen liggen lager dan in de vorige versie van dit rapport: losse woordjes zoals "du" of "-Ar" die
de NER als entiteit aanzag, worden nu weggefilterd.

**Twee soorten tellingen.** Wie wie schreef, wanneer en vanuit welk archief: dat telt de bot exact,
op nodegoat-ID. Wie of wat er in een brief vermeld wordt: dat komt uit Fiens correcties waar die er zijn,
en anders uit de automatische NER en linking. De bot noemt zulke getallen dus een *schatting*.

### 2.3 Fiens correcties uit OpenRefine

Fien verbetert de mention-CSV van de pipeline in OpenRefine en laadt de export op de pagina "Chatbot data".
De app zet die om naar `chatbot/data/corrections/person_mentions.jsonl`; bij elke indexbouw vervangen haar
personen de automatische koppelingen van de brieven in die export.

| Wat Fien invulde | Wat de chatbot ermee doet | Batch 1 |
|---|---|---|
| Nodegoat-naam in `surface_v2`, `ID_Nodegoat`, of een nodegoat-ID in `Notes` | gekoppeld aan die persoon | 7.358 vermeldingen, 346 personen |
| Een andere naam in `surface_v2` (bv. "mon frère" → "Aurèle Robert") | als naam bewaard, zonder nodegoat-ID | 901 |
| `DEL` / "Geen PER" | verwijderd | 164 |
| `GRP` / `ALG` | blijft staan, zonder persoon | 197 / 161 |
| `ART` / `LOC` | verplaatst naar kunstwerken of plaatsen | 64 / 22 |
| Niets gewijzigd | naam blijft als vermelding, zonder persoon | 12.404 |
| `Source_v2` / `Target_v2` | afzender en ontvanger van de brief in de fragmenten | |

Batch 1 (`20262808_KBR_Batch1_PER-and-PER-NOM`) beslaat 1.694 brieven; 21.257 van de 21.258 vermeldingen
vallen exact op hun plek in de tekst. Een nieuwere export van dezelfde brieven vervangt de oude; brieven uit
andere batches blijven staan.

Wat dit oploste, onder meer:

| Persoon | Was | Nu |
|---|---|---|
| François Marius Granet | foutief gekoppeld aan Antoinnette Granet | 132 brieven |
| B. Renard | onterecht gekoppeld aan veel losse "Mr"-vermeldingen | 12 brieven |
| "Madame Navez" | gekoppeld aan François-Joseph Navez | Augustine Flore de Lathuy (182 brieven) |
| "Robert" in de brief over Sturms dood | Louis-Léopold Robert (overleden 1835) | Alexandre Robert |

Vermeldingen die Fien ongewijzigd liet, verliezen hun automatische koppeling. Dat waren er 1.924, vooral
fouten ("L", "Mr" → B. Renard, "de"), maar ook enkele plausibele (Dupré → Louis Dupré).

### 2.4 Geheugen: correcties uit gesprekken en bewaarde gesprekken

| Bestand | Inhoud |
|---|---|
| `chatbot/data/memory/notes.jsonl` | feiten die een onderzoeker in de chat corrigeerde, met brief-ID's, personen, auteur en datum |
| `chatbot/data/memory/conversations.sqlite` | alle gesprekken van de chatpagina, te heropenen via de zijbalk |

Corrigeert Fien de bot op een feit ("Ms. II 70/1/184 is van Alexandre Robert, niet van Portaels"), dan slaat
de bot dat op met het hulpmiddel `remember_correction` en bevestigt hij het onder zijn antwoord. In latere
gesprekken krijgt hij de correcties die bij de vraag horen (zolang er minder dan 30 zijn: alle), en een
correctie over een brief staat ook naast die brief wanneer hij hem leest. Zulke correcties gaan voor op de
metadata, de automatische koppelingen en de transcripties, maar veranderen de tellingen niet: die blijven
op nodegoat steunen. Op de pagina "Chatbot data" kan Fien ze bekijken, aanpassen, verwijderen of zelf
toevoegen.

### 2.5 Fiens correcties delen

Fien is de bron van alle correcties. Alles wat zij verbetert, staat eerst alleen op haar computer. Op de
pagina "Chatbot data" kan ze het, wanneer zij dat wil, publiceren naar de openbare GitHub-repo
[TessDejaeghere/Navapp-corrections](https://github.com/TessDejaeghere/Navapp-corrections):

| Bestand | Inhoud |
|---|---|
| `manifest.json` | wanneer, door wie, hoeveel |
| `notes.jsonl` | de feiten die ze in de chat corrigeerde (zonder de gesprekken of haar vragen) |
| `person_mentions.jsonl.gz` | de nagekeken personen uit haar OpenRefine-exports |

Gesprekken en brieven worden nooit gepubliceerd. Het is bewust een aparte repo naast de code: Fiens
GitHub-sleutel (een fine-grained token met alleen *Contents: Read and write* op die ene repo, bewaard in
`config/github_token.txt`) kan zo alleen correcties schrijven, nooit de code die elke Navapp automatisch
installeert.

Andere Navapps controleren elk uur of er iets nieuws is gepubliceerd en tonen dan een melding. Met
"Use these corrections" downloaden ze alles naar `chatbot/data/shared/` en herbouwen ze de index. Fiens
correcties zijn daar alleen-lezen; eigen correcties blijven lokaal en gaan voor bij brieven die men zelf
corrigeerde.

### 2.6 Herkomst, geschiedenis en twijfel

Navapp bewaart niet alleen de huidige identificatie van een persoon, maar ook hoe die tot stand kwam. Elke
nagekeken vermelding draagt:

| Veld | Inhoud |
|---|---|
| `machine_person` | wat de automatische linking voorstelde |
| `history` | alle lezingen na elkaar: eerst de machine, dan elke lezing van de historicus die verschilt van de vorige, met datum, bron (welke export) en status |
| `note` | haar eigen woorden uit de kolom `Notes` (twijfel, hints), zonder statuscodes of nodegoat-ID's |

Een nieuwe export met een gewijzigde lezing voegt een stap toe aan de geschiedenis; een ongewijzigde lezing voegt
niets toe. Ook correcties in de chat houden hun eerdere formuleringen bij. Die geschiedenis reist mee in Fiens
publicaties.

In de chatbotindex krijgt elk fragment twee extra velden: `provenance` (per vermelding: oppervlaktevorm, lezing
van de machine, lezing van de historicus, datum, bron, geschiedenis) en `doubts` (haar twijfels). De bot:
- vermeldt haar twijfel wanneer hij over zo'n persoon spreekt ("de historicus noteert: niet zeker welke zoon");
- kan met `identification_history` uitleggen hoe een persoon in een brief geïdentificeerd werd en of dat veranderde.

De app toont bij elke bron hoe de personen gelezen werden (machine → historicus, met datum), en de pagina
"Chatbot data" geeft een overzicht van alle lezingen: de vaakst gewijzigde identificaties, alle twijfels, zoeken
op naam of brief, en een CSV met alles.

Voor Batch 1: 21.271 nagekeken vermeldingen, waarvan 5.462 met een andere lezing dan de machine voorstelde, en
210 met een twijfel of opmerking. De vaakste wijzigingen:

| Machine | Historicus | Vermeldingen |
|---|---|---|
| (geen persoon) | François-Joseph Navez | 296 |
| David d'Angers | Jacques-Louis David | 174 |
| François-Joseph Navez | Augustine Flore de Lathuy | 113 |
| Antoinnette Granet | François Marius Granet | 106 |
| (geen persoon) | Pierre III Poisson | 104 |

---

## 3. Hoe de chatbot de database gebruikt

1. **Model:** Claude Sonnet 4.5 (of Opus 4.8, te kiezen in de app), via OpenRouter, met tools.
2. **Inhoudsvragen** (`search_letters`):
   - De bot zoekt hybride: trefwoorden (BM25) en embeddings, samengevoegd met reciprocal rank fusion.
   - Daarna sorteert een lokale reranker de resultaten opnieuw (`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`).
   - Er komen maximaal 3 fragmenten per brief mee, en de bot mag maximaal 2 keer zoeken per vraag.
   - Met `letter_id` leest hij één bepaalde brief in plaats van te zoeken.
   - Filters: correspondent, periode, vermelde persoon en ongedateerde brieven aan of uit.
   - Elk antwoord verwijst met [#n] naar de fragmenten; de app toont die met een link naar de KBR.
3. **Telvragen:** daarvoor zijn er aparte tools. Alle teltools kunnen ook op één brief-ID filteren.

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
4. **Geheugen:** `remember_correction` slaat een gecorrigeerd feit op (zie 2.4).
5. **Taal:** de bot antwoordt in de taal van de vraag (Nederlands, Engels of Frans).
6. **Kostprijs en snelheid:** ongeveer $0,05 en 16 seconden per vraag (evaluatie van 21/11/2026). Tijdens het
   wachten toont de app wisselende boodschapjes ("Sifting through veuves Poignet", "Looking for castles in
   Spain" …).

## 4. Bekende beperkingen van de data

**Automatische koppelingen in nog niet nagekeken brieven.** Voor de brieven buiten Batch 1 staan nog
automatische suggesties in `persons_linked`, met de bekende soorten fouten ("Léonard" → du Bus de Gisignies).

**Een metadatafout:** Ms. II 70/1/184 staat in nodegoat op naam van Portaels, maar is van Robert. Dat is
in nodegoat te verbeteren; in de chat kan de bot het intussen onthouden als correctie.

**Personen zonder nodegoat-ID.** Namen die Fien wel identificeerde maar die niet in de personenlijst staan,
tellen als aparte "persoon" op naam. `find_person` vindt ze niet, dus vragen naar zo iemand lopen via de tekst.

**Relaties tussen personen.** "De vrouw van Navez" vindt de bot niet: hij zoekt personen op naam en gebruikt
de kolommen "Spouse of" en "Family of" uit de personenlijst nog niet.

**HTR-leesfouten.** Ongecorrigeerde HTR (1.467 brieven) bevat leesfouten, vooral in eigennamen.

**Brieven zonder tekst.** 385 brieven tellen mee in de statistieken, maar hun inhoud is niet doorzoekbaar.

## 5. Navapp bijwerken

- **Code:** ontwikkeling gebeurt in de Navez-onderzoeksrepo. `tools/export_navapp_repo.py` kopieert de code
  naar de private repo [TessDejaeghere/Navapp](https://github.com/TessDejaeghere/Navapp) (alleen code, geen
  data of sleutels).
- **Desktoppakket voor Fien:** `tools/build_navapp_dist.py --zip` maakt `dist/navapp.zip` (± 530 MB, met
  model, index en correcties, zonder OpenRouter-sleutel). Dat is alleen nodig voor de eerste installatie.
- **Automatische updates:** Navapp vraagt bij het opstarten (en daarna elk uur) aan de openbare GitHub-repo
  of er een nieuwere versie is (`navapp_version.json`, met versienummer en wat er nieuw is). Is dat zo, dan
  krijgt Fien een melding en een knop "Install the update" in de zijbalk. De app haalt dan de repo op en
  vervangt alleen code; vervangen bestanden komen in `_backup/<oude versie>/`. Zijn de benodigde
  Python-pakketten veranderd, dan installeert de app die ook. Daarna herstart Fien Navapp.
- **Een update uitbrengen:** na een wijziging in de onderzoeksrepo:
  `python tools/export_navapp_repo.py --release "wat er nieuw is"`, daarna commit en push in de Navapp-repo.
  Zonder `--release` krijgt Fien geen melding, zodat tussentijdse pushes haar niet storen.
- **Fiens eigen gegevens:** alles wat Fien in de app doet, staat in `chatbot/data` van haar installatie
  (doorgestuurde brieven, correcties, gesprekken, geheugen). De updater raakt die map nooit aan, net zo min als
  het model, de personenlijst, `data/` en haar sleutel in `config/`.
- **Wat niet via GitHub gaat:** een nieuwe index of nieuwe correcties van jouw kant staan niet in de repo
  (alleen code). Die laadt Fien zelf op de pagina "Chatbot data", of ze krijgt een nieuw pakket.
