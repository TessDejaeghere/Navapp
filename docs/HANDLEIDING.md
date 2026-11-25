# Navapp: handleiding voor Fien

Navapp heeft drie pagina's, die je links in de zijbalk kiest:

- **Pipeline:** van scan of transcriptie naar personen, met een CSV voor OpenRefine.
- **Chatbot:** vragen stellen over de brieven.
- **Chatbot data:** je correcties inladen, bekijken en publiceren.

Alles wat de machine maakt (transcripties, herkende namen, gekoppelde personen, antwoorden) is een voorstel.
Jij hebt het laatste woord.

---

## 1. Opstarten

1. Dubbelklik op **`start_navapp.bat`** in je Navapp-map.
2. Er opent een zwart venster: laat het openstaan zolang je werkt. Navapp opent in je browser.
   Gebeurt dat niet, ga dan naar `http://localhost:8501`.
3. Klaar? Sluit het browsertabblad en het zwarte venster.

**Eerste keer:** vul je OpenRouter-sleutel in via **Settings** in de zijbalk van de pagina Pipeline. Die sleutel
is nodig voor de transcriptie (HTR) en voor de chatbot.

> Installeer Navapp altijd vanuit de **zip** (niet vanuit GitHub): alleen de zip bevat het model, de
> personenlijst en de chatbotdata. Updates komen daarna vanzelf via GitHub.

## 2. Updates

Staat er links in de zijbalk een blauw vak **"A new Navapp is available"**:

1. Klik op **⬇️ Install the update**.
2. Sluit het zwarte venster en start Navapp opnieuw met `start_navapp.bat`.

Je data, je correcties en je sleutel blijven gewoon staan. Zie je het vak niet terwijl er wel een update is?
Start Navapp dan opnieuw op: hij kijkt één keer per uur.

---

## 3. Brieven door de pipeline halen

### Stap 1: brieven inladen

Kies hoe je de brieven aanlevert:

| Keuze | Wanneer |
|---|---|
| **Images of one letter** | één brief, als losse afbeeldingen |
| **Zip with several letters** | meerdere brieven: één map per brief, of bestanden `<brief>_<pagina>.jpg` |
| **Folder on this computer** | grote batches: geef het pad van de map, er wordt niets geüpload |
| **IIIF manifest URL** | een brief rechtstreeks van de KBR-server |
| **Paste an existing transcription** | één brief waarvan je de tekst al hebt |
| **Transcriptions in bulk** | veel bestaande transcripties tegelijk (zie hieronder) |

Klik daarna op **Load letter(s)**.

**Transcripties in bulk:**

- Tekstbestanden (`.txt`): één brief per bestand, de bestandsnaam is de brief-ID, bv. `Letter 144.txt`. Een zip
  met zulke bestanden mag ook.
- Of een tabel (CSV) met één brief per rij: een kolom `letter_id` (of `brief`) en een kolom `text` (of
  `transcriptie`). Afzender, ontvanger en jaar mogen er ook in. Gebruik de tabel voor ID's als
  `Ms. II 70/1/23`: een `/` kan niet in een bestandsnaam.

**Scans en transcripties combineren:** laad eerst het ene, dan het andere. Laat het vinkje **"Add to the
letters already loaded"** aan staan, dan komen ze in één batch en krijg je straks één CSV.

> Gebruik dezelfde brief-ID's als in nodegoat. Anders herkent de chatbot de brief later niet.

### Stap 2: transcriptie

- Brieven met scans: klik op **Transcribe … letter(s)**. De geschatte kost staat erboven (ongeveer $0,035 per
  pagina).
- Brieven die al tekst hebben, worden overgeslagen.
- Pagina's die zijwaarts, ondersteboven of scheef gescand zijn, worden eerst rechtgezet (schakelaar **"Turn
  pages upright first"** in de zijbalk, staat standaard aan). Een klein model op je eigen computer beslist;
  alleen bij twijfel wordt Claude Haiku kort geraadpleegd (ongeveer $0,0002 per pagina). Je originele scans
  blijven onaangeroerd. Onder de brief zie je welke pagina's gedraaid zijn. Bij twijfel blijft een pagina zoals
  ze is.
- Je kunt elke transcriptie, afzender, ontvanger en jaar nog aanpassen in het uitklapvak van de brief.

### Stap 3: personen zoeken

Klik op **Run the pipeline on … letter(s)**. Dit is gratis: het draait op je eigen computer. De eerste keer
duurt het even, omdat het model geladen wordt.

Daarna zie je de gevonden vermeldingen en de knop **⬇️ Download mention CSV**: dat is het bestand voor
OpenRefine, met dezelfde kolommen als bij Batch 1.

**Context:** bij elke vermelding staan een paar volledige zinnen ervoor en erna (`context_before`,
`context_after`). Hoeveel zinnen kies je in de zijbalk met **"Context in the CSV"**.

### Alles blijft bewaard

Elke run wordt op je computer bewaard, ook na herladen, afsluiten of een update. Bovenaan de pagina, in
**📚 Letters already through the pipeline**, vind je:

- welke brieven al door de pipeline gingen, wanneer, en of ze al in de chatbot zitten;
- **Reopen this run**: de resultaten van een eerdere run terughalen;
- **⬇️ All mentions of all runs**: alles in één CSV.

### Stap 4: naar de chatbot sturen

Klik op **➕ Send … letter(s) to the chatbot** als je over deze brieven vragen wilt kunnen stellen. Vink
**"I checked and corrected these transcriptions"** aan als je de tekst nagekeken hebt; dan waarschuwt de
chatbot niet meer voor leesfouten. Dit kost minder dan een cent per tien brieven.

---

## 4. De chatbot

- Stel je vraag in het Nederlands, Engels of Frans. Hij antwoordt in dezelfde taal.
- Elke bewering heeft een bron **[#n]**: klik erop om het fragment en de link naar de scan te zien.
- Vragen over aantallen ("hoeveel brieven schreef Gudin?") worden exact geteld op de nodegoat-metadata,
  met een tabel als bewijs.
- Je kunt vragen naar je eigen opmerkingen: *"Over welke brieven is er twijfel over de OCR?"*, *"Hoe werd
  Robert in Ms. II 70/1/181 geïdentificeerd?"*.
- **Een fout verbeteren:** zeg het gewoon, bv. *"Die brief is van Alexandre Robert, niet van Louis-Léopold."*
  De chatbot onthoudt het en gebruikt het in alle latere antwoorden.
- Gesprekken worden bewaard. Je vindt ze terug in de zijbalk; daar start je ook een nieuw gesprek.

Een vraag kost meestal een paar cent.

---

## 5. Je correcties uit OpenRefine inladen

1. **In OpenRefine:** wis eerst alle filters en facetten (**Remove all**, linksboven), zodat alle rijen
   zichtbaar zijn. OpenRefine exporteert alleen wat je op dat moment ziet.
2. Exporteer als CSV.
3. In Navapp: pagina **Chatbot data** → **Checked persons (OpenRefine export)** → upload het bestand.
4. Kijk het overzicht na en klik op **✅ Use these corrections**. De chatbot werkt meteen met je lezingen.

Wat Navapp leest: `surface_v2` (je lezing), `ID_Nodegoat`, `Notes` (je opmerkingen en twijfels, in je eigen
woorden) en `Source_v2` / `Target_v2` (afzender en ontvanger). Niets gaat verloren: bij elke vermelding blijft
bewaard wat de machine voorstelde en elke lezing van jou, met datum. Dat zie je onder **History of the
readings**.

## 6. Je correcties publiceren

Je correcties blijven op je eigen computer tot jij beslist ze te delen, bv. na een afgewerkte batch.

1. Pagina **Chatbot data** → klap **⬆️ Publish my corrections (for Fien)** open.
2. De eerste keer: plak de GitHub-sleutel die Tess je gaf en klik op **Save key**.
3. Vink de bevestiging aan en klik op **⬆️ Publish**.

Andere installaties van Navapp krijgen dan een melding en kunnen je correcties met één klik inladen. Zij
kunnen ze niet aanpassen. Gesprekken en brieven worden nooit gepubliceerd.

---

## 7. Als er iets misloopt

| Wat je ziet | Wat te doen |
|---|---|
| Stap 3 meldt dat het model ontbreekt | Kopieer de mappen `NER_pipeline\ckpts` en `NER_pipeline\eval` uit de zip naar je Navapp-map en start opnieuw. |
| Fout **402** (credit) bij transcriptie of chat | Kijk op openrouter.ai: staat er tegoed onder **Credits**, en is de limiet van je sleutel onder **Keys** niet opgebruikt? Tegoed en sleutel moeten in hetzelfde account staan. |
| Je resultaten van stap 3 zijn weg | Open **📚 Letters already through the pipeline** bovenaan en klik op **Reopen this run**. |
| De chatbot kent een verbetering of een nieuwe brief niet | Pagina **Chatbot data** → **🔄 Rebuild the index**. |
| Iets anders | Maak een screenshot van de foutmelding én van het zwarte venster, en stuur ze naar Tess. |

Kosten in het kort: transcriptie ongeveer $0,035 per pagina, een chatvraag een paar cent, de pipeline (stap 3)
gratis.
