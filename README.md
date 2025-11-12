# 🍕 Local_AI_Agent_01

Ein **lokaler KI‑Agent**, der Fragen zu Restaurantbewertungen beantwortet.  
Das Projekt nutzt **LangChain**, **Ollama** und **Chroma**, um eine **lokale RAG‑Pipeline** (Retrieval‑Augmented Generation) zu implementieren – ohne Cloud.

---

## 🧩 Überblick

- **Use Case:** Stelle Fragen wie „Wie ist die Pizza bewertet?“ oder „Was bemängeln Gäste am Service?“  
- **Datenquelle:** Lokale CSV-Datei mit Bewertungen (wird **nicht** ins Repo eingecheckt).  
- **Ablauf:** CSV → Embeddings → Vektorspeicher (Chroma) → semantische Suche → Antwort durch LLM.

---

## 🧠 Architektur (kurz)

- `vector.py` lädt die CSV, erzeugt Embeddings (`mxbai-embed-large` über Ollama) und speichert sie in **Chroma**.
- `main.py` baut eine einfache Prompt-Kette mit **LangChain** und einem lokalen LLM (z. B. `llama3.2` über Ollama).  
  Für jede Nutzerfrage werden passende Reviews abgerufen und in die Antwort eingebunden.

---

## 📦 Anforderungen

- **Python** 3.10+
- **Ollama** installiert und laufend (https://ollama.com/)  
  Modelle, die benötigt werden:
  ```bash
  ollama pull llama3.2
  ollama pull mxbai-embed-large
  ```
- Abhängigkeiten aus `requirements.txt`:
  - `langchain`
  - `langchain-ollama`
  - `langchain-chroma`
  - `pandas`

> 💡 Tipp: Erstelle eine virtuelle Umgebung, bevor du installierst.

---

## 🗂 Projektstruktur

```
Local_AI_Agent_01/
├─ main.py                      # Interaktives Q&A im Terminal
├─ vector.py                    # CSV einlesen, Embeddings, Chroma-Retriever
├─ requirements.txt             # Python-Abhängigkeiten
└─ reviews.csv   # 
```

---

## 🧰 Installation & Setup

1. **Repo klonen**
   ```bash
   git clone https://github.com/waistarek/Local_AI_Agent_01.git
   cd Local_AI_Agent_01
   ```

2. **Virtuelle Umgebung**
   ```bash
   python -m venv .venv
   # macOS/Linux
   source .venv/bin/activate
   # Windows (PowerShell)
   .venv\Scripts\Activate.ps1
   ```

3. **Abhängigkeiten installieren**
   ```bash
   pip install -r requirements.txt
   ```

4. **Ollama & Modelle vorbereiten**
   ```bash
   ollama pull llama3.2
   ollama pull mxbai-embed-large
   ```

5. **CSV-Datei lokal hinzufügen**  
   Lege `reviews.csv` ins Projektverzeichnis.  
   **Wichtig:** Die Datei wird **nicht** ins Git-Repo hochgeladen (siehe `.gitignore` unten).

---

## 🧾 Datenschema (CSV)

Erwartete Spaltennamen (Beispiele in `vector.py`):

- **Title** – Kurztitel der Bewertung  
- **Review** – Volltext der Bewertung  
- **Rating** – Numerische Bewertung (z. B. 1–5)  
- **Date** – Datum der Bewertung (z. B. `2024-05-10`)

> Achte auf exakte Spaltennamen. Abweichungen führen zu Fehlern beim Einlesen.

---

## ▶️ Ausführen

Starte zuerst Ollama im Hintergrund (falls nicht automatisch).  
Dann im Projektordner:

```bash
python main.py
```

Du siehst eine Eingabeaufforderung wie:
```
#######################################################
Ask your question (q to quit): Welche Gerichte werden oft gelobt?
```

- Beenden mit: `q`  
- Stelle Fragen auf Deutsch oder Englisch.

---

## 🛠️ Konfiguration & Ablage des Vektorspeichers

Standardmäßig wird Chroma lokal gespeichert.  
Wenn du einen **persistenten Pfad** erzwingen möchtest, ergänze in `vector.py` beim Erstellen des Chroma-Stores einen `persist_directory` (z. B. `./chroma_langchain_database`) und rufe nach dem Einfügen `persist()` auf:

```python
vector_store = Chroma(
    collection_name="restaurant_reviews",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_database",
)
# nach dem Hinzufügen:
vector_store.persist()
```

> Dadurch bleiben die Embeddings auch nach Skriptneustarts erhalten.

---

## 🚫 Wichtiger Git-Hinweis (Dateien nicht hochladen)

Bitte **keine Datenblätter** committen:
- CSV- und Excel-Dateien aus Datenschutz-/Gründen der Repo-Hygiene ausschließen.

Beispiel für `.gitignore` (füge diese Datei im Repo hinzu):
```
# Daten
*.csv
*.xlsx
*.xls
realistic_restaurant_reviews.csv

# Vektordatenbanken / temporäre Artefakte
chroma*
*.chroma
*.db
**/__pycache__/
.venv/
```

---

## 🧪 Beispiel-Fragen

- „Was sagen Gäste über die Pizza Margherita?“  
- „Welche Kritikpunkte kommen am häufigsten vor?“  
- „Wie ist die Stimmung insgesamt im Juni 2024?“

---

## ❗ Fehlerbehebung (FAQ)

- **`ModuleNotFoundError`**: Abhängigkeiten mit `pip install -r requirements.txt` installieren.  
- **Ollama-Fehler / Modell nicht gefunden**: `ollama pull <modellname>` ausführen und prüfen, dass der Ollama-Dienst läuft.  
- **Leere Antworten**: Prüfe, ob die CSV korrekt benannt ist und die Spalten wie oben heißen.  
- **Persistenz funktioniert nicht**: `persist_directory` setzen und nach dem Hinzufügen `persist()` aufrufen (siehe oben).

---

## 🗺️ Roadmap / Verbesserungen

- Eingabedaten validieren (z. B. fehlende Spalten erkennen).  
- Web-GUI (z. B. Streamlit) statt Terminal.  
- Mehr Metriken / Filter (z. B. nach Datum oder Rating).  
- Unit-Tests und Typannotationen.  

> Hinweis zu `vector.py`: Achte darauf, dass `metadata` den Schlüssel `rating` korrekt schreiben sollte.

---

## 📜 Lizenz

MIT-Lizenz – gerne forken, verbessern und teilen.

---

## 👤 Autor

Tarek Wais  · © 2025
