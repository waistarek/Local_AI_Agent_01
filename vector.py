"""
vector.py — Erstellen eines Vektorspeichers (Chroma) aus CSV-Bewertungen
und Bereitstellen eines Retrievers für semantische Suche.

Ablauf:
1) CSV mit Restaurant-Bewertungen laden (Titel, Review, Rating, Date).
2) Für jede Zeile ein LangChain-Document erstellen (Text + Metadaten).
3) Mit Ollama-Embeddings (mxbai-embed-large) die Texte vektorisieren.
4) Alles in eine lokale Chroma-Datenbank schreiben (persistieren).
5) Einen Retriever exportieren, den main.py nutzen kann.

Voraussetzungen:
- Python-Abhängigkeiten aus requirements.txt
- Ollama installiert + Modell "mxbai-embed-large" vorhanden:
    ollama pull mxbai-embed-large
- CSV-Datei liegt im Projektordner: realistic_restaurant_reviews.csv
"""

# --- Importe ---
from langchain_ollama import OllamaEmbeddings          # Embeddings über Ollama (lokal)
from langchain_chroma import Chroma                    # Vektorspeicher (Chroma) für LangChain
from langchain_core.documents import Document          # Dokument-Container von LangChain
import os
import pandas as pd


# --- Konfiguration / Konstanten ---
CSV_PATH = "realistic_restaurant_reviews.csv"          # Name der Eingabe-CSV
CHROMA_DIR = "./chroma_langchain_database"             # Speicherort der Chroma-Datenbank
COLLECTION_NAME = "restaurant_reviews"                 # Name der Chroma-Collection
TOP_K = 5                                              # Anzahl ähnlicher Dokumente bei Abfragen


def _load_dataframe(csv_path: str) -> pd.DataFrame:
    """
    Lädt die CSV-Datei in ein Pandas-DataFrame und prüft die erwarteten Spalten.

    Erwartete Spalten:
      - Title
      - Review
      - Rating
      - Date
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"CSV-Datei '{csv_path}' wurde nicht gefunden. "
            f"Lege die Datei ins Projektverzeichnis oder passe CSV_PATH an."
        )

    df = pd.read_csv(csv_path)

    required_cols = {"Title", "Review", "Rating", "Date"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(
            "Die CSV-Datei enthält nicht alle benötigten Spalten.\n"
            f"Erwartet: {sorted(required_cols)}\n"
            f"Fehlend:  {sorted(missing)}"
        )

    # Optional: Leere Zellen durch leere Strings ersetzen (robuster gegen NaNs)
    df["Title"] = df["Title"].fillna("").astype(str)
    df["Review"] = df["Review"].fillna("").astype(str)

    return df


def _build_documents(df: pd.DataFrame):
    """
    Erzeugt aus dem DataFrame eine Liste von LangChain-Dokumenten und eine Liste passender IDs.

    - page_content: kombinierter Text, hier: "Title" + " " + "Review"
    - metadata: zusätzliche Infos, hilfreich fürs Debuggen/Filtern
    """
    documents = []
    ids = []

    for i, row in df.iterrows():
        # Kurzer, robuster Zusammenschnitt des Textes
        content = f"{row['Title']} {row['Review']}".strip()

        # Metadaten sauber benennen (rating statt ratin)
        metadata = {
            "rating": row["Rating"],
            "date": row["Date"],
            "source_row": int(i),    # praktisch fürs Nachverfolgen
        }

        # Document-Objekt anlegen
        doc = Document(page_content=content, metadata=metadata)

        documents.append(doc)
        ids.append(str(i))          # eindeutige ID als String

    return documents, ids


def _should_add_documents(chroma_dir: str) -> bool:
    """
    Entscheidet, ob Dokumente neu hinzugefügt werden sollen.
    Einfache Heuristik: Wenn das persistente Chroma-Verzeichnis noch nicht existiert,
    gehen wir davon aus, dass noch keine Einträge vorhanden sind.
    """
    return not os.path.exists(chroma_dir)


# --- Embeddings vorbereiten ---
# Wir verwenden OllamaEmbeddings mit dem lokalen Modell "mxbai-embed-large".
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# --- Chroma-Vektorspeicher erstellen/öffnen ---
# persist_directory sorgt dafür, dass die Daten auf der Festplatte gespeichert werden.
vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR,
)

# --- Daten hinzufügen (falls nötig) ---
if _should_add_documents(CHROMA_DIR):
    # 1) CSV laden
    dataframe = _load_dataframe(CSV_PATH)

    # 2) Documents + IDs bauen
    documents, ids = _build_documents(dataframe)

    # 3) In Chroma einfügen und persistieren
    vector_store.add_documents(documents=documents, ids=ids)
    # Persistiert den aktuellen Stand auf die Platte (wichtig!)
    vector_store.persist()

# --- Retriever exportieren ---
# as_retriever stellt eine bequeme .invoke()-Schnittstelle bereit,
# um die TOP_K ähnlichsten Dokumente für eine Anfrage abzurufen.
retriever = vector_store.as_retriever(
    search_kwargs={"k": TOP_K}
)


# --- Optionaler Testlauf ---
# Erlaube einen schnellen Plausibilitätscheck durch `python vector.py`
if __name__ == "__main__":
    print("Chroma-Setup abgeschlossen.")
    print(f"Persistenzordner: {os.path.abspath(CHROMA_DIR)}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Retriever bereit (Top-K = {TOP_K}).")
