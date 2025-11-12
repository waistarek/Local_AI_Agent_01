"""
main.py — Interaktives Q&A-Programm über Restaurantbewertungen (lokal, mit Ollama + LangChain)

Dieses Skript liest eine Nutzerfrage ein, sucht dazu passende Bewertungen im Vektorspeicher
(über den in vector.py definierten Retriever) und lässt ein lokales Sprachmodell (Ollama)
eine Antwort formulieren, die die gefundenen Bewertungen berücksichtigt.

Voraussetzungen:
- Ollama installiert + Modell 'llama3.2' verfügbar
- vector.py erstellt einen 'retriever', der Dokumente zu einer Frage zurückgibt
"""

# --- Importe ---
# OllamaLLM: Schnittstelle zu einem lokalen LLM, das über Ollama läuft (z. B. llama3.2)
from langchain_ollama.llms import OllamaLLM

# ChatPromptTemplate: Erlaubt es uns, Vorlagen (Prompts) mit Platzhaltern zu definieren
from langchain_core.prompts import ChatPromptTemplate

# 'retriever' wird in vector.py erzeugt und kümmert sich um das Auffinden passender Dokumente
from vector import retriever


# --- Modell initialisieren ---
# Wir sagen LangChain/Ollama, welches lokale Modell verwendet werden soll.
# Tipp: Stelle sicher, dass du es mit `ollama pull llama3.2` heruntergeladen hast.
llm = OllamaLLM(model="llama3.2")


# --- Prompt-Vorlage definieren ---
# Diese Vorlage beschreibt dem Modell seine Rolle und wie es antworten soll.
# {reviews} wird später mit den gefundenen Bewertungen ersetzt, {question} mit der Nutzerfrage.
PROMPT_TEMPLATE = """
You are an expert in answering questions about a pizza restaurant.

Use only the information from the provided reviews and do not invent facts.
If the reviews are not sufficient, say what is missing.

Here are some relevant reviews:
{reviews}

Here is the question to answer:
{question}
"""

# Wir bauen aus der reinen Zeichenkette eine Prompt-Template-Instanz.
prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

# --- Kette (Chain) zusammenbauen ---
# Mit dem Pipe-Operator '|' verbinden wir Prompt + Modell zu einer lauffähigen Kette.
# Eingabe: dict mit {"reviews": <Text>, "question": <Text>}
# Ausgabe: String-Antwort des Modells
qa_chain = prompt_template | llm


def _format_reviews_for_prompt(docs, max_chars: int = 1800) -> str:
    """
    Hilfsfunktion: Konvertiert eine Liste von LangChain-Dokumenten (docs) in
    einen gut lesbaren Textblock für den Prompt.

    - docs: Liste von Document-Objekten (typisch Ergebnis von retriever.invoke())
    - max_chars: Sicherheitslimit, damit der Prompt nicht zu groß wird
    """
    # Falls nichts gefunden wurde, dem Modell klar sagen, dass keine Belege vorhanden sind.
    if not docs:
        return "No matching reviews were found for this question."

    # Wir bauen eine kompakte Liste: Titel + kurzer Ausschnitt aus dem Review-Text.
    lines = []
    for idx, d in enumerate(docs, start=1):
        # .page_content enthält normalerweise den Volltext
        content = getattr(d, "page_content", str(d))
        # Kurzen Ausschnitt bilden, damit der Prompt nicht explodiert
        snippet = content.strip().replace("\n", " ")
        if len(snippet) > 400:
            snippet = snippet[:400].rstrip() + "…"
        lines.append(f"- Review #{idx}: {snippet}")

    # Zu einem Textblock zusammenfügen
    joined = "\n".join(lines)

    # Maximalgröße sicherstellen (einfacher Cutoff)
    if len(joined) > max_chars:
        joined = joined[:max_chars].rstrip() + "\n… (truncated)"

    return joined


def _is_exit_command(user_input: str) -> bool:
    """
    Prüft, ob der Nutzer beenden möchte.
    Akzeptiert 'q', 'quit', 'exit' – Groß/Kleinschreibung egal.
    """
    if not user_input:
        return False
    return user_input.strip().lower() in {"q", "quit", "exit"}


def main_loop() -> None:
    """
    Einfacher REPL (Read-Eval-Print-Loop):
    - Fragt nach einer Nutzerfrage
    - Holt passende Reviews über den Retriever
    - Lässt das Modell eine Antwort formulieren
    - Gibt die Antwort aus
    - Beenden mit q/quit/exit
    """
    print("\nWillkommen! Stelle Fragen zu den Restaurantbewertungen.")
    print("Beenden mit: q | quit | exit\n")

    while True:
        print("\n" + "#" * 55)
        user_question = input("Ask your question (q to quit): ").strip()
        print()

        # Beenden?
        if _is_exit_command(user_question):
            print("Programm beendet. Bis bald! 👋")
            break

        # Leere Eingaben überspringen (freundliche Erinnerung)
        if not user_question:
            print("Bitte gib eine Frage ein (oder 'q' zum Beenden).")
            continue

        try:
            # 1) Relevante Dokumente über semantische Suche holen
            retrieved_docs = retriever.invoke(user_question)

            # 2) Dokumente in einen gut lesbaren Prompt-Text verwandeln
            reviews_block = _format_reviews_for_prompt(retrieved_docs)

            # 3) Modell aufrufen: Prompt-Template + Platzhalter füllen
            response_text = qa_chain.invoke(
                {
                    "reviews": reviews_block,
                    "question": user_question,
                }
            )

            # 4) Antwort ausgeben
            print("Antwort:")
            print(response_text)

        except Exception as exc:
            # Fehler robust und verständlich kommunizieren,
            # damit Einsteiger wissen, woran es liegen könnte.
            print("Es ist ein Fehler aufgetreten.")
            print(f"Details: {exc}")
            print(
                "Tipps:\n"
                "- Läuft Ollama und ist das Modell 'llama3.2' installiert?\n"
                "- Existiert die CSV und wurde der Vektorspeicher durch 'vector.py' erstellt?\n"
                "- Stimmen die Spaltennamen in der CSV (Title, Review, Rating, Date)?"
            )


# --- Einstiegspunkt ---
if __name__ == "__main__":
    main_loop()
