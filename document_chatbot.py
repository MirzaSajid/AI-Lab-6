"""Task 3 interactive document chatbot (Lab 6 reference, extended).

Retrieves candidate evidence from corpus.json and shows ranked source IDs and
character offsets. The user must choose a candidate before its exact passage is
displayed; pressing Enter or an invalid choice gives the fallback. A candidate
is never presented as an independently verified answer, and unanswerable
queries are not answered automatically.

Configuration: the strategy/method selected on development data by
task3_retrieval.py is read from selected_config.json when it exists; otherwise
"sentence"/"embedding" (the reference default). Override with --strategy/--backend
("0", "10" or "sentence"; "tfidf" or "embedding"). Input is echoed when it is
piped, so a scripted session gives a readable transcript.
Run: python document_chatbot.py
"""
import argparse
import json
import sys
from pathlib import Path

from lab6_search import Retriever
from task2_chunks import make_chunks


def ask(prompt):
    value = input(prompt)
    if not sys.stdin.isatty():
        print(value)
    return value


def default_config():
    path = Path("selected_config.json")
    if path.exists():
        cfg = json.loads(path.read_text(encoding="utf-8"))
        return cfg["strategy"], cfg["method"]
    return "sentence", "embedding"


if __name__ == "__main__":
    d_strategy, d_backend = default_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=["0", "10", "sentence"], default=d_strategy)
    parser.add_argument("--backend", choices=["tfidf", "embedding"], default=d_backend)
    args = parser.parse_args()
    with open("corpus.json", encoding="utf-8") as f:
        docs = json.load(f)
    engine = Retriever(make_chunks(docs, args.strategy), args.backend)
    print(f"Evidence assistant (strategy={args.strategy}, backend={args.backend}). "
          "Type exit to stop.")
    while True:
        try:
            query = ask("Question: ")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() == "exit":
            break
        hits = engine.search(query)
        if not hits:
            print("No candidate evidence; rephrase or contact a human.")
            continue
        print("Candidate evidence; relevance must be checked:")
        for number, (r, score) in enumerate(hits, 1):
            print(number, r["id"], f"{score:.3f}", "characters", r["start"], r["end"])
        chosen = ask("Choose 1-3 to inspect; Enter if none apply: ")
        if not chosen.isdigit() or not 1 <= int(chosen) <= len(hits):
            print("No answer selected. Rephrase or contact a human.")
            continue
        r, score = hits[int(chosen) - 1]
        print("Retrieved passage:", r["text"])
        print("Source:", r["doc"], "characters", r["start"], r["end"])
        print("This is retrieved text to inspect, not a verified answer.")
