"""Guided example 6.7: recommend related FAQ questions (supplied code), plus the
manual's TF-IDF experiment with a partial query. Needs guided_chatbot.py.
Run: python guided_suggestions.py   (add --backend embedding for the semantic backend)
"""
import argparse

from guided_chatbot import build_bot

parser = argparse.ArgumentParser()
parser.add_argument("--backend", choices=["tfidf", "embedding"], default="tfidf")
args = parser.parse_args()
engine = build_bot(args.backend)
for query in ["assignment upload", "forgot login"]:
    print(f"Query: {query!r} (backend={args.backend})")
    print("Suggested questions:")
    shown = 0
    for record, score in engine.search(query, k=3):
        # The score > 0 rule is for lexical overlap only; it is not a semantic cutoff.
        if args.backend == "embedding" or score > 0:
            print(" ", record["question"], f"({score:.3f})")
            shown += 1
    if not shown:
        print("  (no suggestions: no weighted words in common)")
    print()
