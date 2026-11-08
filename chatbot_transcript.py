"""Scripted replay of the FAQ chatbot for the report transcript.

This calls the SAME reply() function as guided_chatbot.py, but feeds a fixed
list of queries so the transcript is reproducible. (For a live session, run
python guided_chatbot.py.) Backend: tfidf by default; --backend embedding needs
the model.
Run: python chatbot_transcript.py
"""
import argparse

from guided_chatbot import build_bot, reply

QUERIES = [
    ("exact FAQ answer", "How do I reset my password?"),
    ("exact FAQ answer (added FAQ)", "How do I connect to the campus Wi-Fi?"),
    ("paraphrase -> suggestions", "password recovery"),
    ("paraphrase with little overlap", "I forgot my login"),
    ("empty input", ""),
    ("unsupported input", "volcano geology"),
]

parser = argparse.ArgumentParser()
parser.add_argument("--backend", choices=["tfidf", "embedding"], default="tfidf")
args = parser.parse_args()
engine = build_bot(args.backend)
print(f"FAQ assistant transcript (backend={args.backend})")
for label, query in QUERIES:
    print(f"[{label}]")
    print("You:", query)
    print("Bot:", reply(engine, query))
    print()
