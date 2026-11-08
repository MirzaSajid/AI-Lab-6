"""Guided example 6.6: command-line FAQ chatbot (supplied code).

Task 1 adaptation: two FAQs were ADDED (faq:wifi, faq:library), each with a
source. Everything else is the supplied code. All FAQ text is synthetic.
Only the exact canonical question is answered automatically; anything else
returns candidates for the user to confirm.
Run: python guided_chatbot.py   (add --backend embedding for the semantic backend)
"""
from lab6_search import Retriever

FAQ = [
    {"id": "faq:password", "question": "How do I reset my password?",
     "answer": "Use the recovery link on the login page.",
     "source": "it_help.txt"},
    {"id": "faq:submit", "question": "Where do I submit my assignment?",
     "answer": "Use the assignment upload page in the learning portal.",
     "source": "assessment.txt"},
    {"id": "faq:late", "question": "Can illness justify a late submission?",
     "answer": "Request an extension with medical evidence before the deadline.",
     "source": "handbook.txt"},
    # --- added for Task 1 (synthetic) ---
    {"id": "faq:wifi", "question": "How do I connect to the campus Wi-Fi?",
     "answer": "Select the campus network and sign in with your student ID.",
     "source": "it_help.txt"},
    {"id": "faq:library", "question": "What are the library opening hours?",
     "answer": "The library is open from 8 am to 8 pm on weekdays.",
     "source": "library.txt"}]


def build_bot(backend="tfidf"):
    records = [dict(r, text=r["question"] + " " + r["answer"]) for r in FAQ]
    return Retriever(records, backend)


def reply(engine, query):
    if not query.strip():
        return "Please enter a question."
    # Only exact canonical FAQ text is answered automatically here.
    for r in FAQ:
        if query.strip().casefold() == r["question"].casefold():
            return r["answer"] + " [source: " + r["source"] + "]"
    hits = engine.search(query)
    if not hits:
        return "No matching FAQ. Please contact the help desk."
    lines = ["Possible matches; choose or rephrase your question:"]
    for r, score in hits:
        lines.append(r["question"] + f" (similarity {score:.3f})")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["tfidf", "embedding"],
                        default="tfidf")
    args = parser.parse_args()
    engine = build_bot(args.backend)
    print("FAQ assistant. Type exit to stop.")
    while True:
        try:
            query = input("You: ")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() == "exit":
            break
        print("Bot:", reply(engine, query))
