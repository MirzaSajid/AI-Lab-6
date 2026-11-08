"""Task 2 semantic retrieval check: TF-IDF vs. a real sentence encoder.

Part A  the manual's small wiring example (two one-sentence documents).
Part B  chunks of the project corpus (sentence-aware strategy), three
        paraphrased queries, top-2 chunks per backend with chunk IDs, scores,
        offsets and whether the chunk contains the labelled evidence.
The embedding backend uses the real pretrained encoder
(sentence-transformers/all-MiniLM-L6-v2); no hand-made vectors are used. If the
model is unavailable the embedding rows are reported as NOT RUN.
Run: python task2_semantic_check.py   (add --require-embedding to make that an error)
"""
import json
import sys

from lab6_search import Retriever, show_hits
from task2_chunks import make_chunks

PARAPHRASES = ["How can I recover my login access?",
               "How should I hand in my coursework?",
               "What proof is needed when I was ill?"]


def build(records, backend):
    try:
        return Retriever(records, backend), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def main():
    require = "--require-embedding" in sys.argv
    print("== Part A: wiring example from the manual ==")
    texts = {"it": "A forgotten password can be reset using the recovery link.",
             "assessment": "Assignments are uploaded through the learning portal."}
    records = make_chunks(texts, "sentence")
    for c in records:
        assert c["text"] == texts[c["doc"]][c["start"]:c["end"]]
    print("source spans match the originals: True")
    missing = False
    for backend in ["tfidf", "embedding"]:
        engine, err = build(records, backend)
        print("Backend:", backend)
        if engine is None:
            print("NOT RUN:", err)
            missing = True
            continue
        show_hits(engine.search("How can I regain access to my account?", k=2))
    print()

    print("== Part B: project corpus, sentence-aware chunks, 3 paraphrased queries ==")
    with open("corpus.json", encoding="utf-8") as f:
        docs = json.load(f)
    with open("queries.json", encoding="utf-8") as f:
        items = {q["query"]: q for q in json.load(f)}
    chunks = make_chunks(docs, "sentence")
    for backend in ["tfidf", "embedding"]:
        engine, err = build(chunks, backend)
        print(f"--- backend: {backend} ---")
        if engine is None:
            print("NOT RUN:", err)
            missing = True
            continue
        for query in PARAPHRASES:
            q = items[query]
            print(f"query: {query!r} (expects {q['doc']})")
            print(f"  labelled evidence: {q['evidence']!r}")
            hits = engine.search(query, k=2)
            if not hits:
                print("  no matching records (no shared vocabulary)")
            for rank, (c, score) in enumerate(hits, 1):
                whole = q["evidence"] in c["text"] and c["doc"] == q["doc"]
                print(f"  {rank}. {c['id']} score {score:.3f} chars {c['start']}-{c['end']} "
                      f"evidence-complete={whole}")
                print(f"     {c['text'][:110].strip()!r}")
        print()
    if missing and require:
        print("--require-embedding was given and the model is unavailable.")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
