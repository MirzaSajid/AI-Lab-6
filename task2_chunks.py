"""Task 2 (intermediate): chunking with source offsets, boundary tests and a
comparison on the project corpus.

windows(), sentence_chunks() and make_chunks() follow the Lab 6 reference
solution; the boundary tests, corpus comparison and abbreviation demonstration
in __main__ are my own additions. Chunks are (start, end) character spans, so
text == source[start:end]. Sizes and overlaps are measured in whitespace-
separated words ("word units").

Strategies: "0" = 40-word windows, overlap 0; "10" = 40-word windows, overlap
10; "sentence" = sentence-aware packing with a 40-word budget (a sentence longer
than the budget falls back to 40-word windows). The sentence splitter is a
teaching splitter: punctuation followed by whitespace.
Run: python task2_chunks.py
"""
import json
import re


def word_spans(text):
    return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def windows(text, size=40, overlap=10, offset=0):
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("Require size > 0 and 0 <= overlap < size")
    spans = word_spans(text)
    result = []
    for i in range(0, len(spans), size - overlap):
        j = min(i + size, len(spans))
        result.append((offset + spans[i][0], offset + spans[j - 1][1]))
        if j == len(spans):
            break
    return result


def sentence_chunks(text, size=40):
    if size <= 0:
        raise ValueError("size must be positive")
    # Teaching splitter: punctuation followed by whitespace.
    cuts = [0] + [m.end() for m in
                  re.finditer(r"[.!?]\s+", text)] + [len(text)]
    out, pending, count = [], None, 0
    for a, b in zip(cuts, cuts[1:]):
        segment = text[a:b]
        n = len(segment.split())
        if not n:
            continue
        if n > size:
            if pending is not None:
                out.append(pending)
            out.extend(windows(segment, size, 0, a))
            pending, count = None, 0
        elif pending is not None and count + n <= size:
            pending, count = (pending[0], b), count + n
        else:
            if pending is not None:
                out.append(pending)
            pending, count = (a, b), n
    if pending is not None:
        out.append(pending)
    return out


def make_chunks(docs, strategy):
    result = []
    for doc_id, text in docs.items():
        spans = (sentence_chunks(text) if strategy == "sentence"
                 else windows(text, 40, int(strategy)))
        for i, (a, b) in enumerate(spans):
            result.append(dict(id=f"{doc_id}:{i}", doc=doc_id,
                               start=a, end=b, text=text[a:b]))
    return result


LABELS = {"0": "window-40, overlap 0", "10": "window-40, overlap 10",
          "sentence": "sentence-aware, budget 40"}


def boundary_tests():
    print("== Boundary tests ==")
    assert windows("") == []
    print("PASS | empty text gives no windows")
    assert sentence_chunks("") == []
    print("PASS | empty text gives no sentence chunks")
    assert windows("short text") == [(0, 10)]
    print("PASS | document shorter than the budget gives one window (0, 10)")
    assert sentence_chunks("Short text.") == [(0, 11)]
    print("PASS | document shorter than the budget gives one sentence chunk")
    for bad in [(2, 2), (2, 5), (5, -1), (0, 0)]:
        try:
            windows("a b c d", *bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid (size, overlap) {bad} was accepted")
    print("PASS | invalid size/overlap combinations raise ValueError")
    try:
        sentence_chunks("a b", 0)
    except ValueError:
        print("PASS | sentence budget 0 raises ValueError")
    long = " ".join(["word"] * 95) + "."
    spans = sentence_chunks(long)
    sizes = [len(long[a:b].split()) for a, b in spans]
    assert all(s <= 40 for s in sizes)
    print(f"PASS | a 95-word sentence falls back to {len(spans)} windows of {sizes} words")
    docs = {"example": "Reset your password. Keep recovery codes private."}
    for strategy in ["0", "10", "sentence"]:
        for c in make_chunks(docs, strategy):
            assert c["text"] == docs[c["doc"]][c["start"]:c["end"]]
    print("PASS | chunk text equals source[start:end] for all three strategies")
    print()


def corpus_comparison(docs, queries):
    print("== Corpus comparison (same 6 documents, same labelled evidence) ==")
    evidence = {q["doc"]: q["evidence"] for q in queries if q["doc"]}
    original_words = sum(len(t.split()) for t in docs.values())
    print(f"documents: {len(docs)}; original word units: {original_words}")
    print("strategy | chunks | mean words | max words | emitted words | "
          "duplicated words | evidence clauses kept whole")
    detail = {}
    for strategy in ["0", "10", "sentence"]:
        chunks = make_chunks(docs, strategy)
        sizes = [len(c["text"].split()) for c in chunks]
        for c in chunks:
            assert c["text"] == docs[c["doc"]][c["start"]:c["end"]]
            assert c["text"].strip()
        emitted = sum(sizes)
        kept = {d: any(e in c["text"] for c in chunks if c["doc"] == d)
                for d, e in evidence.items()}
        detail[strategy] = kept
        print(LABELS[strategy], len(chunks), f"{sum(sizes) / len(sizes):.1f}",
              max(sizes), emitted, emitted - original_words,
              f"{sum(kept.values())} of {len(kept)}", sep=" | ")
    print()
    print("Evidence clause kept inside ONE chunk (True) or split (False)")
    print("document | " + " | ".join(LABELS[s] for s in ["0", "10", "sentence"]))
    for d in evidence:
        print(d, *[detail[s][d] for s in ["0", "10", "sentence"]], sep=" | ")
    print()
    print("Chunks per document")
    print("document | words | " + " | ".join(LABELS[s] for s in ["0", "10", "sentence"]))
    for d, t in docs.items():
        counts = [sum(1 for c in make_chunks({d: t}, s)) for s in ["0", "10", "sentence"]]
        print(d, len(t.split()), *counts, sep=" | ")
    print()
    print("Sentences longer than the 40-word budget (use the fallback windows)")
    for d, t in docs.items():
        long_ones = [len(s.split()) for s in re.split(r"(?<=[.!?])\s+", t)
                     if len(s.split()) > 40]
        print(d, long_ones if long_ones else "none", sep=" | ")
    print()


def abbreviation_demo(docs):
    print("== Where sentence splitting fails: abbreviations ==")
    text = docs["late"]
    cuts = [0] + [m.end() for m in re.finditer(r"[.!?]\s+", text)] + [len(text)]
    for a, b in zip(cuts, cuts[1:]):
        seg = text[a:b].strip()
        if seg.endswith(("Dr.", "e.g.")):
            print(f"false cut after abbreviation at char {b}: {seg[-45:]!r}")
    print("The splitter treats 'Dr.' and 'e.g.' as sentence ends, so one real "
          "sentence is cut into pieces.")
    print()


def main():
    boundary_tests()
    with open("corpus.json", encoding="utf-8") as f:
        docs = json.load(f)
    with open("queries.json", encoding="utf-8") as f:
        queries = json.load(f)
    corpus_comparison(docs, queries)
    abbreviation_demo(docs)


if __name__ == "__main__":
    main()
