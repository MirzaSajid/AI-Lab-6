"""Task 3 (advanced): evaluate document retrieval on a labelled corpus.

Compares 3 chunking strategies x 2 retrieval methods on the SAME corpus.json
and queries.json:
  strategies : "0"        40-word windows, overlap 0
               "10"       40-word windows, overlap 10
               "sentence" sentence-aware packing, 40-word budget
  methods    : "tfidf"     sparse lexical baseline
               "embedding" sentence-transformers/all-MiniLM-L6-v2 on CPU
Relevant = same document AND the chunk contains the full labelled evidence.
Hit@3 = share of answerable queries with a relevant chunk in the top 3;
MRR@3 = mean of 1/rank of the first relevant chunk (0 if none in the top 3).
Unanswerable queries (doc = null) are reported separately: they have no
relevant evidence, so no Hit/MRR is computed for them.

Dev queries choose the configuration; test queries are scored once, afterwards,
without retuning. Model load time is measured separately from encode/search time.
If the model cannot be loaded, only TF-IDF runs and every embedding row is
reported as NOT RUN (never as zero). Use --require-embedding to make that an error.

Outputs: retrieval_log.csv, outputs/task3_metrics.json and, only when both
methods ran, selected_config.json (read by document_chatbot.py).
Run: python task3_retrieval.py --require-embedding
"""
import argparse
import csv
import json
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from task2_chunks import LABELS, make_chunks

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
STRATEGIES = ["0", "10", "sentence"]
METHODS = ["tfidf", "embedding"]


def load_data():
    with open("corpus.json", encoding="utf-8") as f:
        docs = json.load(f)
    with open("queries.json", encoding="utf-8") as f:
        items = json.load(f)
    assert docs and items
    for q in items:
        assert q["split"] in {"dev", "test"}
        if q["doc"] is not None:
            assert q["evidence"] and q["evidence"] in docs[q["doc"]]
    return docs, items


def load_model():
    """Return (model, load_seconds, error). The load time is measured here only."""
    t = time.perf_counter()
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(MODEL_NAME, device="cpu")
    except Exception as exc:                       # not installed / no network
        return None, None, f"{type(exc).__name__}: {exc}"
    return model, time.perf_counter() - t, None


def score_matrix(method, texts, query_texts, model):
    t = time.perf_counter()
    if method == "tfidf":
        vectorizer = TfidfVectorizer()
        C = vectorizer.fit_transform(texts)
        Q = vectorizer.transform(query_texts)
    else:
        C = model.encode(texts, normalize_embeddings=True)
        Q = model.encode(query_texts, normalize_embeddings=True)
    encode_seconds = time.perf_counter() - t
    t = time.perf_counter()
    scores = (Q @ C.T).toarray() if method == "tfidf" else Q @ C.T
    search_seconds = time.perf_counter() - t
    return np.asarray(scores), encode_seconds, search_seconds


def top3(row):
    return np.argsort(-row, kind="stable")[:3]


def is_relevant(chunk, q):
    return (q["doc"] is not None and chunk["doc"] == q["doc"]
            and q["evidence"] in chunk["text"])


def first_relevant_rank(row, chunks, q):
    for rank, i in enumerate(top3(row), 1):
        if is_relevant(chunks[i], q):
            return rank
    return None


def evaluate(scores, chunks, items):
    hits, reciprocal = [], []
    for row, q in zip(scores, items):
        if q["doc"] is None:
            continue
        rank = first_relevant_rank(row, chunks, q)
        hits.append(rank is not None)
        reciprocal.append(1 / rank if rank else 0)
    if not hits:
        return None, None, 0
    return float(np.mean(hits)), float(np.mean(reciprocal)), len(hits)


def fmt(x):
    return "n/a" if x is None else f"{x:.3f}"


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-embedding", action="store_true",
                        help="fail instead of falling back to TF-IDF only")
    parser.add_argument("--out-dir", default=".")
    parser.add_argument("--no-top3", action="store_true",
                        help="do not print the top-3 listing for every query")
    args = parser.parse_args(argv)
    out = Path(args.out_dir)
    (out / "outputs").mkdir(parents=True, exist_ok=True)

    docs, items = load_data()
    query_texts = [q["query"] for q in items]
    n_ans = sum(q["doc"] is not None for q in items)
    print(f"corpus: {len(docs)} documents, {sum(len(t.split()) for t in docs.values())} word units")
    print(f"queries: {len(items)} ({n_ans} answerable: "
          f"{sum(bool(q['doc']) and q['split'] == 'dev' for q in items)} dev + "
          f"{sum(bool(q['doc']) and q['split'] == 'test' for q in items)} test; "
          f"{len(items) - n_ans} unanswerable)")

    model, load_seconds, error = load_model()
    if model is None:
        print("EMBEDDING METHOD NOT RUN: model unavailable (" + error + ")")
        if args.require_embedding:
            print("--require-embedding was given: stopping.")
            return 2
        methods = ["tfidf"]
    else:
        print(f"Model load seconds (measured separately): {load_seconds:.2f}")
        print(f"Model: {MODEL_NAME}; max_seq_length = {model.max_seq_length}")
        methods = METHODS

    results, logs, failures, unanswerable = [], [], [], []
    audit = {}
    tops = {}
    for strategy in STRATEGIES:
        chunks = make_chunks(docs, strategy)
        texts = [c["text"] for c in chunks]
        if model is not None:
            lengths = [len(model.tokenizer(t, truncation=False)["input_ids"])
                       for t in texts + query_texts]
            audit[strategy] = dict(max_tokens=max(lengths), limit=model.max_seq_length,
                                   n_over=sum(n > model.max_seq_length for n in lengths))
            print(f"Token audit {LABELS[strategy]}: longest input {max(lengths)} tokens "
                  f"(limit {model.max_seq_length}, special tokens included)")
            if max(lengths) > model.max_seq_length:
                raise ValueError("Model limit exceeded; rechunk or shorten query")
        for method in methods:
            scores, enc_s, srch_s = score_matrix(method, texts, query_texts, model)
            for split in ["dev", "test"]:
                idx = [i for i, q in enumerate(items) if q["split"] == split]
                hit, mrr, n = evaluate(scores[idx], chunks, [items[i] for i in idx])
                results.append(dict(strategy=strategy, method=method, split=split,
                                    n_answerable=n, hit=hit, mrr=mrr,
                                    chunks=len(chunks), encode_seconds=enc_s,
                                    search_seconds=srch_s))
            tops[(strategy, method)] = (chunks, scores)
            for row, q in zip(scores, items):
                rank1 = first_relevant_rank(row, chunks, q)
                for rank, i in enumerate(top3(row), 1):
                    c = chunks[i]
                    logs.append(dict(strategy=strategy, method=method, query=q["query"],
                                     split=q["split"], answerable=q["doc"] is not None,
                                     rank=rank, score=float(row[i]), chunk_id=c["id"],
                                     doc=c["doc"], start=c["start"], end=c["end"],
                                     relevant=(is_relevant(c, q) if q["doc"] else ""),
                                     text=c["text"]))
                best = chunks[top3(row)[0]]
                if q["doc"] is None:
                    unanswerable.append(dict(strategy=strategy, method=method,
                                             query=q["query"], top1_chunk=best["id"],
                                             top1_doc=best["doc"],
                                             top1_score=float(row[top3(row)[0]])))
                elif rank1 != 1:
                    failures.append(dict(strategy=strategy, method=method,
                                         query=q["query"], split=q["split"],
                                         expected_doc=q["doc"], first_relevant_rank=rank1,
                                         top1_chunk=best["id"], top1_doc=best["doc"],
                                         top1_score=float(row[top3(row)[0]])))

    # ---------------------------------------------------------------- results
    print()
    print("== Metrics (answerable queries only) ==")
    print("strategy | method | split | queries | Hit@3 | MRR@3 | chunks | encode s | search s")
    for r in results:
        print(LABELS[r["strategy"]], r["method"], r["split"], r["n_answerable"],
              fmt(r["hit"]), fmt(r["mrr"]), r["chunks"],
              f"{r['encode_seconds']:.4f}", f"{r['search_seconds']:.6f}", sep=" | ")
    if model is None:
        print("embedding | NOT RUN for all strategies and splits (see the message above)")

    # ----------------------------------------------------------- selection
    print()
    print("== Selection on DEVELOPMENT queries only ==")
    print("rule: highest dev MRR@3, then highest dev Hit@3, then fewest chunks, "
          "then strategy order 0, 10, sentence, then embedding before tfidf")
    dev = [r for r in results if r["split"] == "dev"]
    dev.sort(key=lambda r: (-r["mrr"], -r["hit"], r["chunks"],
                            STRATEGIES.index(r["strategy"]), METHODS.index(r["method"]) * -1))
    chosen = dev[0]
    tag = "" if model is not None else " (PROVISIONAL: lexical methods only; embedding not run)"
    print(f"selected: strategy = {LABELS[chosen['strategy']]}, method = {chosen['method']}{tag}")
    test_row = [r for r in results if r["split"] == "test"
                and r["strategy"] == chosen["strategy"] and r["method"] == chosen["method"]][0]
    print(f"dev  : Hit@3 {fmt(chosen['hit'])}, MRR@3 {fmt(chosen['mrr'])} over {chosen['n_answerable']} queries")
    print(f"test : Hit@3 {fmt(test_row['hit'])}, MRR@3 {fmt(test_row['mrr'])} over "
          f"{test_row['n_answerable']} queries (scored once, no retuning)")
    ties = [r for r in dev if r["mrr"] == chosen["mrr"] and r["hit"] == chosen["hit"]]
    print(f"configurations tied with the selected one on dev MRR and Hit: {len(ties)} of {len(dev)}")
    print("With only 4 dev and 4 test answerable queries, differences of one query "
          "move a metric by 0.25 (Hit) or more; treat the ranking as weak evidence.")

    # --------------------------------------------- manual metric arithmetic
    chunks, scores = tops[(chosen["strategy"], chosen["method"])]
    qi = next(i for i, q in enumerate(items) if q["split"] == "dev" and q["doc"])
    row, q = scores[qi], items[qi]
    print()
    print("== Metric arithmetic check on one top-3 list ==")
    print(f"configuration: {LABELS[chosen['strategy']]} / {chosen['method']}; query: {q['query']!r}")
    for rank, i in enumerate(top3(row), 1):
        print(f"  rank {rank}: {chunks[i]['id']} score {row[i]:.3f} relevant={is_relevant(chunks[i], q)}")
    r1 = first_relevant_rank(row, chunks, q)
    print("  reciprocal rank =", (f"1/{r1} = {1 / r1:.3f}" if r1 else "0 (no relevant chunk in top 3)"))

    # ---------------------------------------------------------------- top-3
    if not args.no_top3:
        print()
        print("== Top-3 results per query (text shortened to 70 characters; full text is in retrieval_log.csv) ==")
        for strategy in STRATEGIES:
            for method in methods:
                chunks, scores = tops[(strategy, method)]
                print(f"--- {LABELS[strategy]} / {method} ---")
                for row, q in zip(scores, items):
                    kind = "unanswerable" if q["doc"] is None else f"expects {q['doc']}"
                    print(f"query [{q['split']}, {kind}]: {q['query']}")
                    for rank, i in enumerate(top3(row), 1):
                        c = chunks[i]
                        mark = "-" if q["doc"] is None else ("REL" if is_relevant(c, q) else "no")
                        print(f"  {rank}. {c['id']} score {row[i]:.3f} chars {c['start']}-{c['end']} "
                              f"[{mark}] {textwrap.shorten(c['text'], 70)}")

    # ------------------------------------------------- failures / unanswerable
    print()
    print("== Failures: answerable queries whose first relevant chunk is not rank 1 ==")
    print("strategy | method | split | query | expected doc | first relevant rank | top-1 chunk | top-1 score")
    for f in failures:
        print(LABELS[f["strategy"]], f["method"], f["split"], f["query"], f["expected_doc"],
              f["first_relevant_rank"] or "none in top 3", f["top1_chunk"],
              f"{f['top1_score']:.3f}", sep=" | ")
    print()
    print("== Unanswerable queries: the highest score still returns text ==")
    print("strategy | method | query | top-1 chunk | top-1 score")
    for u in unanswerable:
        print(LABELS[u["strategy"]], u["method"], u["query"], u["top1_chunk"],
              f"{u['top1_score']:.3f}", sep=" | ")
    print("Ranking always returns the best-scoring chunks; a score measures similarity, "
          "not whether the corpus can answer the question. No rejection threshold was "
          "calibrated: the development split contains no unanswerable query, so a "
          "threshold could not be calibrated on separate development examples.")

    # ----------------------------------------------------------------- files
    with open(out / "retrieval_log.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(logs[0]))
        writer.writeheader()
        writer.writerows(logs)
    metrics = dict(model=dict(name=MODEL_NAME, available=model is not None,
                              load_seconds=load_seconds, error=error,
                              max_seq_length=getattr(model, "max_seq_length", None)),
                   token_audit=audit, results=results, failures=failures,
                   unanswerable=unanswerable,
                   selected=dict(strategy=chosen["strategy"], method=chosen["method"],
                                 provisional=model is None))
    with open(out / "outputs" / "task3_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    if model is not None:
        with open(out / "selected_config.json", "w", encoding="utf-8") as f:
            json.dump(dict(strategy=chosen["strategy"], method=chosen["method"]), f)
    print()
    print(f"wrote {out / 'retrieval_log.csv'} ({len(logs)} rows) and "
          f"{out / 'outputs' / 'task3_metrics.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
