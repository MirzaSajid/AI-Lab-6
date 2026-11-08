"""Guided example 6.5: Hit@3 and MRR@3 (supplied code) plus the manual's experiment.
Run: python guided_metrics.py
"""


def retrieval_metrics(rankings, relevant):
    hits, rr = [], []
    for ids, good in zip(rankings, relevant):
        ranks = [r for r, chunk_id in enumerate(ids[:3], 1)
                 if chunk_id in good]
        hits.append(bool(ranks))
        rr.append(1 / ranks[0] if ranks else 0)
    return sum(hits) / len(hits), sum(rr) / len(rr)


rankings = [["a", "b", "c"], ["c", "a", "b"], ["x", "y", "z"]]
relevant = [{"a"}, {"a"}, {"a"}]
hit, mrr = retrieval_metrics(rankings, relevant)
print(f"Hit@3={hit:.3f}; MRR@3={mrr:.3f}")
assert abs(hit - 2 / 3) < 1e-9 and abs(mrr - 0.5) < 1e-9

# Experiment: a relevant result at rank 4 must count as a miss at cutoff 3.
hit4, mrr4 = retrieval_metrics([["x", "y", "z", "a"]], [{"a"}])
print(f"Relevant item at rank 4: Hit@3={hit4:.3f}; MRR@3={mrr4:.3f}")
assert hit4 == 0 and mrr4 == 0
