"""Validate the dataset contract for corpus.json and queries.json.

Checks: at least five documents of 150-250 words, every labelled evidence clause
is an exact substring of its document, 8 answerable queries (4 dev + 4 test) plus
2 unanswerable queries (doc and evidence null), no duplicate queries, every
document is marked synthetic and no e-mail address or phone number appears.
Run: python dataset_check.py
"""
import json
import re

with open("corpus.json", encoding="utf-8") as f:
    docs = json.load(f)
with open("queries.json", encoding="utf-8") as f:
    items = json.load(f)

print("document | words | marked synthetic | longest sentence (words)")
for d, t in docs.items():
    n = len(t.split())
    longest = max(len(s.split()) for s in re.split(r"(?<=[.!?])\s+", t))
    print(d, n, "synthetic" in t.lower(), longest, sep=" | ")
    assert 150 <= n <= 250, f"{d}: {n} words"
    assert "synthetic" in t.lower()
    assert not re.search(r"\S+@\S+|\+?\d[\d -]{8,}\d", t), "possible personal data"
assert len(docs) >= 5

answerable = [q for q in items if q["doc"] is not None]
unanswerable = [q for q in items if q["doc"] is None]
for q in answerable:
    assert q["doc"] in docs and q["evidence"] in docs[q["doc"]]
for q in unanswerable:
    assert q["evidence"] is None
assert len(answerable) == 8 and len(unanswerable) == 2
assert sum(q["split"] == "dev" for q in answerable) == 4
assert sum(q["split"] == "test" for q in answerable) == 4
assert len({q["query"] for q in items}) == len(items)
print()
print(f"{len(docs)} documents; {len(answerable)} answerable queries "
      f"({sum(q['split'] == 'dev' for q in answerable)} dev, "
      f"{sum(q['split'] == 'test' for q in answerable)} test); "
      f"{len(unanswerable)} unanswerable")
print("Dataset contract satisfied.")
