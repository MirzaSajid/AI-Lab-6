"""Scenario 2: customer support knowledge base (synthetic teaching data).
Supplied guided implementation, plus the manual's experiments (no filter; paraphrase).
Run: python scenario2_support.py
"""
from lab6_search import Retriever, show_hits

records = [
    {"id": "returns:damage", "region": "PK",
     "text": "For a damaged parcel, contact support within 48 hours "
             "and attach photographs of the item and packaging."},
    {"id": "returns:change", "region": "PK",
     "text": "For a change of mind, return unused items within 14 days."},
    {"id": "returns:damage:other", "region": "OTHER",
     "text": "Damaged parcel reports in the other region have a 72 hour window."}]
engine = Retriever(records)
print("Guided query with region filter PK: damaged parcel photographs")
show_hits(engine.search("damaged parcel photographs", k=1, filters={"region": "PK"}))
print()
print("Experiment: same query WITHOUT the region filter, top 2")
show_hits(engine.search("damaged parcel photographs", k=2))
print()
print("Experiment (TF-IDF): My parcel arrived broken")
show_hits(engine.search("My parcel arrived broken", k=1))
