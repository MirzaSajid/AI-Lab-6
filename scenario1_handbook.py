"""Scenario 1: university handbook search (synthetic teaching data).
Supplied guided implementation, plus the manual's TF-IDF experiment query.
Run: python scenario1_handbook.py
"""
from lab6_search import Retriever, show_hits

records = [
    {"id": "assessment:late", "source": "handbook.txt",
     "text": "Late submission is normally penalized. Students with illness "
             "may request an extension with medical evidence before the deadline."},
    {"id": "assessment:format", "source": "handbook.txt",
     "text": "Submit the assignment as a PDF through the learning portal."}]
engine = Retriever(records)
print("Guided query: illness medical evidence late submission")
show_hits(engine.search("illness medical evidence late submission", k=1))
print()
print("Experiment (TF-IDF): I was sick; can I hand it in tomorrow?")
show_hits(engine.search("I was sick; can I hand it in tomorrow?", k=1))
