"""Scenario 4: workplace policy retrieval (synthetic HR text).
Supplied guided implementation, plus the experiment without the audience filter.
Metadata filtering here is NOT a security or access-control system.
Run: python scenario4_workplace.py
"""
from lab6_search import Retriever, show_hits

records = [
    {"id": "remote:staff", "audience": "employee",
     "text": "Employees may request remote work with manager approval."},
    {"id": "remote:contractor", "audience": "contractor",
     "text": "Contractors may work remotely only when the signed contract "
             "explicitly permits remote work."}]
engine = Retriever(records)
print("Guided query with audience filter contractor: remote work")
show_hits(engine.search("remote work", k=1, filters={"audience": "contractor"}))
print()
print("Experiment: same query WITHOUT the filter, top 2")
show_hits(engine.search("remote work", k=2))
