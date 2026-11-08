"""Scenario 5: equipment troubleshooting (synthetic teaching data).
Supplied guided implementation, plus the experiment without device metadata.
Run: python scenario5_equipment.py
"""
from lab6_search import Retriever, show_hits

records = [
    {"id": "printer:blank", "device": "printer",
     "text": "For blank printer pages: 1. Check ink level. "
             "2. Run the printer cleaning utility. "
             "3. Print a test page. Follow the device manual."},
    {"id": "scanner:blank", "device": "scanner",
     "text": "For a blank scanner image, check document placement "
             "and select the correct scanning source."}]
engine = Retriever(records)
print("Guided query with device filter printer: blank pages")
show_hits(engine.search("blank pages", k=1, filters={"device": "printer"}))
print()
print("Experiment: 'blank image' WITHOUT device metadata, top 2")
show_hits(engine.search("blank image", k=2))
print()
print("Experiment: 'blank pages' WITHOUT device metadata, top 2")
show_hits(engine.search("blank pages", k=2))
