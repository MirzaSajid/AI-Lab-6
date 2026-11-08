"""Guided example 6.3: word-window vs. sentence-aware chunks (supplied code).
Needs task2_chunks.py in the same folder.
Run: python guided_chunks.py
"""
from task2_chunks import windows, sentence_chunks

text = ("Late work is penalized. Illness may qualify for an extension "
        "when medical evidence is provided. Ask the coordinator first.")
for name, spans in [
        ("word windows", windows(text, size=12, overlap=4)),
        ("sentence aware", sentence_chunks(text, size=12))]:
    print(name)
    for start, end in spans:
        print(start, end, repr(text[start:end]))
        assert 0 <= start < end <= len(text)

print()
print("Experiment: 20-word budget")
for start, end in sentence_chunks(text, size=20):
    print(start, end, repr(text[start:end]))
