"""Guided example 6.1: whitespace vs. Unicode word-like tokens (supplied code),
plus the manual's experiment with an email, an empty input and Urdu text.
Run: python guided_tokens.py
"""
import re

text = "I can't submit CS4944 late!"
whitespace = text.split()
words = re.findall(r"\b\w+\b", text)
print("Whitespace:", whitespace, len(whitespace))
print("Word-like:", words, len(words))
print("Original:", text)

print()
print("Experiment: email, empty input and Urdu text")
for sample in ["help@example.com", "", "\u06cc\u06c1 \u0627\u06cc\u06a9 \u0645\u062b\u0627\u0644 \u06c1\u06d2"]:
    ws = sample.split()
    wd = re.findall(r"\b\w+\b", sample)
    print(repr(sample), "| whitespace:", ws, len(ws), "| word-like:", wd, len(wd))
