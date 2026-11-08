"""Task 1 FAQ chatbot checks (Lab 6 reference assertions, lexical backend only).
The assertions do not cover the semantic backend: semantic retrieval can still
return unrelated candidates for unrelated queries.
Run: python task1_chatbot_check.py
"""
from guided_chatbot import build_bot, reply

engine = build_bot()
assert "it_help.txt" in reply(engine, "How do I reset my password?")
assert reply(engine, "") == "Please enter a question."
assert "No matching FAQ" in reply(engine, "volcano geology")
assert "Possible matches" in reply(engine, "password recovery")
# Checks for the two FAQs added in Task 1 (exact answers carry their sources).
assert "it_help.txt" in reply(engine, "How do I connect to the campus Wi-Fi?")
assert "library.txt" in reply(engine, "what are the library opening hours?")
print("FAQ behaviour checks passed")
