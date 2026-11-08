"""Task 1 (beginner): compare whitespace splitting with Unicode word-like regex
tokens on ten inputs, with the pretrained model's subword tokenizer when it is
available.

Word-like tokens come from re.findall(r"\\b\\w+\\b", text, flags=re.UNICODE). This
extracts letter/number sequences, drops punctuation and splits contractions; it
is NOT a linguistic or complete multilingual tokenizer. The encoder uses its own
subword vocabulary, so its token count can differ from both. The original text
is always printed unchanged; any normalized search copy is kept separately.
Run: python task1_tokens.py
"""
import re

SAMPLES = [
    ("punctuation", "AI, search!"),
    ("contraction (straight apostrophe)", "I can't log in."),
    ("contraction (curly apostrophe)", "I can\u2019t log in."),
    ("email address", "help@example.com"),
    ("course code", "CS4944"),
    ("identifier with digits first", "2.1 version 4944CS "),
    ("empty string", ""),
    ("Urdu text", "\u06cc\u06c1 \u0627\u06cc\u06a9 \u0645\u062b\u0627\u0644 \u06c1\u06d2"),
    ("Urdu text with full stop", "\u06cc\u06c1 \u0627\u06cc\u06a9 \u0645\u062b\u0627\u0644 \u06c1\u06d2\u06d4"),
    ("URL", "See https://example.com/help?id=42"),
]
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_model_tokenizer():
    """Return the model tokenizer, or (None, reason) if it is unavailable."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(MODEL_NAME, device="cpu")
        return model.tokenizer, None
    except Exception as exc:                      # not installed / no network
        return None, f"{type(exc).__name__}: {exc}"


def search_copy(text):
    """A normalized COPY for lexical search (the original is never altered)."""
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    words = re.findall(r"\w+", text.lower())
    return [w for w in words if w not in ENGLISH_STOP_WORDS]


def main():
    tokenizer, reason = load_model_tokenizer()
    print("== Token comparison on", len(SAMPLES), "inputs ==")
    print("(model subwords are shown only for the first two English examples)")
    print()
    for number, (label, text) in enumerate(SAMPLES, 1):
        ws = text.split()
        words = re.findall(r"\b\w+\b", text, flags=re.UNICODE)
        print(f"{number}. {label}")
        print("   original  :", repr(text))
        print("   whitespace:", ws, "count", len(ws))
        print("   word-like :", words, "count", len(words))
        if number <= 2:
            if tokenizer is not None:
                sub = tokenizer.tokenize(text)
                print("   model subwords:", sub, "count", len(sub))
            else:
                print("   model subwords: NOT AVAILABLE here (" + reason + ")")
    assert re.findall(r"\b\w+\b", "") == []

    print()
    print("== Normalization risk: a search copy can change the meaning ==")
    for text in ["I do not need a password reset", "I can't submit CS4944 late!"]:
        print("original   :", repr(text))
        print("search copy:", search_copy(text), "(lowercase, punctuation dropped, stop words removed)")
    print("Lowercasing also merges identifiers that differ only in case:",
          "CS4944" == "cs4944", "->", "CS4944".lower() == "cs4944")


if __name__ == "__main__":
    main()
