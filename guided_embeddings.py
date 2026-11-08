"""Guided example 6.2: sentence embeddings and cosine similarity (supplied code).
Needs sentence-transformers and a one-time model download (internet).
Run: python guided_embeddings.py
"""
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2", device="cpu")
passages = ["Reset a forgotten password using the recovery link.",
            "Submit assignments through the learning portal.",
            "Check ink levels when a printer produces blank pages."]
C = model.encode(passages, normalize_embeddings=True)
Q = model.encode(["How can I recover my login?"],
                 normalize_embeddings=True)
scores = (Q @ C.T)[0]
print("Embedding shape:", C.shape)
for i in np.argsort(-scores):
    print(i, round(float(scores[i]), 3), passages[i])
assert np.allclose(np.linalg.norm(C, axis=1), 1, atol=1e-5)

# Manual experiment: a negated query may still favour the password passage.
N = model.encode(["I do not need a password reset"], normalize_embeddings=True)
neg_scores = (N @ C.T)[0]
print("Negated query: I do not need a password reset")
for i in np.argsort(-neg_scores):
    print(i, round(float(neg_scores[i]), 3), passages[i])
