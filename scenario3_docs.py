"""Scenario 3: software documentation search (synthetic teaching data).
Supplied guided implementation, plus the manual's filter experiments.
No API is called; this only searches documentation text.
Run: python scenario3_docs.py
"""
from lab6_search import Retriever, show_hits

records = [
    {"id": "auth:v2", "version": "v2",
     "text": "Renew an expired API token with POST /v2/token/refresh. "
             "Send the refresh_token field; never print its value in logs."},
    {"id": "auth:v1", "version": "v1",
     "text": "Renew an expired API token with POST /v1/auth/renew."}]
engine = Retriever(records)
for version in ("v2", "v1", "v9"):
    print(f"Query: renew expired API token, filter version={version}")
    show_hits(engine.search("renew expired API token", k=1,
                            filters={"version": version}))
    print()
