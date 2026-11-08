# Lab 6 - NLP Fundamentals: Tokenization, Embeddings & Text Chunking

**Student:** Sajid Hussain (BSAIM-046) | **Course:** CAI601400 - Artificial Intelligence | **Lab:** 6 (CLO3)

A traceable text-retrieval pipeline for a university help desk: token comparison,
a FAQ chatbot, word-window and sentence-aware chunking with source offsets, and a
retrieval comparison of TF-IDF against real sentence embeddings
(`sentence-transformers/all-MiniLM-L6-v2`, CPU) on a labelled corpus.
**All documents, rules and FAQs are synthetic teaching data; they are not real
university policies.** A retrieved passage is evidence to inspect, not a verified answer.

## Setup (Python 3.10+ recommended)
```bash
python -m venv .venv
# Windows PowerShell:  .venv\Scripts\Activate.ps1
# macOS / Linux:       source .venv/bin/activate
python -m pip install numpy scikit-learn sentence-transformers
python -m pip freeze > requirements.txt      # replace the minimum-version file with your exact versions
```
The first run of anything that uses the encoder downloads the model from the public
Hugging Face hub (internet needed once; later runs use the local cache, which is **not**
committed). No GPU and no paid API are needed. If you redirect output to a file on
Windows, use `python -X utf8 script.py > out.txt` (the token examples print Urdu text).

## Run everything once (recommended)
```bash
python run_all.py --freeze
```
This runs every script below, saves each output to `outputs/<name>.txt`, writes
`retrieval_log.csv`, `outputs/task3_metrics.json`, `selected_config.json` and
`requirements.txt`. Steps that need the encoder report a clear error if it is missing.

## Commands by task
**Guided scenarios and examples**
```bash
python lab6_search.py                   # shared retriever (TF-IDF or embedding backend)
python scenario1_handbook.py            # ... through scenario5_equipment.py
python guided_tokens.py                 # 6.1 tokens
python guided_embeddings.py             # 6.2 embeddings (needs the model)
python guided_chunks.py                 # 6.3 chunks (uses task2_chunks.py)
python guided_limits.py                 # 6.4 model token limit (needs the model)
python guided_metrics.py                # 6.5 Hit@3 / MRR@3 arithmetic
python guided_chatbot.py                # 6.6 FAQ chatbot, live (add --backend embedding)
python guided_suggestions.py            # 6.7 related questions (add --backend embedding)
```
**Task 1 (beginner): tokenization and FAQ chatbot**
```bash
python task1_tokens.py                  # 10 inputs, whitespace vs word-like vs model subwords
python task1_chatbot_check.py           # assertions for the FAQ chatbot (lexical backend)
python chatbot_transcript.py            # scripted replay of reply() for the report
```
**Task 2 (intermediate): chunking and semantic suggestions**
```bash
python task2_chunks.py                  # boundary tests + 3-strategy comparison on corpus.json
python task2_semantic_check.py          # TF-IDF vs embedding on 3 paraphrases (needs the model)
```
**Task 3 (advanced): evaluated retrieval and document chatbot**
```bash
python dataset_check.py                 # validates corpus.json / queries.json
python task3_retrieval.py --require-embedding   # 3 strategies x 2 methods; writes retrieval_log.csv
python document_chatbot.py              # interactive; uses selected_config.json when present
```
`task3_retrieval.py` without `--require-embedding` falls back to TF-IDF only and prints
**NOT RUN** for every embedding row (never zeros); that output is not a complete result.

## Dataset
`corpus.json`: 6 synthetic documents (184-210 words each; one rule with an exception,
long sentences, abbreviations, and related but conflicting passages such as student vs.
staff password reset). `queries.json`: 8 answerable queries (4 `dev`, 4 `test`) with an
exact evidence clause plus 2 unanswerable queries (`doc` and `evidence` are `null`).
Evidence was labelled before any ranking was inspected. `dataset_check.py` verifies this.
The manual's starter `create_dataset.py` was replaced by this final expanded dataset.

## Conventions
Chunk sizes are in whitespace-separated words; a chunk is a `(start, end)` character span
so `text == source[start:end]`. A chunk is **relevant** only if it comes from the labelled
document AND contains the full evidence clause. Strategies: `0` = 40-word windows with
overlap 0; `10` = 40-word windows with overlap 10; `sentence` = sentence-aware packing with
a 40-word budget. The development queries choose the configuration; the test queries are
scored once afterwards without retuning. Timing: model load is measured separately from
encode and search time (single runs; tiny corpora make timings noisy).

## Files
| File | Purpose |
|---|---|
| `lab6_search.py` | Shared `Retriever` (TF-IDF / embedding) - supplied code |
| `scenario1_handbook.py` ... `scenario5_equipment.py` | Guided scenarios + the manual's experiments |
| `guided_*.py` | Guided examples 6.1-6.7 (supplied code; small additions noted in each file) |
| `task1_tokens.py`, `task1_chatbot_check.py`, `chatbot_transcript.py` | Task 1 |
| `task2_chunks.py`, `task2_semantic_check.py` | Task 2 |
| `task3_retrieval.py`, `document_chatbot.py`, `doc_chatbot_script.txt` | Task 3 |
| `corpus.json`, `queries.json`, `dataset_check.py` | Dataset and validation |
| `retrieval_log.csv` | Top-3 log: strategy, method, query, split, rank, score, chunk ID, offsets, relevance, text |
| `run_all.py`, `requirements.txt` | Reproduction |
| `outputs/` | Captured outputs of every script |
| `Lab6_Report.docx` | Report |

## Sources and attribution
Supplied code from the Lab 6 manual: `lab6_search.py`, the five scenario scripts, the
`guided_*.py` examples, the Task 2 chunking reference (`windows`, `sentence_chunks`,
`make_chunks`), the Task 3 evaluation logic and the chatbot reference. My own work: the
expanded dataset, `task1_tokens.py`, the boundary tests and comparison in `task2_chunks.py`,
the extended `task3_retrieval.py` (selection rule, failure and unanswerable reporting,
JSON/CSV logging, NOT RUN handling), `document_chatbot.py` options, `run_all.py` and
`dataset_check.py`. Model: `sentence-transformers/all-MiniLM-L6-v2` (public model card:
https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). No model weights, virtual
environment, credentials or personal data are included.
