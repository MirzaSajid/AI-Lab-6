"""Run every Lab 6 script and save each output to outputs/.

Run this ONCE on your own machine, inside the virtual environment where numpy,
scikit-learn and sentence-transformers are installed (the first run downloads
the encoder). Scripts that need the encoder record a clear error if it is
missing; nothing is invented.

    python run_all.py --freeze     # also writes requirements.txt (pip freeze)

Outputs are written to outputs/<name>.txt; task3_retrieval.py also writes
retrieval_log.csv, outputs/task3_metrics.json and selected_config.json.
Run: python run_all.py
"""
import os
import subprocess
import sys
from pathlib import Path

# (output name, script and arguments, optional stdin file)
STEPS = [
    ("dataset_check_output", ["dataset_check.py"], None),
    ("lab6_search_output", ["lab6_search.py"], None),
    ("scenario1_handbook_output", ["scenario1_handbook.py"], None),
    ("scenario2_support_output", ["scenario2_support.py"], None),
    ("scenario3_docs_output", ["scenario3_docs.py"], None),
    ("scenario4_workplace_output", ["scenario4_workplace.py"], None),
    ("scenario5_equipment_output", ["scenario5_equipment.py"], None),
    ("guided_tokens_output", ["guided_tokens.py"], None),
    ("guided_embeddings_output", ["guided_embeddings.py"], None),
    ("guided_chunks_output", ["guided_chunks.py"], None),
    ("guided_limits_output", ["guided_limits.py"], None),
    ("guided_metrics_output", ["guided_metrics.py"], None),
    ("guided_suggestions_output", ["guided_suggestions.py"], None),
    ("guided_suggestions_embedding_output", ["guided_suggestions.py", "--backend", "embedding"], None),
    ("task1_tokens_output", ["task1_tokens.py"], None),
    ("task1_chatbot_check_output", ["task1_chatbot_check.py"], None),
    ("chatbot_transcript_tfidf", ["chatbot_transcript.py"], None),
    ("chatbot_transcript_embedding", ["chatbot_transcript.py", "--backend", "embedding"], None),
    ("task2_chunks_output", ["task2_chunks.py"], None),
    ("task2_semantic_output", ["task2_semantic_check.py"], None),
    ("task3_output", ["task3_retrieval.py", "--require-embedding"], None),
    ("document_chatbot_transcript_tfidf",
     ["document_chatbot.py", "--backend", "tfidf", "--strategy", "sentence"], "doc_chatbot_script.txt"),
    ("document_chatbot_transcript_embedding", ["document_chatbot.py"], "doc_chatbot_script.txt"),
]


def main():
    Path("outputs").mkdir(exist_ok=True)
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    failed = []
    for name, cmd, stdin_file in STEPS:
        stdin = open(stdin_file, encoding="utf-8").read() if stdin_file else None
        proc = subprocess.run([sys.executable, *cmd], capture_output=True, text=True,
                              encoding="utf-8", env=env, input=stdin)
        text = proc.stdout + (("\n[stderr]\n" + proc.stderr) if proc.returncode else "")
        Path("outputs", name + ".txt").write_text(text, encoding="utf-8")
        status = "ok" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
        print(f"{status:>16} | {' '.join(cmd)}")
        if proc.returncode:
            failed.append(" ".join(cmd))
    if "--freeze" in sys.argv:
        frozen = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                                capture_output=True, text=True).stdout
        Path("requirements.txt").write_text(frozen, encoding="utf-8")
        print("requirements.txt written (pip freeze)")
    print()
    print("All steps succeeded." if not failed else
          "Steps that failed (see their output files): " + "; ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
