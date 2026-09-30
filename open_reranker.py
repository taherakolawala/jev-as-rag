"""Open MiniLM cross-encoder on BM25 top-100, matched to Jev query IDs."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from fastembed.rerank.cross_encoder import TextCrossEncoder
from sklearn.feature_extraction.text import CountVectorizer

from benchmark import load_data, metrics, score_bm25


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--jev-run", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--candidate-k", type=int, default=100)
    a = p.parse_args()
    docs, queries, qrels = load_data(a.data)
    ids = [d["_id"] for d in docs]
    texts = [(d.get("title", "") + ". " + d.get("text", ""))[:1600] for d in docs]
    selected = json.loads(Path(a.jev_run).read_text(encoding="utf-8"))["config"]["query_ids"]
    vectorizer = CountVectorizer(lowercase=True, token_pattern=r"(?u)\b\w\w+\b")
    counts = vectorizer.fit_transform([d.get("title", "") + " " + d.get("text", "") for d in docs]).tocsr()
    qcounts = vectorizer.transform([queries[q] for q in selected]).tocsr()
    bm25 = score_bm25(counts, qcounts)
    start = time.perf_counter()
    model = TextCrossEncoder(model_name="Xenova/ms-marco-MiniLM-L-6-v2", cache_dir=".cache/fastembed")
    model_setup_s = time.perf_counter() - start
    rows = []
    dest = Path(a.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    for i, qid in enumerate(selected, 1):
        candidates = np.argsort(-bm25(i - 1), kind="stable")[:a.candidate_k]
        start = time.perf_counter()
        scores = np.array(list(model.rerank(queries[qid], [texts[j] for j in candidates])), dtype=float)
        order = np.argsort(-scores, kind="stable")[:10]
        elapsed = (time.perf_counter() - start) * 1000
        ranked = [ids[candidates[j]] for j in order]
        row = {"query_id": qid, "ranked_ids": ranked, "latency_ms": elapsed,
               "candidate_recall": sum(ids[j] in qrels[qid] for j in candidates) / len(qrels[qid]),
               **metrics(ranked, qrels[qid])}
        rows.append(row)
        dest.write_text(json.dumps({"model": "Xenova/ms-marco-MiniLM-L-6-v2",
            "dataset": Path(a.data).name, "candidate_source": "BM25", "candidate_k": a.candidate_k,
            "model_setup_s": model_setup_s, "per_query": rows}, indent=2), encoding="utf-8")
        print(f"{i}/{len(selected)} {qid} hit5={row['hit5']} latency={elapsed:.0f}ms", flush=True)


if __name__ == "__main__":
    main()
