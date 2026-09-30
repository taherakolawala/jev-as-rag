"""Add query encoding to an older dense run that timed only vector search."""
import json
import time
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding

from benchmark import load_data


def main():
    path = Path("results/baselines.json")
    output = json.loads(path.read_text(encoding="utf-8"))
    _, queries, qrels = load_data("data/scifact")
    model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5", cache_dir=".cache/fastembed", local_files_only=True)
    encode_ms = {}
    for qid in sorted(qrels):
        start = time.perf_counter()
        list(model.query_embed([queries[qid]]))
        encode_ms[qid] = (time.perf_counter() - start) * 1000
    tfidf = {r["query_id"]: r["latency_ms"] for r in output["per_query"] if r["method"] == "tfidf"}
    dense = {}
    for row in output["per_query"]:
        if row["method"] == "bge-small":
            row["latency_ms"] += encode_ms[row["query_id"]]
            dense[row["query_id"]] = row["latency_ms"]
    for row in output["per_query"]:
        if row["method"] == "tfidf+bge-rrf":
            row["latency_ms"] += tfidf[row["query_id"]] + dense[row["query_id"]]
    for method in ("bge-small", "tfidf+bge-rrf"):
        values = [r["latency_ms"] for r in output["per_query"] if r["method"] == method]
        output["summary"][method]["latency_ms"] = round(float(np.mean(values)), 6)
        output["summary"][method]["p95_latency_ms"] = round(float(np.percentile(values, 95)), 6)
    output["indexing_seconds"]["dense_note"] = "Includes an 83-minute host interruption; do not compare as a normal index build."
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print({m: output["summary"][m] for m in ("bge-small", "tfidf+bge-rrf")})


if __name__ == "__main__":
    main()
