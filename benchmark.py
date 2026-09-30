"""BEIR SciFact local retrieval baselines, with per-query measurements."""
import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def load_data(root):
    ds = Path(root)
    docs = [json.loads(line) for line in (ds / "corpus.jsonl").open(encoding="utf-8")]
    queries = {x["_id"]: x["text"] for x in map(json.loads, (ds / "queries.jsonl").open(encoding="utf-8"))}
    qrels = {}
    with (ds / "qrels" / "test.tsv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if int(row["score"]) > 0:
                qrels.setdefault(row["query-id"], set()).add(row["corpus-id"])
    return docs, queries, qrels


def metrics(rank, relevant):
    hits = [i + 1 for i, docid in enumerate(rank) if docid in relevant]
    return {"hit1": int(any(i <= 1 for i in hits)), "hit5": int(any(i <= 5 for i in hits)),
            "recall5": sum(i <= 5 for i in hits) / len(relevant),
            "recall10": sum(i <= 10 for i in hits) / len(relevant),
            "mrr10": 1 / min((i for i in hits if i <= 10), default=math.inf)}


def score_bm25(counts, qcounts, k1=1.2, b=0.75):
    n = counts.shape[0]
    df = np.asarray((counts > 0).sum(axis=0)).ravel()
    idf = np.log(1 + (n - df + 0.5) / (df + 0.5))
    lens = np.asarray(counts.sum(axis=1)).ravel()
    avg_len = lens.mean()
    norm = k1 * (1 - b + b * lens / avg_len)
    csc = counts.tocsc()
    def query(i):
        scores = np.zeros(n)
        terms = qcounts[i].indices
        for term in terms:
            col = csc.getcol(term)
            rows = col.indices
            tf = col.data
            scores[rows] += idf[term] * (tf * (k1 + 1) / (tf + norm[rows]))
        return scores
    return query


def rank_and_record(name, scorer, docids, queryids, qrels, records):
    for qi, qid in enumerate(queryids):
        start = time.perf_counter()
        scores = np.asarray(scorer(qi)).ravel()
        order = np.argsort(-scores, kind="stable")[:10]
        elapsed = (time.perf_counter() - start) * 1000
        rank = [docids[x] for x in order]
        records.append({"method": name, "query_id": qid, "latency_ms": elapsed,
                        "ranked_ids": rank, **metrics(rank, qrels[qid])})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/scifact")
    p.add_argument("--output", default="results/baselines.json")
    p.add_argument("--dense", action="store_true")
    p.add_argument("--dense-cache", default=".cache/fastembed")
    a = p.parse_args()
    docs, queries, qrels = load_data(a.data)
    queryids = sorted(qrels)
    texts = [(d.get("title", "") + " " + d.get("text", "")).strip() for d in docs]
    docids = [d["_id"] for d in docs]
    records, indexing = [], {}

    start = time.perf_counter()
    count_vectorizer = CountVectorizer(lowercase=True, token_pattern=r"(?u)\b\w\w+\b")
    counts = count_vectorizer.fit_transform(texts).tocsr()
    qcounts = count_vectorizer.transform([queries[q] for q in queryids]).tocsr()
    bm25 = score_bm25(counts, qcounts)
    indexing["bm25_s"] = time.perf_counter() - start
    rank_and_record("bm25", bm25, docids, queryids, qrels, records)

    start = time.perf_counter()
    tfidf_vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1)
    tfidf = tfidf_vectorizer.fit_transform(texts)
    qtfidf = tfidf_vectorizer.transform([queries[q] for q in queryids])
    indexing["tfidf_s"] = time.perf_counter() - start
    tfidf_scores = []
    def tfidf_scorer(i):
        scores = (tfidf @ qtfidf[i].T).toarray().ravel()
        tfidf_scores.append(scores)
        return scores
    rank_and_record("tfidf", tfidf_scorer, docids, queryids, qrels, records)

    if a.dense:
        from fastembed import TextEmbedding
        start = time.perf_counter()
        model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5", cache_dir=a.dense_cache)
        vecs = np.array(list(model.embed(texts, batch_size=64)), dtype=np.float32)
        qvecs = np.array(list(model.query_embed([queries[q] for q in queryids])), dtype=np.float32)
        vecs /= np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12)
        qvecs /= np.maximum(np.linalg.norm(qvecs, axis=1, keepdims=True), 1e-12)
        indexing["dense_s"] = time.perf_counter() - start
        dense_scores = []
        def dense_scorer(i):
            scores = vecs @ qvecs[i]
            dense_scores.append(scores)
            return scores
        rank_and_record("bge-small", dense_scorer, docids, queryids, qrels, records)
        for qi, qid in enumerate(queryids):
            start = time.perf_counter()
            b_rank = np.argsort(-tfidf_scores[qi], kind="stable")
            d_rank = np.argsort(-dense_scores[qi], kind="stable")
            scores = np.empty(len(docids))
            scores[b_rank] = 1 / (60 + np.arange(1, len(docids) + 1))
            scores[d_rank] += 1 / (60 + np.arange(1, len(docids) + 1))
            order = np.argsort(-scores, kind="stable")[:10]
            elapsed = (time.perf_counter() - start) * 1000
            rank = [docids[x] for x in order]
            records.append({"method": "tfidf+bge-rrf", "query_id": qid, "latency_ms": elapsed,
                            "ranked_ids": rank, **metrics(rank, qrels[qid])})

    summary = {}
    for method in sorted({r["method"] for r in records}):
        rows = [r for r in records if r["method"] == method]
        summary[method] = {k: round(float(np.mean([r[k] for r in rows])), 6)
                           for k in ("hit1", "hit5", "recall5", "recall10", "mrr10", "latency_ms")}
        summary[method]["p95_latency_ms"] = round(float(np.percentile([r["latency_ms"] for r in rows], 95)), 6)
    output = {"dataset": f"BEIR {Path(a.data).name} test", "documents": len(docs), "queries": len(queryids),
              "indexing_seconds": indexing, "summary": summary, "per_query": records}
    path = Path(a.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in output.items() if k != "per_query"}, indent=2))


if __name__ == "__main__":
    main()
