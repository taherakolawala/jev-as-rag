"""Full-corpus Jev Choice tournament on BEIR SciFact.

Each round partitions remaining documents into random groups, obtains Choice
probabilities for each group, and retains the top ``keep`` in each group.
The final group yields the ranking. API usage and wall times are recorded.
"""
import argparse
import concurrent.futures
import json
import math
import os
import random
import threading
import time
from pathlib import Path

import requests
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from benchmark import load_data, metrics, score_bm25

API = "https://api.typesafe.ai/v1/systemone"
PRICE_PER_M = 0.042


def chunks(seq, n):
    return [seq[i:i + n] for i in range(0, len(seq), n)]


class JevClient:
    def __init__(self, key, max_cost):
        self.key = key
        self.max_cost = max_cost
        self.cost = 0.0
        self.lock = threading.Lock()

    def rank_group(self, query, group, docs):
        criteria = {
            f"d{i}": (docs[docid].get("title", "") + ". " + docs[docid].get("text", ""))[:1600]
            for i, docid in enumerate(group)
        }
        payload = {"model": "jev-1.13.0", "state": {"claim": query}, "questions": {
            "best": {"type": "choice", "instructions":
                     "Which scientific abstract is most useful as evidence to verify or refute the claim? "
                     "Choose the most directly relevant abstract, not just a related topic.",
                     "criteria": criteria}}}
        # Conservative preflight based on characters, including schema overhead.
        estimated_cost = (len(json.dumps(payload)) / 3 + 3000) / 1_000_000 * PRICE_PER_M
        with self.lock:
            if self.cost + estimated_cost > self.max_cost:
                raise RuntimeError("Spend limit reached before request")
            self.cost += estimated_cost
        headers = {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"}
        for attempt in range(5):
            start = time.perf_counter()
            try:
                response = requests.post(API, json=payload, headers=headers, timeout=45)
                latency = (time.perf_counter() - start) * 1000
                if response.status_code in (429, 500, 502, 503, 504):
                    time.sleep(min(2 ** attempt, 12))
                    continue
                response.raise_for_status()
                body = response.json()
                answer = body["answers"]["best"]
                probabilities = answer["probabilities"]
                ranked = sorted(range(len(group)), key=lambda i: -probabilities.get(f"d{i}", 0.0))
                usage = body.get("usage", {})
                actual_cost = usage.get("input_tokens", 0) / 1_000_000 * PRICE_PER_M
                with self.lock:
                    self.cost += actual_cost - estimated_cost
                return [group[i] for i in ranked], {"latency_ms": latency,
                    "input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens"),
                    "cost_usd": actual_cost, "model": body.get("model"), "group_size": len(group)}
            except requests.RequestException:
                if attempt == 4:
                    raise
                time.sleep(min(2 ** attempt, 12))
        raise RuntimeError(f"Jev API failed after retries: HTTP {response.status_code}: {response.text[:200]}")


def tournament(client, query, ids, docs, group_size, keep, workers, seed):
    rng = random.Random(seed)
    current = list(ids)
    calls = []
    wall_start = time.perf_counter()
    while len(current) > group_size:
        rng.shuffle(current)
        groups = chunks(current, group_size)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            pairs = list(pool.map(lambda g: client.rank_group(query, g, docs), groups))
        current = [docid for ranking, info in pairs for docid in ranking[:keep]]
        calls.extend(info for _, info in pairs)
    ranking, info = client.rank_group(query, current, docs)
    calls.append(info)
    return ranking, calls, (time.perf_counter() - wall_start) * 1000


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/scifact")
    p.add_argument("--key-file", required=True)
    p.add_argument("--output", default="results/raw/jev_full.json")
    p.add_argument("--queries", type=int, default=20)
    p.add_argument("--group-size", type=int, default=32)
    p.add_argument("--keep", type=int, default=5)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--seed", type=int, default=20260930)
    p.add_argument("--max-cost", type=float, default=5.0)
    p.add_argument("--candidate-source", choices=("all", "bm25"), default="all")
    p.add_argument("--candidate-k", type=int, default=100)
    a = p.parse_args()
    if a.keep >= a.group_size or a.keep < 1:
        p.error("keep must be between 1 and group-size - 1")
    key = Path(a.key_file).read_text(encoding="utf-8").strip()
    if not key:
        p.error("key file is empty")
    docs_list, queries, qrels = load_data(a.data)
    docs = {d["_id"]: d for d in docs_list}
    ids = list(docs)
    selected = random.Random(a.seed).sample(sorted(qrels), min(a.queries, len(qrels)))
    bm25_candidates = None
    if a.candidate_source == "bm25":
        texts = [(d.get("title", "") + " " + d.get("text", "")).strip() for d in docs_list]
        vectorizer = CountVectorizer(lowercase=True, token_pattern=r"(?u)\b\w\w+\b")
        counts = vectorizer.fit_transform(texts).tocsr()
        qcounts = vectorizer.transform([queries[q] for q in selected]).tocsr()
        scorer = score_bm25(counts, qcounts)
        bm25_candidates = {qid: [ids[j] for j in np.argsort(-scorer(i), kind="stable")[:a.candidate_k]]
                           for i, qid in enumerate(selected)}
    client = JevClient(key, a.max_cost)
    result = {"config": {"dataset": "BEIR SciFact test", "model": "jev-1.13.0",
              "query_ids": selected, "group_size": a.group_size, "keep": a.keep,
              "workers": a.workers, "seed": a.seed, "candidate_source": a.candidate_source,
              "candidate_k": a.candidate_k if bm25_candidates else len(ids)}, "per_query": []}
    dest = Path(a.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    for number, qid in enumerate(selected, 1):
        candidate_ids = bm25_candidates[qid] if bm25_candidates else ids
        ranking, calls, elapsed = tournament(client, queries[qid], candidate_ids, docs,
            a.group_size, a.keep, a.workers, a.seed + number)
        row = {"query_id": qid, "ranked_ids": ranking[:10], "wall_ms": elapsed,
               "calls": len(calls), "input_tokens": sum(c["input_tokens"] or 0 for c in calls),
               "cost_usd": sum(c["cost_usd"] for c in calls),
               "mean_call_ms": sum(c["latency_ms"] for c in calls) / len(calls),
               "resolved_models": sorted({c["model"] for c in calls}),
               "candidate_recall": sum(x in qrels[qid] for x in candidate_ids) / len(qrels[qid]),
               **metrics(ranking[:10], qrels[qid])}
        result["per_query"].append(row)
        dest.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"{number}/{len(selected)} {qid} hit5={row['hit5']} "
              f"wall={elapsed/1000:.1f}s calls={len(calls)} cost=${row['cost_usd']:.4f}", flush=True)
    print(f"Total billed estimate: ${client.cost:.4f}")


if __name__ == "__main__":
    main()
