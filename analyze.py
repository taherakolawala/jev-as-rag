"""Summarize matched retrieval runs and paired Hit@5 uncertainty."""
import json
from pathlib import Path

import numpy as np

ROOT = Path("results")


def load(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def summarize(rows, latency_key):
    values = np.array([r["hit5"] for r in rows], dtype=float)
    latencies = np.array([r[latency_key] for r in rows], dtype=float)
    # A wall-clock interruption is not a representative model latency.
    normal = latencies[latencies < 60_000]
    out = {"n": len(rows), "hit5": float(values.mean()), "hits": int(values.sum()),
           "recall5": float(np.mean([r["recall5"] for r in rows])),
           "mrr10": float(np.mean([r["mrr10"] for r in rows])),
           "mean_latency_ms": float(normal.mean()), "p95_latency_ms": float(np.percentile(normal, 95)),
           "latency_interruptions": int(len(latencies) - len(normal))}
    if "cost_usd" in rows[0]:
        out.update(total_cost_usd=float(sum(r["cost_usd"] for r in rows)),
                   cost_per_query_usd=float(np.mean([r["cost_usd"] for r in rows])),
                   calls_per_query=float(np.mean([r["calls"] for r in rows])),
                   input_tokens_per_query=float(np.mean([r["input_tokens"] for r in rows])))
    return out


def paired_ci(first, second):
    common = sorted(set(first) & set(second))
    diffs = np.array([first[q]["hit5"] - second[q]["hit5"] for q in common], dtype=float)
    rng = np.random.default_rng(20260930)
    samples = rng.choice(diffs, size=(20_000, len(diffs)), replace=True).mean(axis=1)
    return {"difference": float(diffs.mean()), "ci95": [float(x) for x in np.quantile(samples, [0.025, 0.975])],
            "wins": int((diffs > 0).sum()), "losses": int((diffs < 0).sum()), "ties": int((diffs == 0).sum())}


def main():
    output = {}
    datasets = {
        "scifact": ("baselines.json", "raw/jev_full20.json", "raw/jev_bm25_100.json"),
        "nfcorpus": ("nfcorpus_baselines.json", "raw/nfcorpus_jev_full20.json", "raw/nfcorpus_jev_bm25_100.json"),
    }
    for name, (base_file, full_file, hybrid_file) in datasets.items():
        base, full, hybrid = map(load, (base_file, full_file, hybrid_file))
        selected = set(full["config"]["query_ids"])
        assert selected == set(hybrid["config"]["query_ids"])
        methods = {}
        for method in sorted({r["method"] for r in base["per_query"]}):
            rows = [r for r in base["per_query"] if r["method"] == method and r["query_id"] in selected]
            methods[method] = summarize(rows, "latency_ms")
        methods["jev_full"] = summarize(full["per_query"], "wall_ms")
        methods["bm25+jev"] = summarize(hybrid["per_query"], "wall_ms")
        open_run = ROOT / f"{name}_bm25_minilm20.json"
        if open_run.exists():
            model_rows = json.loads(open_run.read_text(encoding="utf-8"))["per_query"]
            if len(model_rows) == len(selected):
                methods["bm25+minilm"] = summarize(model_rows, "latency_ms")
        base_bm25 = {r["query_id"]: r for r in base["per_query"] if r["method"] == "bm25" and r["query_id"] in selected}
        full_rows = {r["query_id"]: r for r in full["per_query"]}
        hybrid_rows = {r["query_id"]: r for r in hybrid["per_query"]}
        output[name] = {"documents": base["documents"], "sampled_queries": len(selected),
                        "methods": methods,
                        "paired_full_minus_bm25": paired_ci(full_rows, base_bm25),
                        "paired_full_minus_hybrid": paired_ci(full_rows, hybrid_rows),
                        "candidate_recall_bm25_100": float(np.mean([r["candidate_recall"] for r in hybrid["per_query"]]))}
    variants = {}
    for name, file in (("keep1", "raw/jev_keep1_20.json"), ("keep5_seed2", "raw/jev_keep5_seed2_20.json")):
        if (ROOT / file).exists():
            data = load(file)
            if len(data["per_query"]) == 20:
                variants[name] = summarize(data["per_query"], "wall_ms")
                base_rows = {r["query_id"]: r for r in load("raw/jev_full20.json")["per_query"]}
                variants[name]["paired_minus_keep5_seed1"] = paired_ci(
                    {r["query_id"]: r for r in data["per_query"]}, base_rows)
    output["scifact_variants"] = variants
    (ROOT / "analysis.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
