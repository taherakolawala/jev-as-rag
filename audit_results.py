"""Re-evaluate published rankings using graded qrels; no model calls."""
import csv
import json
import math
from pathlib import Path


def evaluate(rank, grades, k):
    if len(set(rank)) != len(rank):
        raise ValueError("Duplicate retrieved document")
    positive = {doc for doc, grade in grades.items() if grade > 0}
    dcg = sum((2 ** grades.get(doc, 0) - 1) / math.log2(i + 2)
              for i, doc in enumerate(rank[:k]))
    ideal = sum((2 ** grade - 1) / math.log2(i + 2)
                for i, grade in enumerate(sorted(grades.values(), reverse=True)[:k]))
    hits = sum(doc in positive for doc in rank[:k])
    return {f"ndcg{k}": dcg / ideal if ideal else 0,
            f"precision{k}": hits / k,
            f"recall{k}": hits / len(positive) if positive else 0,
            f"hit{k}": int(hits > 0)}


def main():
    result = {}
    for dataset, baseline_file in (("scifact", "baselines.json"), ("nfcorpus", "nfcorpus_baselines.json")):
        qrels = {}
        with Path(f"data/{dataset}/qrels/test.tsv").open(encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                qrels.setdefault(row["query-id"], {})[row["corpus-id"]] = int(row["score"])
        full = json.loads(Path(f"results/{dataset}_jev_full20.json").read_text())
        selected = set(full["config"]["query_ids"])
        rows_by_method = {"jev_full": full["per_query"]}
        for method, suffix in (("bm25+jev", "bm25_jev20"), ("bm25+minilm", "bm25_minilm20")):
            rows_by_method[method] = json.loads(Path(f"results/{dataset}_{suffix}.json").read_text())["per_query"]
        baseline = json.loads(Path(f"results/{baseline_file}").read_text())
        for row in baseline["per_query"]:
            if row["query_id"] in selected:
                rows_by_method.setdefault(row["method"], []).append(row)
        summaries, per_query = {}, {}
        for method, rows in rows_by_method.items():
            assert len(rows) == len(selected) and {r["query_id"] for r in rows} == selected
            evaluated = {r["query_id"]: {**evaluate(r["ranked_ids"], qrels[r["query_id"]], 5),
                                         **evaluate(r["ranked_ids"], qrels[r["query_id"]], 10)} for r in rows}
            for row in rows:
                assert abs(row["hit5"] - evaluated[row["query_id"]]["hit5"]) < 1e-10
                assert abs(row["recall5"] - evaluated[row["query_id"]]["recall5"]) < 1e-10
            summaries[method] = {metric: sum(r[metric] for r in evaluated.values()) / len(evaluated)
                                 for metric in next(iter(evaluated.values()))}
            per_query[method] = evaluated
        paired = {}
        for method in rows_by_method:
            if method == "jev_full":
                continue
            diffs = [per_query["jev_full"][q]["hit5"] - per_query[method][q]["hit5"] for q in sorted(selected)]
            wins, losses = diffs.count(1), diffs.count(-1)
            n = wins + losses
            p = min(1., 2 * sum(math.comb(n, j) for j in range(min(wins, losses) + 1)) / 2 ** n) if n else 1.
            paired[method] = {"wins": wins, "losses": losses, "ties": len(diffs) - n,
                              "exact_two_sided_p": p, "multiplicity_adjusted": False}
        result[dataset] = {"n": len(selected), "summary": summaries, "full_jev_vs": paired, "per_query": per_query}
    Path("results/graded_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({d: {"summary": v["summary"], "full_jev_vs": v["full_jev_vs"]} for d, v in result.items()}, indent=2))


if __name__ == "__main__":
    main()
