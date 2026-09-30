# Jev as a primary retriever

Reproducible experiments on whether a round-based Jev tournament can retrieve relevant documents from a corpus without a first-stage search index.

## Design

For each query, randomly partition the entire corpus into groups of 32 documents. Each Jev `Choice` call sees the query and the title plus the first 1,600 characters of each abstract as its options. Retain the five highest-probability documents from each group, regroup, and repeat until a final group produces the ranking. The documented `Choice` output includes a probability for every option, allowing more than one survivor per group. This experiment also tests a one-survivor variant and alternate random groupings.

The comparison methods are BM25, word/bigram TF-IDF, BGE-small dense embedding retrieval, reciprocal-rank fusion, and BM25 top-100 followed by the same Jev tournament. All retrieval metrics use the official BEIR test qrels. Indexing time is separate from query latency. Jev cost is calculated from returned input-token usage at TypeSafe's published $0.042 per million input tokens. Local baseline costs exclude hardware and electricity.

## Data

Download the official [BEIR SciFact](https://github.com/beir-cellar/beir) and [BEIR NFCorpus](https://github.com/beir-cellar/beir) ZIPs and unpack into `data/scifact` and `data/nfcorpus`. Dataset files and credentials are ignored by Git.

## Run

Use Python 3.12 and install `requirements.txt`. For local baselines:

```sh
python benchmark.py --data data/scifact --output results/baselines.json --dense
python benchmark.py --data data/nfcorpus --output results/nfcorpus_baselines.json
```

Put a TypeSafe API key in a one-line file **outside this repository**. For example, from the repository root:

```sh
python jev_tournament.py --data data/scifact --key-file ../typesafe_api_key.txt --queries 20 --max-cost 5 --output results/raw/jev_full20.json
python jev_tournament.py --data data/scifact --key-file ../typesafe_api_key.txt --queries 20 --candidate-source bm25 --candidate-k 100 --max-cost 1 --output results/raw/jev_bm25_100.json
```

`--query-seed` fixes which labeled queries are sampled. `--seed` fixes grouping and can be varied independently. Raw result files are checkpointed after every query and excluded from Git. The published `results/` files contain sanitized run records and the summary.

## Interpretation

`Hit@5` asks whether at least one judged relevant document appears in the top five. `Recall@5` is the fraction of judged relevant documents retrieved. These measure retrieval, not generated-answer quality. SciFact and NFCorpus have different qrel densities, so compare systems within each dataset. The 20-query Jev sample is exploratory; differences of a few queries have wide uncertainty.

TypeSafe's [Choice documentation](https://docs.typesafe.ai/primitives/choice) defines the API response used here. Its [launch note](https://typesafe.ai/blog/introducing-system-one-models-and-jev) gives the published per-call latency and token price. The [BEIR repository](https://github.com/beir-cellar/beir) documents the benchmark format and datasets.
