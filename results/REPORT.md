# Jev tournament retrieval pilot — 30 September 2026

**Research audit, 1 October 2026:** These are exploratory results, not a completed paper evaluation. The NFCorpus run reused a claim-verification prompt; Choice probabilities were used as a top-five ranking without validating that interpretation; latency boundaries differ across methods; and failed calls are incompletely accounted for. See `RESEARCH_PROTOCOL.md` for the limitations and required follow-up. The new `graded_audit.json` re-evaluates the saved rankings without new model calls: on the 20 SciFact queries, full Jev Hit@10 was 20/20 versus 18/20 for BM25→Jev. This suggests potential recall gains that Hit@5 alone did not reveal, but does not establish a general advantage.

## Question and setup

Can Jev retrieve from an entire corpus by repeatedly eliminating documents in bounded groups, without a separate first-stage retriever? We tested TypeSafe `jev-1.13.0` on two BEIR test collections: SciFact (5,183 abstracts) and NFCorpus (3,633 documents). The Jev sample is 20 randomly selected labeled queries per collection. BM25 and TF-IDF were also evaluated on every labeled query (300 SciFact, 323 NFCorpus), but the tables below compare the **same 20 queries** as the Jev runs.

The primary tournament presents 32 documents per Jev Choice call, advances the five highest-probability options, randomizes surviving candidates, and repeats until a final group yields the top five. Jev sees each title plus up to 1,600 characters of document text. There is no retrieval index in this lane. A separate BM25→Jev lane starts with BM25's top 100 and applies the same tournament. The open MiniLM cross-encoder scores BM25's top 100 independently; it is a reranking comparator, not a full-corpus replacement test. Every Jev call is charged from actual returned input-token usage at the published $0.042 per million input tokens. Costs below omit local hardware.

## Matched results

| Dataset | Method | Hit@5 | Recall@5 | Mean query latency | Jev calls/query | Jev cost/query |
|---|---|---:|---:|---:|---:|---:|
| SciFact | BM25 | 15/20 | 0.725 | 1.9 ms | 0 | $0 |
| SciFact | TF-IDF | 10/20 | 0.500 | 13.9 ms | 0 | $0 |
| SciFact | BGE-small dense | 14/20 | 0.700 | 9.0 ms | 0 | $0 |
| SciFact | TF-IDF + BGE RRF | 14/20 | 0.700 | 24.0 ms | 0 | $0 |
| SciFact | BM25 top 100 → MiniLM | 13/20 | 0.625 | 9.20 s | 0 | $0 |
| SciFact | Full-corpus Jev, 5/32 | **18/20** | **0.900** | **8.67 s** | 194 | **$0.0844** |
| SciFact | BM25 top 100 → Jev | **18/20** | 0.875 | 0.77 s | 5 | $0.00170 |
| NFCorpus | BM25 | **14/20** | 0.054 | 0.3 ms | 0 | $0 |
| NFCorpus | TF-IDF | 13/20 | 0.057 | 4.6 ms | 0 | $0 |
| NFCorpus | BM25 top 100 → MiniLM | **14/20** | **0.065** | 9.06 s | 0 | $0 |
| NFCorpus | Full-corpus Jev, 5/32 | 13/20 | **0.059** | **6.33 s** | 136 | **$0.0626** |
| NFCorpus | BM25 top 100 → Jev | **14/20** | 0.057 | 0.75 s | 5 | $0.00174 |

NFCorpus has many relevant documents per query. That makes its Recall@5 values small even when Hit@5 is high; compare systems within a dataset, not raw recall across datasets. The BM25→Jev latency excludes the BM25 index build and adds only a few milliseconds for BM25 query scoring.

SciFact's full Jev lane beat BM25 by three hits in this sample, but the paired bootstrap 95% interval for the Hit@5 difference is **−5 to +35 percentage points**. It tied BM25→Jev on Hit@5, with one query won and one lost. On NFCorpus, full Jev lost one hit to BM25; its paired interval against BM25 is **−20 to +10 points**. These samples cannot establish a reliable accuracy edge. The open MiniLM cross-encoder was weaker on SciFact and tied BM25 Hit@5 on NFCorpus, though its NFCorpus Recall@5 was higher. It ran locally on CPU, so its time is not an API-hosted latency comparison.

## Tournament choices and scaling

| SciFact tournament | Hit@5 | Mean normal latency | Calls/query | Jev cost/query |
|---|---:|---:|---:|---:|
| Advance 5 of 32, grouping seed 1 | **18/20** | 8.67 s | 194 | $0.0844 |
| Advance 5 of 32, grouping seed 2 | 17/20 | 8.79 s | 194 | $0.0844 |
| Advance 1 of 32 | 15/20 | 7.61 s | 169 | $0.0735 |

One one-survivor query experienced a roughly 83-minute machine interruption. That wall-time outlier is excluded from the normal latency mean, but its accuracy and token use remain in the table. A five-survivor tournament keeps more relevant candidates alive; the one-survivor setting saves only about 13% of token cost while losing three hits on this sample.

Each full-corpus SciFact query sent roughly **2.0 million input tokens** over 194 API calls. Jev calls averaged about **449 ms** in this run, consistent with TypeSafe's published 70–500 ms per-call range. Even with 12 parallel requests, multiple rounds and the first-round scan dominate end-to-end time. Repeating a scan for every query makes both tokens and calls grow approximately linearly with corpus size. An extrapolation at similar document length is roughly **$16 per query and tens of minutes at 1 million documents** with the same 12-worker setting; this is a scaling estimate, not a measurement.

The completed Jev runs in this report plus the one-query pilot used about **$6.25** at the published token rate, below the authorized $50 cap. This is an estimate from API-reported token usage, not a billing invoice. An attempted 48-document-group variant then received HTTP 402 before completing its first query; any successful calls preceding that error are not included in this total, and no result is claimed for that variant.

## Assessment

The experiment shows that a Jev tournament **can** act as a primary retriever on collections of a few thousand documents and can outperform BM25 on some query sets. It did not show a consistent accuracy advantage across the two datasets. On SciFact, BM25→Jev matched full-corpus Jev Hit@5 at about **1/50 of the Jev cost** and **1/11 of the latency**. For large, frequently queried corpora, the full rescan is unlikely to be competitive without another way to prune or reuse work. Jev remains promising as a judge of a bounded candidate set.

These are retrieval measurements, not answer-generation accuracy. The qrels judge only known relevant documents, and only one query sample and two grouping seeds were tested. The Jev input is truncated to 1,600 characters per document; local lexical baselines use full text. Timing compares a remote Jev API with local CPU baselines, so deployment hardware and network placement can change the numbers. Dense query latency includes query encoding and vector search; its substantial one-time corpus embedding work is excluded. The dense index build crossed the machine interruption, so its recorded wall time is not a valid uninterrupted build measurement. Dense retrieval and RRF were run on SciFact only.

For a separate uninterrupted throughput check, embedding 200 SciFact documents with BGE-small took 37.6 seconds on the local CPU, or about 5.3 documents/second. That would put a fresh 5,183-document index near 16 minutes at the same throughput; actual batch scaling can differ. The Jev full-corpus lane has no index build, but pays for the scan on every query.

## Reproduction and sources

The code, run configurations, per-query rankings, token counts, costs, and local baseline outputs are in this repository. Benchmark datasets come from the [BEIR project](https://github.com/beir-cellar/beir). The Jev request format and per-option probabilities follow [TypeSafe's Choice documentation](https://docs.typesafe.ai/primitives/choice). The per-token price and published per-call latency are from [TypeSafe's launch note](https://typesafe.ai/blog/introducing-system-one-models-and-jev).
