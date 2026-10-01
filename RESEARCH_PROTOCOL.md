# Research status and next experiment protocol

Audit date: 2026-10-01. The existing results are exploratory pilot results. No new model runs were performed for this audit.

## What the pilot establishes

There are 40 unique Jev-evaluated queries: 20 SciFact and 20 NFCorpus. Repeated runs on the same queries are not additional independent questions. The full-corpus method is feasible on these collections, but the sample is too small and the setup too limited to establish a general advantage. It replaces retrieval; the pilot did not test generated answers or replacement of the whole RAG pipeline.

## Issues to resolve before confirmatory testing

- **Prompt fit:** The same scientific claim-verification prompt was used for NFCorpus search questions. Use an appropriate relevance prompt per task, frozen using development data. The existing NFCorpus results must retain this caveat.
- **Choice versus relevance:** Choice probabilities concern which single option is best. The top five are not necessarily the five relevant documents. Compare Choice tournaments against independent per-document relevance decisions and flat full-corpus scoring using the same model.
- **Input parity:** Jev sees the first 1,600 characters of title plus text. Lexical retrieval sees full text; dense models have their own truncation. Compare a controlled equal-evidence track and a practical best-configuration track. Log every truncation and budget batches by tokens.
- **Sample size and tuning:** Only 20 queries per dataset were tested with Jev. Freeze settings on development queries; treat all existing pilot queries as already inspected. Evaluate held-out queries across additional domains and multiple corpus sizes. Set sample size from the smallest useful effect and paired power estimates before running.
- **Baselines:** The existing BM25 and TF-IDF are simple implementations. Dense retrieval used BGE-small; the fusion used TF-IDF plus dense rather than the common BM25 plus dense combination. Add validated lexical, dense, hybrid and reranking baselines with documented configurations. Laya has not been tested or verified for compatibility.
- **Ranking quality:** Retain graded qrels and evaluate nDCG@10, Recall@5/10/100, precision, and Hit@5. Hit@5 alone can hide loss of supporting evidence. Do not silently treat unjudged documents as confirmed irrelevant in error analysis.
- **Latency fairness:** Lexical query preprocessing and hybrid candidate retrieval were excluded from some timers. Local CPU and remote API performance are different deployment conditions. Rerun query-to-results timing, with idle hardware, recorded thread counts, warmup, repeated measurements, median/p95, and request/query concurrency. Report index building and updates separately.
- **Interruptions:** The existing analysis drops all latencies above 60 seconds. Replace this generic cutoff with explicitly identified interruptions; publish raw, inclusive and justified exclusion results. Do not exclude genuine slow requests.
- **Accounting:** Existing logs aggregate successful calls only after a completed query. Failed/in-flight requests and retries can be missing from cost. Add a durable per-attempt ledger, cumulative budget reservations, response validation and a global budget cap. Stop promptly on authentication/billing errors. Missing usage is unknown, not zero. Pricing-derived cost is distinct from billed cost.
- **Reproducibility:** Save package/model revisions, hardware, dataset hashes, prompts, grouping membership, probabilities, survivor lists and round timings. The current aggregate logs cannot explain exactly which round discarded evidence. Analysis must run from published files rather than ignored raw paths.

## Planned comparisons

1. Freeze output k and task prompts using development data.
2. Compare full-corpus tournaments with BM25, dense, BM25+dense fusion, and matched candidate-set rerankers. Include flat model scoring to isolate whether repeated elimination helps.
3. Vary group capacity, survivors, shuffled/grouped candidates, independent relevance versus Choice, and concurrency. Repeat a fixed set across at least five grouping seeds; keep query-level clustering in statistical analysis.
4. Evaluate Laya under the same evidence, query set and tournament policies after confirming its model, license, context limits, outputs and compute requirements. Cross-encoder reranking is a separate comparator, not a substitute for this test.
5. Trace relevant-document survival after every round, probability ties, and position effects. Include no-answer questions, multiple-evidence questions and difficult distractors with independently justified labels.
6. Test larger real corpora. Label subsampled-corpus experiments and analytical projections explicitly; do not infer million-document performance from the current few-thousand-document runs.
7. Measure retrieval plus a fixed answer generator on a suitable held-out QA benchmark. Assess answer correctness and evidence support separately from retrieval scores.
8. Report paired uncertainty and effect sizes across datasets, including failures and negative results. Specify primary comparisons before evaluation and account for multiple comparisons.

## Resource boundary

The original total paid authorization remains $50; it has not reset. Completed recorded Jev calls were approximately $6.25 at published pricing, plus potentially unrecorded calls from the failed 48-document attempt. The last TypeSafe attempt returned HTTP 402. Further paid runs require restored API access and an audited cumulative ledger. A full 300-query SciFact plus 323-query NFCorpus run alone is projected around $46 at the pilot per-query rates, before other variants and past spending; the full proposed study does not fit in the remaining budget at those rates. Stage the study and use local/open-model experiments where feasible.
