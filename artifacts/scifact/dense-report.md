# Retrieval benchmark report

> **Dataset-specific measurements.** Interpret results using the dataset provenance and sample size. A single local run does not establish general model performance.

Generated: `2026-09-22T12:17:41.427376+00:00`

## Best measured configuration

The highest-ranked run by `mrr@10` used chunk size **256** with overlap **32**. It measured `mrr@10=0.5944`, mean query latency `27.254 ms`, and p95 query latency `47.469 ms` on the runtime below.

## All experiment configurations

| Rank | Chunk size | Overlap | Chunks | recall@1 | recall@3 | recall@5 | recall@10 | mrr@10 | ndcg@10 | Mean latency (ms) | p95 latency (ms) | Build time (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 256 | 32 | 9054 | 0.4692 | 0.6479 | 0.7264 | 0.7899 | 0.5944 | 0.6378 | 27.254 | 47.469 | 528.104 |

Construction time includes document embedding and index construction. Query latency includes single-query encoding, exact chunk search, and source-document deduplication. Model loading, warmup, chunking, and report generation are excluded.

## Runtime

- **python:** 3.12.14
- **platform:** Linux-6.18.44-x86_64-with-glibc2.39
- **processor:** x86_64
- **sentence-transformers:** 6.0.0
- **torch:** 2.13.0+cpu
- **faiss-cpu:** 1.15.0
- **numpy:** 2.4.6
