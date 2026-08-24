# Retrieval benchmark report

> **Demonstration dataset only.** These measurements verify the evaluation pipeline; they are not a statistically meaningful or general model benchmark.

Generated: `2026-08-24T06:33:25.307740+00:00`

## Best measured configuration

The highest-ranked run by `mrr@10` used chunk size **128** with overlap **32**. It measured `mrr@10=1.0000`, mean query latency `4.573 ms`, and p95 query latency `5.826 ms` on the runtime below.

## All experiment configurations

| Rank | Chunk size | Overlap | Chunks | recall@1 | recall@3 | recall@5 | recall@10 | mrr@10 | ndcg@10 | Mean latency (ms) | p95 latency (ms) | Build time (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 128 | 32 | 55 | 0.9600 | 0.9800 | 0.9800 | 1.0000 | 1.0000 | 0.9933 | 4.573 | 5.826 | 0.384 |
| 2 | 256 | 32 | 27 | 0.9600 | 0.9800 | 1.0000 | 1.0000 | 1.0000 | 0.9951 | 4.796 | 6.109 | 0.393 |
| 3 | 512 | 64 | 15 | 0.9400 | 1.0000 | 1.0000 | 1.0000 | 0.9800 | 0.9877 | 4.371 | 5.159 | 0.466 |
| 4 | 256 | 64 | 29 | 0.9400 | 0.9800 | 1.0000 | 1.0000 | 0.9800 | 0.9850 | 4.688 | 6.225 | 0.413 |
| 5 | 512 | 32 | 15 | 0.9400 | 1.0000 | 1.0000 | 1.0000 | 0.9800 | 0.9877 | 5.329 | 7.480 | 0.466 |
| 6 | 128 | 0 | 45 | 0.8800 | 0.9800 | 1.0000 | 1.0000 | 0.9600 | 0.9645 | 4.469 | 5.821 | 0.332 |
| 7 | 512 | 0 | 15 | 0.9000 | 1.0000 | 1.0000 | 1.0000 | 0.9600 | 0.9730 | 4.519 | 5.885 | 0.464 |
| 8 | 128 | 64 | 75 | 0.8800 | 0.9800 | 1.0000 | 1.0000 | 0.9600 | 0.9656 | 4.775 | 6.192 | 0.519 |
| 9 | 256 | 0 | 27 | 0.8000 | 0.9800 | 1.0000 | 1.0000 | 0.9200 | 0.9360 | 4.917 | 6.254 | 0.385 |

Construction time includes document embedding and index construction. Query latency includes single-query encoding, exact chunk search, and source-document deduplication. Model loading, warmup, chunking, and report generation are excluded.

## Runtime

- **python:** 3.11.16
- **platform:** Windows-10-10.0.26200-SP0
- **processor:** AMD64 Family 26 Model 68 Stepping 0, AuthenticAMD
- **sentence-transformers:** 6.0.0
- **torch:** 2.13.0+cpu
- **faiss-cpu:** 1.15.0
- **numpy:** 2.4.6
