# Project Status

## Task 1 — Extraction REBEL sur test ✅
- **Done**: REBEL extraction on test (300 questions) → `kg_raw_test_part0.jsonl` (2243 triplets)
- **Done**: Normalization + labeling → `kg_triplets.jsonl` (18897 triplets total, 3687 positives)
- **Stats**: `reports/j4_kg_stats.md` updated
- **Key metric**: 30.75% questions sans triplet positif (615/2000)

## Task 2 — Peupler Neo4j ✅
- **Done**: docker compose up, load_neo4j.py executed successfully
- **Counts**: 17 257 nodes, 43 040 edges, 4 865 edges label=1, 15 302 entities, 1 955 questions
- **Cypher verification**: 2-hop neighborhood queries working
- **Report**: `reports/j11_neo4j.md`
- **Note**: 45 questions missing (no triplets extracted)

## Task 3 — Prédictions top-k sur test ✅
- **Done**: Regenerated PyG datasets (train: 1466, val: 196, test: 293 graphs)
- **Done**: GNN A8 (gnn_best.pt) predictions → `gnn_topk_test.jsonl` (293 qids, k=5)
- **Format**: identical to val (qid, k, triplets with score)
- **7 questions without triplets** (293/300)

## Task 4 — Évaluation finale sur test ✅
- **Done**: Run1 fused 4-bit, config C5, metrics computed
- **Test metrics**: EM=0.1633, F1=0.2636, Format OK=99.7%, Latency=0.607s
- **Val metrics**: EM=0.1950, F1=0.2784, Format OK=100%, Latency=0.618s
- **Supporting facts recall@5**: Val=0.9357, Test=0.9009
- **Report**: `reports/j12_final_test.md`

## Task 5 — Ablations manquantes ✅
- **A) BM25 seul**: R@5=0.7074 (-0.0218 vs GNN A8)
- **B) Neo4j sans GNN**: 1-2 hops + BM25, R@5=0.6634 (-0.0658)
- **C) Neo4j + Cross-Encoder**: R@5=0.4426 (-0.2866, reranker fails on short triplets)
- **Report**: `reports/j13_ablations_final.md` with full comparison table

## Task 6 — Taux d'hallucination ✅
- **Done**: Judge = embedding (bge-small) threshold 0.5 on 100 val responses Run1_GNN
- **Result**: 41% responses hallucinated, 59% claims supported
- **Lexical judge (comparison)**: 76% hallucinated (too strict)
- **Report**: `reports/j14_hallucination.md` with 5 detailed examples

---
Last updated: 2026-10-11