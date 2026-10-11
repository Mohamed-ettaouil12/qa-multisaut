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

## Task 4 — Évaluation finale sur test ⏳
- Pending: Run1 fused 4-bit, config C5, metrics EM/F1/recall/format/latency

## Task 5 — Ablations manquantes ⏳
- A) BM25 seul
- B) Neo4j sans GNN (voisinage 1-2 sauts)
- C) Neo4j + reranker (cross-encoder ou BM25)

## Task 6 — Taux d'hallucination ⏳
- Pending: juge lexical/embedding sur 100 réponses val Run1_GNN

---
Last updated: 2026-10-11