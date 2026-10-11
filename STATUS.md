# Project Status

## Task 1 — Extraction REBEL sur test ✅
- **Done**: REBEL extraction on test (300 questions) → `kg_raw_test_part0.jsonl` (2243 triplets)
- **Done**: Normalization + labeling → `kg_triplets.jsonl` (18897 triplets total, 3687 positives)
- **Stats**: `reports/j4_kg_stats.md` updated
- **Key metric**: 30.75% questions sans triplet positif (615/2000)

## Task 2 — Peupler Neo4j ⏳
- Pending: docker compose up, load_neo4j.py, Cypher verification

## Task 3 — Prédictions top-k sur test ⏳
- Pending: gnn_best.pt (A8) → gnn_topk_test.jsonl (k=5, 300 qids)

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