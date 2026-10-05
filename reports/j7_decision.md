# Décision J7 - GNN Multi-sauts

**Meilleure config :** A8 (gatv2_h128_l2 + BM25 feature)
**Val R@5 :** 0.7292 (vs cosinus 0.7269, BM25 0.7074) → **+0.0023 vs cosinus**
**k retenu :** 5 (métrique cible, aligné sur early stopping et baseline)

**Go / No-go options J8-J10 :**
- **Labels phrase-level** : GO — le GNN utilise déjà sentence-level via kg_triplets ; gain marginal attendu
- **RAGAS eval** : GO — nécessaire pour valider la chaîne complète retrieval→LLM
- **DPO sur LLM** : NO-GO — budget GPU insuffisant (4 Go VRAM) ; prioriser retrieval

**Fichiers créés :**
- checkpoints/gnn_best.pt (A8, best epoch 4)
- data/processed/gnn_topk_val.jsonl (196 qids, k=5)
- data/processed/gnn_topk_test.jsonl (0 qids — pas de triplets test)
- reports/j7_ablations.md
- MLflow runs : A1–A8 dans exp "qa-multisaut"

**Prochaine étape (J8) :** Pipeline retrieval→LLM avec top-5 GNN + prompt.py, éval EM/F1 val.