# AGENTS.md — Guide pour Sessions Futures

## Contexte Projet
Pipeline QA multi-sauts HotpotQA (distractor) :
**REBEL → Neo4j → GNN GATv2 → Qwen2.5-1.5B QLoRA 4-bit**

GPU : RTX 3050 Ti 4 Go. **NE RETRAÎNER RIEN** — modèles figés dans `checkpoints/`.

## Modèles Entraînés (figés)

| Modèle | Chemin | Config | Métrique clé |
|--------|--------|--------|--------------|
| GNN A8 | `checkpoints/gnn_best.pt` | gatv2_h128_l2 + BM25 feat | Val R@5=0.7292 |
| LoRA Run1 | `checkpoints/qlora_run1/final/` | r=16, 2ep, lr=2e-4 | Val F1=0.2784 |
| LoRA Run2 | `checkpoints/qlora_run2/final/` | r=32, 4ep, lr=2e-4 | Val F1=0.2489 |
| **Fusionné (prod)** | `checkpoints/merged_final/` | Run1 merged 4-bit | Test F1=0.2636 |

## Commandes Utiles

### Lancer l'API
```bash
docker compose up -d neo4j && sleep 20
.venv/bin/python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 1
```

### Test Rapide
```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the capital of France?"}'
```

### Démo Locale
```bash
.venv/bin/python scripts/demo.py
```

### Tests
```bash
.venv/bin/python -m pytest tests/ -v
```

## Structure Clés

```
src/
├── api/main.py              # FastAPI endpoints
├── orchestrator/            # LangGraph pipeline
│   ├── state.py             # PipelineState, PipelineConfig
│   ├── nodes.py             # 7 nœuds (entity, retrieval, gnn, topk, abstain, gen)
│   └── graph.py             # create_pipeline(), run_pipeline()
├── gnn/
│   ├── model.py             # TripleScorerGNN (GATv2/GCN)
│   ├── predict_topk.py      # Inférence GNN + BM25
│   └── dataset.py           # build_graph_data() pour PyG
├── llm/
│   ├── load_llm.py          # load_base_model(), generate_answer()
│   └── prompt.py            # build_prompt_from_triplets()
└── kg/
    └── load_neo4j.py        # Peuplement Neo4j
```

## Points d'Attention

1. **VRAM 4 Go** : toujours `use_4bit=True`, `batch=1`, `workers=1`
2. **PyG Data** : `build_graph_data()` dans `src/gnn/dataset.py` pour inférence ad-hoc
3. **GNN A8** : utilise `TripleScorerGNN_BM25` (voir `src/gnn/predict_topk.py:12`) — recréer `edge_proj` et `edge_scorer` avec dims étendues
4. **Abstention** : seuil `max_score_threshold=0.0` par défaut, `min_triplets_for_generation=1`
4. **Neo4j** : URI `bolt://localhost:7687`, auth `neo4j/REDACTED`

## Rapports de Référence

| Fichier | Contenu |
|---------|---------|
| `reports/j12_final_test.md` | Test EM/F1/recall/latence (val vs test) |
| `reports/j13_ablations_final.md` | 3 ablations + GNN ref (R@5) |
| `reports/j14_hallucination.md` | Hallucination 41% (embedding 0.5) |
| `reports/j10_selection.md` | Run1 vs Run2 vs Base, fusion speedup 1.81x |
| `reports/j7_ablations.md` | 8 configs GNN (A1-A8) |
| `rapport.md` | Rapport final complet |

## Reproductibilité

- Graines : `SEED=42` partout
- `requirements.lock` : versions exactes
- `pip freeze > requirements.lock` pour maj

## Prochaines Étapes Possibles

1. Labelling phrase-level (réduire 30.75% questions sans positif)
2. GNN top-10 + reranker cross-encoder fine-tuné sur triplets
3. Plus de données SFT (data augmentation)
4. DPO si GPU > 8 Go
5. Modèle 3B/7B avec quantization 4-bit

## Contact / Historique

- Commits atomiques par tâche (git log --oneline)
- MLflow : `mlruns/` (expériences GNN, LLM, eval)
- Dernière MAJ : 2026-10-11