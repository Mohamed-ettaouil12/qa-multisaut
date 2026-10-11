# QA Multi-Sauts : Retrieval Appris (GNN) et LLM Fine-tuné (QLoRA)

Pipeline : KG (REBEL + Neo4j) → GNN GATv2 de scoring des triplets → LLM fine-tuné QLoRA.
Dataset : HotpotQA (distractor). Environnement : RTX 3050 Ti 4 Go, Ubuntu.

## Installation

```bash
# Cloner
git clone <repo>
cd qa-multisaut

# Créer environnement
python -m venv .venv
source .venv/bin/activate

# Dépendances
pip install -r requirements.txt

# Modèles HuggingFace (premiers lancements)
# REBEL, Qwen2.5-1.5B, bge-small seront téléchargés automatiquement
```

## Reproduction Complète (dans l'ordre)

### 1. Données HotpotQA → splits
```bash
PYTHONPATH=. .venv/bin/python src/prep/sample_splits.py
# Crée data/processed/train.jsonl (1500), val.jsonl (200), test.jsonl (300)
```

### 2. Extraction REBEL (triplets bruts)
```bash
PYTHONPATH=. .venv/bin/python src/kg/run_extraction.py
# ~22 min total (train 14min + test 8min, val déjà fait)
# Crée data/processed/kg_raw_{split}_part*.jsonl
```

### 3. Normalisation + Étiquetage
```bash
PYTHONPATH=. .venv/bin/python src/kg/normalize_label.py
# Crée data/processed/kg_triplets.jsonl (18897 triplets)
# Rapport: reports/j4_kg_stats.md
```

### 4. Neo4j (optionnel, pour ablation B/C)
```bash
docker compose up -d neo4j
sleep 20
PYTHONPATH=. .venv/bin/python src/kg/load_neo4j.py
# 17257 nœuds, 43040 arêtes
# Rapport: reports/j11_neo4j.md
```

### 5. Construction datasets PyG (pour GNN)
```bash
PYTHONPATH=. .venv/bin/python src/gnn/dataset.py
# Crée data/processed/pyg_{train,val,test}.pt
```

### 6. Entraînement GNN (abslations A1-A8)
```bash
PYTHONPATH=. .venv/bin/python src/gnn/train.py --config A1_gatv2_h128_l2_avec_question
# Best: A8_gatv2_h128_l2_bm25 (config A8 + feature BM25)
# Checkpoint: checkpoints/gnn_best.pt
# Rapports: reports/j7_ablations.md, j7_decision.md
```

### 7. Prédictions top-k (val + test)
```bash
PYTHONPATH=. .venv/bin/python src/gnn/predict_topk.py
# Crée data/processed/gnn_topk_{val,test}.jsonl
```

### 8. Construction dataset SFT
```bash
PYTHONPATH=. .venv/bin/python src/llm/build_sft.py
# Crée data/processed/sft_train.jsonl (686 ex)
# Rapports: reports/j8_sft_stats.md, j8_sft_samples.md
```

### 9. Fine-tuning QLoRA (Run1)
```bash
PYTHONPATH=. .venv/bin/python src/llm/train_qlora.py --lora_r 16 --epochs 2 --lr 2e-4
# Checkpoints: checkpoints/qlora_run1/final/
# Modèle fusionné: checkpoints/merged_final/
```

### 10. Évaluation Baselines LLM (val)
```bash
PYTHONPATH=. .venv/bin/python src/llm/eval_baselines.py
# C1: sans contexte, C2: cosinus top-5, C3: GNN top-5
# Rapport: reports/j8_baselines_llm.md
```

### 11. Évaluation Fine-tuné (val)
```bash
PYTHONPATH=. .venv/bin/python src/llm/eval_finetuned.py
# Compare Base+GNN vs Run1 vs Run2
# Rapport: reports/j10_selection.md (Run1 sélectionné)
```

### 12. Évaluation Finale (test)
```bash
PYTHONPATH=. .venv/bin/python -m src.eval.run_evaluation --config C5 --split test
# Résultats: results/C5_test.csv
# Rapport: reports/j12_final_test.md
```

### 13. Ablations Finales
```bash
# BM25 seul, Neo4j sans GNN, Neo4j + Cross-Encoder
# (scripts ad-hoc dans l'historique, résultats dans reports/j13_ablations_final.md)
```

### 14. Taux d'Hallucination
```bash
# Juge embedding bge-small sur 100 réponses val
# Rapport: reports/j14_hallucination.md
```

## Lancement de l'API

```bash
# Nécessite: Neo4j running, modèles entraînés
docker compose up -d neo4j
sleep 20

# Lancer API (1 worker, batch 1 pour VRAM 4GB)
.venv/bin/python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 1
```

### Endpoints
- `GET /health` → statut + version + modèle chargé
- `POST /ask` `{question}` → `{answer, sources[], latency_ms, config, abstained}`
- `GET /docs` → Swagger UI

### Test API
```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the capital of France?"}'
```

## Démo Locale (sans API)

```bash
.venv/bin/python scripts/demo.py
```

## Tests

```bash
.venv/bin/python -m pytest tests/ -v
```

## Structure du Dépôt

```
qa-multisaut/
├── checkpoints/           # Modèles entraînés
│   ├── gnn_best.pt        # GNN A8 (gatv2_h128_l2 + BM25)
│   ├── qlora_run1/final/  # LoRA Run1 (r=16, 2ep)
│   ├── qlora_run2/final/  # LoRA Run2 (r=32, 4ep)
│   └── merged_final/      # Modèle fusionné 4-bit (Run1)
├── configs/
│   └── config.yaml        # Config globale
├── data/
│   ├── processed/         # Données intermédiaires (gitignored)
│   │   ├── train/val/test.jsonl
│   │   ├── kg_triplets.jsonl
│   │   ├── pyg_{train,val,test}.pt
│   │   ├── gnn_topk_{val,test}.jsonl
│   │   └── sft_train.jsonl
│   └── raw/               # Données brutes (gitignored)
├── reports/               # Rapports d'étapes
│   ├── j4_kg_stats.md
│   ├── j5_baseline_triplet.md
│   ├── j7_ablations.md
│   ├── j7_decision.md
│   ├── j8_baselines_llm.md
│   ├── j8_sft_stats.md
│   ├── j8_sft_samples.md
│   ├── j10_selection.md
│   ├── j11_neo4j.md
│   ├── j12_final_test.md
│   ├── j13_ablations_final.md
│   └── j14_hallucination.md
├── results/               # CSV d'évaluation (gitignored)
├── scripts/
│   └── demo.py            # Démo 3 exemples
├── src/
│   ├── api/               # FastAPI
│   ├── eval/              # Évaluation
│   ├── gnn/               # GNN (model, train, dataset, predict)
│   ├── kg/                # Knowledge Graph (REBEL, Neo4j, normalize)
│   ├── llm/               # LLM (prompt, load, train, eval)
│   ├── orchestrator/      # LangGraph pipeline
│   ├── pipeline/          # (vide, réservé)
│   ├── prep/              # Préparation données
│   └── retrieval/         # Baselines BM25/cosinus
├── tests/
│   └── test_api.py        # Tests API
├── mlruns/                # MLflow (gitignored)
├── docker-compose.yml     # Neo4j
├── requirements.txt
├── requirements.lock      # À générer: pip freeze > requirements.lock
├── STATUS.md              # État d'avancement
├── rapport.md             # Rapport final
└── README.md
```

## Versions Clés (reproductibilité)

```bash
# Générer le lock
pip freeze > requirements.lock
```

Versions utilisées :
- torch 2.14.0
- torch_geometric 2.6.1
- transformers 4.44.0
- peft 0.12.0
- trl 0.11.1
- bitsandbytes 0.43.3
- sentence-transformers 3.0.1
- langgraph 0.2.0
- fastapi 0.112.0
- neo4j 5.15.0
- python 3.14

## Graines Fixées
- `SEED = 42` partout (Python, NumPy, PyTorch, random)
- Split HotpotQA : seed 42 (sample_splits.py)
- Dataset SFT : seed 42 (build_sft.py)
- Entraînement GNN/QLoRA : seed 42

## Licence
MIT