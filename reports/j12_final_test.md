# Évaluation Finale Test - J12

## Configuration
- **Modèle** : Run1 fusionné 4-bit (Qwen2.5-1.5B-Instruct, LoRA r=16, 2 epochs)
- **Contexte** : GNN top-k (k=5, config A8: gatv2_h128_l2 + BM25)
- **Split** : test (300 questions)
- **Max new tokens** : 64
- **Seed** : 42

## Résultats

| Métrique | Val (200) | Test (300) | Δ |
|----------|-----------|------------|---|
| **EM** | 0.1950 | **0.1633** | -0.0317 |
| **F1** | 0.2784 | **0.2636** | -0.0148 |
| **Format OK %** | 100.0% | **99.7%** | -0.3% |
| **Latence moy (s)** | 0.618 | **0.607** | -0.011 |
| **Supporting Facts Recall@5** | 0.9357 | **0.9009** | -0.0348 |

## Analyse

- **Performances stables** : légère baisse attendue val→test (distribution légèrement différente)
- **Format excellente** : 99.7% des réponses respectent le format "Sources: [n]"
- **Recall supporting facts** : 90% des questions avec triplets positifs ont au moins 1 triplet pertinent dans le top-5 GNN
- **Latence** : ~0.6s/question sur RTX 3050 Ti 4GB (4-bit fused)

## Comparaison avec baselines (val)

| Config | EM | F1 | Format OK |
|--------|-----|-----|-----------|
| C1_sans_contexte (base) | 0.0000 | 0.0183 | 0.0% |
| C2_top5_cosinus (base) | 0.0000 | 0.0232 | 0.0% |
| C3_gnn_top5 (base) | 0.0000 | 0.0242 | 0.0% |
| **C5_run1_gnn (fine-tuned)** | **0.1950** | **0.2784** | **100.0%** |

## Fichiers générés
- `results/C5_test.csv` : prédictions détaillées (300 lignes)
- `results/C5_val.csv` : prédictions val (200 lignes)
- `data/processed/gnn_topk_test.jsonl` : 293 qids avec top-5 GNN
- `data/processed/gnn_topk_val.jsonl` : 196 qids avec top-5 GNN

## Conclusion
Le pipeline complet (REBEL → KG → GNN → QLoRA) fonctionne **de bout en bout** sur HotpotQA distractor. Le fine-tuning QLoRA apporte un gain massif vs base model (+0.26 F1, +19.5 EM) et le modèle apprend à citer ses sources correctement.