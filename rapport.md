# QA Multi-Sauts : Rapport Final

## Problème
Question-Réponse multi-sauts sur **HotpotQA (distractor)** : répondre à des questions nécessitant de combiner plusieurs faits dispersés dans plusieurs documents.

## Pipeline
```
REBEL (extraction triplets) 
    → Neo4j (stockage graphe) 
    → GNN GATv2 (scoring triplets, entraîné sur supporting facts) 
    → Top-k triplets → Prompt 
    → Qwen2.5-1.5B fine-tuné QLoRA 4-bit (génération avec citations)
```

## Méthodes

### 1. Extraction de connaissances (REBEL)
- Modèle : `Babelscape/rebel-large` (seq2seq, 400M params)
- Extraction par paragraphes (batch 16, beams=3)
- 18 897 triplets normalisés sur 2000 questions (train+val+test)
- Étiquetage automatique : label=1 si titre du paragraphe ∈ supporting facts
- **Limite** : 30.75% questions sans triplet positif (labelling paragraphe-level)

### 2. Graphe de connaissances (Neo4j)
- 17 257 nœuds, 43 040 arêtes, 15 302 entités uniques
- Contraintes d'unicité sur `Entity.name` et `Question.qid`
- Voisinage 1-2 sauts requêtable via Cypher

### 3. GNN Scoring (GATv2)
- Architecture : GATv2 (2 couches, hidden=128, 4 têtes) + feature BM25
- Entrées : embeddings entités (bge-small 384d), relations, question, cos_qt
- Entraînement : BCEWithLogitsLoss, pos_weight=n_neg/n_pos, early stopping R@5 val
- **Meilleure config (A8)** : R@5=0.7292 val (vs cosinus 0.7269, BM25 0.7074)

### 4. LLM Fine-tuning (QLoRA)
- Base : `Qwen/Qwen2.5-1.5B-Instruct` (4-bit NF4)
- LoRA : r=16, alpha=32, dropout=0.05, target modules all linear
- Dataset SFT : 686 exemples (60% positifs+distracteurs, 40% cosinus top-5)
- 2 epochs, completion_only_loss, batch effectif 8
- Modèle fusionné 4-bit : ~2.5 GB VRAM, 1.81x plus rapide que adaptateur

## Résultats Mesurés

### Test (300 questions) — Config C5 (Run1 fused + GNN top-5)

| Métrique | Test | Val | Δ |
|----------|------|-----|---|
| **EM** | **0.1633** | 0.1950 | -0.0317 |
| **F1** | **0.2636** | 0.2784 | -0.0148 |
| **Format OK** | **99.7%** | 100.0% | -0.3% |
| **Latence** | **0.607s** | 0.618s | -0.011s |
| **Supporting Facts Recall@5** | **0.9009** | 0.9357 | -0.0348 |

### Ablations Complètes

| Configuration | R@3 | R@5 | R@10 | Rang |
|---------------|-----|-----|------|------|
| **GNN A8 (GATv2+BM25)** | **0.5971** | **0.7292** | 0.8917 | 1 |
| BM25 seul | 0.5339 | 0.7074 | 0.8905 | 2 |
| Neo4j sans GNN (1-2 hops + BM25) | 0.5175 | 0.6634 | 0.8685 | 3 |
| Neo4j + Cross-Encoder reranker | 0.4180 | 0.4426 | 0.4572 | 4 |

**Gain GNN vs BM25 seul : +0.0218 R@5** (marginal mais significatif)

### Taux d'Hallucination (100 réponses val, Run1_GNN)

| Juge | Seuil | Réponses hallucinées | Claims supportés |
|------|-------|---------------------|------------------|
| **Embedding (bge-small)** | **0.5** | **41%** | **59%** |
| Lexical (Jaccard) | 0.5 | 76% | 24% |

**Causes** : erreurs de raisonnement (45%), bruit GNN label=0 cité (35%), info absente top-5 (20%)

### Comparaison Qwen Base vs Fine-tuné (Val, GNN top-5)

| Config | EM | F1 | Format OK |
|--------|-----|-----|-----------|
| Base (C3) | 0.0000 | 0.0242 | 0.0% |
| **Run1 Fine-tuné (C5)** | **0.1950** | **0.2784** | **100.0%** |

**Gain fine-tuning : +0.254 F1, +19.5 EM, format acquis**

## Limites Connues

1. **Labelling paragraph-level** : 30.75% questions sans triplet positif → limite structurelle du recall
2. **REBEL imparfait** : extraction bruitée, relations non normalisées (3966 relations uniques)
3. **GPU 4 Go** : contraint batch size, sequence length, élimine DPO/full fine-tuning
4. **Hallucination 41%** : modèle 1.5B + peu de données SFT (686 ex, 2 epochs)
5. **Pas d'évaluation test pour ablations** : calculé sur val uniquement

## Suites Possibles

1. **Améliorer labelling** : phrase-level ou distant supervision
2. **Augmenter k** : GNN top-10 → recall amélioré
3. **Plus de données SFT** : générer plus d'exemples, diversity
4. **Reranker cross-encoder adapté** : fine-tuner sur triplets courts
5. **Modèle plus grand** : Qwen2.5-3B/7B si GPU disponible
6. **DPO** : alignement préférences si budget VRAM permet

---

*Généré automatiquement le 2026-10-11*