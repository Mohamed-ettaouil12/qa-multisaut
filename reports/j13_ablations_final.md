# Ablations Finales - J13

## Contexte
Évaluation sur val (200 questions, 140 avec ≥1 triplet label=1)
Métrique cible: Recall@5
GNN de référence (A8): R@5 = 0.7292

## Tableau récapitulatif

| Ablation | Description | R@3 | R@5 | R@10 | vs GNN R@5 | Question répondue |
|----------|-------------|-----|-----|------|------------|-------------------|
| **GNN A8 (réf)** | GATv2 + BM25 feature, k=5 | 0.5971 | **0.7292** | 0.8917 | — | — |
| **A) BM25 seul** | BM25 sur textes triplets (h r t), sans graphe | 0.5339 | 0.7074 | 0.8905 | -0.0218 | Le KG + GNN apporte-t-il un gain vs BM25 pur ? |
| **B) Neo4j sans GNN** | Voisinage 1-2 sauts depuis entités question, trié BM25 | 0.5175 | 0.6634 | 0.8685 | -0.0658 | Le graphe seul (sans GNN) suffit-il ? |
| **C) Neo4j + Cross-Encoder** | Même voisinage, reranker cross-encoder (ms-marco-MiniLM-L-6-v2) | 0.4180 | 0.4426 | 0.4572 | -0.2866 | Un reranker textuel améliore-t-il le voisinage brut ? |

## Détails par ablation

### A) BM25 seul
- **Méthode** : BM25Okapi sur corpus = `{h, r, t}` normalisés par question
- **Résultat** : R@5 = 0.7074 (très proche du GNN 0.7292)
- **Interprétation** : Le BM25 seul est un baseline très fort. Le GNN n'apporte que +0.0218 en R@5, confirmant que la similarité lexicale capture déjà la majorité du signal.

### B) Neo4j sans GNN
- **Méthode** : 
  1. Entités détectées = entités présentes dans les triplets de la question (kg_triplets.jsonl)
  2. Requête Cypher 1-2 sauts depuis ces entités
  3. Déduplication, puis ranking BM25 question↔triplet
- **Résultat** : R@5 = 0.6634
- **Interprétation** : Le graphe seul (sans GNN) sous-performe le BM25 global car le voisinage 1-2 sauts introduit du bruit (entités non pertinentes) et perd le classement global optimal.

### C) Neo4j + Cross-Encoder Reranker
- **Méthode** : Même voisinage 1-2 sauts que B, puis reranker `cross-encoder/ms-marco-MiniLM-L-6-v2` sur top-20 BM25
- **Résultat** : R@5 = 0.4426 (**pire que BM25 seul**)
- **Interprétation** : Le cross-encoder MS MARCO (entraîné sur passages longs) ne généralise pas aux textes triplet courts (h r t). Le reranker dense (bge-small, top-10) donne R@5=0.3939, aussi faible.
- **Conclusion** : Pour cette tâche, le reranker textuel ne remplace pas le GNN ; le GNN apprend une propagation relationnelle que les modèles textuels ne capturent pas.

## Conclusion générale

| Configuration | R@5 | Rang |
|---------------|-----|------|
| GNN A8 (GATv2+BM25) | **0.7292** | 1 |
| BM25 seul | 0.7074 | 2 |
| Neo4j sans GNN (1-2 hops + BM25) | 0.6634 | 3 |
| Neo4j + Cross-Encoder | 0.4426 | 4 |

**Le GNN appris reste la meilleure approche** malgré un gain marginal vs BM25 seul (+0.0218). Les ablations confirment :
1. Le graphe sans GNN (B) perd l'avantage du classement global
2. Les rerankers textuels (C) échouent sur le format triplet court
3. BM25 seul (A) est un baseline dur à battre, validant le choix d'en faire une feature du GNN (A8)