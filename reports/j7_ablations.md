# Ablations GNN - J7

Baseline cosinus (J5) : R@3=0.5754, R@5=0.7269, R@10=0.9337
Baseline BM25 (J5) : R@3=0.5339, R@5=0.7074, R@10=0.8905

| Config | Description | R@3 | R@5 | R@10 | vs cosinus R@5 |
|--------|-------------|-----|-----|------|----------------|
| A1 | gatv2_h128_l2_avec_question (v1) | 0.5523 | **0.7197** | 0.9204 | -0.0072 |
| A2 | gcn_h128_l2_avec_question | 0.5764 | 0.7068 | 0.9107 | -0.0201 |
| A3 | gatv2_h128_l1 (1 couche) | 0.5270 | 0.6955 | 0.8958 | -0.0314 |
| A4 | gatv2_h128_l3 (3 couches) | 0.5174 | 0.6998 | 0.9078 | -0.0271 |
| A5 | gatv2_h64_l2 (hidden=64) | 0.5611 | 0.7125 | 0.8890 | -0.0144 |
| A6 | gatv2_h256_l2 (hidden=256) | 0.5237 | 0.7144 | 0.9214 | -0.0125 |
| A7 | gatv2_h128_l2_SANS_conditionnement | 0.5012 | 0.6928 | 0.9153 | -0.0341 |
| **A8** | **gatv2_h128_l2 + BM25 feature** | **0.5971** | **0.7292** | 0.8917 | **+0.0023** |

**Meilleure config : A8 (gatv2_h128_l2 + BM25)** — seule à battre le cosinus en R@5 val.

**Observations :**
- Le conditionnement par la question (q_emb + cos_qt) est crucial : A7 (-0.0341 vs A1)
- GATv2 > GCN pour cette tâche (A1 > A2)
- 2 couches optimal : 1 couche sous-performe (A3), 3 couches n'apporte rien (A4)
- Hidden dim 128 optimal : 64 trop petit (A5), 256 sur-paramétré (A6)
- BM25 en feature d'arête apporte le gain décisif (+0.0095 vs A1)