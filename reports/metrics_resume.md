# Résumé des Métriques Clés

> Source unique de vérité : tous les chiffres ci-dessous proviennent exclusivement des rapports `j10–j14` et `rapport.md`. **Aucun chiffre inventé**. Métrique absente → `MANQUANT`.

---

## 1. Résultats TEST (300 questions) — Config C5 (Run1 fusionné 4-bit + GNN top-5)

| Métrique | Valeur | Source |
|----------|--------|--------|
| **EM** | 0.1633 | `j12_final_test.md` |
| **F1** | 0.2636 | `j12_final_test.md` |
| **Format OK %** | 99.7% | `j12_final_test.md` |
| **Latence moyenne** | 0.607 s | `j12_final_test.md` |
| **Supporting Facts Recall@5** | 0.9009 | `j12_final_test.md` |
| **Taux d'hallucination** | N/A sur test | `MANQUANT` (mesuré seulement sur val) |

---

## 2. Résultats VAL (200 questions) — Config C5 (Run1 fusionné 4-bit + GNN top-5)

| Métrique | Valeur | Source |
|----------|--------|--------|
| **EM** | 0.1950 | `j12_final_test.md` / `j10_selection.md` |
| **F1** | 0.2784 | `j12_final_test.md` / `j10_selection.md` |
| **Format OK %** | 100.0% | `j12_final_test.md` / `j10_selection.md` |
| **Latence moyenne** | 0.618 s | `j12_final_test.md` |
| **Supporting Facts Recall@5** | 0.9357 | `j12_final_test.md` |
| **Taux d'hallucination (embedding 0.5)** | 41% | `j14_hallucination.md` |
| **Claims supportés (embedding 0.5)** | 59% | `j14_hallucination.md` |

---

## 3. Tableau d'Ablation Complet (VAL, 140 questions évaluées)

| Configuration | Description | R@3 | R@5 | R@10 | Question répondue | Source |
|---------------|-------------|-----|-----|------|-------------------|--------|
| **GNN A8 (réf)** | GATv2 + BM25 feature, k=5 | 0.5971 | **0.7292** | 0.8917 | — | `j13_ablations_final.md` |
| **A) BM25 seul** | BM25 sur textes triplets (h r t), sans graphe | 0.5339 | 0.7074 | 0.8905 | Le KG + GNN apporte-t-il un gain vs BM25 pur ? | `j13_ablations_final.md` |
| **B) Neo4j sans GNN** | Voisinage 1-2 sauts depuis entités question, trié BM25 | 0.5175 | 0.6634 | 0.8685 | Le graphe seul (sans GNN) suffit-il ? | `j13_ablations_final.md` |
| **C) Neo4j + Cross-Encoder** | Même voisinage, reranker cross-encoder (ms-marco-MiniLM-L-6-v2) | 0.4180 | 0.4426 | 0.4572 | Un reranker textuel améliore-t-il le voisinage brut ? | `j13_ablations_final.md` |
| **Cosinus (bge-small)** | Dense retrieval baseline | 0.5754 | 0.7269 | 0.9337 | Baseline dense | `j5_baseline_triplet.md` |
| **BM25 (baseline)** | BM25 baseline | 0.5339 | 0.7074 | 0.8905 | Baseline sparse | `j5_baseline_triplet.md` |

**Gain GNN vs BM25 seul : +0.0218 R@5** (marginal mais significatif)

---

## 4. Comparaisons Clés

### Run1 vs Run2 (VAL, GNN top-5)

| Config | EM | F1 | Format OK % | Insuffisant à tort % | Latence (s) | Source |
|--------|-----|-----|-------------|----------------------|-------------|--------|
| Base_GNN | 0.0000 | 0.0242 | 0.0 | 0.0 | 1.596 | `j10_selection.md` |
| **Run1_GNN** | **0.1950** | **0.2784** | **100.0** | 0.0 | 0.623 | `j10_selection.md` |
| Run2_GNN | 0.1700 | 0.2489 | 99.5 | 0.7 | 0.621 | `j10_selection.md` |

**→ Run1 sélectionné (r=16, 2 epochs)** — Run2 surapprend légèrement.

### Adaptateur vs Fusionné (Inférence)

| Modèle | Latence/question | Tokens/s | VRAM | Source |
|--------|------------------|----------|------|--------|
| Adaptateur Run1 (4-bit) | 0.658s | 15.6 | ~2.5 GB | `j10_selection.md` |
| **Fusionné (4-bit)** | **0.364s** | **32.7** | **~2.5 GB** | `j10_selection.md` |
| Fusionné (fp16) | N/A (OOM) | N/A | >4 GB | `j10_selection.md` |

**Speedup fusionné vs adaptateur (tous deux 4-bit) : 1.81x**

### Base vs Fine-tuné (VAL, GNN top-5)

| Config | EM | F1 | Format OK % | Source |
|--------|-----|-----|-------------|--------|
| Base (C3) | 0.0000 | 0.0242 | 0.0% | `j10_selection.md` / `j8_baselines_llm.md` |
| **Run1 Fine-tuné (C5)** | **0.1950** | **0.2784** | **100.0%** | `j10_selection.md` / `j12_final_test.md` |

**Gain fine-tuning : +0.254 F1, +19.5 EM, format acquis**

---

## 5. Taux d'Hallucination (VAL, 100 réponses Run1_GNN)

| Juge | Seuil | Réponses hallucinées | Claims supportés | Source |
|------|-------|---------------------|------------------|--------|
| **Embedding (bge-small)** | **0.5** | **41%** | **59%** (59/100) | `j14_hallucination.md` |
| Embedding | 0.6 | 77% | 23% | `j14_hallucination.md` |
| Embedding | 0.7 | 83% | 17% | `j14_hallucination.md` |
| Lexical (Jaccard) | 0.5 | 76% | 24% | `j14_hallucination.md` |

**Définition du juge principal** : Similarité cosinus max entre embedding de la claim et embeddings des triplets cités (bge-small-en-v1.5). Seuil 0.5 choisi pour capturer similarité sémantique au-delà du lexical exact.

---

## 6. Métriques de Retrieval (GNN)

| Métrique | Val | Test | Source |
|----------|-----|------|--------|
| R@3 | 0.5971 | N/A | `j13_ablations_final.md` / `j12_final_test.md` |
| **R@5** | **0.7292** | N/A (mesuré sur val) | `j13_ablations_final.md` |
| R@10 | 0.8917 | N/A | `j13_ablations_final.md` |
| Supporting Facts Recall@5 | 0.9357 | **0.9009** | `j12_final_test.md` |

---

## 7. Statistiques KG (Train+Val+Test)

| Statistique | Valeur | Source |
|-------------|--------|--------|
| Total triplets | 18 897 | `j4_kg_stats.md` |
| Nœuds uniques (entités) | 15 302 | `j4_kg_stats.md` |
| Relations uniques | 6 183 | `j4_kg_stats.md` |
| Triplets/question (moy) | 9.67 | `j4_kg_stats.md` |
| Taux positifs (label=1) | 19.51% (3687/18897) | `j4_kg_stats.md` |
| % questions sans positif | **30.75%** (615/2000) | `j4_kg_stats.md` |
| Neo4j nœuds | 17 257 | `j11_neo4j.md` |
| Neo4j arêtes | 43 040 | `j11_neo4j.md` |
| Neo4j questions | 1 955 / 2 000 | `j11_neo4j.md` |

---

## 8. Métriques SFT / Fine-tuning

| Paramètre | Valeur | Source |
|-----------|--------|--------|
| Modèle de base | Qwen/Qwen2.5-1.5B-Instruct | `j10_selection.md` |
| Quantization | 4-bit NF4 | `j10_selection.md` |
| LoRA r | 16 (Run1) / 32 (Run2) | `j10_selection.md` |
| Epochs | 2 (Run1) / 4 (Run2) | `j10_selection.md` |
| LR | 2e-4 | `j10_selection.md` |
| Dataset SFT taille | 686 exemples | `j8_sft_stats.md` |
| Split SFT | 60% positifs+distracteurs, 40% cosinus | `j8_sft_stats.md` |
| Completion-only loss | Oui | `j10_selection.md` |
| Seed | 42 | Partout |

---

## 9. Métriques Manquantes / Non Mesurées

| Métrique | Statut | Note |
|----------|--------|------|
| Hallucination sur **test** | `MANQUANT` | Mesuré seulement sur val (100 échantillons) |
| Ablations sur **test** | `MANQUANT` | Calculées seulement sur val |
| EM/F1 par type de question (comparison/bridge) | `MANQUANT` | Non détaillé |
| Latence P50 / P90 / P99 | `MANQUANT` | Seule moyenne rapportée |
| VRAM peak réelle (monitoring) | `MANQUANT` | Estimation ~2.5 GB |

---

*Généré le 2026-10-11 à partir des rapports sources j10–j14, j12, rapport.md*