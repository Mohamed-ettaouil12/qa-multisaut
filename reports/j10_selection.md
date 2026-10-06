# Sélection Adaptateur - J10

Comparaison: Base+GNN (J8-C3) vs Run1 (r=16, 2ep) vs Run2 (r=32, 4ep)

Modèle de base: Qwen/Qwen2.5-1.5B-Instruct (4-bit)
Contexte: GNN top-k (val)
Max new tokens: 64
Seed: 42

| Config   | EM     | F1     | Format OK % | Insuffisant à tort % | Latence (s) |
| -------- | ------ | ------ | ----------- | -------------------- | ----------- |
| Base_GNN | 0.0000 | 0.0242 | 0.0         | 0.0                  | 1.596       |
| Run1_GNN | 0.1950 | 0.2784 | 100.0       | 0.0                  | 0.623       |
| Run2_GNN | 0.1700 | 0.2489 | 99.5        | 0.7                  | 0.621       |

## Analyse

- **Baseline F1 (J8-C3)**: 0.0242
- **Run 1 F1**: 0.2784 (↑ +0.2541)
- **Run 2 F1**: 0.2489 (↑ +0.2247)

**✅ Run 1 améliore le F1 vs baseline (meilleur: +0.2541)** → Sélection: Run 1 (r=16, 2 epochs)

**Meilleur config**: Run1_GNN

Run 2 (r=32, 4 epochs) surapprend légèrement (F1 plus bas que Run 1).

## Fusion et Vitesse d'Inférence

Modèle fusionné: `checkpoints/merged_final/` (fp16, merge_and_unload sur CPU)

| Modèle | Latence/question | Tokens/s | VRAM |
|--------|------------------|----------|------|
| Adaptateur Run1 (4-bit) | 0.658s | 15.6 | ~2.5 GB |
| Fusionné (4-bit) | 0.364s | 32.7 | ~2.5 GB |
| Fusionné (fp16) | N/A (OOM) | N/A | >4 GB |

**Speedup fusionné (4-bit vs adaptateur 4-bit): 1.81x**

Le modèle fusionné en fp16 ne tient pas en VRAM 4GB → utilisation en 4-bit pour la démo.
L'adaptateur 4-bit reste une option valide (plus lent mais même VRAM).

## Questions d'Entretien (3)

**Q1: Pourquoi QLoRA plutôt que fine-tuning complet ?**
R: QLoRA (4-bit NF4 + LoRA) réduit la VRAM de ~16GB à ~4GB pour un modèle 1.5B, rendant l'entraînement possible sur GPU consumer. Le LoRA (r=16) n'entraîne que 1.2% des paramètres, évitant l'oubli catastrophique et permettant des checkpoints légers (~30MB vs ~3GB).

**Q2: Pourquoi perte sur la réponse seule (completion_only_loss) ?**
R: Le prompt (contexte + question) est fixe et ne nécessite pas d'apprentissage. La perte sur la completion seule évite de backpropager sur des tokens de contexte invariants, accélère la convergence et réduit le risque de surapprentissage sur le format de prompt.

**Q3: Pourquoi 40% de contextes bruités (cosinus) dans le dataset SFT ?**
R: Les 60% restants (positifs + distracteurs aléatoires) créent un signal "trop propre" où les triplets label=1 sont toujours en tête. Les 40% cosinus introduisent du bruit réaliste (triplets pertinents sémantiquement mais pas nécessairement label=1), forçant le modèle à apprendre à filtrer l'information plutôt qu'à mémoriser la position des labels.
