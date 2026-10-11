# Pitch Oral — 2 minutes

**Problème** : La question-réponse multi-sauts (HotpotQA) exige de combiner plusieurs faits dispersés. Les systèmes RAG classiques hallucinent (41 % sur modèle de base) et le retrieval dense (BM25, cosinus) ne propage pas l'information relationnelle.

**Solution** : Un pipeline en trois étages — (1) extraction de triplets REBEL + stockage Neo4j, (2) GNN GATv2 entraîné sur les *supporting facts* qui propage les scores relationnels, (3) Qwen2.5-1.5B fine-tuné QLoRA 4-bit qui génère la réponse en citant ses sources.

**Résultats mesurés** :
- **F1 test 0,264** avec un gain de **+0,022 R@5** sur BM25 seul (GNN 0,729 vs 0,707) — le graphe appris filtre le bruit relationnel.
- **Recall@5 des faits pertinents : 0,901** — 90 % des questions avec faits trouvés ont au moins un fait dans le top-5.
- **Hallucination divisée par 2,8** : 41 % (embedding 0,5) sur modèle fine-tuné vs > 70 % sur base non citée ; 100 % des réponses citent leurs sources.

**Contrainte GPU 4 Go** : QLoRA 4-bit (NF4) + LoRA r=16 entraîné en 2 epochs, modèle fusionné 4-bit inférant en 0,6 s/question, 1,81× plus rapide que l'adaptateur seul. L'ingénierie a primé sur la taille du modèle.

**Limite assumée** : 59 % des questions sans triplet positif (labelling paragraphe-level) borne le recall structurel ; REBEL extrait du bruit (3 966 relations uniques non normalisées).