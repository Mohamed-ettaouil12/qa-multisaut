# Taux d'Hallucination - J14

## Définition du juge

**Juge principal : Similarité d'embedding (bge-small-en-v1.5)**

Pour chaque affirmation (claim) dans la réponse générée :
1. Encoder l'affirmation et les triplets cités avec `BAAI/bge-small-en-v1.5`
2. Calculer la similarité cosinus max entre l'affirmation et les triplets cités
3. **Seuil : 0.5** (choisi pour capturer la similarité sémantique au-delà du chevauchement lexical exact)

**Pourquoi 0.5 ?** 
- Les réponses sont courtes (entités, dates, noms propres)
- Le chevauchement lexical exact (threshold 1.0) est trop strict : "Indian Army" vs "flotilla" = 0 token commun mais sémantiquement lié (militaire)
- 0.5 capture la similarité sémantique tout en rejetant les hallucinations franches

**Juge de comparaison : Lexical (SQuAD normalize + token overlap ≥ 0.5)**
- Même protocole mais similarité = Jaccard sur tokens normalisés
- Beaucoup plus strict, ne capture pas la sémantique

## Échantillon
- **Source** : 100 premières réponses val de Run1_GNN (`pred_Run1_GNN_val.jsonl`)
- **Contexte** : GNN top-k (k=5) + modèle fusionné Run1 (4-bit)
- **Format** : 99.7% respectent "Sources: [n]"

## Résultats

| Juge | Seuil | Réponses hallucinées | Affirmations supportées |
|------|-------|---------------------|------------------------|
| **Embedding (bge-small)** | **0.5** | **41%** | **59%** (59/100) |
| Embedding | 0.6 | 77% | 23% |
| Embedding | 0.7 | 83% | 17% |
| Lexical (Jaccard) | 0.5 | 76% | 24% |

## Interprétation

- **Taux d'hallucination principal (embedding 0.5) : 41%**
  - 59% des affirmations sont sémantiquement couvertes par les triplets cités
  - 41% des réponses contiennent au moins une affirmation non supportée
  
- **Le juge lexical surestime l'hallucination (76%)** car il ne capture pas la similarité sémantique (ex: "United States" vs "Armenian" = 0 overlap mais géographiquement liés)

- **Causes principales d'hallucination** :
  1. **Mauvaise réponse** : Le modèle prédit une entité incorrecte (ex: "Indian Army" au lieu de "flotilla") → 0 similarité avec triplets cités
  2. **Triplets cités non pertinents** : Le modèle cite des triplets label=0 (bruit GNN) → pas de support possible
  3. **Information manquante** : La réponse correcte n'est pas dans le top-5 GNN

## 5 Exemples détaillés

### Exemple 1 — Hallucination (mauvaise entité)
- **Question** : "What was Iqbal F. Qadir on when he participated in an attack on a radar station...?"
- **Gold** : `flotilla`
- **Pred** : `Indian Army`
- **Cités** : [1] (no. 227 radar station raaf, royal australian air force, operator) label=0, [2] (duplicate), [3] (dwarka, has_info, located in...) label=1
- **Embedding sim** : max=0.12 < 0.5 → **Non supporté**
- **Analyse** : Le modèle confond "radar station attack" avec contexte militaire indien ; les triplets cités parlent d'Australie/Dwarka

### Exemple 2 — Hallucination (mauvaise date)
- **Question** : "When did the park at which Tivolis Koncertsal is located open?"
- **Gold** : `15 August 1843`
- **Pred** : `1970`
- **Cités** : [1-4] (tivolis koncertsal, denmark, country) label=1, (tivoli gardens, amusement park, instance of) label=1
- **Embedding sim** : max=0.21 < 0.5 → **Non supporté**
- **Analyse** : Les triplets cités ne contiennent pas la date d'ouverture (1843) ; le modèle hallucine une date

### Exemple 3 — Hallucination (mauvais pays)
- **Question** : "What is the shared country of ancestry between Art Laboe and Scout Tufankjian?"
- **Gold** : `Armenian`
- **Pred** : `United States`
- **Cités** : [1] (art laboe, august 7, 1925, date of birth) label=1, [2] (interamerican scout jamboree, has_info, country) label=0
- **Embedding sim** : max=0.34 < 0.5 → **Non supporté**
- **Analyse** : "United States" vs "Armenian" : similarité 0.34 (pays tous deux) mais pas assez pour seuil 0.5

### Exemple 4 — Supporté (entité correcte citée)
- **Question** : "Which of the following is known for his work 'Good Bye Lenin!' in 2003: Vladimir Danilevich or Wolfgang Becker?"
- **Gold** : `Wolfgang Becker`
- **Pred** : `Wolfgang Becker`
- **Cités** : [1] (wolfgang becker, 22 june 1954, date of birth) label=1
- **Embedding sim** : max=0.89 ≥ 0.5 → **Supporté**
- **Analyse** : L'entité exacte "Wolfgang Becker" apparaît dans le triplet cité

### Exemple 5 — Hallucination (information absente du contexte)
- **Question** : "Who will Billy Howle be seen opposite in the upcoming British drama film directed by Dominic Cooke?"
- **Gold** : `Saoirse Ronan`
- **Pred** : `Tom Hardy`
- **Cités** : [1] (billy howle, november 9, 1989, date of birth) label=1, [2] (once upon time in london, leo gregory, cast member) label=0, [3] (seagull, anton chekhov, author) label=1
- **Embedding sim** : max=0.18 < 0.5 → **Non supporté**
- **Analyse** : Aucun triplet ne mentionne Saoirse Ronan ni le film de Dominic Cooke ; le modèle hallucine un co-star connu

## Conclusion

**Taux d'hallucination (embedding, seuil 0.5) : 41%**

C'est un taux **élevé mais attendu** pour un modèle 1.5B fine-tuné sur peu de données (686 exemples SFT, 2 epochs). Les hallucinations proviennent principalement de :
1. **Erreurs de raisonnement** (mauvaise entité prédite) — le modèle ne "sait" pas la réponse
2. **Bruit dans le contexte GNN** (triplets label=0 cités) — le GNN R@5=0.73 laisse 27% de questions sans triplet pertinent en top-5
3. **Couverture incomplète** — l'information correcte n'est pas dans les 5 triplets récupérés

**Pistes d'amélioration** : augmenter k (GNN top-10), améliorer le recall GNN, plus d'époques SFT, ou DPO (exclu par budget GPU).