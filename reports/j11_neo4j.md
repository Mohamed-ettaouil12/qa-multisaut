# Neo4j Population Report - J11

## Comptages globaux
- **Nœuds (Entity + Question)** : 17 257
- **Arêtes (REL + HAS_TRIPLET)** : 43 040
- **Arêtes label=1 (supporting facts)** : 4 865
- **Entités uniques** : 15 302
- **Questions** : 1 955 (sur 2 000 attendues)

## Vérification Cypher - Voisinage 2 sauts (exemples)

### Question: 5a7a567255429941d65f25bd
```
(Question)-[:HAS_TRIPLET]->(operator)-[:REL {type: "new york city transit"}]->(roosevelt island tramway)
(roosevelt island tramway)-[:REL {type: "new york city transit"}]->(operator)
```
Autres exemples: SEPTA media-sharon hill trolley line, Amtrak northeast regional, USPS zip code, Korean consulate, KMB route 1a, Indian Railways shipra express, Wesleyan University van vleck observatory, Eastern Air Lines Lockheed super constellation, BLM castle mountains, NPS harry s. truman, Imperial Navy "nassau", KMB kowloon motor bus, Royal Navy "duke of edinburgh"-class, NPS pipe spring, BLM mojave trails, NPS truman home, Imperial Navy nassau-class battleship, NPS canyons of ancients

### Question: 5abca1a55542993a06baf937
(Aucun résultat 2 sauts retourné pour cette question dans les 20 premiers)

## Configuration
- **Image** : neo4j:5.15
- **Auth** : neo4j / REDACTED
- **Ports** : 7474 (HTTP), 7687 (Bolt)
- **Volumes** : neo4j_data, neo4j_logs, neo4j_import
- **Plugin APOC** : installé

## Notes
- 45 questions manquantes (1 955 vs 2 000) → probablement questions sans triplets extraits
- Le graphe est prêt pour les ablations Neo4j (Tâche 5B/5C)