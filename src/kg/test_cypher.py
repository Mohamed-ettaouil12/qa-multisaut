"""Test de requêtes Cypher sur le KG Neo4j.

Requête 2 sauts : MATCH (e:Entity {name:$name})-[r1:REL]-(m)-[r2:REL]-(o)
Testée sur 3 entités réelles de questions de val.jsonl + comptes globaux.
"""
import json
from pathlib import Path
from neo4j import GraphDatabase


URI = "bolt://localhost:7687"
AUTH = ("neo4j", "REDACTED")


def get_val_entities(limit: int = 3) -> list[str]:
    """Récupère quelques entités réelles depuis val.jsonl qui existent dans Neo4j."""
    # Entités connues présentes dans val et normalisées
    known_entities = [
        "tivoli gardens", "art laboe", "billy howle", "radar station",
        "trimingham", "wilmslow", "chesil beach", "saoirse ronan"
    ]
    # Normaliser
    normalized = [normalize_entity(e) for e in known_entities]
    return normalized[:limit]


def normalize_entity(text: str) -> str:
    text = text.lower().strip()
    words = text.split()
    words = [w for w in words if w not in {"a", "an", "the"}]
    return " ".join(words)


def run_2hop_query(driver, entity_name: str, limit: int = 30):
    """Exécute la requête 2 sauts pour une entité."""
    query = """
    MATCH (e:Entity {name: $name})-[r1:REL]-(m:Entity)-[r2:REL]-(o:Entity)
    RETURN e.name as source, r1.type as rel1, m.name as middle, r2.type as rel2, o.name as target
    LIMIT $limit
    """
    with driver.session() as session:
        result = session.run(query, name=entity_name, limit=limit)
        return [dict(record) for record in result]


def get_global_counts(driver):
    """Retourne les comptes globaux."""
    with driver.session() as session:
        nodes = session.run("MATCH (n) RETURN count(n) as c").single()["c"]
        edges = session.run("MATCH ()-[r]->() RETURN count(r) as c").single()["c"]
        edges_label1 = session.run("MATCH ()-[r:REL {label: 1}]->() RETURN count(r) as c").single()["c"]
    return nodes, edges, edges_label1


def main():
    driver = GraphDatabase.driver(URI, auth=AUTH)

    try:
        driver.verify_connectivity()
        print("Connecté à Neo4j\n")

        # Comptes globaux
        nodes, edges, edges_label1 = get_global_counts(driver)
        print("=== Comptes globaux ===")
        print(f"  Nœuds totaux: {nodes}")
        print(f"  Arêtes totales: {edges}")
        print(f"  Arêtes label=1: {edges_label1}\n")

        # Entités de test
        test_entities = get_val_entities(3)
        print(f"Entités de test: {test_entities}\n")

        for entity in test_entities:
            print(f"--- Requête 2 sauts pour: '{entity}' ---")
            results = run_2hop_query(driver, entity)
            if results:
                for r in results[:5]:  # Afficher max 5 résultats
                    print(f"  {r['source']} -[{r['rel1']}]-> {r['middle']} -[{r['rel2']}]-> {r['target']}")
                if len(results) > 5:
                    print(f"  ... et {len(results) - 5} autres")
            else:
                print("  Aucun résultat")
            print()

    finally:
        driver.close()


if __name__ == "__main__":
    main()