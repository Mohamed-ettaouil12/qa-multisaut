"""Charge les triplets normalisés dans Neo4j.

Contraintes d'unicité :
- Entity.name
- Question.qid

MERGE nœuds (:Entity {name}) et arêtes [:REL {type, source_title, qid, sentence, label}]
Nœuds (:Question {qid})
"""
import json
from pathlib import Path
from neo4j import GraphDatabase


URI = "bolt://localhost:7687"
AUTH = ("neo4j", "REDACTED")


def create_constraints(driver):
    """Crée les contraintes d'unicité."""
    with driver.session() as session:
        # Contrainte sur Entity.name
        session.run("""
            CREATE CONSTRAINT entity_name_unique IF NOT EXISTS
            FOR (e:Entity) REQUIRE e.name IS UNIQUE
        """)
        # Contrainte sur Question.qid
        session.run("""
            CREATE CONSTRAINT question_qid_unique IF NOT EXISTS
            FOR (q:Question) REQUIRE q.qid IS UNIQUE
        """)
    print("Contraintes créées")


def load_triplets(driver, triplets_path: Path):
    """Charge les triplets dans Neo4j par batch."""
    with open(triplets_path) as f:
        triplets = [json.loads(line) for line in f]

    print(f"Chargement de {len(triplets)} triplets...")

    batch_size = 1000
    for i in range(0, len(triplets), batch_size):
        batch = triplets[i:i + batch_size]
        with driver.session() as session:
            session.execute_write(_load_batch, batch)
        if (i // batch_size + 1) % 5 == 0:
            print(f"  {min(i + batch_size, len(triplets))}/{len(triplets)}")

    print("Chargement terminé")


def _load_batch(tx, batch):
    """Charge un batch de triplets en une transaction."""
    # Préparer les données pour UNWIND
    data = []
    for t in batch:
        data.append({
            "qid": t["qid"],
            "h": t["h"],
            "r": t["r"],
            "t": t["t"],
            "source_title": t["source_title"],
            "sentence": t["sentence"],
            "label": t["label"],
        })

    query = """
    UNWIND $batch AS row
    MERGE (q:Question {qid: row.qid})
    MERGE (h:Entity {name: row.h})
    MERGE (t:Entity {name: row.t})
    MERGE (h)-[rel:REL {
        type: row.r,
        source_title: row.source_title,
        qid: row.qid,
        sentence: row.sentence,
        label: row.label
    }]->(t)
    MERGE (q)-[:HAS_TRIPLET]->(h)
    MERGE (q)-[:HAS_TRIPLET]->(t)
    """
    tx.run(query, batch=data)


def get_counts(driver):
    """Retourne les comptes globaux."""
    with driver.session() as session:
        nodes = session.run("MATCH (n) RETURN count(n) as c").single()["c"]
        edges = session.run("MATCH ()-[r]->() RETURN count(r) as c").single()["c"]
        edges_label1 = session.run("MATCH ()-[r:REL {label: 1}]->() RETURN count(r) as c").single()["c"]
        entities = session.run("MATCH (e:Entity) RETURN count(e) as c").single()["c"]
        questions = session.run("MATCH (q:Question) RETURN count(q) as c").single()["c"]
    return {
        "nodes": nodes,
        "edges": edges,
        "edges_label1": edges_label1,
        "entities": entities,
        "questions": questions,
    }


def main():
    driver = GraphDatabase.driver(URI, auth=AUTH)

    try:
        driver.verify_connectivity()
        print("Connecté à Neo4j")

        create_constraints(driver)

        triplets_path = Path("data/processed/kg_triplets.jsonl")
        load_triplets(driver, triplets_path)

        counts = get_counts(driver)
        print("\n=== Comptes globaux ===")
        for k, v in counts.items():
            print(f"  {k}: {v}")

    finally:
        driver.close()


if __name__ == "__main__":
    main()