"""Test REBEL sur 50 premiers paragraphes de val.jsonl"""
import json
import random
from pathlib import Path

from src.kg.extract_rebel import load_extractor, extract_paragraph


def load_val_paragraphs(n: int = 50) -> list[dict]:
    """Charge les n premiers paragraphes de val.jsonl avec leur texte concaténé."""
    paragraphs = []
    with open("data/processed/val.jsonl") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            ex = json.loads(line)
            for p in ex["paragraphs"]:
                text = p["title"] + " " + " ".join(p["sentences"])
                paragraphs.append({
                    "qid": ex["id"],
                    "title": p["title"],
                    "text": text,
                })
    return paragraphs


def main():
    random.seed(42)

    print("Chargement du modèle REBEL...")
    model, tokenizer = load_extractor()

    print("Chargement des paragraphes...")
    paragraphs = load_val_paragraphs(50)
    print(f"Total paragraphes: {len(paragraphs)}")

    texts = [p["text"] for p in paragraphs]

    print("Extraction des triplets...")
    all_triplets = extract_paragraph(model, tokenizer, texts)

    # Statistiques
    triplet_counts = [len(t) for t in all_triplets]
    avg_triplets = sum(triplet_counts) / len(triplet_counts) if triplet_counts else 0
    pct_empty = sum(1 for c in triplet_counts if c == 0) / len(triplet_counts) * 100

    print(f"\n=== Statistiques REBEL ===")
    print(f"Triplets moyens par paragraphe: {avg_triplets:.2f}")
    print(f"% paragraphes sans triplet: {pct_empty:.1f}%")

    # Échantillon de 30 triplets aléatoires pour validation manuelle
    all_triplet_entries = []
    for p, triplets in zip(paragraphs, all_triplets):
        for t in triplets:
            all_triplet_entries.append({
                "source": p["text"][:120],
                "h": t["h"],
                "r": t["r"],
                "t": t["t"],
            })

    random.shuffle(all_triplet_entries)
    sample_30 = all_triplet_entries[:30]

    # Génération du rapport markdown
    report_path = Path("reports/j3_triplets_a_verifier.md")
    report_path.parent.mkdir(exist_ok=True)

    with open(report_path, "w") as f:
        f.write("# Triplets à vérifier manuellement (seed 42)\n\n")
        f.write("| source (tronqué 120c) | sujet | relation | objet |\n")
        f.write("|---|---|---|---|\n")
        for entry in sample_30:
            source = entry["source"].replace("|", "\\|")
            h = entry["h"].replace("|", "\\|")
            r = entry["r"].replace("|", "\\|")
            t = entry["t"].replace("|", "\\|")
            f.write(f"| {source} | {h} | {r} | {t} |\n")

    print(f"\nRapport généré: {report_path}")
    print(f"Total triplets extraits: {len(all_triplet_entries)}")
    print(f"Échantillon pour validation: {len(sample_30)}")


if __name__ == "__main__":
    main()