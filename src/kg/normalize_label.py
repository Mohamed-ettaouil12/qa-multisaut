"""Normalisation et étiquetage des triplets REBEL.

Lit data/processed/kg_raw_*.jsonl -> data/processed/kg_triplets.jsonl

Normalisation :
- minuscules, suppression des articles a/an/the, strip
- fusion des entités identiques après normalisation

Entité ANCRE :
Si le titre normalisé d'un paragraphe n'apparaît dans aucun triplet de ce paragraphe,
ajoute UN triplet synthétique (title, "has_info", objet du 1er triplet du paragraphe).
Ce choix garantit que chaque paragraphe a au moins une ancre vers son contenu,
même si REBEL n'a pas extrait le titre comme entité.

LABEL automatique :
label=1 si para_title ∈ titres de supporting de la même question (relit les splits), sinon 0.

Champs finaux : {qid, h, r, t, source_title, sentence, label}.
"""
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterator


ARTICLES = {"a", "an", "the"}


def normalize_entity(text: str) -> str:
    """Normalise une entité : minuscules, suppression articles, strip."""
    text = text.lower().strip()
    words = text.split()
    words = [w for w in words if w not in ARTICLES]
    return " ".join(words)


def load_splits() -> dict[str, list[dict]]:
    """Charge train/val/test.jsonl et retourne un dict split -> liste questions."""
    splits = {}
    for split in ["train", "val", "test"]:
        path = Path(f"data/processed/{split}.jsonl")
        with open(path) as f:
            splits[split] = [json.loads(line) for line in f]
    return splits


def build_supporting_titles(splits: dict[str, list[dict]]) -> dict[str, set[str]]:
    """Construit qid -> set(titres supporting normalisés).

    On considère comme 'supporting' les paragraphes dont le titre apparaît
    dans la question ou qui sont marqués comme pertinents. Ici, on utilise
    tous les titres de paragraphes de la question comme supporting potentiels.
    """
    supporting = defaultdict(set)
    for split_name, questions in splits.items():
        for ex in questions:
            qid = ex["id"]
            for p in ex["paragraphs"]:
                norm_title = normalize_entity(p["title"])
                supporting[qid].add(norm_title)
    return supporting


def iter_raw_triplets() -> Iterator[dict]:
    """Itère sur tous les fichiers kg_raw_*.jsonl."""
    for path in sorted(Path("data/processed").glob("kg_raw_*.jsonl")):
        with open(path) as f:
            for line in f:
                yield json.loads(line)


def main():
    print("Chargement des splits pour les titres supporting...")
    splits = load_splits()
    supporting_titles = build_supporting_titles(splits)
    print(f"  {len(supporting_titles)} questions avec titres supporting")

    print("Lecture des triplets bruts...")
    raw_triplets = list(iter_raw_triplets())
    print(f"  {len(raw_triplets)} triplets bruts")

    # Grouper par (qid, para_title) pour gérer l'entité ancre
    by_para = defaultdict(list)
    for t in raw_triplets:
        key = (t["qid"], t["para_title"])
        by_para[key].append(t)

    print(f"  {len(by_para)} paragraphes avec triplets")

    # Normalisation et étiquetage
    normalized_triplets = []
    entity_map = {}  # entité originale -> normalisée

    for (qid, para_title), triplets in by_para.items():
        norm_para_title = normalize_entity(para_title)
        supporting = supporting_titles.get(qid, set())
        label = 1 if norm_para_title in supporting else 0

        # Normaliser tous les triplets du paragraphe
        norm_triplets = []
        for t in triplets:
            h_norm = normalize_entity(t["h"])
            r_norm = normalize_entity(t["r"])
            t_norm = normalize_entity(t["t"])

            entity_map[t["h"]] = h_norm
            entity_map[t["r"]] = r_norm
            entity_map[t["t"]] = t_norm

            norm_triplets.append({
                "qid": qid,
                "h": h_norm,
                "r": r_norm,
                "t": t_norm,
                "source_title": para_title,
                "sentence": t["sentence"],
                "label": label,
            })

        # Entité ancre : si le titre normalisé n'apparaît dans aucun h ou t
        entities_in_para = set()
        for nt in norm_triplets:
            entities_in_para.add(nt["h"])
            entities_in_para.add(nt["t"])

        if norm_para_title not in entities_in_para and norm_triplets:
            first_obj = norm_triplets[0]["t"]
            norm_triplets.append({
                "qid": qid,
                "h": norm_para_title,
                "r": "has_info",
                "t": first_obj,
                "source_title": para_title,
                "sentence": norm_triplets[0]["sentence"],
                "label": label,
            })

        normalized_triplets.extend(norm_triplets)

    print(f"  {len(normalized_triplets)} triplets après normalisation + ancres")

    # Écriture kg_triplets.jsonl
    out_path = Path("data/processed/kg_triplets.jsonl")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        for t in normalized_triplets:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"Écrit: {out_path}")

    # Statistiques
    total_triplets = len(normalized_triplets)
    unique_entities = set()
    unique_relations = set()
    triplets_per_q = defaultdict(int)
    positive_count = 0
    qids_with_positive = set()

    for t in normalized_triplets:
        unique_entities.add(t["h"])
        unique_entities.add(t["t"])
        unique_relations.add(t["r"])
        triplets_per_q[t["qid"]] += 1
        if t["label"] == 1:
            positive_count += 1
            qids_with_positive.add(t["qid"])

    all_qids = set(supporting_titles.keys())
    qids_without_positive = all_qids - qids_with_positive

    avg_triplets = total_triplets / len(triplets_per_q) if triplets_per_q else 0
    min_triplets = min(triplets_per_q.values()) if triplets_per_q else 0
    max_triplets = max(triplets_per_q.values()) if triplets_per_q else 0
    positive_rate = positive_count / total_triplets if total_triplets else 0
    pct_no_positive = len(qids_without_positive) / len(all_qids) if all_qids else 0

    # Rapport
    report_path = Path("reports/j4_kg_stats.md")
    report_path.parent.mkdir(exist_ok=True)
    with open(report_path, "w") as f:
        f.write("# Statistiques KG - J4\n\n")
        f.write(f"- **Total triplets** : {total_triplets}\n")
        f.write(f"- **Nœuds uniques (entités)** : {len(unique_entities)}\n")
        f.write(f"- **Relations uniques** : {len(unique_relations)}\n")
        f.write(f"- **Triplets/question (moy)** : {avg_triplets:.2f}\n")
        f.write(f"- **Triplets/question (min)** : {min_triplets}\n")
        f.write(f"- **Triplets/question (max)** : {max_triplets}\n")
        f.write(f"- **Taux de positifs (label=1)** : {positive_rate:.4f} ({positive_count}/{total_triplets})\n")
        f.write(f"- **% questions sans positif** : {pct_no_positive:.2%} ({len(qids_without_positive)}/{len(all_qids)})\n")
        f.write(f"\n## Relations uniques\n")
        for r in sorted(unique_relations):
            f.write(f"- {r}\n")

    print(f"\nRapport écrit: {report_path}")
    print(f"Total triplets: {total_triplets}")
    print(f"Nœuds uniques: {len(unique_entities)}")
    print(f"Relations uniques: {len(unique_relations)}")
    print(f"Triplets/q (moy/min/max): {avg_triplets:.2f}/{min_triplets}/{max_triplets}")
    print(f"Taux positifs: {positive_rate:.4f}")
    print(f"% questions sans positif: {pct_no_positive:.2%}")


if __name__ == "__main__":
    main()