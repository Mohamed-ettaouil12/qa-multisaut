"""Test extraction REBEL sur petit échantillon pour générer kg_raw val."""
import json
import torch
from pathlib import Path
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.kg.extract_rebel import extract_paragraph, load_extractor, parse_rebel_output


def paragraph_text(p: dict) -> str:
    return p["title"] + " " + " ".join(p["sentences"])


def find_sentence_with_object(sentences: list[str], obj: str) -> str:
    obj_lower = obj.lower()
    for s in sentences:
        if obj_lower in s.lower():
            return s
    return ""


def main():
    print("Chargement REBEL...")
    model, tokenizer = load_extractor()

    # Charger seulement val.jsonl, premières 5 questions
    with open("data/processed/val.jsonl") as f:
        val_data = [json.loads(line) for line in f][:5]

    print(f"Test sur {len(val_data)} questions")

    all_texts = []
    meta = []

    for ex in val_data:
        qid = ex["id"]
        for p in ex["paragraphs"]:
            all_texts.append(paragraph_text(p))
            meta.append((qid, p["title"], p["sentences"]))

    print(f"  {len(all_texts)} paragraphes")
    triplets_per_para = extract_paragraph(model, tokenizer, all_texts)

    results = []
    for (qid, para_title, sentences), triplets in zip(meta, triplets_per_para):
        for t in triplets:
            sentence = find_sentence_with_object(sentences, t["t"])
            results.append({
                "qid": qid,
                "para_title": para_title,
                "sentence": sentence,
                "h": t["h"],
                "r": t["r"],
                "t": t["t"],
            })

    out_path = Path("data/processed/kg_raw_val_part0.jsonl")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        for t in results:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    print(f"Écrit {len(results)} triplets dans {out_path}")


if __name__ == "__main__":
    main()