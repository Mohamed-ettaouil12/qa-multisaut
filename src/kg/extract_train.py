"""Extraction REBEL sur train complet (1500 questions)."""
import json
from pathlib import Path
from src.kg.extract_rebel import extract_paragraph, load_extractor
from src.kg.run_extraction import paragraph_text, find_sentence_with_object


def main():
    print("Chargement REBEL...")
    model, tokenizer = load_extractor()

    with open("data/processed/train.jsonl") as f:
        train_data = [json.loads(line) for line in f]

    print(f"Extraction sur {len(train_data)} questions (train)")

    all_texts = []
    meta = []

    for ex in train_data:
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

    out_path = Path("data/processed/kg_raw_train_part0.jsonl")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        for t in results:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    print(f"Écrit {len(results)} triplets dans {out_path}")


if __name__ == "__main__":
    main()