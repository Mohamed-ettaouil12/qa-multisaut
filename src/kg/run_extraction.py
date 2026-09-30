"""Extraction complète REBEL par blocs de 500 questions avec reprise."""
import json
import time
from pathlib import Path
from typing import Iterator

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.kg.extract_rebel import extract_paragraph, load_extractor, parse_rebel_output


SPLITS = {"train": 1500, "val": 200, "test": 300}
BLOCK_SIZE = 500


def load_split(split: str) -> list[dict]:
    """Charge un split complet (train/val/test).jsonl"""
    path = f"data/processed/{split}.jsonl"
    with open(path) as f:
        return [json.loads(line) for line in f]


def iter_blocks(data: list[dict], block_size: int) -> Iterator[tuple[int, list[dict]]]:
    """Itère sur les blocs de questions."""
    for i in range(0, len(data), block_size):
        yield i // block_size, data[i:i + block_size]


def paragraph_text(p: dict) -> str:
    return p["title"] + " " + " ".join(p["sentences"])


def find_sentence_with_object(sentences: list[str], obj: str) -> str:
    """Retourne la phrase contenant l'objet (insensible à la casse), sinon ''."""
    obj_lower = obj.lower()
    for s in sentences:
        if obj_lower in s.lower():
            return s
    return ""


def extract_block(model, tokenizer, questions: list[dict]) -> list[dict]:
    """Extrait les triplets pour un bloc de questions."""
    all_texts = []
    meta = []  # (qid, para_title, sentences)

    for ex in questions:
        qid = ex["id"]
        for p in ex["paragraphs"]:
            all_texts.append(paragraph_text(p))
            meta.append((qid, p["title"], p["sentences"]))

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
    return results


def main():
    print("Chargement du modèle REBEL...")
    model, tokenizer = load_extractor()

    for split in ["train", "val", "test"]:
        print(f"\n=== Split: {split} ===")
        data = load_split(split)
        n_questions = len(data)
        n_blocks = (n_questions + BLOCK_SIZE - 1) // BLOCK_SIZE
        print(f"Questions: {n_questions}, Blocs: {n_blocks}")

        out_dir = Path("data/processed")
        out_dir.mkdir(exist_ok=True)

        start_time = time.time()

        for block_idx, block_questions in iter_blocks(data, BLOCK_SIZE):
            part_path = out_dir / f"kg_raw_{split}_part{block_idx}.jsonl"

            if part_path.exists():
                print(f"  Bloc {block_idx+1}/{n_blocks} : existe déjà, ignoré.")
                continue

            print(f"  Bloc {block_idx+1}/{n_blocks} ({len(block_questions)} questions)...")
            block_start = time.time()

            triplets = extract_block(model, tokenizer, block_questions)

            with open(part_path, "w") as f:
                for t in triplets:
                    f.write(json.dumps(t, ensure_ascii=False) + "\n")

            elapsed = time.time() - block_start
            total_elapsed = time.time() - start_time
            blocks_done = block_idx + 1
            if blocks_done < n_blocks:
                eta = total_elapsed / blocks_done * (n_blocks - blocks_done)
                print(f"    → {len(triplets)} triplets, {elapsed:.1f}s, ETA: {eta/60:.1f}min")
            else:
                print(f"    → {len(triplets)} triplets, {elapsed:.1f}s, TERMINÉ")

        print(f"Split {split} terminé en {(time.time() - start_time)/60:.1f}min")


if __name__ == "__main__":
    main()