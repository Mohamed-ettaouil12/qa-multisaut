"""Analyse exploratoire de data/processed/train.jsonl"""
import json
from collections import Counter


def main():
    path = "data/processed/train.jsonl"

    n_total = 0
    type_counter = Counter()
    yes_no_count = 0
    total_context_len = 0
    supporting_title_counts = []

    with open(path) as f:
        for line in f:
            ex = json.loads(line)
            n_total += 1

            type_counter[ex["type"]] += 1

            ans = ex["answer"].strip().lower()
            if ans in ("yes", "no"):
                yes_no_count += 1

            ctx_len = sum(len("".join(p["sentences"])) for p in ex["paragraphs"])
            total_context_len += ctx_len

            supporting_titles = {s["title"] for s in ex["supporting"]}
            supporting_title_counts.append(len(supporting_titles))

    print(f"Total exemples: {n_total}")
    print(f"Distribution types: {dict(type_counter)}")
    print(f"% réponses yes/no: {yes_no_count / n_total * 100:.1f}%")
    print(f"Longueur moyenne contexte (chars): {total_context_len / n_total:.0f}")
    print(f"Nb titres uniques dans supporting (distribution): {Counter(supporting_title_counts)}")


if __name__ == "__main__":
    main()