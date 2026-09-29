"""Échantillonnage des splits HotpotQA (seed 42) -> data/processed/{train,val,test}.jsonl"""
import json, random
from datasets import load_dataset

SPLITS = {"train": 1500, "val": 200, "test": 300}
HF_SPLITS = {"train": "train", "val": "validation", "test": "validation"}

def to_record(ex):
    return {
        "id": ex["id"],
        "question": ex["question"],
        "answer": ex["answer"],
        "type": ex["type"],
        "paragraphs": [
            {"title": t, "sentences": sents}
            for t, sents in zip(ex["context"]["title"], ex["context"]["sentences"])
        ],
        "supporting": [
            {"title": t, "sent_id": i}
            for t, i in zip(ex["supporting_facts"]["title"], ex["supporting_facts"]["sent_id"])
        ],
    }

def main():
    random.seed(42)
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor")
    all_val_indices = list(range(len(ds["validation"])))
    random.shuffle(all_val_indices)
    val_idx = all_val_indices[:SPLITS["val"]]
    test_idx = all_val_indices[SPLITS["val"]:SPLITS["val"] + SPLITS["test"]]
    train_idx = random.sample(range(len(ds["train"])), SPLITS["train"])

    for name, n in SPLITS.items():
        if name == "train":
            idx = train_idx
        elif name == "val":
            idx = val_idx
        else:
            idx = test_idx
        path = f"data/processed/{name}.jsonl"
        hf_split = HF_SPLITS[name]
        with open(path, "w") as f:
            for ex in ds[hf_split].select(idx):
                f.write(json.dumps(to_record(ex), ensure_ascii=False) + "\n")
        print(f"OK {path}: {n} exemples")

if __name__ == "__main__":
    main()
