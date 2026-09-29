"""Baseline retrieval niveau paragraphe (distractor) sur val.jsonl"""
import json
from collections import defaultdict
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import mlflow
import mlflow.tracking
import torch


def load_val_data(path="data/processed/val.jsonl"):
    with open(path) as f:
        return [json.loads(line) for line in f]


def bm25_rank(paragraphs, query):
    corpus = []
    for p in paragraphs:
        tokens = p["title"].lower().split() + " ".join(p["sentences"]).lower().split()
        corpus.append(tokens)
    bm25 = BM25Okapi(corpus)
    query_tokens = query.lower().split()
    scores = bm25.get_scores(query_tokens)
    ranked = sorted(zip(range(len(paragraphs)), scores), key=lambda x: x[1], reverse=True)
    return [i for i, _ in ranked]


def dense_rank(paragraphs, query, model):
    texts = [p["title"] + " " + " ".join(p["sentences"]) for p in paragraphs]
    query_text = "Represent this sentence for searching relevant passages: " + query
    with torch.no_grad():
        para_emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False, batch_size=10)
        query_emb = model.encode([query_text], normalize_embeddings=True, show_progress_bar=False, batch_size=1)
    scores = (para_emb @ query_emb.T).squeeze()
    ranked = sorted(zip(range(len(paragraphs)), scores), key=lambda x: x[1], reverse=True)
    return [i for i, _ in ranked]


def rrf_rank(bm25_order, dense_order, k=60):
    scores = defaultdict(float)
    for rank, idx in enumerate(bm25_order):
        scores[idx] += 1 / (k + rank + 1)
    for rank, idx in enumerate(dense_order):
        scores[idx] += 1 / (k + rank + 1)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in ranked]


def recall_at_k(gold_titles, pred_titles, k):
    pred_set = set(pred_titles[:k])
    return int(gold_titles.issubset(pred_set))


def main():
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("qa-multisaut")
    mlflow.start_run(run_name="baseline_paragraphe_val")

    val_data = load_val_data()
    model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cuda")

    metrics = {
        "bm25_r2": 0, "bm25_r3": 0,
        "dense_r2": 0, "dense_r3": 0,
        "rrf_r2": 0, "rrf_r3": 0,
    }
    n = len(val_data)

    for ex in val_data:
        paragraphs = ex["paragraphs"]
        gold = {s["title"] for s in ex["supporting"]}
        query = ex["question"]

        bm25_idx = bm25_rank(paragraphs, query)
        bm25_titles = [paragraphs[i]["title"] for i in bm25_idx]

        dense_idx = dense_rank(paragraphs, query, model)
        dense_titles = [paragraphs[i]["title"] for i in dense_idx]

        rrf_idx = rrf_rank(bm25_idx, dense_idx)
        rrf_titles = [paragraphs[i]["title"] for i in rrf_idx]

        metrics["bm25_r2"] += recall_at_k(gold, bm25_titles, 2)
        metrics["bm25_r3"] += recall_at_k(gold, bm25_titles, 3)
        metrics["dense_r2"] += recall_at_k(gold, dense_titles, 2)
        metrics["dense_r3"] += recall_at_k(gold, dense_titles, 3)
        metrics["rrf_r2"] += recall_at_k(gold, rrf_titles, 2)
        metrics["rrf_r3"] += recall_at_k(gold, rrf_titles, 3)

    for k in metrics:
        metrics[k] /= n
        mlflow.log_metric(k, metrics[k])

    print(f"{'Méthode':<10} {'Recall@2':>10} {'Recall@3':>10}")
    print("-" * 32)
    print(f"{'BM25':<10} {metrics['bm25_r2']:>10.3f} {metrics['bm25_r3']:>10.3f}")
    print(f"{'Dense':<10} {metrics['dense_r2']:>10.3f} {metrics['dense_r3']:>10.3f}")
    print(f"{'RRF':<10} {metrics['rrf_r2']:>10.3f} {metrics['rrf_r3']:>10.3f}")

    mlflow.end_run()


if __name__ == "__main__":
    main()