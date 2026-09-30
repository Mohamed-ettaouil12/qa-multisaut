"""Baseline retrieval niveau triplet sur val.jsonl (référence pour GNN).

Candidats = triplets de kg_triplets.jsonl groupés par qid.
Gold = label=1.
Exclut les questions sans positif.
"""
import json
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Tuple

import mlflow
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


def load_triplets_by_qid(path: str) -> Dict[str, List[Dict]]:
    """Charge kg_triplets.jsonl et groupe par qid."""
    by_qid = defaultdict(list)
    with open(path) as f:
        for line in f:
            t = json.loads(line)
            by_qid[t["qid"]].append(t)
    return by_qid


def load_val_qids(path: str) -> List[str]:
    """Retourne la liste des qid du split val."""
    with open(path) as f:
        return [json.loads(line)["id"] for line in f]


def triplet_text(t: Dict) -> str:
    """Texte pour embedding: 'h r t'."""
    return f"{t['h']} {t['r']} {t['t']}"


def triplet_tokens(t: Dict) -> List[str]:
    """Tokens pour BM25: tokens de 'h r t'."""
    return triplet_text(t).lower().split()


def compute_recall(gold_set: set, pred_list: List[int], k: int) -> float:
    pred_set = set(pred_list[:k])
    if not gold_set:
        return 0.0
    return len(gold_set & pred_set) / len(gold_set)


def compute_precision(gold_set: set, pred_list: List[int], k: int) -> float:
    if k == 0:
        return 0.0
    pred_set = set(pred_list[:k])
    return len(gold_set & pred_set) / k


def compute_mrr(gold_set: set, pred_list: List[int]) -> float:
    for rank, idx in enumerate(pred_list, 1):
        if idx in gold_set:
            return 1.0 / rank
    return 0.0


def evaluate_method(
    method_name: str,
    qid_to_triplets: Dict[str, List[Dict]],
    val_qids: List[str],
    k_values: List[int],
) -> Dict[str, float]:
    """Évalue une méthode de ranking sur val."""
    recalls = {k: [] for k in k_values}
    precisions = {k: [] for k in k_values}
    mrrs = []

    excluded_no_positive = 0
    evaluated = 0

    for qid in val_qids:
        triplets = qid_to_triplets.get(qid, [])
        if not triplets:
            continue

        gold_indices = [i for i, t in enumerate(triplets) if t["label"] == 1]
        if not gold_indices:
            excluded_no_positive += 1
            continue

        gold_set = set(gold_indices)
        evaluated += 1

        # Cette fonction sera remplacée par la méthode spécifique
        ranked_indices = list(range(len(triplets)))

        for k in k_values:
            recalls[k].append(compute_recall(gold_set, ranked_indices, k))
            precisions[k].append(compute_precision(gold_set, ranked_indices, k))
        mrrs.append(compute_mrr(gold_set, ranked_indices))

    return {
        "method": method_name,
        "evaluated": evaluated,
        "excluded_no_positive": excluded_no_positive,
        **{f"R@{k}": sum(recalls[k]) / len(recalls[k]) if recalls[k] else 0.0 for k in k_values},
        **{f"P@{k}": sum(precisions[k]) / len(precisions[k]) if precisions[k] else 0.0 for k in k_values},
        "MRR": sum(mrrs) / len(mrrs) if mrrs else 0.0,
    }


def evaluate_cosine(
    qid_to_triplets: Dict[str, List[Dict]],
    val_qids: List[str],
    val_questions: Dict[str, str],
    model: SentenceTransformer,
    k_values: List[int],
) -> Dict[str, float]:
    """Évalue la méthode cosinus (bge-small-en-v1.5)."""
    recalls = {k: [] for k in k_values}
    precisions = {k: [] for k in k_values}
    mrrs = []

    excluded_no_positive = 0
    evaluated = 0

    for qid in val_qids:
        triplets = qid_to_triplets.get(qid, [])
        if not triplets:
            continue

        gold_indices = [i for i, t in enumerate(triplets) if t["label"] == 1]
        if not gold_indices:
            excluded_no_positive += 1
            continue

        gold_set = set(gold_indices)
        evaluated += 1

        question = val_questions[qid]
        query_text = f"Represent this sentence for searching relevant passages: {question}"

        triplet_texts = [triplet_text(t) for t in triplets]
        passage_texts = [f"Represent this passage for retrieval: {text}" for text in triplet_texts]

        with torch.no_grad():
            query_emb = model.encode([query_text], normalize_embeddings=True, show_progress_bar=False, batch_size=1, convert_to_tensor=True)
            triplet_embs = model.encode(passage_texts, normalize_embeddings=True, show_progress_bar=False, batch_size=64, convert_to_tensor=True)

        scores = (triplet_embs @ query_emb.T).squeeze()
        if scores.ndim == 0:
            scores = scores.unsqueeze(0)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i].item(), reverse=True)

        for k in k_values:
            recalls[k].append(compute_recall(gold_set, ranked_indices, k))
            precisions[k].append(compute_precision(gold_set, ranked_indices, k))
        mrrs.append(compute_mrr(gold_set, ranked_indices))

    return {
        "method": "cosine",
        "evaluated": evaluated,
        "excluded_no_positive": excluded_no_positive,
        **{f"R@{k}": sum(recalls[k]) / len(recalls[k]) if recalls[k] else 0.0 for k in k_values},
        **{f"P@{k}": sum(precisions[k]) / len(precisions[k]) if precisions[k] else 0.0 for k in k_values},
        "MRR": sum(mrrs) / len(mrrs) if mrrs else 0.0,
    }


def evaluate_bm25(
    qid_to_triplets: Dict[str, List[Dict]],
    val_qids: List[str],
    val_questions: Dict[str, str],
    k_values: List[int],
) -> Dict[str, float]:
    """Évalue la méthode BM25."""
    recalls = {k: [] for k in k_values}
    precisions = {k: [] for k in k_values}
    mrrs = []

    excluded_no_positive = 0
    evaluated = 0

    for qid in val_qids:
        triplets = qid_to_triplets.get(qid, [])
        if not triplets:
            continue

        gold_indices = [i for i, t in enumerate(triplets) if t["label"] == 1]
        if not gold_indices:
            excluded_no_positive += 1
            continue

        gold_set = set(gold_indices)
        evaluated += 1

        question = val_questions[qid]
        query_tokens = question.lower().split()

        corpus = [triplet_tokens(t) for t in triplets]
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(query_tokens)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        for k in k_values:
            recalls[k].append(compute_recall(gold_set, ranked_indices, k))
            precisions[k].append(compute_precision(gold_set, ranked_indices, k))
        mrrs.append(compute_mrr(gold_set, ranked_indices))

    return {
        "method": "bm25",
        "evaluated": evaluated,
        "excluded_no_positive": excluded_no_positive,
        **{f"R@{k}": sum(recalls[k]) / len(recalls[k]) if recalls[k] else 0.0 for k in k_values},
        **{f"P@{k}": sum(precisions[k]) / len(precisions[k]) if precisions[k] else 0.0 for k in k_values},
        "MRR": sum(mrrs) / len(mrrs) if mrrs else 0.0,
    }


def load_val_questions(path: str) -> Dict[str, str]:
    """Charge qid -> question depuis val.jsonl."""
    q_dict = {}
    with open(path) as f:
        for line in f:
            ex = json.loads(line)
            q_dict[ex["id"]] = ex["question"]
    return q_dict


def format_table(results: List[Dict], k_values: List[int]) -> str:
    """Formate le tableau markdown."""
    headers = ["Méthode"] + [f"R@{k}" for k in k_values] + [f"P@{k}" for k in k_values] + ["MRR"]
    rows = []
    for r in results:
        row = [r["method"]]
        for k in k_values:
            row.append(f"{r[f'R@{k}']:.4f}")
        for k in k_values:
            row.append(f"{r[f'P@{k}']:.4f}")
        row.append(f"{r['MRR']:.4f}")
        rows.append(row)

    # Calculer largeurs
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    lines = []
    lines.append("| " + " | ".join(h.ljust(w) for h, w in zip(headers, col_widths)) + " |")
    lines.append("| " + " | ".join("-" * w for w in col_widths) + " |")
    for row in rows:
        lines.append("| " + " | ".join(c.ljust(w) for c, w in zip(row, col_widths)) + " |")
    return "\n".join(lines)


import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

def main():
    print("Chargement des données...")
    qid_to_triplets = load_triplets_by_qid("data/processed/kg_triplets.jsonl")
    val_qids = load_val_qids("data/processed/val.jsonl")
    val_questions = load_val_questions("data/processed/val.jsonl")

    print(f"  {len(qid_to_triplets)} qids avec triplets")
    print(f"  {len(val_qids)} questions val")

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("qa-multisaut")
    mlflow.start_run(run_name="baseline_triplet_val")

    model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cuda")
    k_values = [3, 5, 10]

    print("\n=== Évaluation Cosinus (bge-small-en-v1.5) ===")
    cosine_res = evaluate_cosine(qid_to_triplets, val_qids, val_questions, model, k_values)
    print(f"  Questions évaluées: {cosine_res['evaluated']}, exclues (sans positif): {cosine_res['excluded_no_positive']}")

    print("\n=== Évaluation BM25 ===")
    bm25_res = evaluate_bm25(qid_to_triplets, val_qids, val_questions, k_values)
    print(f"  Questions évaluées: {bm25_res['evaluated']}, exclues (sans positif): {bm25_res['excluded_no_positive']}")

    # Log MLflow
    for k in k_values:
        mlflow.log_metric(f"cosine_R{k}", cosine_res[f"R@{k}"])
        mlflow.log_metric(f"cosine_P{k}", cosine_res[f"P@{k}"])
        mlflow.log_metric(f"bm25_R{k}", bm25_res[f"R@{k}"])
        mlflow.log_metric(f"bm25_P{k}", bm25_res[f"P@{k}"])
    mlflow.log_metric("cosine_MRR", cosine_res["MRR"])
    mlflow.log_metric("bm25_MRR", bm25_res["MRR"])
    mlflow.log_metric("evaluated_questions", cosine_res["evaluated"])
    mlflow.log_metric("excluded_no_positive", cosine_res["excluded_no_positive"])

    # Tableau final
    table = format_table([cosine_res, bm25_res], k_values)
    print("\n=== TABLEAU FINAL ===")
    print(table)

    # Sauvegarde rapport
    report_path = Path("reports/j5_baseline_triplet.md")
    report_path.parent.mkdir(exist_ok=True)
    with open(report_path, "w") as f:
        f.write("# Baseline Triplet - Val\n\n")
        f.write(f"Questions évaluées: {cosine_res['evaluated']}\n")
        f.write(f"Questions exclues (sans triplet label=1): {cosine_res['excluded_no_positive']}\n\n")
        f.write(table + "\n")

    print(f"\nRapport sauvé: {report_path}")

    mlflow.end_run()


if __name__ == "__main__":
    main()