"""Évaluation baselines LLM sur val (J8) : sans contexte / cosinus / GNN."""
import json
import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import mlflow
from pathlib import Path
from typing import List, Dict, Any, Tuple

import torch
from src.llm.prompt import build_prompt_from_triplets
from src.llm.load_llm import load_base_model, generate_answer
from src.eval.metrics import exact_match, f1

SEED = 42
MAX_NEW_TOKENS = 64
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

torch.manual_seed(SEED)


def load_val_data() -> List[Dict]:
    with open("data/processed/val.jsonl") as f:
        return [json.loads(line) for line in f]


def load_triplets_by_qid() -> Dict[str, List[Dict]]:
    from collections import defaultdict
    by_qid = defaultdict(list)
    with open("data/processed/kg_triplets.jsonl") as f:
        for line in f:
            t = json.loads(line)
            by_qid[t["qid"]].append(t)
    return by_qid


def load_cos_topk_val() -> Dict[str, List[int]]:
    """Charge les indices top-k cosinus pour val."""
    result = {}
    with open("data/processed/cos_topk_val.jsonl") as f:
        for line in f:
            d = json.loads(line)
            result[d["qid"]] = d["topk_indices"]
    return result


def load_gnn_topk_val(k: int = 5) -> Dict[str, List[Dict]]:
    """Charge les prédictions GNN top-k pour val."""
    result = {}
    with open("data/processed/gnn_topk_val.jsonl") as f:
        for line in f:
            d = json.loads(line)
            result[d["qid"]] = d["triplets"]
    return result


def parse_sources(response: str) -> Tuple[str, List[int]]:
    """Parse la réponse pour extraire answer et sources.
    
    Returns:
        (answer_clean, sources_list)
    """
    if "Sources:" in response:
        parts = response.split("Sources:")
        answer = parts[0].strip()
        sources_str = parts[1].strip()
        # Extraire indices [1][2]...
        import re
        sources = [int(x) for x in re.findall(r"\[(\d+)\]", sources_str)]
    else:
        answer = response.strip()
        sources = []
    return answer, sources


def evaluate_config(
    config_name: str,
    model,
    tokenizer,
    val_data: List[Dict],
    qid_to_triplets: Dict[str, List[Dict]],
    get_context_triplets_fn,
) -> Dict[str, Any]:
    """Évalue une configuration de contexte."""
    em_scores = []
    f1_scores = []
    format_ok = 0
    insuffisant_a_tort = 0  # question a des positifs mais modèle répond Insuffisant
    total_with_pos = 0
    predictions = []

    for ex in val_data:
        qid = ex["id"]
        question = ex["question"]
        gold = ex["answer"]

        triplets = qid_to_triplets.get(qid, [])
        pos_indices = [i for i, t in enumerate(triplets) if t.get("label", 0) == 1]
        has_pos = len(pos_indices) > 0

        if has_pos:
            total_with_pos += 1

        # Obtenir triplets de contexte
        context_triplets = get_context_triplets_fn(qid, triplets)

        # Construire prompt
        prompt = build_prompt_from_triplets(question, context_triplets)

        # Générer
        response = generate_answer(model, tokenizer, prompt, MAX_NEW_TOKENS)
        pred_answer, pred_sources = parse_sources(response)

        # Vérifier format
        has_sources_format = "Sources:" in response
        if has_sources_format:
            format_ok += 1

        # Vérifier Insuffisant à tort
        if pred_answer == "Insuffisant" and has_pos:
            insuffisant_a_tort += 1

        # Calculer métriques
        em = exact_match(pred_answer, gold)
        f1_score = f1(pred_answer, gold)

        em_scores.append(em)
        f1_scores.append(f1_score)

        predictions.append({
            "qid": qid,
            "question": question,
            "gold": gold,
            "pred": pred_answer,
            "sources": pred_sources,
            "raw_response": response,
            "em": em,
            "f1": f1_score,
            "format_ok": has_sources_format,
        })

    avg_em = sum(em_scores) / len(em_scores) if em_scores else 0
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
    format_pct = format_ok / len(val_data) * 100 if val_data else 0
    insuffisant_pct = insuffisant_a_tort / total_with_pos * 100 if total_with_pos else 0

    return {
        "config": config_name,
        "n": len(val_data),
        "EM": avg_em,
        "F1": avg_f1,
        "format_ok_pct": format_pct,
        "insuffisant_a_tort_pct": insuffisant_pct,
        "predictions": predictions,
    }


def main():
    print(f"Chargement modèle {MODEL_NAME} (4-bit)...")
    model, tokenizer = load_base_model("cuda", use_4bit=True)
    print(f"Modèle chargé sur {model.device}")

    val_data = load_val_data()
    qid_to_triplets = load_triplets_by_qid()

    print(f"Questions val: {len(val_data)}")

    # Config C1: sans contexte
    def get_context_none(qid, triplets):
        return []

    # Config C2: top-5 cosinus
    cos_topk = load_cos_topk_val()
    def get_context_cos(qid, triplets):
        indices = cos_topk.get(qid, [])
        return [triplets[i] for i in indices if i < len(triplets)]

    # Config C3: GNN top-k (k=5)
    gnn_topk = load_gnn_topk_val(k=5)
    def get_context_gnn(qid, triplets):
        return gnn_topk.get(qid, [])

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("qa-multisaut")
    mlflow.start_run(run_name="baselines_llm_val")

    configs = [
        ("C1_sans_contexte", get_context_none),
        ("C2_top5_cosinus", get_context_cos),
        ("C3_gnn_top5", get_context_gnn),
    ]

    results = {}
    for name, fn in configs:
        print(f"\n=== {name} ===")
        res = evaluate_config(name, model, tokenizer, val_data, qid_to_triplets, fn)
        results[name] = res

        print(f"  EM: {res['EM']:.4f}")
        print(f"  F1: {res['F1']:.4f}")
        print(f"  Format OK: {res['format_ok_pct']:.1f}%")
        print(f"  Insuffisant à tort: {res['insuffisant_a_tort_pct']:.1f}%")

        # Log MLflow
        mlflow.log_metric(f"{name}_EM", res["EM"])
        mlflow.log_metric(f"{name}_F1", res["F1"])
        mlflow.log_metric(f"{name}_format_ok_pct", res["format_ok_pct"])
        mlflow.log_metric(f"{name}_insuffisant_a_tort_pct", res["insuffisant_a_tort_pct"])

    # Tableau final
    headers = ["Config", "EM", "F1", "Format OK %", "Insuffisant à tort %"]
    rows = []
    for name, res in results.items():
        rows.append([
            name,
            f"{res['EM']:.4f}",
            f"{res['F1']:.4f}",
            f"{res['format_ok_pct']:.1f}",
            f"{res['insuffisant_a_tort_pct']:.1f}",
        ])

    col_widths = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    lines = []
    lines.append("| " + " | ".join(h.ljust(w) for h, w in zip(headers, col_widths)) + " |")
    lines.append("| " + " | ".join("-" * w for w in col_widths) + " |")
    for row in rows:
        lines.append("| " + " | ".join(c.ljust(w) for c, w in zip(row, col_widths)) + " |")

    table = "\n".join(lines)
    print("\n=== TABLEAU FINAL ===")
    print(table)

    # Sauvegarder rapport
    report_path = Path("reports/j8_baselines_llm.md")
    report_path.parent.mkdir(exist_ok=True)
    with open(report_path, "w") as f:
        f.write("# Baselines LLM sur Val - J8\n\n")
        f.write(f"Modèle: {MODEL_NAME} (4-bit)\n")
        f.write(f"Max new tokens: {MAX_NEW_TOKENS}\n")
        f.write(f"Seed: {SEED}\n\n")
        f.write(table + "\n")

    # Sauvegarder prédictions détaillées
    for name, res in results.items():
        pred_path = Path(f"data/processed/pred_{name}_val.jsonl")
        with open(pred_path, "w") as f:
            for p in res["predictions"]:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(f"Prédictions sauvées: {pred_path}")

    mlflow.end_run()
    print(f"\nRapport sauvé: {report_path}")


if __name__ == "__main__":
    main()