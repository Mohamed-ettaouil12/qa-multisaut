"""Évaluation modèles fine-tunés sur val (J10).

Compare: base+GNN (J8-C3) vs run1 vs run2
Métriques: EM, F1, format_ok, insuffisant_à_tort, latence moyenne
"""
import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import json
import time
import torch
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict

import mlflow
from src.llm.prompt import build_prompt_from_triplets
from src.llm.load_llm import load_base_model, load_qlora_model, generate_answer
from src.eval.metrics import exact_match, f1

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
RUN1_ADAPTER = "checkpoints/qlora_run1/final"
RUN2_ADAPTER = "checkpoints/qlora_run2/final"
MAX_NEW_TOKENS = 64
SEED = 42

torch.manual_seed(SEED)


def load_val_data() -> List[Dict]:
    with open("data/processed/val.jsonl") as f:
        return [json.loads(line) for line in f]


def load_triplets_by_qid() -> Dict[str, List[Dict]]:
    by_qid = defaultdict(list)
    with open("data/processed/kg_triplets.jsonl") as f:
        for line in f:
            t = json.loads(line)
            by_qid[t["qid"]].append(t)
    return by_qid


def load_gnn_topk_val() -> Dict[str, List[Dict]]:
    result = {}
    with open("data/processed/gnn_topk_val.jsonl") as f:
        for line in f:
            d = json.loads(line)
            result[d["qid"]] = d["triplets"]
    return result


def parse_sources(response: str) -> Tuple[str, List[int]]:
    if "Sources:" in response:
        parts = response.split("Sources:")
        answer = parts[0].strip()
        sources_str = parts[1].strip()
        import re
        sources = [int(x) for x in re.findall(r"\[(\d+)\]", sources_str)]
    else:
        answer = response.strip()
        sources = []
    return answer, sources


def evaluate_model(
    config_name: str,
    model,
    tokenizer,
    val_data: List[Dict],
    qid_to_triplets: Dict[str, List[Dict]],
    gnn_topk: Dict[str, List[Dict]],
) -> Dict[str, Any]:
    em_scores = []
    f1_scores = []
    format_ok = 0
    insuffisant_a_tort = 0
    total_with_pos = 0
    latencies = []
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

        gnn_triplets = gnn_topk.get(qid, [])

        prompt = build_prompt_from_triplets(question, gnn_triplets)

        start = time.time()
        response = generate_answer(model, tokenizer, prompt, MAX_NEW_TOKENS)
        latency = time.time() - start
        latencies.append(latency)

        pred_answer, pred_sources = parse_sources(response)

        has_sources_format = "Sources:" in response
        if has_sources_format:
            format_ok += 1

        if pred_answer == "Insuffisant" and has_pos:
            insuffisant_a_tort += 1

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
            "em": int(em),
            "f1": round(f1_score, 4),
            "format_ok": has_sources_format,
            "latency": round(latency, 3),
        })

    avg_em = sum(em_scores) / len(em_scores) if em_scores else 0
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
    format_pct = format_ok / len(val_data) * 100 if val_data else 0
    insuffisant_pct = insuffisant_a_tort / total_with_pos * 100 if total_with_pos else 0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0

    return {
        "config": config_name,
        "n": len(val_data),
        "EM": avg_em,
        "F1": avg_f1,
        "format_ok_pct": format_pct,
        "insuffisant_a_tort_pct": insuffisant_pct,
        "avg_latency": avg_latency,
        "predictions": predictions,
    }


def main():
    print("Chargement données val...")
    val_data = load_val_data()
    qid_to_triplets = load_triplets_by_qid()
    gnn_topk = load_gnn_topk_val()

    print(f"Questions val: {len(val_data)}")

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("qa-multisaut")
    mlflow.start_run(run_name="eval_finetuned_val")

    results = {}

    # 1. Base model + GNN (J8-C3)
    print("\n=== Base + GNN (J8-C3) ===")
    base_model, base_tokenizer = load_base_model("cuda", use_4bit=True)
    res_base = evaluate_model("Base_GNN", base_model, base_tokenizer, val_data, qid_to_triplets, gnn_topk)
    results["Base_GNN"] = res_base
    print(f"  EM: {res_base['EM']:.4f}")
    print(f"  F1: {res_base['F1']:.4f}")
    print(f"  Format OK: {res_base['format_ok_pct']:.1f}%")
    print(f"  Insuffisant à tort: {res_base['insuffisant_a_tort_pct']:.1f}%")
    print(f"  Latence moy: {res_base['avg_latency']:.3f}s")

    del base_model
    torch.cuda.empty_cache()

    # 2. Run 1 adapter + GNN
    print("\n=== Run 1 + GNN ===")
    run1_model, run1_tokenizer = load_qlora_model(RUN1_ADAPTER, "cuda")
    res_run1 = evaluate_model("Run1_GNN", run1_model, run1_tokenizer, val_data, qid_to_triplets, gnn_topk)
    results["Run1_GNN"] = res_run1
    print(f"  EM: {res_run1['EM']:.4f}")
    print(f"  F1: {res_run1['F1']:.4f}")
    print(f"  Format OK: {res_run1['format_ok_pct']:.1f}%")
    print(f"  Insuffisant à tort: {res_run1['insuffisant_a_tort_pct']:.1f}%")
    print(f"  Latence moy: {res_run1['avg_latency']:.3f}s")

    del run1_model
    torch.cuda.empty_cache()

    # 3. Run 2 adapter + GNN
    print("\n=== Run 2 + GNN ===")
    run2_model, run2_tokenizer = load_qlora_model(RUN2_ADAPTER, "cuda")
    res_run2 = evaluate_model("Run2_GNN", run2_model, run2_tokenizer, val_data, qid_to_triplets, gnn_topk)
    results["Run2_GNN"] = res_run2
    print(f"  EM: {res_run2['EM']:.4f}")
    print(f"  F1: {res_run2['F1']:.4f}")
    print(f"  Format OK: {res_run2['format_ok_pct']:.1f}%")
    print(f"  Insuffisant à tort: {res_run2['insuffisant_a_tort_pct']:.1f}%")
    print(f"  Latence moy: {res_run2['avg_latency']:.3f}s")

    # Log MLflow
    for name, res in results.items():
        mlflow.log_metric(f"{name}_EM", res["EM"])
        mlflow.log_metric(f"{name}_F1", res["F1"])
        mlflow.log_metric(f"{name}_format_ok_pct", res["format_ok_pct"])
        mlflow.log_metric(f"{name}_insuffisant_a_tort_pct", res["insuffisant_a_tort_pct"])
        mlflow.log_metric(f"{name}_avg_latency", res["avg_latency"])

    # Tableau final
    headers = ["Config", "EM", "F1", "Format OK %", "Insuffisant à tort %", "Latence (s)"]
    rows = []
    for name, res in results.items():
        rows.append([
            name,
            f"{res['EM']:.4f}",
            f"{res['F1']:.4f}",
            f"{res['format_ok_pct']:.1f}",
            f"{res['insuffisant_a_tort_pct']:.1f}",
            f"{res['avg_latency']:.3f}",
        ])

    col_widths = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    lines = []
    lines.append("| " + " | ".join(h.ljust(w) for h, w in zip(headers, col_widths)) + " |")
    lines.append("| " + " | ".join("-" * w for w in col_widths) + " |")
    for row in rows:
        lines.append("| " + " | ".join(c.ljust(w) for c, w in zip(row, col_widths)) + " |")

    table = "\n".join(lines)
    print("\n=== TABLEAU COMPARATIF ===")
    print(table)

    # Sauvegarder rapport
    report_path = Path("reports/j10_selection.md")
    report_path.parent.mkdir(exist_ok=True)
    with open(report_path, "w") as f:
        f.write("# Sélection Adaptateur - J10\n\n")
        f.write("Comparaison: Base+GNN (J8-C3) vs Run1 (r=16, 2ep) vs Run2 (r=32, 4ep)\n\n")
        f.write(f"Modèle de base: {MODEL_NAME} (4-bit)\n")
        f.write(f"Contexte: GNN top-k (val)\n")
        f.write(f"Max new tokens: {MAX_NEW_TOKENS}\n")
        f.write(f"Seed: {SEED}\n\n")
        f.write(table + "\n\n")

        # Analyse
        baseline_f1 = results["Base_GNN"]["F1"]
        run1_f1 = results["Run1_GNN"]["F1"]
        run2_f1 = results["Run2_GNN"]["F1"]

        f.write("## Analyse\n\n")
        f.write(f"- **Baseline F1 (J8-C3)**: {baseline_f1:.4f}\n")
        f.write(f"- **Run 1 F1**: {run1_f1:.4f} ({'↑' if run1_f1 > baseline_f1 else '↓'} {run1_f1 - baseline_f1:+.4f})\n")
        f.write(f"- **Run 2 F1**: {run2_f1:.4f} ({'↑' if run2_f1 > baseline_f1 else '↓'} {run2_f1 - baseline_f1:+.4f})\n\n")

        if run2_f1 >= baseline_f1:
            f.write("**✅ Run 2 améliore le F1 vs baseline** → Sélection: Run 2\n")
            best = "Run2_GNN"
        elif run1_f1 >= baseline_f1:
            f.write("**✅ Run 1 améliore le F1 vs baseline** → Sélection: Run 1\n")
            best = "Run1_GNN"
        else:
            f.write("**❌ Aucun run ne dépasse le baseline** → Résultat négatif documenté\n")
            f.write("Possibles causes: format non acquis (peu d'époques), données SFT bruitées (40% cosinus),\n")
            f.write("perte completion-only trop restrictive, ou modèle de base déjà fort sur ce task.\n")
            best = "Base_GNN"

        f.write(f"\n**Meilleur config**: {best}\n")

    print(f"\nRapport sauvé: {report_path}")

    # Sauvegarder prédictions détaillées
    for name, res in results.items():
        pred_path = Path(f"data/processed/pred_{name}_val.jsonl")
        with open(pred_path, "w") as f:
            for p in res["predictions"]:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(f"Prédictions sauvées: {pred_path}")

    mlflow.end_run()


if __name__ == "__main__":
    main()