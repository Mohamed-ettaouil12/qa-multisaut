"""Construction dataset SFT pour fine-tuning LLM (J8)."""
import json
import random
import torch
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any, Tuple

from sentence_transformers import SentenceTransformer
from src.llm.prompt import build_prompt_from_triplets

SEED = 42
MAX_TRIPLETS_PER_QUESTION = 15
RANDOM_DISTRACTOR_RATIO = 0.6  # 60% avec positifs + distracteurs aléatoires
COSINE_TOPK_RATIO = 0.4  # 40% avec top-5 cosinus
COSINE_K = 5

random.seed(SEED)
torch.manual_seed(SEED)


def load_questions(split: str) -> List[Dict]:
    with open(f"data/processed/{split}.jsonl") as f:
        return [json.loads(line) for line in f]


def load_triplets_by_qid() -> Dict[str, List[Dict]]:
    by_qid = defaultdict(list)
    with open("data/processed/kg_triplets.jsonl") as f:
        for line in f:
            t = json.loads(line)
            by_qid[t["qid"]].append(t)
    return by_qid


def triplet_text(t: Dict) -> str:
    return f"{t['h']} {t['r']} {t['t']}"


def compute_cosine_topk(
    questions: Dict[str, str],
    qid_to_triplets: Dict[str, List[Dict]],
    model: SentenceTransformer,
    k: int = COSINE_K,
) -> Dict[str, List[int]]:
    """Retourne qid -> liste des indices top-k par similarité cosinus."""
    results = {}
    for qid, question in questions.items():
        triplets = qid_to_triplets.get(qid, [])
        if not triplets:
            results[qid] = []
            continue

        query_text = f"Represent this sentence for searching relevant passages: {question}"
        triplet_texts = [triplet_text(t) for t in triplets]
        passage_texts = [f"Represent this passage for retrieval: {text}" for text in triplet_texts]

        with torch.no_grad():
            query_emb = model.encode([query_text], normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=False)
            triplet_embs = model.encode(passage_texts, normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=False, batch_size=64)

        scores = (triplet_embs @ query_emb.T).squeeze()
        if scores.ndim == 0:
            scores = scores.unsqueeze(0)
        topk_indices = torch.topk(scores, k=min(k, len(scores))).indices.tolist()
        results[qid] = topk_indices
    return results


def build_sft_example(
    qid: str,
    question: str,
    answer: str,
    triplets: List[Dict],
    context_triplet_indices: List[int],
    is_cosine_example: bool,
) -> Dict[str, Any]:
    """Construit un exemple SFT avec prompt et réponse cible."""
    # Contexte : triplets sélectionnés
    context_triplets = [triplets[i] for i in context_triplet_indices]

    # Prompt
    prompt = build_prompt_from_triplets(question, context_triplets, max_triplets=MAX_TRIPLETS_PER_QUESTION)

    # Déterminer quels triplets label=1 sont dans le contexte
    pos_in_context = []
    for idx, t_idx in enumerate(context_triplet_indices, 1):
        if triplets[t_idx].get("label", 0) == 1:
            pos_in_context.append(idx)

    # Construire la réponse cible
    is_yes_no = answer.lower().strip() in ["yes", "no"]

    if not pos_in_context:
        # Aucun triplet positif dans le contexte
        if is_yes_no:
            target = f"{answer.lower().strip()}"
        else:
            target = "Insuffisant"
    else:
        # Il y a des triplets positifs
        if is_yes_no:
            target = f"{answer.lower().strip()}"
        else:
            target = answer
        sources_str = "".join(f"[{i}]" for i in pos_in_context)
        target += f" Sources: {sources_str}"

    return {
        "qid": qid,
        "question": question,
        "answer": answer,
        "context_indices": context_triplet_indices,
        "pos_in_context": pos_in_context,
        "is_cosine_example": is_cosine_example,
        "prompt": prompt,
        "target": target,
    }


def main():
    print("Chargement des données...")
    train_questions = load_questions("train")
    qid_to_triplets = load_triplets_by_qid()

    print(f"Questions train: {len(train_questions)}")
    print(f"QIDs avec triplets: {len(qid_to_triplets)}")

    # Filtrer questions avec au moins 1 triplet
    train_qids_with_triplets = [ex["id"] for ex in train_questions if ex["id"] in qid_to_triplets]
    print(f"Questions train avec triplets: {len(train_qids_with_triplets)}")

    # Charger modèle pour cosinus
    print("Chargement modèle d'embedding...")
    embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cuda")

    # Questions dict pour cosinus
    train_q_dict = {ex["id"]: ex["question"] for ex in train_questions}

    # Calculer top-k cosinus pour TOUTES les questions train (pour les 40%)
    print("Calcul top-k cosinus train...")
    cos_topk_train = compute_cosine_topk(train_q_dict, qid_to_triplets, embed_model)

    # Sauvegarder cos_topk_train pour référence
    with open("data/processed/cos_topk_train.jsonl", "w") as f:
        for qid, indices in cos_topk_train.items():
            f.write(json.dumps({"qid": qid, "topk_indices": indices}) + "\n")
    print("  Sauvé: data/processed/cos_topk_train.jsonl")

    # Aussi calculer pour val (pour l'étape 2)
    val_questions = load_questions("val")
    val_q_dict = {ex["id"]: ex["question"] for ex in val_questions}
    cos_topk_val = compute_cosine_topk(val_q_dict, qid_to_triplets, embed_model)
    with open("data/processed/cos_topk_val.jsonl", "w") as f:
        for qid, indices in cos_topk_val.items():
            f.write(json.dumps({"qid": qid, "topk_indices": indices}) + "\n")
    print("  Sauvé: data/processed/cos_topk_val.jsonl")

    # Construire dataset SFT
    print("\nConstruction dataset SFT...")
    sft_examples = []

    for ex in train_questions:
        qid = ex["id"]
        if qid not in qid_to_triplets:
            continue

        triplets = qid_to_triplets[qid]
        pos_indices = [i for i, t in enumerate(triplets) if t.get("label", 0) == 1]

        if not pos_indices:
            # Pas de triplet positif -> on ne peut pas créer d'exemple utile
            continue

        question = ex["question"]
        answer = ex["answer"]

        # Décider du type d'exemple (60/40)
        use_cosine = random.random() < COSINE_TOPK_RATIO

        if use_cosine:
            # 40% : top-5 cosinus
            context_indices = cos_topk_train.get(qid, [])
            if not context_indices:
                # Fallback : prendre positifs + quelques négatifs
                neg_indices = [i for i in range(len(triplets)) if i not in pos_indices]
                context_indices = pos_indices + random.sample(neg_indices, min(5 - len(pos_indices), len(neg_indices)))
        else:
            # 60% : tous les positifs + distracteurs aléatoires
            neg_indices = [i for i in range(len(triplets)) if i not in pos_indices]
            n_distractors = min(len(neg_indices), MAX_TRIPLETS_PER_QUESTION - len(pos_indices))
            distractors = random.sample(neg_indices, n_distractors) if n_distractors > 0 else []
            context_indices = pos_indices + distractors
            random.shuffle(context_indices)

        # Tronquer à MAX_TRIPLETS_PER_QUESTION
        context_indices = context_indices[:MAX_TRIPLETS_PER_QUESTION]

        sft_ex = build_sft_example(
            qid=qid,
            question=question,
            answer=answer,
            triplets=triplets,
            context_triplet_indices=context_indices,
            is_cosine_example=use_cosine,
        )
        sft_examples.append(sft_ex)

    print(f"Total exemples SFT: {len(sft_examples)}")

    # Statistiques
    n_cosine = sum(1 for e in sft_examples if e["is_cosine_example"])
    n_random = sum(1 for e in sft_examples if not e["is_cosine_example"])
    n_insufficient = sum(1 for e in sft_examples if e["target"] == "Insuffisant" or e["target"].startswith("Insuffisant"))
    n_yes_no = sum(1 for e in sft_examples if e["target"].startswith("yes") or e["target"].startswith("no"))

    avg_context_len = sum(len(e["context_indices"]) for e in sft_examples) / len(sft_examples) if sft_examples else 0
    avg_pos_in_context = sum(len(e["pos_in_context"]) for e in sft_examples) / len(sft_examples) if sft_examples else 0

    print(f"  Cosinus (40%): {n_cosine}")
    print(f"  Aléatoire (60%): {n_random}")
    print(f"  Insuffisant: {n_insufficient} ({n_insufficient/len(sft_examples)*100:.1f}%)")
    print(f"  Yes/No: {n_yes_no}")
    print(f"  Longueur contexte moy: {avg_context_len:.1f}")
    print(f"  Positifs dans contexte moy: {avg_pos_in_context:.1f}")

    # Token stats (approximatif)
    all_text = " ".join(e["prompt"] + " " + e["target"] for e in sft_examples)
    approx_tokens = len(all_text.split()) * 1.3  # approximation grossière
    print(f"  Tokens approx (mots*1.3): {approx_tokens:.0f}")

    # Sauvegarder SFT
    out_path = Path("data/processed/sft_train.jsonl")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        for e in sft_examples:
            # Format pour SFTTrainer : messages ou prompt/completion
            f.write(json.dumps({
                "qid": e["qid"],
                "prompt": e["prompt"],
                "completion": e["target"],
            }, ensure_ascii=False) + "\n")
    print(f"\nDataset SFT sauvé: {out_path}")

    # Rapport
    report_path = Path("reports/j8_sft_stats.md")
    report_path.parent.mkdir(exist_ok=True)
    with open(report_path, "w") as f:
        f.write("# Statistiques Dataset SFT - J8\n\n")
        f.write(f"- **Total exemples** : {len(sft_examples)}\n")
        f.write(f"- **Exemples cosinus (40%)** : {n_cosine}\n")
        f.write(f"- **Exemples aléatoires (60%)** : {n_random}\n")
        f.write(f"- **% Insuffisant** : {n_insufficient/len(sft_examples)*100:.1f}%\n")
        f.write(f"- **Questions Yes/No** : {n_yes_no}\n")
        f.write(f"- **Longueur contexte moyenne** : {avg_context_len:.1f} triplets\n")
        f.write(f"- **Positifs dans contexte moyen** : {avg_pos_in_context:.1f}\n")
        f.write(f"- **Tokens approx (mots*1.3)** : {approx_tokens:.0f}\n")
        f.write(f"- **Seed** : {SEED}\n")

    # 10 exemples complets pour validation
    samples_path = Path("reports/j8_sft_samples.md")
    with open(samples_path, "w") as f:
        f.write("# Exemples SFT - J8 (10 premiers)\n\n")
        for i, e in enumerate(sft_examples[:10]):
            qid = e["qid"]
            f.write(f"## Exemple {i+1} (qid: {qid}, {'cosinus' if e['is_cosine_example'] else 'aléatoire'})\n\n")
            f.write(f"**Question:** {e['question']}\n\n")
            f.write(f"**Réponse gold:** {e['answer']}\n\n")
            f.write(f"**Triplets dans contexte ({len(e['context_indices'])}):**\n")
            for idx, t_idx in enumerate(e["context_indices"], 1):
                t = qid_to_triplets[qid][t_idx]
                label_str = " ✓" if t.get("label", 0) == 1 else ""
                f.write(f"  [{idx}] ({t['h']}, {t['r']}, {t['t']}){label_str}\n")
            f.write(f"\n**Prompt:**\n```\n{e['prompt']}\n```\n\n")
            f.write(f"**Cible (completion):**\n```\n{e['target']}\n```\n\n")
            f.write("---\n\n")

    print(f"Exemples de validation: {samples_path}")
    print(f"Stats: {report_path}")


if __name__ == "__main__":
    main()