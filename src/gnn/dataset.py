"""Datasets PyG pour GNN (anti-fuite stricte).

Pour chaque question avec ≥1 triplet, un torch_geometric Data :
- nœuds = entités uniques, x = embedding bge du nom (384d)
- edge_index, edge_attr = embedding bge du texte de la RELATION r seule
- q_emb = embedding de la question
- cos_qt = cosinus(question, "h r t") par arête
- y = label par arête (float)

ANTI-FUITE : source_title, sentence, label et titres de paragraphes
NE doivent apparaître dans AUCUNE feature (ils ne servent qu'à calculer y).

Sauvegarde data/processed/pyg_{train,val,test}.pt (listes de Data, tenseurs sur CPU).
"""
import json
import torch
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple

from torch_geometric.data import Data
from sentence_transformers import SentenceTransformer


SPLITS = ["train", "val", "test"]
MODEL_NAME = "BAAI/bge-small-en-v1.5"


def load_split(split: str) -> List[Dict]:
    """Charge un split (train/val/test).jsonl"""
    with open(f"data/processed/{split}.jsonl") as f:
        return [json.loads(line) for line in f]


def load_triplets_by_qid() -> Dict[str, List[Dict]]:
    """Charge kg_triplets.jsonl et groupe par qid."""
    by_qid = defaultdict(list)
    with open("data/processed/kg_triplets.jsonl") as f:
        for line in f:
            t = json.loads(line)
            by_qid[t["qid"]].append(t)
    return by_qid


def build_graph_data(
    qid: str,
    question: str,
    triplets: List[Dict],
    model: SentenceTransformer,
) -> Data:
    """Construit un objet Data PyG pour une question."""
    # Entités uniques
    entities = set()
    for t in triplets:
        entities.add(t["h"])
        entities.add(t["t"])
    entity_list = sorted(entities)
    entity_to_idx = {e: i for i, e in enumerate(entity_list)}

    # x = embeddings des entités (noms seulement)
    with torch.no_grad():
        x = model.encode(entity_list, normalize_embeddings=True, show_progress_bar=False, convert_to_tensor=True, batch_size=64)
    x = x.cpu()

    # edge_index et edge_attr (embedding de la relation r seule)
    edge_index = []
    edge_attr_list = []
    edge_labels = []
    cos_qt_list = []

    question_emb = model.encode(
        [f"Represent this sentence for searching relevant passages: {question}"],
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_tensor=True,
    ).cpu()  # [1, 384]
    question_emb_squeezed = question_emb.squeeze(0)  # [384]

    relation_texts = []
    triplet_texts = []
    for t in triplets:
        h_idx = entity_to_idx[t["h"]]
        t_idx = entity_to_idx[t["t"]]
        edge_index.append([h_idx, t_idx])
        relation_texts.append(t["r"])
        triplet_texts.append(f"{t['h']} {t['r']} {t['t']}")
        edge_labels.append(float(t["label"]))

    # edge_attr = embedding de la relation SEULE
    with torch.no_grad():
        edge_attr = model.encode(
            relation_texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_tensor=True,
            batch_size=64,
        )
    edge_attr = edge_attr.cpu()

    # cos_qt = cosinus(question, triplet_text) pour chaque arête
    with torch.no_grad():
        triplet_embs = model.encode(
            [f"Represent this passage for retrieval: {text}" for text in triplet_texts],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_tensor=True,
            batch_size=64,
        )
    triplet_embs = triplet_embs.cpu()
    cos_qt = (triplet_embs @ question_emb_squeezed).squeeze()
    if cos_qt.ndim == 0:
        cos_qt = cos_qt.unsqueeze(0)

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    y = torch.tensor(edge_labels, dtype=torch.float)
    cos_qt = cos_qt.cpu()

    # Assertions anti-fuite
    assert x.shape[0] == len(entity_list), "x shape mismatch"
    assert edge_index.shape[1] == len(triplets), "edge_index shape mismatch"
    assert edge_attr.shape[0] == len(triplets), "edge_attr shape mismatch"
    assert edge_attr.shape[1] == 384, f"edge_attr dim mismatch: {edge_attr.shape[1]} != 384"
    assert y.shape[0] == len(triplets), "y shape mismatch"
    assert cos_qt.shape[0] == len(triplets), "cos_qt shape mismatch"
    assert x.shape[1] == 384, f"x dim mismatch: {x.shape[1]} != 384"

    # Vérifier qu'aucune feature ne contient source_title, sentence, label, para_title
    # (ces infos ne sont utilisées que pour y, déjà vérifié ci-dessus)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        cos_qt=cos_qt,
        q_emb=question_emb,  # [1, 384] pour batching PyG correct
        qid=qid,
        num_nodes=len(entity_list),
    )

    # Stocker les mapping pour debug si nécessaire
    data.entity_list = entity_list
    data.triplets = triplets

    return data


def build_dataset(split: str, model: SentenceTransformer) -> List[Data]:
    """Construit la liste de Data pour un split."""
    questions = load_split(split)
    qid_to_triplets = load_triplets_by_qid()

    data_list = []
    for ex in questions:
        qid = ex["id"]
        triplets = qid_to_triplets.get(qid, [])
        if not triplets:
            continue
        question = ex["question"]
        data = build_graph_data(qid, question, triplets, model)
        data_list.append(data)

    return data_list


def main():
    print(f"Chargement modèle {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME, device="cuda")

    for split in SPLITS:
        print(f"\n=== Construction {split} ===")
        data_list = build_dataset(split, model)
        print(f"  {len(data_list)} graphes")

        out_path = Path(f"data/processed/pyg_{split}.pt")
        out_path.parent.mkdir(exist_ok=True)
        torch.save(data_list, out_path)
        print(f"  Sauvé: {out_path}")

    # Test DataLoader sur val (seul split avec données pour l'instant)
    print("\n=== Test DataLoader (val) ===")
    from torch_geometric.loader import DataLoader

    val_data = torch.load("data/processed/pyg_val.pt", weights_only=False)
    if len(val_data) > 0:
        loader = DataLoader(val_data, batch_size=8, shuffle=True)
        batch = next(iter(loader))

        print("Batch shapes:")
        print(f"  x: {batch.x.shape}")
        print(f"  edge_index: {batch.edge_index.shape}")
        print(f"  edge_attr: {batch.edge_attr.shape}")
        print(f"  y: {batch.y.shape}")
        print(f"  cos_qt: {batch.cos_qt.shape}")
        print(f"  q_emb: {batch.q_emb.shape}")
        print(f"  batch: {batch.batch.shape}")
        print(f"  num_graphs: {batch.num_graphs}")
    else:
        print("  Pas de données val pour test")


if __name__ == "__main__":
    main()