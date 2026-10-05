"""Prédiction top-k triplets par question avec le GNN entraîné."""
import json
import torch
import pickle
from pathlib import Path
from collections import defaultdict
from torch_geometric.loader import DataLoader

from src.gnn.model import TripleScorerGNN


class TripleScorerGNN_BM25(TripleScorerGNN):
    """Modèle GNN avec feature BM25 additionnelle."""
    def forward(self, data):
        x, edge_index, edge_attr, q_emb, cos_qt, batch, bm25_score = (
            data.x, data.edge_index, data.edge_attr, data.q_emb, data.cos_qt, data.batch, data.bm25_score
        )
        h = self.node_proj(x)
        if self.use_question_conditioning:
            q_emb_expanded = q_emb[batch[edge_index[0]]]
            cos_qt_expanded = cos_qt.unsqueeze(1)
            bm25_expanded = bm25_score.unsqueeze(1)
            edge_attr = torch.cat([edge_attr, q_emb_expanded, cos_qt_expanded, bm25_expanded], dim=-1)
        edge_attr = self.edge_proj(edge_attr)
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            if self.use_gatv2:
                h = conv(h, edge_index, edge_attr)
            else:
                h = conv(h, edge_index)
            h = norm(h)
            if i < self.num_layers - 1:
                h = torch.relu(h)
                h = torch.nn.functional.dropout(h, p=self.dropout, training=self.training)
        h_src = h[edge_index[0]]
        h_dst = h[edge_index[1]]
        edge_feat = torch.cat([h_src, h_dst, edge_attr, cos_qt.unsqueeze(1), bm25_score.unsqueeze(1)], dim=-1)
        logits = self.edge_scorer(edge_feat).squeeze(-1)
        return logits


def load_triplets_by_qid(path: str) -> dict:
    """Charge kg_triplets.jsonl et groupe par qid."""
    by_qid = defaultdict(list)
    with open(path) as f:
        for line in f:
            t = json.loads(line)
            by_qid[t["qid"]].append(t)
    return by_qid


def load_split_qids(split: str) -> list:
    """Retourne la liste des qid du split."""
    with open(f"data/processed/{split}.jsonl") as f:
        return [json.loads(line)["id"] for line in f]


def load_questions(split: str) -> dict:
    """Charge qid -> question."""
    q_dict = {}
    with open(f"data/processed/{split}.jsonl") as f:
        for line in f:
            ex = json.loads(line)
            q_dict[ex["id"]] = ex["question"]
    return q_dict


def compute_bm25_scores(questions: dict, qid_to_triplets: dict) -> dict:
    """Calcule scores BM25 pour chaque question."""
    from rank_bm25 import BM25Okapi
    
    def triplet_tokens(t):
        return f"{t['h']} {t['r']} {t['t']}".lower().split()
    
    bm25_scores = {}
    for qid, question in questions.items():
        triplets = qid_to_triplets.get(qid, [])
        if not triplets:
            bm25_scores[qid] = []
            continue
        corpus = [triplet_tokens(t) for t in triplets]
        bm25 = BM25Okapi(corpus)
        query_tokens = question.lower().split()
        scores = bm25.get_scores(query_tokens)
        bm25_scores[qid] = scores.tolist() if hasattr(scores, 'tolist') else list(scores)
    return bm25_scores


def predict_topk(
    model,
    data_list: list,
    qid_to_triplets: dict,
    k: int,
    device: str,
) -> dict:
    """Retourne dict qid -> liste de top-k triplets avec scores."""
    model.eval()
    loader = DataLoader(data_list, batch_size=32, shuffle=False)
    
    all_predictions = {}
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            scores = torch.sigmoid(logits)
            
            # batch.batch donne l'index du graphe par nœud
            # On a besoin de l'index par arête
            edge_batch = batch.batch[batch.edge_index[0]]
            
            num_graphs = batch.num_graphs
            for q_idx in range(num_graphs):
                mask = (edge_batch == q_idx)
                q_scores = scores[mask].cpu()
                qid = batch.qid[q_idx] if hasattr(batch, 'qid') else None
                
                if qid is None or qid not in qid_to_triplets:
                    continue
                
                triplets = qid_to_triplets[qid]
                topk = min(k, len(q_scores))
                _, topk_indices = torch.topk(q_scores, k=topk)
                
                topk_triplets = []
                for idx in topk_indices.tolist():
                    t = triplets[idx].copy()
                    t["score"] = float(q_scores[idx])
                    topk_triplets.append(t)
                
                all_predictions[qid] = {
                    "qid": qid,
                    "k": k,
                    "triplets": topk_triplets,
                }
    
    return all_predictions


def main():
    k = 5  # R@5 est la métrique cible
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Device: {device}")
    print(f"k = {k}")
    
    # Charger checkpoint
    checkpoint = torch.load("checkpoints/gnn_best.pt", map_location=device, weights_only=False)
    config = checkpoint["config"]
    config_name = checkpoint["config_name"]
    
    print(f"Config: {config_name}")
    print(f"Best epoch: {checkpoint['best_epoch']}")
    print(f"Val R@5: {checkpoint['val_metrics']['R@5']:.4f}")
    
    # Recréer modèle
    model = TripleScorerGNN_BM25(in_dim=384, **config).to(device)
    model.edge_proj = torch.nn.Linear(770, config['hidden_dim']).to(device)
    model.edge_scorer = torch.nn.Sequential(
        torch.nn.Linear(386, config['hidden_dim']),
        torch.nn.ReLU(),
        torch.nn.Dropout(config['dropout']),
        torch.nn.Linear(config['hidden_dim'], 1),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Modèle chargé: {sum(p.numel() for p in model.parameters()):,} paramètres")
    
    # Charger données
    print("Chargement triplets...")
    qid_to_triplets = load_triplets_by_qid("data/processed/kg_triplets.jsonl")
    
    # Charger BM25 scores
    print("Calcul BM25...")
    val_questions = load_questions("val")
    test_questions = load_questions("test")
    all_questions = {**val_questions, **test_questions}
    bm25_scores = compute_bm25_scores(all_questions, qid_to_triplets)
    
    # Val
    print("\n=== Prédictions VAL ===")
    val_data = torch.load("data/processed/pyg_val.pt", weights_only=False)
    for d in val_data:
        qid = d.qid
        d.bm25_score = torch.tensor(bm25_scores.get(qid, [0]*d.edge_index.shape[1]), dtype=torch.float)
    
    val_preds = predict_topk(model, val_data, qid_to_triplets, k, device)
    print(f"  {len(val_preds)} questions prédites")
    
    out_val = Path("data/processed/gnn_topk_val.jsonl")
    with open(out_val, "w") as f:
        for qid, pred in val_preds.items():
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")
    print(f"  Sauvé: {out_val}")
    
    # Test
    print("\n=== Prédictions TEST ===")
    test_data = torch.load("data/processed/pyg_test.pt", weights_only=False)
    for d in test_data:
        qid = d.qid
        d.bm25_score = torch.tensor(bm25_scores.get(qid, [0]*d.edge_index.shape[1]), dtype=torch.float)
    
    test_preds = predict_topk(model, test_data, qid_to_triplets, k, device)
    print(f"  {len(test_preds)} questions prédites")
    
    out_test = Path("data/processed/gnn_topk_test.jsonl")
    with open(out_test, "w") as f:
        for qid, pred in test_preds.items():
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")
    print(f"  Sauvé: {out_test}")
    
    print("\nTerminé !")


if __name__ == "__main__":
    main()