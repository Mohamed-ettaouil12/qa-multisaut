"""Boucle d'entraînement GNN (src/gnn/train.py).

- DataLoader PyG (batch 32 graphes)
- TripleScorerGNN, BCEWithLogitsLoss avec pos_weight = n_neg/n_pos
- Adam lr 1e-3, weight decay 1e-4, 30 époques max
- Early stopping (patience 5) sur Recall@5 val
- Recall@k : classement arêtes par score décroissant par question
- Logs MLflow : perte et Recall@5 val par époque
- Sauvegarde checkpoints/gnn_v1.pt (state_dict + config)
"""
import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import json
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple

import mlflow
from torch_geometric.loader import DataLoader

from src.gnn.model import TripleScorerGNN, get_model_config


SEED = 42
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 30
PATIENCE = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int = SEED):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_pos_weight(train_data: List) -> float:
    """Calcule pos_weight = n_neg / n_pos depuis train."""
    total_pos = 0
    total_neg = 0
    for data in train_data:
        total_pos += (data.y == 1).sum().item()
        total_neg += (data.y == 0).sum().item()
    if total_pos == 0:
        return 1.0
    return total_neg / total_pos


def recall_at_k_per_question(logits: torch.Tensor, y: torch.Tensor, batch: torch.Tensor, k: int) -> List[float]:
    """Calcule Recall@k par question (moyenné sur questions avec ≥1 positif).

    logits: [E] scores
    y: [E] labels (0/1)
    batch: [E] index de question par arête
    """
    recalls = []
    num_graphs = batch.max().item() + 1

    for q_idx in range(num_graphs):
        mask = (batch == q_idx)
        q_logits = logits[mask]
        q_y = y[mask]

        gold_indices = (q_y == 1).nonzero(as_tuple=True)[0]
        if len(gold_indices) == 0:
            continue  # Skip questions without positive (same as baseline)

        # Top-k indices by score
        _, topk_indices = torch.topk(q_logits, k=min(k, len(q_logits)))
        pred_set = set(topk_indices.tolist())
        gold_set = set(gold_indices.tolist())

        recall = len(pred_set & gold_set) / len(gold_set)
        recalls.append(recall)

    return recalls


def evaluate(model, loader, device, k_values=(3, 5, 10)) -> Dict[str, float]:
    """Évalue le modèle sur un DataLoader."""
    model.eval()
    all_recalls = {k: [] for k in k_values}

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            y = batch.y
            # batch.batch is per-node; we need per-edge batch indices
            edge_batch = batch.batch[batch.edge_index[0]]

            for k in k_values:
                recalls = recall_at_k_per_question(logits, y, edge_batch, k)
                all_recalls[k].extend(recalls)

    results = {}
    for k in k_values:
        results[f"R@{k}"] = sum(all_recalls[k]) / len(all_recalls[k]) if all_recalls[k] else 0.0
    return results


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch)
        loss = loss_fn(logits, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0.0


def main(config_name: str = "A1_gatv2_h128_l2_avec_question", run_name: str = "gnn_v1"):
    set_seed()

    print(f"Device: {DEVICE}")
    print(f"Config: {config_name}")

    # Charger données
    print("Chargement données...")
    train_data = torch.load("data/processed/pyg_train.pt", weights_only=False)
    val_data = torch.load("data/processed/pyg_val.pt", weights_only=False)

    if len(train_data) == 0:
        print("ERREUR: Train data vide ! Extraction train nécessaire.")
        return

    print(f"  Train: {len(train_data)} graphes")
    print(f"  Val: {len(val_data)} graphes")

    # pos_weight
    pos_weight = compute_pos_weight(train_data)
    print(f"  pos_weight (n_neg/n_pos): {pos_weight:.4f}")

    # DataLoaders
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

    # Modèle
    config = get_model_config(config_name)
    model = TripleScorerGNN(**config).to(DEVICE)
    print(f"Modèle: {sum(p.numel() for p in model.parameters()):,} paramètres")

    # Optimiseur & loss
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    pos_weight_tensor = torch.tensor([pos_weight], device=DEVICE)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    # MLflow
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("qa-multisaut")
    mlflow.start_run(run_name=run_name)

    # Log hyperparams
    mlflow.log_params({
        "config": config_name,
        "hidden_dim": config["hidden_dim"],
        "num_layers": config["num_layers"],
        "heads": config["heads"],
        "use_gatv2": config["use_gatv2"],
        "use_question_conditioning": config["use_question_conditioning"],
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "pos_weight": pos_weight,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
    })

    # Early stopping
    best_val_r5 = 0.0
    best_epoch = 0
    patience_counter = 0
    best_state = None

    print("\n=== Entraînement ===")
    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, DEVICE)
        val_metrics = evaluate(model, val_loader, DEVICE)
        val_r5 = val_metrics["R@5"]

        mlflow.log_metric("train_loss", train_loss, step=epoch)
        for k, v in val_metrics.items():
            metric_name = k.replace("@", "")
            mlflow.log_metric(f"val_{metric_name}", v, step=epoch)

        improved = val_r5 > best_val_r5
        if improved:
            best_val_r5 = val_r5
            best_epoch = epoch
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"Epoch {epoch:2d} | Loss: {train_loss:.4f} | Val R@5: {val_r5:.4f} ⭐")
        else:
            patience_counter += 1
            print(f"Epoch {epoch:2d} | Loss: {train_loss:.4f} | Val R@5: {val_r5:.4f} (patience {patience_counter}/{PATIENCE})")

        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping à l'epoch {epoch} (best epoch {best_epoch})")
            break

    # Restaurer meilleur modèle
    if best_state is not None:
        model.load_state_dict(best_state)

    # Métriques finales
    print("\n=== Métriques finales ===")
    train_metrics = evaluate(model, train_loader, DEVICE)
    val_metrics = evaluate(model, val_loader, DEVICE)

    for k in [3, 5, 10]:
        mlflow.log_metric(f"final_train_R{k}", train_metrics[f"R@{k}"])
        mlflow.log_metric(f"final_val_R{k}", val_metrics[f"R@{k}"])

    print(f"Train: R@3={train_metrics['R@3']:.4f} R@5={train_metrics['R@5']:.4f} R@10={train_metrics['R@10']:.4f}")
    print(f"Val:   R@3={val_metrics['R@3']:.4f} R@5={val_metrics['R@5']:.4f} R@10={val_metrics['R@10']:.4f}")

    # Sauvegarder checkpoint
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_path = checkpoint_dir / "gnn_v1.pt"

    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "config_name": config_name,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "best_epoch": best_epoch,
        "pos_weight": pos_weight,
    }, checkpoint_path)
    print(f"\nCheckpoint sauvé: {checkpoint_path}")

    mlflow.end_run()

    return train_metrics, val_metrics


if __name__ == "__main__":
    main()