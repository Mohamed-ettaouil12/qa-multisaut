"""Modèle GNN pour scoring de triplets (TripleScorerGNN).

Architecture : GATv2 (ou GCN) sur graphe de triplets par question.
Entrées par Data PyG :
- x : [N, 384] embeddings entités
- edge_index : [2, E]
- edge_attr : [E, 384] embedding relation seule
- q_emb : [384] embedding question
- cos_qt : [E] cosinus(question, triplet)

Conditionnement par la question : concaténation de q_emb (broadcast) et cos_qt
aux features d'arête avant le message passing.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GCNConv
from torch_geometric.utils import softmax


class TripleScorerGNN(nn.Module):
    """GNN pour prédire label=1 (triplet pertinent) par arête."""

    def __init__(
        self,
        in_dim: int = 384,
        hidden_dim: int = 128,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.2,
        use_gatv2: bool = True,
        use_question_conditioning: bool = True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.heads = heads
        self.dropout = dropout
        self.use_gatv2 = use_gatv2
        self.use_question_conditioning = use_question_conditioning

        # Projection initiale des nœuds
        self.node_proj = nn.Linear(in_dim, hidden_dim)

        # Projection edge_attr (relation) + conditionnement question
        edge_in_dim = in_dim
        if use_question_conditioning:
            edge_in_dim += in_dim + 1  # q_emb (384) + cos_qt (1)
        self.edge_proj = nn.Linear(edge_in_dim, hidden_dim)

        # Couches GNN
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            if use_gatv2:
                conv = GATv2Conv(
                    hidden_dim,
                    hidden_dim // heads,
                    heads=heads,
                    dropout=dropout,
                    add_self_loops=False,
                    edge_dim=hidden_dim,
                )
            else:
                conv = GCNConv(hidden_dim, hidden_dim, add_self_loops=False)
            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(hidden_dim))

        # Tête de scoring par arête
        # Input: h_src [hidden_dim] + h_dst [hidden_dim] + edge_attr [hidden_dim] + cos_qt [1]
        self.edge_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data):
        x, edge_index, edge_attr, q_emb, cos_qt, batch = (
            data.x, data.edge_index, data.edge_attr, data.q_emb, data.cos_qt, data.batch
        )

        # Projection nœuds
        h = self.node_proj(x)

        # Préparer edge_attr conditionné par question
        if self.use_question_conditioning:
            # q_emb: [num_graphs, 384] -> broadcast par arête via batch
            num_edges = edge_index.shape[1]
            q_emb_expanded = q_emb[batch[edge_index[0]]]  # [E, 384]
            cos_qt_expanded = cos_qt.unsqueeze(1)  # [E, 1]
            edge_attr = torch.cat([edge_attr, q_emb_expanded, cos_qt_expanded], dim=-1)

        edge_attr = self.edge_proj(edge_attr)

        # Message passing
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            if self.use_gatv2:
                h = conv(h, edge_index, edge_attr)
            else:
                h = conv(h, edge_index)
            h = norm(h)
            if i < self.num_layers - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)

        # Scoring par arête: concatène [h_src, h_dst, edge_attr, cos_qt]
        h_src = h[edge_index[0]]
        h_dst = h[edge_index[1]]
        edge_feat = torch.cat([h_src, h_dst, edge_attr, cos_qt.unsqueeze(1)], dim=-1)
        logits = self.edge_scorer(edge_feat).squeeze(-1)
        return logits


def get_model_config(model_name: str) -> dict:
    """Retourne la config pour une ablation."""
    configs = {
        "A1_gatv2_h128_l2_avec_question": {
            "hidden_dim": 128,
            "num_layers": 2,
            "heads": 4,
            "use_gatv2": True,
            "use_question_conditioning": True,
        },
        "A2_gcn_h128_l2_avec_question": {
            "hidden_dim": 128,
            "num_layers": 2,
            "heads": 4,
            "use_gatv2": False,
            "use_question_conditioning": True,
        },
        "A3_gatv2_h128_l1": {
            "hidden_dim": 128,
            "num_layers": 1,
            "heads": 4,
            "use_gatv2": True,
            "use_question_conditioning": True,
        },
        "A4_gatv2_h128_l3": {
            "hidden_dim": 128,
            "num_layers": 3,
            "heads": 4,
            "use_gatv2": True,
            "use_question_conditioning": True,
        },
        "A5_gatv2_h64_l2": {
            "hidden_dim": 64,
            "num_layers": 2,
            "heads": 4,
            "use_gatv2": True,
            "use_question_conditioning": True,
        },
        "A6_gatv2_h256_l2": {
            "hidden_dim": 256,
            "num_layers": 2,
            "heads": 4,
            "use_gatv2": True,
            "use_question_conditioning": True,
        },
        "A7_gatv2_h128_l2_SANS_conditionnement": {
            "hidden_dim": 128,
            "num_layers": 2,
            "heads": 4,
            "use_gatv2": True,
            "use_question_conditioning": False,
        },
    }
    if model_name not in configs:
        raise ValueError(f"Config inconnue: {model_name}")
    return configs[model_name]


if __name__ == "__main__":
    # Test rapide
    model = TripleScorerGNN()
    print(f"Modèle: {sum(p.numel() for p in model.parameters()):,} paramètres")

    # Test forward avec données factices
    from torch_geometric.data import Data, Batch
    d1 = Data(
        x=torch.randn(10, 384),
        edge_index=torch.randint(0, 10, (2, 8)),
        edge_attr=torch.randn(8, 384),
        q_emb=torch.randn(384),
        cos_qt=torch.randn(8),
        y=torch.randint(0, 2, (8,)).float(),
        num_nodes=10,
    )
    d2 = Data(
        x=torch.randn(12, 384),
        edge_index=torch.randint(0, 12, (2, 10)),
        edge_attr=torch.randn(10, 384),
        q_emb=torch.randn(384),
        cos_qt=torch.randn(10),
        y=torch.randint(0, 2, (10,)).float(),
        num_nodes=12,
    )
    batch = Batch.from_data_list([d1, d2])
    logits = model(batch)
    print(f"Logits shape: {logits.shape} (attendu: 18)")