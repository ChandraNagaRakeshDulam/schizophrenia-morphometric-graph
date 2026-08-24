import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, SAGEConv, GINConv, GATConv, GATv2Conv, global_mean_pool

class GraphClassifier(nn.Module):
    def __init__(self, in_dim, hidden, kind="gcn", dropout=0.3):
        super().__init__()
        self.kind = kind
        self.dropout = nn.Dropout(dropout)
        def gin_layer(a, b):
            return GINConv(nn.Sequential(nn.Linear(a, b), nn.ReLU(), nn.Linear(b, b)))
        if kind == "gcn":
            self.c1, self.c2 = GCNConv(in_dim, hidden), GCNConv(hidden, hidden)
        elif kind == "sage":
            self.c1, self.c2 = SAGEConv(in_dim, hidden), SAGEConv(hidden, hidden)
        elif kind == "gin":
            self.c1, self.c2 = gin_layer(in_dim, hidden), gin_layer(hidden, hidden)
        elif kind == "gat":
            self.c1, self.c2 = GATConv(in_dim, hidden, heads=4, concat=False), GATConv(hidden, hidden, heads=4, concat=False)
        elif kind == "gatv2":
            self.c1, self.c2 = GATv2Conv(in_dim, hidden, heads=4, concat=False), GATv2Conv(hidden, hidden, heads=4, concat=False)
        else:
            raise ValueError(kind)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x, edge_index, batch=None, **kwargs):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        x = torch.relu(self.c1(x, edge_index))
        x = self.dropout(x)
        x = torch.relu(self.c2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.out(x).view(-1)
