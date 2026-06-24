"""
train_thgt.py
Temporal Heterogeneous Graph Transformer (THGT) for detecting
structural prompt-injection propagation ("temporal confused deputy")
in LLM Multi-Agent System execution traces.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_recall_curve, auc
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# --------------------------------------------------------------------------
# Architecture defined by Claude
# --------------------------------------------------------------------------
class TimeEncoder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.w = nn.Parameter(torch.from_numpy(
            1 / 10 ** torch.linspace(0, 9, dim).numpy()
        ).float())
        self.b = nn.Parameter(torch.zeros(dim))

    def forward(self, delta_t: torch.Tensor) -> torch.Tensor:
        return torch.cos(delta_t.unsqueeze(-1) * self.w + self.b)

class NodeMemoryBank(nn.Module):
    def __init__(self, node_types: list[str], num_nodes: dict[str, int],
                 raw_feat_dim: int, mem_dim: int):
        super().__init__()
        self.mem_dim = mem_dim
        self.cells = nn.ModuleDict({
            ntype: nn.GRUCell(raw_feat_dim, mem_dim) for ntype in node_types
        })
        self.memory = {
            ntype: torch.zeros(num_nodes[ntype], mem_dim)
            for ntype in node_types
        }
        self.last_update_t = {
            ntype: torch.zeros(num_nodes[ntype])
            for ntype in node_types
        }

    def get(self, node_type: str, node_idx: torch.Tensor) -> torch.Tensor:
        return self.memory[node_type][node_idx]

    def delta_since_last(self, node_type: str, node_idx: torch.Tensor,
                          t: torch.Tensor) -> torch.Tensor:
        return t - self.last_update_t[node_type][node_idx]

    @torch.no_grad()
    def update(self, node_type: str, node_idx: torch.Tensor,
               raw_feat: torch.Tensor, t: torch.Tensor):
        old = self.memory[node_type][node_idx]
        new = self.cells[node_type](raw_feat, old)
        self.memory[node_type][node_idx] = new.detach()
        self.last_update_t[node_type][node_idx] = t.detach()

class HeteroTemporalAttention(nn.Module):
    def __init__(self, edge_types: list[str], mem_dim: int, time_dim: int,
                 edge_feat_dim: int, out_dim: int):
        super().__init__()
        in_dim = mem_dim * 2 + time_dim + edge_feat_dim
        self.proj = nn.ModuleDict({
            etype: nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
            ) for etype in edge_types
        })
        self.attn_score = nn.ModuleDict({
            etype: nn.Linear(out_dim, 1) for etype in edge_types
        })

    def forward(self, edge_type: str, src_mem, dst_mem, time_enc, edge_feat):
        x = torch.cat([src_mem, dst_mem, time_enc, edge_feat], dim=-1)
        h = self.proj[edge_type](x)
        score = self.attn_score[edge_type](h)  
        return h, score

class EventContextTransformer(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int = 4, n_layers: int = 2,
                 max_window: int = 32):
        super().__init__()
        self.max_window = max_window
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=hidden_dim * 2, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.pos_time = TimeEncoder(hidden_dim)

    def forward(self, window_embeds: torch.Tensor, window_deltas: torch.Tensor):
        pos = self.pos_time(window_deltas)
        x = window_embeds + pos
        ctx = self.encoder(x)
        return ctx[:, -1, :]  

class THGT(nn.Module):
    def __init__(self, node_types, edge_types, num_nodes,
                 raw_feat_dim=16, mem_dim=64, time_dim=32,
                 edge_feat_dim=8, hidden_dim=64, n_classes=2):
        super().__init__()
        self.memory_bank = NodeMemoryBank(node_types, num_nodes,
                                           raw_feat_dim, mem_dim)
        self.time_encoder = TimeEncoder(time_dim)
        self.hetero_attn = HeteroTemporalAttention(
            edge_types, mem_dim, time_dim, edge_feat_dim, hidden_dim
        )
        self.seq_context = EventContextTransformer(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, batch):
        src_mem = self.memory_bank.get(batch.src_type, batch.src_idx)
        dst_mem = self.memory_bank.get(batch.dst_type, batch.dst_idx)
        delta = self.memory_bank.delta_since_last(
            batch.dst_type, batch.dst_idx, batch.t
        )
        time_enc = self.time_encoder(delta)

        h_event, raw_score = self.hetero_attn(
            batch.edge_type, src_mem, dst_mem, time_enc, batch.edge_feat
        )
        h_context = self.seq_context(
            batch.dst_window_embeds, batch.dst_window_deltas
        )
        logits = self.classifier(torch.cat([h_event, h_context], dim=-1))
        return logits, h_event  

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.75):  # Adjusted alpha
        super().__init__()
        self.gamma, self.alpha = gamma, alpha

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none')
        p = torch.softmax(logits, dim=-1).gather(1, targets.unsqueeze(1)).squeeze(1)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        return (alpha_t * (1 - p) ** self.gamma * ce).mean()

def train_one_epoch(model, event_stream, optimizer, loss_fn, accumulation_steps=32):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    
    for i, batch in enumerate(event_stream):               
        logits, h_event = model(batch)
        loss = loss_fn(logits, batch.label) / accumulation_steps
        loss.backward()
        
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(event_stream):
            optimizer.step()
            optimizer.zero_grad()

        model.memory_bank.update(
            batch.dst_type, batch.dst_idx, batch.edge_feat, batch.t
        )
        total_loss += loss.item() * accumulation_steps
    return total_loss / max(len(event_stream), 1)


# --------------------------------------------------------------------------
# Pipeline Execution
# --------------------------------------------------------------------------

class EventBatch:
    def __init__(self, src_type, src_idx, dst_type, dst_idx, edge_type, edge_feat, t, label):
        self.src_type = src_type
        self.src_idx = torch.tensor([src_idx], dtype=torch.long)
        self.dst_type = dst_type
        self.dst_idx = torch.tensor([dst_idx], dtype=torch.long)
        self.edge_type = edge_type
        self.edge_feat = torch.tensor([edge_feat], dtype=torch.float)
        self.t = torch.tensor([t], dtype=torch.float)
        self.label = torch.tensor([label], dtype=torch.long)
        
        # Dummy context window (padding) for proof of concept
        # A full system tracks actual K past embeds dynamically per node
        self.dst_window_embeds = torch.zeros((1, 1, 64))
        self.dst_window_deltas = torch.zeros((1, 1))

def prepare_event_stream(csv_path="agent_training_data.csv"):
    print("Loading and preparing chronological event stream...")
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    node_types = ['user', 'agent', 'tool']
    node_mapping = {ntype: {} for ntype in node_types}
    num_nodes = {ntype: 0 for ntype in node_types}
    
    for node_id in pd.concat([df['source'], df['target']]).unique():
        if 'user' in node_id: ntype = 'user'
        elif 'tool' in node_id: ntype = 'tool'
        else: ntype = 'agent'
        
        node_mapping[ntype][node_id] = num_nodes[ntype]
        num_nodes[ntype] += 1
        
    def get_node_info(node_id):
        if 'user' in node_id: ntype = 'user'
        elif 'tool' in node_id: ntype = 'tool'
        else: ntype = 'agent'
        return ntype, node_mapping[ntype][node_id]
        
    edge_types = df['action'].unique().tolist()
    feature_cols = ['src_out_degree', 'tgt_in_degree', 'src_pagerank', 'tgt_pagerank']
    df[feature_cols] = StandardScaler().fit_transform(df[feature_cols])
    
    t0 = df['timestamp'].min()
    df['t_seconds'] = (df['timestamp'] - t0).dt.total_seconds()
    
    event_stream = []
    for _, row in df.iterrows():
        src_t, src_i = get_node_info(row['source'])
        dst_t, dst_i = get_node_info(row['target'])
        feat = row[feature_cols].values.tolist()
        
        batch = EventBatch(
            src_type=src_t, src_idx=src_i,
            dst_type=dst_t, dst_idx=dst_i,
            edge_type=row['action'],
            edge_feat=feat,
            t=row['t_seconds'],
            label=int(row['is_anomaly'])
        )
        event_stream.append(batch)
        
    return event_stream, node_types, edge_types, num_nodes, len(feature_cols)

def evaluate_model(model, event_stream):
    model.eval()
    y_true, y_pred, y_scores = [], [], []
    
    with torch.no_grad():
        for batch in event_stream:
            logits, _ = model(batch)
            probs = torch.softmax(logits, dim=-1)
            pred = torch.argmax(probs, dim=-1).item()
            
            y_true.append(batch.label.item())
            y_pred.append(pred)
            y_scores.append(probs[0, 1].item())
            
            # Model continues to maintain temporal state through evaluation
            model.memory_bank.update(
                batch.dst_type, batch.dst_idx, batch.edge_feat, batch.t
            )
            
    print("\n--- THGT Model Evaluation ---")
    print(f"Accuracy: {accuracy_score(y_true, y_pred) * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Benign', 'Anomaly']))
    
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(f"True Negatives:  {cm[0][0]}")
    print(f"False Positives: {cm[0][1]}")
    print(f"False Negatives: {cm[1][0]}")
    print(f"True Positives:  {cm[1][1]}")
    
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    print(f"\nPR-AUC (Precision-Recall Area Under Curve): {pr_auc:.4f}")

if __name__ == "__main__":
    event_stream, node_types, edge_types, num_nodes, edge_feat_dim = prepare_event_stream()
    
    # Increase from 2500 to 10000 events for better training
    event_stream = event_stream[:10000]
    
    print(f"\nInitializing Temporal Heterogeneous Graph Transformer (THGT) on {len(event_stream)} events...")
    model = THGT(
        node_types=node_types,
        edge_types=edge_types,
        num_nodes=num_nodes,
        raw_feat_dim=edge_feat_dim, 
        edge_feat_dim=edge_feat_dim
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # Lowered LR
    loss_fn = FocalLoss()
    
    split_idx = int(len(event_stream) * 0.8)
    train_stream = event_stream[:split_idx]
    test_stream = event_stream[split_idx:]
    
    print(f"Training on {len(train_stream)} events chronologically...")
    
    # Run Training for 2 epochs instead of 1
    for epoch in range(2):
        print(f"Epoch {epoch + 1}/2...")
        loss = train_one_epoch(model, train_stream, optimizer, loss_fn)
        print(f"Training Loss: {loss:.4f}")
    
    # Run Evaluation
    print("\nEvaluating chronologically on unseen Test Stream...")
    evaluate_model(model, test_stream)
