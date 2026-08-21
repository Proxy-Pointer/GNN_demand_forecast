import torch
import torch.nn.functional as F
from torch_geometric.nn import HGTConv
from torch import nn


class STHGT(nn.Module):
    """
    Spatio-Temporal Heterogeneous Graph Transformer

    - HGT for spatial (heterogeneous) aggregation
    - LSTM for temporal aggregation
    - Designed for log1p regression (NO Softplus)
    """

    def __init__(
        self,
        node_features: int,
        hidden_dim: int,
        out_dim: int,
        metadata,
        num_heads: int = 1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.metadata = metadata
        node_types, _ = metadata

        # --------------------------------------------------
        # 1. Input projections
        # --------------------------------------------------
        # Temporal features only exist for SKU nodes
        self.sku_proj = nn.Linear(node_features, hidden_dim)

        # Static embeddings for non-temporal nodes
        self.node_embeds = nn.ModuleDict()
        for ntype in node_types:
            if ntype != "sku":
                self.node_embeds[ntype] = nn.Embedding(1_000, hidden_dim)

        # --------------------------------------------------
        # 2. Two HGT layers (matching SAGE depth)
        # --------------------------------------------------
        self.hgt1 = HGTConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            metadata=metadata,
            heads=num_heads,
        )
        self.hgt2 = HGTConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            metadata=metadata,
            heads=num_heads,
        )

        # --------------------------------------------------
        # 3. Temporal encoder
        # --------------------------------------------------
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        # --------------------------------------------------
        # 4. Regressor (log1p space + Softplus for stability)
        # --------------------------------------------------
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.Softplus()
        )

    # --------------------------------------------------
    # Helper: build static node embeddings
    # --------------------------------------------------
    def _build_static_nodes(self, edge_index_dict, device):
        node_dict_static = {}

        for ntype, emb in self.node_embeds.items():
            max_id = 0
            for (src, _, dst), ei in edge_index_dict.items():
                if src == ntype or dst == ntype:
                    max_id = max(max_id, int(ei.max()))

            ids = torch.arange(max_id + 1, device=device)
            node_dict_static[ntype] = emb(ids)

        return node_dict_static

    # --------------------------------------------------
    # Helper: batch heterogeneous edges
    # --------------------------------------------------
    def _batch_edges(self, edge_index_dict, num_nodes_dict, B, device):
        batched_edge_index_dict = {}
        for (src, rel, dst), edge_index in edge_index_dict.items():
            num_src = num_nodes_dict[src]
            num_dst = num_nodes_dict[dst]
            
            src_offsets = torch.arange(B, device=device) * num_src
            dst_offsets = torch.arange(B, device=device) * num_dst
            
            # (2, E) -> (E, B) broadcast
            src_idx = edge_index[0].unsqueeze(1) + src_offsets.unsqueeze(0)
            dst_idx = edge_index[1].unsqueeze(1) + dst_offsets.unsqueeze(0)
            
            batched_src = src_idx.T.flatten()
            batched_dst = dst_idx.T.flatten()
            
            batched_edge_index_dict[(src, rel, dst)] = torch.stack([batched_src, batched_dst])
            
        return batched_edge_index_dict

    # --------------------------------------------------
    # Forward
    # --------------------------------------------------
    def forward(self, x_dict, edge_index_dict):
        """
        x_dict["sku"]:
            (N, S, F) or (B, N, S, F)
        """

        # --------------------------------------------------
        # 0. Preparation
        # --------------------------------------------------
        x = x_dict["sku"]
        if x.dim() == 3:
            x = x.unsqueeze(0)

        B, N_sku, S, num_features = x.shape
        device = x.device

        # --------------------------------------------------
        # 1. Infer num_nodes for ALL types 
        #    (needed for batch offsets)
        # --------------------------------------------------
        num_nodes_dict = {"sku": N_sku}
        for ntype in self.node_embeds:
            max_id = 0
            for (src, _, dst), ei in edge_index_dict.items():
                if src == ntype: max_id = max(max_id, int(ei[0].max()))
                if dst == ntype: max_id = max(max_id, int(ei[1].max()))
            num_nodes_dict[ntype] = max_id + 1

        # --------------------------------------------------
        # 2. Batch the Graph Structure (One huge disjoint graph)
        # --------------------------------------------------
        edge_index_batched = self._batch_edges(edge_index_dict, num_nodes_dict, B, device)

        # --------------------------------------------------
        # 3. Batch Static Embeddings
        # --------------------------------------------------
        node_dict_static_batched = {}
        for ntype, emb in self.node_embeds.items():
            N_t = num_nodes_dict[ntype]
            ids = torch.arange(N_t, device=device)
            base_emb = emb(ids) # (N_t, H)
            # Repeat for B batches: (B*N_t, H)
            node_dict_static_batched[ntype] = base_emb.repeat(B, 1)

        spatial_out = []

        # --------------------------------------------------
        # 4. Temporal Loop (Optimized: 1 Conv call per step)
        # --------------------------------------------------
        for t in range(S):
            xt_flat = x[:, :, t, :].reshape(B * N_sku, -1)
            xt_proj = self.sku_proj(xt_flat)
            
            node_dict = {"sku": xt_proj}
            node_dict.update(node_dict_static_batched)
            
            # Layer 1
            h1 = self.hgt1(node_dict, edge_index_batched)
            h1 = {k: F.gelu(v) for k, v in h1.items()}
            # Dropout 1
            h1 = {k: F.dropout(v, p=0.4, training=self.training) for k, v in h1.items()}
            
            # Layer 2
            h_dict = self.hgt2(h1, edge_index_batched)
            
            # Residual (on flattened tensors)
            h_dict["sku"] = 0.7 * h_dict["sku"] + 0.3 * xt_proj
            
            # Activation
            for k in h_dict:
                h_dict[k] = F.gelu(h_dict[k])
                # Dropout 2
                #h_dict[k] = F.dropout(h_dict[k], p=0.2, training=self.training)
                
            # Keep only SKU nodes: (B*N, H)
            spatial_out.append(h_dict["sku"])

        # --------------------------------------------------
        # 5. Temporal Aggregation
        # --------------------------------------------------
        # Stack: (B*N, S, H)
        spatial_seq = torch.stack(spatial_out, dim=1)
        
        lstm_out, _ = self.lstm(spatial_seq)
        last = lstm_out[:, -1, :] # (B*N, H)
        out = self.regressor(last) # (B*N, 1)
        
        return out.view(B, N_sku)


# --------------------------------------------------
# Sanity check
# --------------------------------------------------
if __name__ == "__main__":
    metadata = (
        ["sku", "plant", "product_group"],
        [
            ("sku", "produced_at", "plant"),
            ("plant", "produces", "sku"),
            ("sku", "belongs_to", "product_group"),
            ("product_group", "contains", "sku"),
        ],
    )

    model = STHGT(
        node_features=6,
        hidden_dim=32,
        out_dim=1,
        metadata=metadata,
        num_heads=2,
    )

    x = torch.randn(20, 14, 6)
    edge_index_dict = {
        ("sku", "produced_at", "plant"): torch.randint(0, 20, (2, 100)),
        ("plant", "produces", "sku"): torch.randint(0, 20, (2, 100)),
        ("sku", "belongs_to", "product_group"): torch.randint(0, 20, (2, 60)),
        ("product_group", "contains", "sku"): torch.randint(0, 20, (2, 60)),
    }

    out = model({"sku": x}, edge_index_dict)
    print("Output shape:", out.shape)  # (20,)
