import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch import nn

class STSAGE(nn.Module):
    """
    Spatio-Temporal GraphSAGE
    Combination of GraphSAGE for spatial features and LSTM for temporal sequence.
    """
    def __init__(self, node_features, hidden_dim, out_dim, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Spatial Encoder: GraphSAGE
        # We use a homogeneous SAGEConv. 
        # If the graph is hetero, we can use to_homogeneous() or HeteroConv.
        # Given the "shared_plant" type edges, merging them into a single graph is often reasonable for ST tasks.
        self.conv1 = SAGEConv(node_features, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        
        # Temporal Encoder: LSTM
        # Input: (Batch, Seq_Len, Hidden_Dim) -> Output: (Batch, Hidden_Dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        
        # Regressor
        # Regressor
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.Softplus()
        )

    def forward(self, x, edge_index):
        """
        x: (Num_Nodes, Seq_Len, Features) OR (Batch, Num_Nodes, Seq_Len, Features)
        """
        if x.dim() == 3:
            # Single sample: (Nodes, Seq, Feats)
            num_nodes, seq_len, num_features = x.shape
            batch_size = 1
            x_in = x.unsqueeze(0) # (1, N, S, F)
        else:
            # Batch: (Batch, Nodes, Seq, Feats)
            batch_size, num_nodes, seq_len, num_features = x.shape
            x_in = x

        # Prepare batched edge_index if batch_size > 1
        device = x.device
        if batch_size > 1:
            offsets = torch.arange(batch_size, device=device) * num_nodes
            edge_index_batched = edge_index.unsqueeze(0) + offsets.view(-1, 1, 1)
            # edge_index_batched is (B, 2, E)
            edge_index_in = edge_index_batched.transpose(0, 1).reshape(2, -1)
        else:
            edge_index_in = edge_index

        # Process each timestep
        spatial_embeddings = []
        for t in range(seq_len):
            # x_in is (B, N, S, F) -> xt is (B, N, F)
            xt = x_in[:, :, t, :]
            # Flatten to (B*N, F)
            xt_flat = xt.reshape(batch_size * num_nodes, -1)
            
            h = self.conv1(xt_flat, edge_index_in)
            h = F.relu(h)
            h = F.dropout(h, p=0.5, training=self.training)
            h = self.conv2(h, edge_index_in)
            h = F.relu(h)
            
            # Reshape back to (B, N, H)
            spatial_embeddings.append(h.view(batch_size, num_nodes, -1))
            
        # Stack: (B, N, S, H)
        spatial_seq = torch.stack(spatial_embeddings, dim=2)
        
        # Temporal Aggregation
        # LSTM input: (Batch*Nodes, Seq, Hidden)
        spatial_seq_flat = spatial_seq.reshape(batch_size * num_nodes, seq_len, -1)
        lstm_out, _ = self.lstm(spatial_seq_flat)
        
        # Take last time step
        last_out = lstm_out[:, -1, :] # (B*N, H)
        
        # Prediction
        out = self.regressor(last_out) # (B*N, 1)
        
        if batch_size > 1:
            return out.view(batch_size, num_nodes)
        else:
            return out.view(num_nodes)

if __name__ == "__main__":
    # Test model shape
    model = STSAGE(node_features=4, hidden_dim=16, out_dim=1)
    # Dummy data
    x = torch.randn(40, 14, 4) # 40 nodes, 14 days, 4 features
    edge_index = torch.randint(0, 40, (2, 100))
    
    out = model(x, edge_index)
    print(f"Model Output Shape: {out.shape} (Expected: 40)")
