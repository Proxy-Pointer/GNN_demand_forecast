
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
import pandas as pd

class GNNInterpreter:
    def __init__(self, model, dataset, device=None):
        self.model = model
        self.dataset = dataset
        self.data = dataset.load_graph()
        
        self.device = device if device else torch.device("cpu")
        self.model.to(self.device).eval()
        
        self.edge_index_dict = {k: v.to(self.device) for k, v in self.data.edge_index_dict.items()}
        
        self.feature_names = [
            "Sales", 
            "Delivery", 
            "Factory Issue", 
            "Production", 
            "Rolling Mean (7d)", 
            "Sales Lag (7d)"
        ] 

    def _compute_gradients(self, target_idx, x_sample=None):
        num_nodes = self.data['sku'].num_nodes
        lookback = 14 
        num_features = 6 
        
        if x_sample is not None:
            if not isinstance(x_sample, torch.Tensor):
                x_sample = torch.tensor(x_sample, dtype=torch.float32)
            x_input = x_sample.clone().detach().to(self.device).requires_grad_(True)
        else:
            x_input = torch.randn(num_nodes, lookback, num_features, requires_grad=True, device=self.device)
            
        x_dict = {'sku': x_input}
        out = self.model(x_dict, self.edge_index_dict)
        
        if out.dim() == 2:
            pred_scalar = out[0, target_idx]
        else:
            pred_scalar = out[target_idx]
            
        self.model.zero_grad()
        pred_scalar.backward()
        
        importance_dict = {}
        if x_input.grad is not None:
            sku_imp = x_input.grad.abs().sum(dim=(1, 2)).detach().cpu().numpy()
            importance_dict['sku'] = sku_imp
        
        if hasattr(self.model, 'node_embeds'):
            for ntype, embed_module in self.model.node_embeds.items():
                if embed_module.weight.grad is not None:
                    entity_imp = embed_module.weight.grad.abs().sum(dim=1).detach().cpu().numpy()
                    importance_dict[ntype] = entity_imp
                else:
                    importance_dict[ntype] = np.zeros(embed_module.num_embeddings)
                    
        return x_input, importance_dict

    def get_structural_graph(self, target_idx, top_k=5, x_sample=None):
        _, importance_dict = self._compute_gradients(target_idx, x_sample=x_sample)
        
        # Normalize Neighbor Importance
        target_imp_val = importance_dict['sku'][target_idx]
        sku_imp_no_target = importance_dict['sku'].copy()
        sku_imp_no_target[target_idx] = 0
        
        max_neighbor_imp = sku_imp_no_target.max()
        for ntype, imp in importance_dict.items():
            if ntype != 'sku' and len(imp) > 0:
                max_neighbor_imp = max(max_neighbor_imp, imp.max())
        
        if max_neighbor_imp == 0: max_neighbor_imp = 1.0
        norm_importance = {k: v / max_neighbor_imp for k, v in importance_dict.items()}
        
        G = nx.Graph()
        G.add_node(target_idx, type='target', importance=1.0)
        
        for edge_type, edge_index in self.edge_index_dict.items():
            src_type, rel, dst_type = edge_type
            src_np = edge_index[0].cpu().numpy()
            dst_np = edge_index[1].cpu().numpy()
            
            vals = [] # (neighbor_idx, ntype, weight)
            
            # Source is target -> Dest is neighbor
            if src_type == 'sku':
                mask = (src_np == target_idx)
                indices = np.where(mask)[0]
                for i in indices:
                    neighbor_idx = dst_np[i]
                    weight = norm_importance[dst_type][neighbor_idx]
                    vals.append((neighbor_idx, dst_type, weight))
            
            # Dest is target -> Source is neighbor
            if dst_type == 'sku':
                mask = (dst_np == target_idx)
                indices = np.where(mask)[0]
                for i in indices:
                    neighbor_idx = src_np[i]
                    weight = norm_importance[src_type][neighbor_idx]
                    vals.append((neighbor_idx, src_type, weight))
            
            vals.sort(key=lambda x: x[2], reverse=True)
            top_k_vals = vals[:top_k]
            
            for n_idx, n_type, w in top_k_vals:
                node_id = f"{n_type}_{n_idx}"
                G.add_node(node_id, type=n_type, importance=float(w))
                G.add_edge(target_idx, node_id, weight=float(w), relation=rel)
                
        return G

    def get_feature_importance(self, target_idx, x_sample=None):
        x_input, _ = self._compute_gradients(target_idx, x_sample=x_sample)
        target_grad = x_input.grad[target_idx]
        feat_imp = target_grad.abs().sum(dim=0).detach().cpu().numpy() 
        if feat_imp.max() > 0: feat_imp /= feat_imp.max()
        return dict(zip(self.feature_names, feat_imp))

    def get_temporal_importance(self, target_idx, x_sample=None):
        x_input, _ = self._compute_gradients(target_idx, x_sample=x_sample)
        target_grad = x_input.grad[target_idx] 
        time_imp = target_grad.abs().sum(dim=1).detach().cpu().numpy()
        if time_imp.max() > 0: time_imp /= time_imp.max()
        return time_imp
