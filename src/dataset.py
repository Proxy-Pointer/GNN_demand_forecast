import torch
import pandas as pd
import os
from torch_geometric.data import HeteroData

class SupplyChainDataset:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.nodes_path = os.path.join(root_dir, "Nodes", "NodesIndex.csv")
        self.edges_index_dir = os.path.join(root_dir, "Edges", "EdgesIndex")

    def load_graph(self):
        # 1. Load Nodes
        # We assume nodes are indexed 0 to N-1
        nodes_df = pd.read_csv(self.nodes_path)
        num_nodes = len(nodes_df)
        print(f"Loaded {num_nodes} nodes.")

        # 2. Initialize HeteroData
        # Although all nodes are SKUs, we use HeteroData to handle different edge types easily.
        data = HeteroData()
        
        # We define a single node type 'sku'
        data['sku'].num_nodes = num_nodes
        # We will add features later from temporal data
        data['sku'].x = torch.zeros((num_nodes, 1)) # Placeholder

        # 3. Load Edges from distinct files
        # Map filenames to relation names
        # Filename format: "Edges (Plant).csv" -> relation: (sku, shared_plant, sku)
        edge_files = {
            "Edges (Plant).csv": "shared_plant",
            "Edges (Product Group).csv": "shared_group",
            "Edges (Product Sub-Group).csv": "shared_subgroup",
            "Edges (Storage Location).csv": "shared_storage"
        }

        for filename, relation_name in edge_files.items():
            file_path = os.path.join(self.edges_index_dir, filename)
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                # Expecting columns: "node1", "node2" (indices)
                # The file headers I saw earlier were: Plant, node1, node2 in one file
                # But EdgesIndex/Edges (Plant).csv might be cleaner. 
                # Let's assume 'node1' and 'node2' columns exist and contain the integer indices.
                
                # Check columns
                if 'node1' in df.columns and 'node2' in df.columns:
                    src = torch.tensor(df['node1'].values, dtype=torch.long)
                    dst = torch.tensor(df['node2'].values, dtype=torch.long)

                    edge_index = torch.cat(
                        [
                            torch.stack([src, dst], dim=0),
                            torch.stack([dst, src], dim=0)
                        ],
                        dim=1
                    )
                    
                    data['sku', relation_name, 'sku'].edge_index = edge_index
                else:
                    print(f"Warning: Columns 'node1'/'node2' not found in {filename}")
            else:
                print(f"Warning: File {filename} not found.")

        return data

if __name__ == "__main__":
    # Test loading
    _src = os.path.dirname(os.path.abspath(__file__))
    dataset = SupplyChainDataset(os.path.abspath(os.path.join(_src, "..", "dataset", "Raw Dataset")))
    graph = dataset.load_graph()
    print("\nGraph Structure:")
    print(graph)
