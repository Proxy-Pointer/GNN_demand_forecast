
print("Step 0: Imports starting...")
try:
    import matplotlib
    matplotlib.use('Agg') # Force headless backend
    import matplotlib.pyplot as plt
    print("Step 0a: matplotlib imported")
    import networkx as nx
    print("Step 0b: networkx imported")
    import torch
    print("Step 0c: torch imported")
    import os
    import sys
    import pandas as pd
    import random
    import numpy as np
except Exception as e:
    print(f"IMPORT ERROR: {e}")
    sys.exit(1)

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import SupplyChainDataset

def visualize_graph_subset(num_nodes_to_sample=12):
    print("Step 1: Loading Graph Data...")
    root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset", "Raw Dataset")
    dataset = SupplyChainDataset(root_dir)
    data = dataset.load_graph()
    
    total_nodes = data['sku'].num_nodes
    print(f"Total Nodes: {total_nodes}")

    # 1. Select Subset of Nodes
    # Strategy: Pick nodes with high degree to ensure they have connections to show
    # Calculate degree for each node across all edge types
    degrees = np.zeros(total_nodes)
    
    for etype in data.edge_types:
        edge_index = data[etype].edge_index
        src = edge_index[0].numpy()
        dst = edge_index[1].numpy()
        for n in src: degrees[n] += 1
        for n in dst: degrees[n] += 1
        
    # Get indices of top connected nodes
    top_indices = np.argsort(degrees)[-num_nodes_to_sample:]
    print(f"Selected Top {num_nodes_to_sample} Nodes indices: {top_indices}")
    
    subset_nodes = set(top_indices)

    # 2. Build Subgraph
    G = nx.MultiGraph()
    G.add_nodes_from(subset_nodes)
    
        # Load labels
    try:
        nodes_df = pd.read_csv(os.path.join(root_dir, "Nodes", "Nodes.csv"))
        all_labels = nodes_df['Node'].to_dict()
        labels = {i: all_labels[i] for i in subset_nodes}
    except:
        labels = {i: str(i) for i in subset_nodes}

    edge_types = data.edge_types
    # Colors for different relations
    colors = ['r', 'b', 'g', 'purple', 'orange']
    edge_styles = {}
    
    print("Step 3: Filtering Edges...")
    
    for i, etype in enumerate(edge_types):
        rel_name = etype[1]
        edge_index = data[etype].edge_index
        src = edge_index[0].numpy()
        dst = edge_index[1].numpy()
        
        edges = []
        for s, d in zip(src, dst):
            if s in subset_nodes and d in subset_nodes:
                edges.append((s, d))
        
        if edges:
            color = colors[i % len(colors)]
            G.add_edges_from(edges, type=rel_name, color=color)
            edge_styles[rel_name] = color
            print(f"  - Added {len(edges)} {rel_name} edges")

    print(f"Subgraph constructed: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")

    # 3. Draw
    print("Step 4: Drawing...")
    plt.figure(figsize=(10, 10))
    
    # Use spring layout for organic look
    pos = nx.spring_layout(G, seed=42, k=1.5) # k controls spacing 
    
    # Draw Nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightgray', node_size=500, edgecolors='black')
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_weight='bold')
    
    # Draw Edges (curved if multiple edges between nodes)
    ax = plt.gca()
    for rel_name, color in edge_styles.items():
        target_edges = [
            (u, v) for u, v, k, d in G.edges(keys=True, data=True) 
            if d.get('type') == rel_name
        ]
        
        if target_edges:
            # Use rad to curve edges slightly so they don't overlap perfectly
            nx.draw_networkx_edges(
                G, pos, 
                edgelist=target_edges, 
                edge_color=color, 
                width=1.5, 
                alpha=0.6,
                connectionstyle=f"arc3,rad={0.1 + (0.1 * list(edge_styles.keys()).index(rel_name))}" 
            )

    # Manual Legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color=c, lw=2, label=r) for r, c in edge_styles.items()]
    plt.legend(handles=legend_elements, loc='upper right', title="Relationship Types")

    plt.title(f"Graph Topology (Subset: {num_nodes_to_sample} Most Connected SKUs)", fontsize=14)
    plt.axis('off')
    
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "visuals")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "graph_structure_sample.png")
    
    plt.savefig(output_path)
    print(f"Sample Graph visualization saved to {output_path}")
    plt.close()

if __name__ == "__main__":
    try:
        visualize_graph_subset()
    except Exception as e:
        import traceback
        traceback.print_exc()
