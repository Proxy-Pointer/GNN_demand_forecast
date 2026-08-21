import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd
import seaborn as sns

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import SupplyChainDataset
from temporal import TimeSeriesProcessor
from dataset import SupplyChainDataset
from temporal import TimeSeriesProcessor
from model_sage import STSAGE
from model_hgt import STHGT
import argparse

def visualize(model_type="SAGE"):
    print(f"Initializing Visualizations for {model_type}...")
    src_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(src_dir, "..", "dataset", "Raw Dataset"))
    model_path = os.path.abspath(os.path.join(src_dir, "..", "model", f"st_gnn_model_{model_type}.pth"))
    output_dir = os.path.abspath(os.path.join(src_dir, "..", "visuals"))
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load SKU Names
    nodes_df = pd.read_csv(os.path.join(root_dir, "Nodes", "Nodes.csv"))
    sku_names = nodes_df['Node'].tolist()

    # 2. Load Data (Same as evaluation)
    graph_dataset = SupplyChainDataset(root_dir)
    graph_data = graph_dataset.load_graph()
    
    # Prepare edge inputs for both types
    edge_indices = []
    edge_index_dict = {}
    for edge_type in graph_data.edge_types:
        edge_index = graph_data[edge_type].edge_index
        edge_indices.append(edge_index)
        edge_index_dict[edge_type] = edge_index
    full_edge_index = torch.cat(edge_indices, dim=1)

    temp_proc = TimeSeriesProcessor(root_dir)
    features, common_index = temp_proc.load_features(log_transform=True, normalize=False, add_lags=True)

    lookback = 14
    horizon = 1
    x_all, y_all = temp_proc.create_sliding_window(features, lookback, horizon)
    num_samples = x_all.shape[0]
    test_size = int(0.2 * num_samples)
    train_size = num_samples - test_size
    
    print(f"Total Samples: {num_samples}, Train: {train_size}, Test: {test_size}")
    print(f"Features: {features.shape}, common_index: {len(common_index)}")

    x_test = x_all[train_size:]
    y_test = y_all[train_size:]
    
    # Each sample i in x_all predicts day lookback + i
    # So test sample 0 (at index train_size) predicts day lookback + train_size
    start_idx = train_size + lookback
    test_dates = common_index[start_idx : start_idx + test_size]
    
    print(f"Test dates range: {test_dates[0]} to {test_dates[-1]} (Count: {len(test_dates)})")

    # 3. Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_type == "SAGE":
        model = STSAGE(node_features=x_all.shape[-1], hidden_dim=32, out_dim=1).to(device)
        edge_input = full_edge_index.to(device)
    else:
        metadata = graph_data.metadata()
        model = STHGT(node_features=x_all.shape[-1], hidden_dim=32, out_dim=1, metadata=metadata).to(device)
        edge_input = {k: v.to(device) for k, v in edge_index_dict.items()}

    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # 4. Inference
    preds, actuals = [], []
    with torch.no_grad():
        for i in range(test_size):
            x_batch = x_test[i].to(device)
            if model_type == "SAGE":
                out = model(x_batch, edge_input)
            else:
                out = model({"sku": x_batch}, edge_input)
                
            preds.append(out.cpu().numpy().squeeze())
            actuals.append(y_test[i].squeeze().cpu().numpy())

    preds = np.stack(preds)      # (T, N)
    actuals = np.stack(actuals)  # (T, N)

    # 5. Denormalize and Correct (Real-Space logic from evaluate_gnn.py)
    preds_denorm = np.expm1(preds)
    actuals_denorm = np.expm1(actuals)
    
    K = 14
    preds_corr = preds_denorm.copy()
    for t in range(test_size):
        start = max(0, t - K)
        if t > 0:
            hist_bias = preds_denorm[start:t] - actuals_denorm[start:t]
            sku_bias = hist_bias.mean(axis=0)
            preds_corr[t] -= sku_bias
    preds_corr = np.maximum(preds_corr, 0)

    # ------------------------------------------------------------------
    # CHART 1: Top 4 SKU Dashboard
    # ------------------------------------------------------------------
    print("Generating Dashboard...")
    # Find top 4 volume SKUs
    mean_sales = actuals_denorm.mean(axis=0)
    top_4_idx = np.argsort(mean_sales)[-4:][::-1]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    for i, idx in enumerate(top_4_idx):
        ax = axes[i]
        ax.plot(test_dates, actuals_denorm[:, idx], label='Actual', color='blue', alpha=0.6)
        ax.plot(test_dates, preds_corr[:, idx], label='Predicted (Corrected)', color='red', linestyle='--')
        ax.set_xlabel("Date", fontweight='bold')
        ax.set_ylabel("Sales", fontweight='bold')
        ax.set_title(f"SKU: {sku_names[idx]} (Mean Vol: {mean_sales[idx]:.1f})", fontweight='bold')
        ax.legend(prop={'weight': 'bold'})
        ax.grid(True, linestyle=':', alpha=0.7)
        plt.setp(ax.get_xticklabels(), rotation=30)

    plt.tight_layout()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"forecast_dashboard_{model_type}.png"))
    plt.close()

    # ------------------------------------------------------------------
    # CHART 2: Global Actual vs Pred Scatter
    # ------------------------------------------------------------------
    print("Generating Scatter...")
    plt.figure(figsize=(10, 8))
    plt.scatter(actuals_denorm.flatten(), preds_corr.flatten(), alpha=0.3, color='forestgreen', s=10)
    max_val = max(actuals_denorm.max(), preds_corr.max())
    plt.plot([0, max_val], [0, max_val], 'r--', label='Perfect Forecast')
    plt.xlabel("Actual Sales", fontweight='bold')
    plt.ylabel("Predicted Sales", fontweight='bold')
    plt.title( f"Global Forecast Accuracy (All SKUs, All Days) - {model_type}", fontweight='bold')
    plt.legend(prop={'weight': 'bold'})
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, f"actual_vs_pred_scatter_{model_type}.png"))
    plt.close()

    # ------------------------------------------------------------------
    # CHART 2.5: PRO VIEW - Log-Log Forecast Scatter
    # ------------------------------------------------------------------
    print("Generating Pro View (Log-Log Scatter)...")
    plt.figure(figsize=(10, 8))
    # Using log1p to handle zeros
    act_log = np.log1p(actuals_denorm.flatten())
    pre_log = np.log1p(preds_corr.flatten())
    
    plt.scatter(act_log, pre_log, alpha=0.2, color='royalblue', s=10)
    
    max_log = max(act_log.max(), pre_log.max())
    plt.plot([0, max_log], [0, max_log], 'r--', label='Perfect Forecast')
    
    plt.xlabel("Log(Actual Sales + 1)", fontweight='bold')
    plt.ylabel("Log(Predicted Sales + 1)", fontweight='bold')
    plt.title(f"Pro View: Forecast Accuracy across Magnitudes (Log-Log) - {model_type}", fontweight='bold')
    plt.legend(prop={'weight': 'bold'})
    plt.grid(True, alpha=0.3)
    
    # Add annotations for context
    plt.text(0.5, max_log*0.9, "Small Volume SKUs", fontsize=9, alpha=0.7)
    plt.text(max_log*0.7, max_log*0.9, "High Volume SKUs", fontsize=9, alpha=0.7)
    
    plt.savefig(os.path.join(output_dir, f"actual_vs_pred_log_scatter_{model_type}.png"))
    plt.close()

    # ------------------------------------------------------------------
    # CHART 3: SKU Performance Ranking (WAPE)
    # ------------------------------------------------------------------
    print("Generating Rankings...")
    num_skus_in_data = actuals_denorm.shape[1]
    active_sku_names = sku_names[:num_skus_in_data]
    
    sku_wapes = []
    for n in range(num_skus_in_data):
        sad = np.sum(np.abs(actuals_denorm[:, n] - preds_corr[:, n]))
        st = np.sum(np.abs(actuals_denorm[:, n]))
        w = sad / st if st > 0 else 0
        sku_wapes.append(w)

    wape_df = pd.DataFrame({'SKU': active_sku_names, 'WAPE': sku_wapes})
    wape_df = wape_df[wape_df['WAPE'] > 0].sort_values('WAPE')
    
    plt.figure(figsize=(10, 12))
    colors = ['green' if x < 0.6 else 'orange' if x < 0.75 else 'red' for x in wape_df['WAPE']]
    sns.barplot(data=wape_df, x='WAPE', y='SKU', palette=colors)
    plt.axvline(x=0.75, color='red', linestyle='--', label='Target Threshold (75%)')
    plt.axvline(x=0.6, color='green', linestyle='--', label='Target Threshold (60%)')
    plt.xlabel("WAPE (Weighted MAPE)", fontweight='bold')
    plt.ylabel("SKU ID", fontweight='bold')
    plt.title(f"SKU Performance Ranking (WAPE) - {model_type}", fontweight='bold')
    plt.legend(prop={'weight': 'bold'})
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"sku_performance_rank_{model_type}.png"))
    plt.close()

    # ------------------------------------------------------------------
    # CHART 4: Error Distribution
    # ------------------------------------------------------------------
    print("Generating Error Distribution...")
    errors = (preds_corr - actuals_denorm).flatten()
    # Trim outliers for better histogram view
    q1, q3 = np.percentile(errors, [5, 95])
    errors_trimmed = errors[(errors > q1) & (errors < q3)]

    plt.figure(figsize=(10, 6))
    sns.histplot(errors_trimmed, kde=True, color='purple', bins=40)
    plt.axvline(x=0, color='black', linestyle='-')
    plt.title( f"Frequency Distribution of Prediction Errors (Middle 90%) - {model_type}", fontweight='bold')
    plt.xlabel("Prediction Error (Pred - Actual)", fontweight='bold')
    plt.ylabel("Frequency", fontweight='bold')
    plt.savefig(os.path.join(output_dir, f"error_distribution_{model_type}.png"))
    plt.close()

    print(f"Visualizations saved to: {output_dir}")

def visualize_comparison():
    print("Initializing Comparison Visualization (SAGE vs HGT)...")
    src_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(src_dir, "..", "dataset", "Raw Dataset"))
    output_dir = os.path.abspath(os.path.join(src_dir, "..", "visuals"))
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Data
    graph_dataset = SupplyChainDataset(root_dir)
    graph_data = graph_dataset.load_graph()
    
    edge_indices = []
    edge_index_dict = {}
    for edge_type in graph_data.edge_types:
        edge_index = graph_data[edge_type].edge_index
        edge_indices.append(edge_index)
        edge_index_dict[edge_type] = edge_index
    full_edge_index = torch.cat(edge_indices, dim=1)

    temp_proc = TimeSeriesProcessor(root_dir)
    features, common_index = temp_proc.load_features(log_transform=True, normalize=False, add_lags=True)

    lookback = 14
    horizon = 1
    x_all, y_all = temp_proc.create_sliding_window(features, lookback, horizon)
    num_samples = x_all.shape[0]
    test_size = int(0.2 * num_samples)
    train_size = num_samples - test_size
    
    x_test = x_all[train_size:]
    y_test = y_all[train_size:]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # helper for inference
    def get_preds(model_type):
        model_path = os.path.abspath(os.path.join(src_dir, "..", "model", f"st_gnn_model_{model_type}.pth"))
        if not os.path.exists(model_path):
            print(f"Warning: Model {model_type} not found at {model_path}")
            return None
            
        if model_type == "SAGE":
            model = STSAGE(node_features=x_all.shape[-1], hidden_dim=32, out_dim=1).to(device)
            edge_input = full_edge_index.to(device)
        else:
            metadata = graph_data.metadata()
            model = STHGT(node_features=x_all.shape[-1], hidden_dim=32, out_dim=1, metadata=metadata).to(device)
            edge_input = {k: v.to(device) for k, v in edge_index_dict.items()}

        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        
        preds = []
        with torch.no_grad():
            for i in range(test_size):
                x_batch = x_test[i].to(device)
                if model_type == "SAGE":
                    out = model(x_batch, edge_input)
                else:
                    out = model({"sku": x_batch}, edge_input)
                preds.append(out.cpu().numpy().squeeze())
        return np.stack(preds)

    print("Running Inference for SAGE...")
    sage_raw = get_preds("SAGE")
    print("Running Inference for HGT...")
    hgt_raw = get_preds("HGT")
    
    actuals = np.array([y_test[i].squeeze().cpu().numpy() for i in range(test_size)])
    actuals_denorm = np.expm1(actuals)
    
    # Denorm & Correct
    def process_preds(raw):
        if raw is None: return None
        denorm = np.expm1(raw)
        corr = denorm.copy()
        K = 14
        for t in range(test_size):
            start = max(0, t - K)
            if t > 0:
                hist_bias = denorm[start:t] - actuals_denorm[start:t]
                sku_bias = hist_bias.mean(axis=0)
                corr[t] -= sku_bias
        return np.maximum(corr, 0)

    sage_corr = process_preds(sage_raw)
    hgt_corr = process_preds(hgt_raw)
    
    # Plot
    print("Generating Comparison Log-Log Scatter...")
    plt.figure(figsize=(10, 8))
    
    act_log = np.log1p(actuals_denorm.flatten())
    max_log = act_log.max()
    
    # Plot SAGE (Bright Red -> Deep Red)
    if sage_corr is not None:
        sage_log = np.log1p(sage_corr.flatten())
        plt.scatter(act_log, sage_log, alpha=0.3, color='purple', s=10, label='SAGE')
        max_log = max(max_log, sage_log.max())

    # Plot HGT (Deep Blue)
    if hgt_corr is not None:
        hgt_log = np.log1p(hgt_corr.flatten())
        plt.scatter(act_log, hgt_log, alpha=0.3, color='cyan', s=10, label='HGT')
        max_log = max(max_log, hgt_log.max())
        
    plt.plot([0, max_log], [0, max_log], 'k--', label='Perfect Forecast', linewidth=2)
    
    plt.xlabel("Log(Actual Sales + 1)", fontweight='bold')
    plt.ylabel("Log(Predicted Sales + 1)", fontweight='bold')
    plt.title("Model Comparison: SAGE vs HGT (Log-Log Scatter)", fontweight='bold')
    plt.legend(prop={'weight': 'bold'})
    plt.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(output_dir, "actual_vs_pred_log_scatter_comparison.png"))
    plt.close()
    print(f"Comparison chart saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="ALL", choices=["SAGE", "HGT", "ALL", "COMPARE"])
    args = parser.parse_args()
    
    if args.model_type == "ALL":
        for m in ["SAGE", "HGT"]:
            try:
                visualize(m)
            except Exception as e:
                print(f"Error visualizing {m}: {e}")
        # Also run comparison
        try:
            visualize_comparison()
        except Exception as e:
            print(f"Error executing comparison: {e}")
            
    elif args.model_type == "COMPARE":
        visualize_comparison()
    else:
        visualize(args.model_type)
