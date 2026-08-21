import torch
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import SupplyChainDataset
from temporal import TimeSeriesProcessor
from model_sage import STSAGE
from model_hgt import STHGT


def evaluate(model_type="HGT", model=None):
    print(f"\nEvaluating model type: {model_type}\n")

    src_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(src_dir, "..", "dataset", "Raw Dataset"))
    
    # Loads model from disk only if not passed in memory
    if model is None:
        model_path = os.path.abspath(
            os.path.join(src_dir, "..", "model", f"st_gnn_model_{model_type}.pth")
        )

    # ------------------------------------------------------------
    # 1. Load Graph
    # ------------------------------------------------------------
    graph_dataset = SupplyChainDataset(root_dir)
    graph_data = graph_dataset.load_graph()

    edge_indices = []
    edge_index_dict = {}

    for edge_type in graph_data.edge_types:
        edge_index = graph_data[edge_type].edge_index
        edge_indices.append(edge_index)
        edge_index_dict[edge_type] = edge_index

    full_edge_index = torch.cat(edge_indices, dim=1)

    # ------------------------------------------------------------
    # 2. Load Temporal Data
    # ------------------------------------------------------------
    temp_proc = TimeSeriesProcessor(root_dir)
    features, _ = temp_proc.load_features(
        log_transform=True,
        normalize=False,
        add_lags=True
    )

    lookback = 14
    horizon = 1
    x_all, y_all = temp_proc.create_sliding_window(features, lookback, horizon)

    num_samples = x_all.shape[0]
    test_size = int(0.2 * num_samples)
    train_size = num_samples - test_size

    x_test = x_all[train_size:]
    y_test = y_all[train_size:]

    # ------------------------------------------------------------
    # 3. Load Model (if needed)
    # ------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model is None:
        if model_type == "SAGE":
            model = STSAGE(
                node_features=x_all.shape[-1],
                hidden_dim=32,
                out_dim=1
            ).to(device)
            edge_input = full_edge_index.to(device)

        else:
            metadata = graph_data.metadata()
            model = STHGT(
                node_features=x_all.shape[-1],
                hidden_dim=32,
                out_dim=1,
                metadata=metadata
            ).to(device)
            edge_input = {k: v.to(device) for k, v in edge_index_dict.items()}

        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded robust checkpoint from Epoch {checkpoint.get('epoch', '?')}")
        else:
            model.load_state_dict(checkpoint)
    else:
        # Prepare edge inputs for passed model
        if model_type == "SAGE":
            edge_input = full_edge_index.to(device)
        else:
            edge_input = {k: v.to(device) for k, v in edge_index_dict.items()}

    model.eval()

    preds, actuals, baselines = [], [], []

    # ------------------------------------------------------------
    # 4. Inference
    # ------------------------------------------------------------
    with torch.no_grad():
        for i in range(test_size):
            x_batch = x_test[i].to(device)
            y_batch = y_test[i].to(device)

            if model_type == "SAGE":
                out = model(x_batch, edge_input)
            else:
                out = model({"sku": x_batch}, edge_input)

            naive_pred = x_batch[:, -1, 0]  # last-day log1p sales

            preds.append(out.cpu().numpy().squeeze())
            actuals.append(y_batch.squeeze().cpu().numpy())
            baselines.append(naive_pred.cpu().numpy())

    preds = np.stack(preds).astype(np.float64)
    actuals = np.stack(actuals).astype(np.float64)
    baselines = np.stack(baselines).astype(np.float64)

    # ------------------------------------------------------------
    # 5. Metrics + Debug
    # ------------------------------------------------------------
    results_dir = os.path.abspath(os.path.join(src_dir, "..", "results"))
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    results_path = os.path.join(results_dir, f"eval_report_{model_type}.txt")

    with open(results_path, "w") as f:

        # --------------------------------------------------------
        # NOTE:
        # Rolling bias correction is DIAGNOSTIC ONLY.
        # Do NOT apply this logic in production inference.
        # --------------------------------------------------------

        K = 14

        preds_denorm = np.maximum(np.expm1(preds), 0)
        actuals_denorm = np.maximum(np.expm1(actuals), 0)
        baselines_denorm = np.maximum(np.expm1(baselines), 0)

        preds_corr = preds_denorm.copy()

        for t in range(test_size):
            start = max(0, t - K)
            if t > 0:
                hist_bias = preds_denorm[start:t] - actuals_denorm[start:t]
                
                # Careful mean reduction
                if hist_bias.ndim == 3: # (T, 1, N)
                     sku_bias = hist_bias.mean(axis=0) # -> (1, N)
                else: 
                     sku_bias = hist_bias.mean(axis=0)

                # DEBUG SHAPES
                # print(f"DEBUG: t={t}, preds_corr[t]={preds_corr[t].shape}, sku_bias={sku_bias.shape}")

                # Assignment
                preds_corr[t] -= sku_bias

        preds_corr = np.maximum(preds_corr, 0)

        f.write("\nDEBUG: RAW MODEL OUTPUTS (LOG1P SPACE)\n")
        f.write(f"Preds   | Mean: {preds.mean():.4f}, Min: {preds.min():.4f}, Max: {preds.max():.4f}\n")
        f.write(f"Actuals | Mean: {actuals.mean():.4f}, Min: {actuals.min():.4f}, Max: {actuals.max():.4f}\n")

        f.write("\nDEBUG: REAL SALES (DENORMALIZED + BIAS CORRECTED)\n")
        f.write(f"Preds   | Mean: {preds_corr.mean():.4f}, Min: {preds_corr.min():.4f}, Max: {preds_corr.max():.4f}\n")
        f.write(f"Actuals | Mean: {actuals_denorm.mean():.4f}, Min: {actuals_denorm.min():.4f}, Max: {actuals_denorm.max():.4f}\n")
        f.write("-" * 60 + "\n")

        # ---------------- Metric helpers ----------------
        def wape(y_true, y_pred, name):
            sad = np.sum(np.abs(y_true - y_pred))
            st = np.sum(np.abs(y_true))
            f.write(
                f"DEBUG WAPE ({name}): "
                f"Sum Abs Diff: {sad:.4f}, Sum True: {st:.4f}\n"
            )
            return 0.0 if st == 0 else sad / st

        def bias(y_true, y_pred, name):
            b = np.mean(y_pred - y_true)
            f.write(f"DEBUG BIAS ({name}): Mean Diff: {b:.4f}\n")
            return b

        # ---------------- Metrics ----------------
        wape_raw = wape(actuals_denorm, preds_denorm, "Model (Raw)")
        wape_corr = wape(actuals_denorm, preds_corr, "Model (Log-Corrected)")
        bias_corr = bias(actuals_denorm, preds_corr, "Model (Log-Corrected)")

        wape_base = wape(actuals_denorm, baselines_denorm, "Baseline")
        bias_base = bias(actuals_denorm, baselines_denorm, "Baseline")

        # ---------------- Final report ----------------
        f.write("\n" + "=" * 60 + "\n")
        f.write("EVALUATION RESULTS (Log1p Training)\n")
        f.write("=" * 60 + "\n")
        f.write(f"{'Metric':<20} | {'ST-GNN Model':<15} | {'Naïve Baseline':<15}\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'WAPE':<20} | {wape_corr:.4f}         | {wape_base:.4f}\n")
        f.write(f"{'Bias':<20} | {bias_corr:.4f}          | {bias_base:.4f}\n")
        f.write("=" * 60 + "\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("EXPLAINABILITY ANALYSIS (HGT Only)\n")
        f.write("=" * 60 + "\n")
            
        if model_type == "HGT":
            from explainability import GNNInterpreter
            
            interpreter = GNNInterpreter(model, graph_dataset, device)
            
            # -----------------------------------------------
            # 1. Identify Top 4 SKUs by Volume (in Test Set)
            # -----------------------------------------------
            # actuals_denorm shape: (TestSize, NumNodes)
            sku_volumes = actuals_denorm.sum(axis=0) # (NumNodes,)
            top_4_indices = np.argsort(sku_volumes)[-4:][::-1] # Descending
            
            # Load Node Map
            nodes_path = os.path.join(root_dir, "Nodes", "NodesIndex.csv")
            idx_to_name = {}
            if os.path.exists(nodes_path):
                import pandas as pd
                nodes_df = pd.read_csv(nodes_path)
                idx_to_name = dict(zip(nodes_df["NodeIndex"], nodes_df["Node"]))
            
            # -----------------------------------------------
            # 2. Aggregated Analysis for TOP Volume SKU (First 7 Days)
            # -----------------------------------------------
            target_idx = top_4_indices[0] # Top 1
            sku_name = idx_to_name.get(target_idx, f"SKU_{target_idx}")
            vol = sku_volumes[target_idx]
            
            f.write(f"\nAGGREGATED WEEKLY ANALYSIS: Top Volume SKU ({sku_name})\n")
            f.write(f"Total Test Volume: {vol:.0f}\n")
            f.write("Averaging importance over Day 0 to Day 6 (First Week of Test Set).\n")
            
            days_to_analyze = 7 # Day 0 to 6
            
            # Volume Check
            period_actuals = actuals_denorm[:days_to_analyze, target_idx]
            period_preds = preds_corr[:days_to_analyze, target_idx]
            total_act = period_actuals.sum()
            total_pred = period_preds.sum()
            
            f.write(f"\nPeriod Volume (7 Days): Actual {total_act:.0f} vs Predicted {total_pred:.0f}\n")
            f.write("-" * 40 + "\n")

            
            # Accumulators
            agg_feat_imp = {}
            agg_temp_imp = None
            agg_neighbor_imp = {} # name -> weight sum
            agg_neighbor_rel = {} # name -> relation
            
            for t_idx in range(days_to_analyze):
                x_sample = x_test[t_idx] # (Nodes, Lookback, Feats)
                
                # A. Feature Importance
                feat_imp = interpreter.get_feature_importance(target_idx, x_sample=x_sample)
                for k, v in feat_imp.items():
                    agg_feat_imp[k] = agg_feat_imp.get(k, 0.0) + v
                    
                # B. Temporal Importance
                temp_imp = interpreter.get_temporal_importance(target_idx, x_sample=x_sample)
                if agg_temp_imp is None:
                    agg_temp_imp = np.zeros_like(temp_imp)
                agg_temp_imp += temp_imp
                
                # C. Structural Importance
                # Get wider net (Top 10) to catch varying neighbors
                G = interpreter.get_structural_graph(target_idx, top_k=10, x_sample=x_sample)
                
                for n in G.nodes():
                    if n == target_idx: continue
                    
                    # Resolve Name
                    display_name = n
                    if "_" in str(n):
                        prefix, idx_str = str(n).split("_", 1)
                        if prefix == "sku" and idx_str.isdigit():
                             nid = int(idx_str)
                             display_name = idx_to_name.get(nid, f"SKU_{nid}")
                    
                    # Get Weight
                    if G.has_edge(target_idx, n):
                        w = G[target_idx][n].get('weight', 0.0)
                        rel = G[target_idx][n].get('relation', 'unknown')
                        
                        agg_neighbor_imp[display_name] = agg_neighbor_imp.get(display_name, 0.0) + w
                        agg_neighbor_rel[display_name] = rel # Assume constant relation

            # Average Results
            f.write("\n--- Average Drivers (Over 7 Days) ---\n")
            
            # 1. Feature (Normalize by Days)
            sorted_feats = sorted(agg_feat_imp.items(), key=lambda x: x[1], reverse=True)
            f.write("1. Feature Importance:\n")
            for k, total_v in sorted_feats:
                avg_v = total_v / days_to_analyze
                f.write(f"   - {k:<20}: {avg_v:.4f}\n")
                
            # 2. Temporal
            f.write("\n2. Temporal Importance:\n")
            agg_temp_imp /= days_to_analyze
            top_lags = np.argsort(agg_temp_imp)[-5:][::-1]
            for lag in top_lags:
                days_ago = 14 - lag
                f.write(f"   - {days_ago} Days Ago (t-{days_ago}): {agg_temp_imp[lag]:.4f}\n")
            
            # 3. Structural
            f.write("\n3. Key Neighbors (Structural):\n")
            # Sort by total weight
            sorted_neighbors = sorted(agg_neighbor_imp.items(), key=lambda x: x[1], reverse=True)
            for name, total_w in sorted_neighbors[:5]:
                avg_w = total_w / days_to_analyze
                rel = agg_neighbor_rel.get(name, '?')
                f.write(f"   - {name} ({avg_w:.2f}) [{rel}]\n")
                
            # -----------------------------------------------
            # 3. Aggregated Analysis for TOP Volume SKU (LAST 7 Days)
            # -----------------------------------------------
            f.write("\n" + "="*40 + "\n")
            f.write(f"AGGREGATED WEEKLY ANALYSIS: Top Volume SKU ({sku_name}) - LAST WEEK\n")
            f.write(f"Averaging importance over Last 7 Days of Test Set.\n")
            
            # Start/End Indices
            test_len = x_test.shape[0]
            start_last = test_len - 7
            end_last = test_len
            
            # Volume Check
            period_actuals = actuals_denorm[start_last:end_last, target_idx]
            period_preds = preds_corr[start_last:end_last, target_idx]
            total_act = period_actuals.sum()
            total_pred = period_preds.sum()
            
            f.write(f"\nPeriod Volume (Last 7 Days): Actual {total_act:.0f} vs Predicted {total_pred:.0f}\n")
            f.write("-" * 40 + "\n")
            
            # Accumulators (Reset)
            agg_feat_imp = {}
            agg_temp_imp = None
            agg_neighbor_imp = {} 
            agg_neighbor_rel = {}
            
            for t_idx in range(start_last, end_last):
                x_sample = x_test[t_idx] 
                
                # A. Feature
                feat_imp = interpreter.get_feature_importance(target_idx, x_sample=x_sample)
                for k, v in feat_imp.items():
                    agg_feat_imp[k] = agg_feat_imp.get(k, 0.0) + v
                    
                # B. Temporal
                temp_imp = interpreter.get_temporal_importance(target_idx, x_sample=x_sample)
                if agg_temp_imp is None:
                    agg_temp_imp = np.zeros_like(temp_imp)
                agg_temp_imp += temp_imp
                
                # C. Structural
                G = interpreter.get_structural_graph(target_idx, top_k=10, x_sample=x_sample)
                for n in G.nodes():
                    if n == target_idx: continue
                    display_name = n
                    if "_" in str(n):
                        prefix, idx_str = str(n).split("_", 1)
                        if prefix == "sku" and idx_str.isdigit():
                             nid = int(idx_str)
                             display_name = idx_to_name.get(nid, f"SKU_{nid}")
                    
                    if G.has_edge(target_idx, n):
                        w = G[target_idx][n].get('weight', 0.0)
                        rel = G[target_idx][n].get('relation', 'unknown')
                        agg_neighbor_imp[display_name] = agg_neighbor_imp.get(display_name, 0.0) + w
                        agg_neighbor_rel[display_name] = rel

            # Average Results (Log)
            f.write("\n--- Average Drivers (Last 7 Days) ---\n")
            
            # 1. Feature
            sorted_feats = sorted(agg_feat_imp.items(), key=lambda x: x[1], reverse=True)
            f.write("1. Feature Importance:\n")
            for k, total_v in sorted_feats:
                avg_v = total_v / 7
                f.write(f"   - {k:<20}: {avg_v:.4f}\n")
                
            # 2. Temporal
            f.write("\n2. Temporal Importance:\n")
            agg_temp_imp /= 7
            top_lags = np.argsort(agg_temp_imp)[-5:][::-1]
            for lag in top_lags:
                days_ago = 14 - lag
                f.write(f"   - {days_ago} Days Ago (t-{days_ago}): {agg_temp_imp[lag]:.4f}\n")
            
            # 3. Structural
            f.write("\n3. Key Neighbors (Structural):\n")
            sorted_neighbors = sorted(agg_neighbor_imp.items(), key=lambda x: x[1], reverse=True)
            for name, total_w in sorted_neighbors[:5]:
                avg_w = total_w / 7
                rel = agg_neighbor_rel.get(name, '?')
                f.write(f"   - {name} ({avg_w:.2f}) [{rel}]\n")

                
        else:
            f.write("Explainability available for HGT models only.\n")
                


    print(f"Evaluation written to: {results_path}")

    return {
        "wape": wape_corr,
        "bias": bias_corr,
        "mean_sales": actuals_denorm.mean(),
        "epoch": None # Can be filled by caller
    }


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="HGT", choices=["SAGE", "HGT"])
    args = parser.parse_args()
    
    evaluate(model_type=args.model_type)
