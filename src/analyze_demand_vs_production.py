
import os
import sys
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Setup Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

from dataset import SupplyChainDataset
from temporal import TimeSeriesProcessor
from model_hgt import STHGT

# Configuration
SKU_TARGET = "SE500G24P"
MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "st_gnn_model_HGT.pth")
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "dataset", "Raw Dataset")
PRODUCTION_FILE = os.path.join(RAW_DATA_DIR, "Temporal Data", "Unit", "Production .csv")
OUTPUT_CSV = os.path.join(PROJECT_ROOT, "results", "demand_vs_production_SE500G24P.csv")
DEVICE = torch.device("cpu")

def analyze():
    print(f"Analyzing {SKU_TARGET} using Demand Model...")
    
    # 1. Load Data (Demand/Sales Features)
    processor = TimeSeriesProcessor(RAW_DATA_DIR)
    # Note: load_features loads SALES as the primary target for the Demand Model
    features, common_index = processor.load_features(log_transform=True, normalize=False, add_lags=True)
    
    # Get SKU Index
    nodes_df = pd.read_csv(processor.nodes_path)
    if SKU_TARGET not in nodes_df['Node'].values:
        print(f"Error: SKU {SKU_TARGET} not found in nodes.")
        return
    sku_idx = nodes_df[nodes_df['Node'] == SKU_TARGET]['NodeIndex'].values[0]
    print(f"SKU Index: {sku_idx}")

    # 2. Load Graph Data
    dataset = SupplyChainDataset(RAW_DATA_DIR)
    graph_data = dataset.load_graph()
    metadata = graph_data.metadata()
    edge_index_dict = graph_data.edge_index_dict
    
    # 3. Load Model
    model = STHGT(
        node_features=features.shape[-1], 
        hidden_dim=32, 
        out_dim=1, 
        metadata=metadata
    ).to(DEVICE)
    
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}")
        return
        
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    # 4. Inference (Full Range)
    # We want to predict sales for every possible day
    lookback = 14
    X, y_sales = processor.create_sliding_window(features, lookback, 1)
    
    print("Running Inference...")
    with torch.no_grad():
        # Simple batching
        batch_size = 32
        preds_list = []
        for i in range(0, len(X), batch_size):
            batch_x = X[i:i+batch_size].to(DEVICE)
            out = model({"sku": batch_x}, edge_index_dict)
            preds_list.append(out.cpu().numpy())
            
    preds = np.concatenate(preds_list, axis=0) # (Samples, Nodes)
    
    # Extract Data for Target SKU
    # Adjust index for lookback
    dates = common_index[lookback:]
    
    pred_sales_log = preds[:, sku_idx]
    actual_sales_log = y_sales[:, sku_idx].numpy().flatten()
    
    # Inverse Log
    pred_sales = np.expm1(pred_sales_log)
    actual_sales = np.expm1(actual_sales_log)
    
    # 5. Load Production Data (The "Sunset" Truth)
    if os.path.exists(PRODUCTION_FILE):
        prod_df = pd.read_csv(PRODUCTION_FILE)
        prod_df['Date'] = pd.to_datetime(prod_df['Date'])
        # Filter for dates in our inference range
        prod_series = prod_df.set_index('Date')[SKU_TARGET]
        # Align with inference dates
        aligned_production = prod_series.reindex(dates).fillna(0).values
    else:
        print("Production file not found.")
        aligned_production = np.zeros_like(pred_sales)

    # 6. Construct Dataframe
    results_df = pd.DataFrame({
        'Date': dates,
        'Predicted_Demand': pred_sales,
        'Actual_Sales': actual_sales,
        'Actual_Production': aligned_production
    })
    
    # 7. Analysis: Active vs Sunset
    sunset_date = pd.Timestamp('2023-06-01')
    active_mask = results_df['Date'] < sunset_date
    sunset_mask = results_df['Date'] >= sunset_date
    
    active = results_df[active_mask]
    sunset = results_df[sunset_mask]
    
    print("\n" + "="*40)
    print("DEMAND FORECAST ANALYSIS (Aggregated)")
    print("="*40)
    
    print(f"Model: STGNN (HGT) - Demand/Sales Forecaster")
    print(f"Target SKU: {SKU_TARGET}")
    
    print("\n--- Phase 1: Active (Pre-Jun 1st) ---")
    print(f"Avg Actual Sales: {active['Actual_Sales'].mean():.0f}")
    print(f"Avg Pred Sales:   {active['Predicted_Demand'].mean():.0f}")
    print(f"Avg Production:   {active['Actual_Production'].mean():.0f}")
    print(f"Bias (Pred - Sales): {(active['Predicted_Demand'] - active['Actual_Sales']).mean():.2f}")
    
    print("\n--- Phase 2: Sunset (Post-Jun 1st) ---")
    print(f"Avg Actual Sales: {sunset['Actual_Sales'].mean():.0f}")
    print(f"Avg Pred Sales:   {sunset['Predicted_Demand'].mean():.0f}")
    print(f"Avg Production:   {sunset['Actual_Production'].mean():.0f}")
    
    # The Key Metric: Predicted Demand vs Production Capacity
    gap = sunset['Predicted_Demand'].sum() - sunset['Actual_Production'].sum()
    print(f"\n--- The 'Phantom Demand' Gap ---")
    print(f"Total Predicted Demand (Jun-Aug): {sunset['Predicted_Demand'].sum():,.0f}")
    print(f"Total Actual Production (Jun-Aug): {sunset['Actual_Production'].sum():,.0f}")
    print(f"Unfulfilled Demand Gap: {gap:,.0f} units")
    
    # Save Results
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nDetailed results saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    analyze()
