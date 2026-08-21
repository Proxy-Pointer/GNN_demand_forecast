# GNN Demand Forecasting

A **Spatio-Temporal Graph Neural Network** for supply chain demand forecasting.
The model predicts daily sales for each SKU by combining graph-based spatial aggregation (relationships between SKUs via shared plants, storage locations, and product groups) with LSTM-based temporal modelling.

Two model architectures are implemented and compared:

| Model            | Architecture                                 | Graph Layer   |
| ---------------- | -------------------------------------------- | ------------- |
| **STSAGE** | GraphSAGE + LSTM                             | Homogeneous   |
| **STHGT**  | Heterogeneous Graph Transformer (HGT) + LSTM | Heterogeneous |

---

## Architecture

Read more about the architectural concepts behind this implementation:
- [Time Series Isn’t Enough: How Graph Neural Networks Change Demand Forecasting](https://towardsdatascience.com/time-series-isnt-enough-how-graph-neural-networks-change-demand-forecasting/)
- [From Connections to Meaning: Why Heterogeneous Graph Transformers (HGT) Change Demand Forecasting](https://towardsdatascience.com/from-connections-to-meaning-why-heterogeneous-graph-transformers-hgt-change-demand-forecasting/)

---
## Project Structure

```
GNN_demand/
├── dataset/
│   └── Raw Dataset/
│       ├── Nodes/
│       │   ├── Nodes.csv          # SKU master list
│       │   └── NodesIndex.csv     # SKU → integer index mapping
│       ├── Edges/
│       │   └── EdgesIndex/        # SKU–SKU edge files by relation type
│       └── Temporal Data/
│           └── Unit/              # Daily time-series CSVs per feature
│               ├── Sales Order.csv
│               ├── Delivery To distributor.csv
│               ├── Factory Issue.csv
│               └── Production .csv
├── model/                         # Saved model checkpoints (.pth)
├── results/                       # Evaluation reports and epoch CSV logs
├── visuals/                       # All generated plots and charts
└── src/
    ├── model_sage.py              # STSAGE model definition
    ├── model_hgt.py               # STHGT model definition
    ├── dataset.py                 # Graph loader (HeteroData)
    ├── temporal.py                # Feature engineering + sliding window
    ├── train.py                   # Training pipeline
    ├── evaluate_gnn.py            # Evaluation metrics + explainability
    ├── explainability.py          # Gradient-based GNN interpreter
    ├── analyze_demand_vs_production.py  # Demand vs production gap analysis
    ├── visualize_results.py       # Forecast dashboards and scatter plots
    ├── visualize_graph.py         # Graph topology visualisation
    ├── visualize_explainability.py # Explainability pie charts
    └── plot_metrics.py            # Training metrics plots
```

---

## Setup

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install torch torch-geometric pandas numpy matplotlib seaborn networkx
```

> **PyTorch Geometric** installation varies by platform and CUDA version.
> See the official guide: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

---

## Usage

### 1. Train

```bash
# Train HGT (default)
python src/train.py --model_type HGT

# Train SAGE
python src/train.py --model_type SAGE

# Resume training from an existing checkpoint
python src/train.py --model_type HGT --continue --seed 42
```

Checkpoints are saved to `model/st_gnn_model_{HGT|SAGE}.pth`.
Epoch metrics (loss, WAPE, bias) are logged to `results/epoch_results_{model_type}.csv`.

### 2. Evaluate

```bash
python src/evaluate_gnn.py --model_type HGT
```

Writes a full evaluation report (WAPE, Bias, Explainability) to `results/eval_report_{model_type}.txt`.

### 3. Visualize Forecasts

```bash
# Both models + comparison chart
python src/visualize_results.py --model_type ALL

# Single model
python src/visualize_results.py --model_type HGT

# Head-to-head scatter comparison only
python src/visualize_results.py --model_type COMPARE
```

### 4. Other Visualisations

```bash
# Graph topology (12 most-connected SKUs)
python src/visualize_graph.py

# Training metrics (loss / WAPE / bias ratio over epochs)
python src/plot_metrics.py

# Explainability pie charts (feature & neighbor importance)
python src/visualize_explainability.py

# Demand vs production gap analysis for a specific SKU
python src/analyze_demand_vs_production.py
```

---

## Script Outputs

Quick reference: what each script writes to `results/` or `visuals/`.

### `src/train.py`

| Output file | Location |
|-------------|----------|
| `model/st_gnn_model_{HGT\|SAGE}.pth` | Trained model checkpoint |
| `results/epoch_results_{model_type}.csv` | Per-epoch: Train Loss, Val Loss, WAPE, Bias, BiasRatio |
| `visuals/training_curve_{model_type}.png` | Loss curve (train vs val) |
| `results/eval_report_{model_type}.txt` | Full eval report (called automatically at end of training) |

### `src/evaluate_gnn.py`

| Output file | Location |
|-------------|----------|
| `results/eval_report_HGT.txt` | WAPE, Bias + full explainability analysis (HGT) |
| `results/eval_report_SAGE.txt` | WAPE, Bias only (no explainability for SAGE) |

### `src/visualize_results.py`

| Output file | Notes |
|-------------|-------|
| `visuals/forecast_dashboard_{model_type}.png` | Top-4 SKUs: Actual vs Predicted |
| `visuals/actual_vs_pred_scatter_{model_type}.png` | Global scatter (linear scale) |
| `visuals/actual_vs_pred_log_scatter_{model_type}.png` | Log-log scatter |
| `visuals/sku_performance_rank_{model_type}.png` | Per-SKU WAPE bar chart |
| `visuals/error_distribution_{model_type}.png` | Residual histogram (middle 90%) |
| `visuals/actual_vs_pred_log_scatter_comparison.png` | SAGE vs HGT overlaid (`--model_type COMPARE`) |

### `src/plot_metrics.py`

| Output file | Notes |
|-------------|-------|
| `visuals/training_metrics_HGT.png` | 3-panel: Loss / WAPE / Bias Ratio over epochs |
| `visuals/training_metrics_SAGE.png` | Same for SAGE |

> Reads from `results/epoch_results_{model_type}.csv` — run `train.py` first.

### `src/visualize_graph.py`

| Output file | Notes |
|-------------|-------|
| `visuals/graph_structure_sample.png` | Subgraph of the 12 most-connected SKUs |

### `src/visualize_explainability.py`

| Output file | Notes |
|-------------|-------|
| `visuals/explainability_pie_charts.png` | Feature & neighbour importance: first 7 vs last 7 test days |

> Uses hardcoded data extracted from a prior `eval_report_HGT.txt` run. Re-edit the data dicts in the script to refresh.

### `src/analyze_demand_vs_production.py`

| Output file | Notes |
|-------------|-------|
| `results/demand_vs_production_{SKU_TARGET}.csv` | Date / Predicted Demand / Actual Sales / Actual Production |

> Target SKU is set via the `SKU_TARGET` constant at the top of the script (default: `SE500G24P`).

---


## Dataset Format

The model expects the following directory layout under `dataset/Raw Dataset/`:

**Nodes** — `Nodes/NodesIndex.csv` with columns `Node` (SKU name) and `NodeIndex` (0-based integer).

**Edges** — one CSV per relationship type in `Edges/EdgesIndex/`, each with `node1` and `node2` integer columns:

- `Edges (Plant).csv`
- `Edges (Product Group).csv`
- `Edges (Product Sub-Group).csv`
- `Edges (Storage Location).csv`

**Temporal data** — one CSV per feature in `Temporal Data/Unit/`, rows = dates, columns = SKU names:

- `Sales Order.csv`
- `Delivery To distributor.csv`
- `Factory Issue.csv`
- `Production .csv`

---

## Model Details

### Features (per SKU, per day)

| Feature           | Description                                |
| ----------------- | ------------------------------------------ |
| Sales             | Units sold (log1p transformed)             |
| Delivery          | Units delivered to distributor             |
| Factory Issue     | Factory quality / supply disruption signal |
| Production        | Units produced                             |
| Rolling Mean (7d) | 7-day rolling average of sales (causal)    |
| Sales Lag (7d)    | Sales value 7 days prior                   |

### Training

- **Loss**: Weighted Huber loss in log1p space + optional bias regularisation (HGT only)
- **Optimiser**: Adam with StepLR decay
- **Early stopping**: triggered when WAPE ≤ 0.57 **and** bias ratio ≤ 5%
- **Lookback window**: 14 days → predict 1 day ahead
- **Train / test split**: 80 / 20 (chronological)

### Results (HGT, Epoch 100)

| Metric | ST-GNN (HGT)   | Naïve Baseline |
| ------ | -------------- | --------------- |
| WAPE   | 0.5811         | 0.8593          |
| Bias   | +8.4 units/day | 0.0             |

---

## Explainability

`evaluate_gnn.py` runs gradient-based attribution (via `GNNInterpreter`) on the top-volume SKU for the first and last 7 days of the test set, reporting:

- **Feature importance** — which of the 6 input features drives predictions most
- **Temporal importance** — which lookback days matter most
- **Structural importance** — which graph neighbours (SKUs / plants / product groups) influence the forecast

---

## License

MIT
