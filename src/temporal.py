import torch
import pandas as pd
import numpy as np
import os

class TimeSeriesProcessor:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.temporal_dir = os.path.join(root_dir, "Temporal Data", "Unit")
        self.nodes_path = os.path.join(root_dir, "Nodes", "NodesIndex.csv")

        self.min_vals = None
        self.max_vals = None

        # Explicit feature order
        self.feature_names = [
            "sales",
            "delivery",
            "factory_issue",
            "production",
            "rolling_mean_7",
            "sales_lag7"
        ]

    def load_features(self, log_transform=True, normalize=False, add_lags=True):
        # -----------------------------
        # Load node index mapping
        # -----------------------------
        nodes_df = pd.read_csv(self.nodes_path)
        node_map = dict(zip(nodes_df["Node"], nodes_df["NodeIndex"]))
        all_indices = sorted(node_map.values())

        feature_files = [
            "Sales Order.csv",
            "Delivery To distributor.csv",
            "Factory Issue.csv",
            "Production .csv"
        ]

        dfs = []

        for fname in feature_files:
            fpath = os.path.join(self.temporal_dir, fname)
            if not os.path.exists(fpath):
                raise FileNotFoundError(f"{fname} not found")

            df = pd.read_csv(fpath)
            if "Date" not in df.columns:
                raise ValueError(f"'Date' column missing in {fname}")

            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date").set_index("Date")

            # Keep only known SKUs
            valid_cols = [c for c in df.columns if c in node_map]
            df = df[valid_cols]

            # Rename to node indices and enforce column order
            df = df.rename(columns=node_map)
            # Ensure all 41 indices are present
            df = df.reindex(columns=all_indices, fill_value=0)

            # Enforce daily frequency
            df = df.asfreq("D", fill_value=0)

            dfs.append(df.astype(np.float32))

        # -----------------------------
        # Align dates across features
        # -----------------------------
        common_index = dfs[0].index
        for d in dfs[1:]:
            common_index = common_index.intersection(d.index)

        dfs = [d.loc[common_index] for d in dfs]

        # -----------------------------
        # Stack features
        # Shape: (Days, Nodes, Features)
        # -----------------------------
        feature_arrays = [d.values for d in dfs]
        combined = np.stack(feature_arrays, axis=-1)

        # -----------------------------
        # Log transform
        # -----------------------------
        if log_transform:
            combined = np.log1p(combined)

        # -----------------------------
        # Add causal lag features (SALES ONLY)
        # -----------------------------
        if add_lags:
            days, nodes, feats = combined.shape
            if days <= 7:
                raise ValueError("Not enough history to add lag features")

            sales = combined[:, :, 0]  # (Days, Nodes)

            # Helper for rolling mean along Day axis
            def fast_rolling_mean(arr, window):
                res = np.empty_like(arr)
                for n in range(arr.shape[1]):
                    res[:, n] = pd.Series(arr[:, n]).rolling(window).mean().values
                return res

            rm7 = fast_rolling_mean(sales, 7) # (Days, Nodes)
            
            valid_days = days - 7
            rm7_aligned = rm7[6:6 + valid_days, :] # RM7 ending at t-1 for Day 7
            lag7_aligned = sales[:valid_days, :]    # t-7

            combined_trimmed = combined[7:7 + valid_days, :, :]

            rm7_aligned = rm7_aligned[:, :, None]
            lag7_aligned = lag7_aligned[:, :, None]

            # New shape: (Days-7, Nodes, Feats+2)
            combined = np.concatenate(
                [combined_trimmed, rm7_aligned, lag7_aligned],
                axis=-1
            )

            common_index = common_index[7:]

        # -----------------------------
        # Optional normalization
        # -----------------------------
        days, nodes, feats = combined.shape

        if normalize:
            reshaped = combined.reshape(-1, feats)
            self.min_vals = reshaped.min(axis=0)
            self.max_vals = reshaped.max(axis=0)

            ranges = self.max_vals - self.min_vals
            ranges[ranges == 0] = 1.0

            combined = ((reshaped - self.min_vals) / ranges).reshape(days, nodes, feats)

        # -----------------------------
        # Final tensor
        # Shape: (Nodes, Days, Features)
        # -----------------------------
        final_tensor = torch.tensor(
            combined,
            dtype=torch.float32
        ).permute(1, 0, 2)

        return final_tensor, common_index

    def create_sliding_window(self, tensor, lookback_window=14, horizon=1):
        num_nodes, num_days, num_feats = tensor.shape

        samples = []
        targets = []

        target_idx = 0  # sales

        for t in range(lookback_window, num_days - horizon + 1):
            x = tensor[:, t - lookback_window:t, :]
            y = tensor[:, t:t + horizon, target_idx]
            samples.append(x)
            targets.append(y)

        x = torch.stack(samples)
        y = torch.stack(targets)

        return x, y
