import torch
import torch.nn as nn
import torch.optim as optim
from dataset import SupplyChainDataset
from temporal import TimeSeriesProcessor
from model_sage import STSAGE
from model_hgt import STHGT
from evaluate_gnn import evaluate
import matplotlib.pyplot as plt
import os



def set_seed(seed=42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train(model_type="HGT", continue_train=False, seed=42):
    set_seed(seed)  # Enforce reproducibility
    print(f"Random seed set to: {seed}")


    # ------------------------------------------------------------------
    # 1. Setup Data
    # ------------------------------------------------------------------
    src_dir  = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(src_dir, "..", "dataset", "Raw Dataset"))

    print("Loading Graph...")
    graph_dataset = SupplyChainDataset(root_dir)
    graph_data = graph_dataset.load_graph()

    edge_indices = []
    edge_index_dict = {}

    for edge_type in graph_data.edge_types:
        ei = graph_data[edge_type].edge_index
        edge_indices.append(ei)
        edge_index_dict[edge_type] = ei

    full_edge_index = torch.cat(edge_indices, dim=1)

    # ------------------------------------------------------------------
    # 2. Temporal Features
    # ------------------------------------------------------------------
    print("Loading Temporal Features (Log1p)...")
    temp_proc = TimeSeriesProcessor(root_dir)

    features, _ = temp_proc.load_features(
        log_transform=True,
        normalize=False,
        add_lags=True
    )

    # ------------------------------------------------------------------
    # 3. Sliding Window
    # ------------------------------------------------------------------
    lookback = 14
    horizon = 1

    x_all, y_all = temp_proc.create_sliding_window(
        features, lookback, horizon
    )

    num_samples = x_all.shape[0]
    test_size = int(0.2 * num_samples)
    train_size = num_samples - test_size

    x_train, y_train = x_all[:train_size], y_all[:train_size]
    x_test, y_test = x_all[train_size:], y_all[train_size:]

    print(f"Train samples: {train_size}, Test samples: {test_size}")

    # ------------------------------------------------------------------
    # 4. SKU Weights (CORRECTLY ALIGNED)
    # ------------------------------------------------------------------
    start_day = lookback
    end_day = lookback + train_size

    raw_sales = torch.expm1(features[:, start_day:end_day, 0])
    sku_weights = raw_sales.mean(dim=1)
    sku_weights = sku_weights / sku_weights.mean()
    sku_weights = sku_weights.clamp(min=0.1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sku_weights = sku_weights.to(device)

    print(
        f"SKU Weights | Min: {sku_weights.min():.3f}, "
        f"Max: {sku_weights.max():.3f}"
    )

    # ------------------------------------------------------------------
    # 5. Model
    # ------------------------------------------------------------------
    print(f"Model Type: {model_type}")

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

    model_path = os.path.abspath(os.path.join(src_dir, "..", "model", f"st_gnn_model_{model_type}.pth"))
    
    from torch.optim.lr_scheduler import StepLR

    # Init Optimizer *before* loading so we can load state
    # SAGE prefers lower LR (0.001) and constant schedule
    # HGT prefers higher LR (0.002) and decay to settle
    initial_lr = 0.001 if model_type == "SAGE" else 0.003 #0.003
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    
    # Init Scheduler
    if model_type == "SAGE":
        # Effectively constant LR (decay only after 1000 epochs)
        scheduler = StepLR(optimizer, step_size=1000, gamma=1.0)
    else:
        # HGT: Reduce LR by half every 20 epochs
        scheduler = StepLR(optimizer, step_size=30, gamma=0.5)
    
    start_epoch = 0

    if continue_train and os.path.exists(model_path):
        print(f"Loading existing model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=device)
        
        # Check for new checkpoint format (dict) vs legacy (state_dict only)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load Scheduler State (if handling resumes)
            if 'scheduler_state_dict' in checkpoint:
                 scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            start_epoch = checkpoint.get('epoch', 0)
            print(f"  - Restored model weights, optimizer, and scheduler. Resuming from Epoch {start_epoch}")
            
            # FORCE LR UPDATE REMOVED: Allow resumption with decayed LR
            # for param_group in optimizer.param_groups:
            #     param_group['lr'] = initial_lr
            # print(f"  - Forced Optimizer LR to {initial_lr}")

        else:
            model.load_state_dict(checkpoint)
            print("  - Restored model weights only (Legacy format). Optimizer reset.")
            start_epoch = 0

    # Log-space regression loss
    huber = nn.SmoothL1Loss(beta=0.1, reduction="none")

    # Bias regularization strength (SAFE)
    # SAGE does not need this (it degrades WAPE). HGT needs it.
    if model_type == "SAGE":
        BIAS_LAMBDA = 0.0
        batch_size = 16
    else:
        BIAS_LAMBDA = 0.004 #0.004
        batch_size = 8

    # ------------------------------------------------------------------
    # 6. Training Loop
    # ------------------------------------------------------------------
    if continue_train:
        added_epochs = 20
    else:
        added_epochs = 80 if model_type == "HGT" else 90

    end_epoch = start_epoch + added_epochs

    # ------------------------------------------------------------------
    # Initialize Results File (Before Loop)
    # ------------------------------------------------------------------
    results_dir = os.path.abspath(os.path.join(src_dir, "..", "results"))
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"epoch_results_{model_type}.csv")
    
    # If starting fresh (and not continuing), clear file and write header
    # If continuing, keep file and append
    if not continue_train:
         with open(results_file, "w") as f:
             f.write("Model,Epoch,TrainLoss,ValLoss,WAPE,Bias,BiasRatio\n")
    elif not os.path.exists(results_file):
         # Safety: if continuing but file missing, write header
         with open(results_file, "w") as f:
             f.write("Model,Epoch,TrainLoss,ValLoss,WAPE,Bias,BiasRatio\n")

    train_losses, val_losses = [], []
    
    early_stopped = False

    # Run from logic start -> end
    for epoch in range(start_epoch, end_epoch):
        model.train()
        perm = torch.randperm(train_size)
        total_loss = 0.0

        for i in range(0, train_size, batch_size):
            idx = perm[i:i + batch_size]

            x_batch = x_train[idx].to(device)
            y_batch = y_train[idx].to(device).squeeze(-1)

            optimizer.zero_grad()

            if model_type == "SAGE":
                preds = model(x_batch, edge_input)
            else:
                preds = model({"sku": x_batch}, edge_input)

            # ----------------------------
            # 1. Log-space loss
            # ----------------------------
            log_loss_raw = huber(preds, y_batch)
            log_loss = (log_loss_raw * sku_weights).mean()

            # ----------------------------
            # 2. SAFE bias regularization
            # ----------------------------
            preds_real = torch.expm1(preds)
            y_real = torch.expm1(y_batch)

            scale = y_real.mean(dim=1, keepdim=True).clamp(min=1.0)
            norm_bias = (preds_real - y_real) / scale

            bias_loss = torch.mean(torch.abs(norm_bias))

            # ----------------------------
            # Final loss
            # ----------------------------
            loss = log_loss  + BIAS_LAMBDA * bias_loss #* epoch/30
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(idx)
        
        # Scheduler Step
        scheduler.step()

        avg_train = total_loss / train_size
        train_losses.append(avg_train)

        # ------------------ Validation ------------------
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for i in range(0, test_size, batch_size):
                x_batch = x_test[i:i + batch_size].to(device)
                y_batch = y_test[i:i + batch_size].to(device).squeeze(-1)

                if model_type == "SAGE":
                    preds = model(x_batch, edge_input)
                else:
                    preds = model({"sku": x_batch}, edge_input)

                log_loss_raw = huber(preds, y_batch)
                loss = (log_loss_raw * sku_weights).mean()
                val_loss += loss.item() * x_batch.size(0)

        avg_val = val_loss / test_size
        val_losses.append(avg_val)

        # --------------------------------------------------------
        # Early Stopping & Evaluation (Every 10 Epochs)
        # --------------------------------------------------------
        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch+1:03d}/{end_epoch} | "
                f"Train: {avg_train:.6f} | "
                f"Val: {avg_val:.6f}"
            )
            
            # Run Full Evaluation
            metrics = evaluate(model_type, model=model)
            
            wape = metrics['wape']
            bias = metrics['bias']
            mean_sales = metrics['mean_sales']
            bias_ratio = abs(bias) / mean_sales if mean_sales > 0 else 1.0
            
            print(f"  >> Eval: WAPE={wape:.4f}, Bias={bias:.4f} (Ratio: {bias_ratio:.2%})")

            # Save Epoch Results
            # (File intialized before loop, just append now)
            with open(results_file, "a") as f:
                f.write(f"{model_type},{epoch+1},{avg_train:.6f},{avg_val:.6f},{wape:.4f},{bias:.4f},{bias_ratio:.4f}\n")

            # Check Stop Condition
            # Condition: Bias <= 5% of mean AND WAPE <= 0.6
            if bias_ratio <= 0.05 and wape <= 0.57:
                print("\n" + "="*60)
                print(f"EARLY STOPPING TRIGGERED AT EPOCH {epoch+1}")
                print(f"Condition Met: WAPE {wape:.4f} <= 0.57 AND Bias Ratio {bias_ratio:.2%} <= 5%")
                print("="*60 + "\n")
                
                # Force Save
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': avg_val,
                }
                torch.save(checkpoint, model_path)
                print(f"Model saved to {model_path}")
                early_stopped = True
                break

    # ------------------------------------------------------------------
    # 7. Save + Plot
    # ------------------------------------------------------------------
    os.makedirs(os.path.abspath(os.path.join(src_dir, "..", "model")), exist_ok=True)

    # Save Robust Checkpoint
    # Save Robust Checkpoint
    checkpoint = {
        'epoch': end_epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': avg_val,
    }
    torch.save(checkpoint, model_path)
    print(f"Model checkpoint saved (Epoch {end_epoch}) to {model_path}")

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.legend()
    plt.title(f"Training Curve ({model_type}, log1p)")
    plt.tight_layout()
    plt.savefig(f"training_curve_{model_type}.png")

    evaluate(model_type, model=model)

    return early_stopped


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default="HGT", choices=["SAGE", "HGT"])
    parser.add_argument("--continue", dest="continue_train", action="store_true")
    # ... (rest of parser)
    parser.add_argument("--seed", type=int, default=81, help="Random seed for initialization")
    args = parser.parse_args()

    train(args.model_type, args.continue_train, args.seed)
