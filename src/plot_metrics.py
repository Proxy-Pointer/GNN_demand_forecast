import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_metrics(model_type, csv_path, output_dir):
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        print(f"Empty data for {model_type}")
        return
        
    # Sort by Epoch just in case
    df = df.sort_values("Epoch")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'{model_type} Training Metrics over Epochs', fontsize=16)

    # 1. Loss
    axes[0].plot(df['Epoch'], df['TrainLoss'], label='Train Loss', marker='o')
    axes[0].plot(df['Epoch'], df['ValLoss'], label='Val Loss', marker='o')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (Log1p Huber)')
    axes[0].legend()
    axes[0].grid(True)

    # 2. WAPE
    axes[1].plot(df['Epoch'], df['WAPE'], label='WAPE', color='green', marker='o')
    axes[1].set_title('WAPE (Target <= 0.6)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('WAPE')
    axes[1].axhline(y=0.6, color='r', linestyle='--', label='Goal (0.6)')
    axes[1].legend()
    axes[1].grid(True)

    # 3. Bias Ratio
    axes[2].plot(df['Epoch'], df['BiasRatio'], label='Bias Ratio', color='purple', marker='o')
    axes[2].set_title('Bias Ratio (Target <= 5%)')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Bias Ratio')
    axes[2].axhline(y=0.05, color='r', linestyle='--', label='Goal (5%)')
    axes[2].legend()
    axes[2].grid(True)
    
    # Format Bias Ratio as percentage on Y axis if possible, but keep simple for now.

    plt.tight_layout()
    output_path = os.path.join(output_dir, f'training_metrics_{model_type}.png')
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")
    plt.close()

def main():
    src_dir     = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.abspath(os.path.join(src_dir, "..", "results"))
    visuals_dir = os.path.abspath(os.path.join(src_dir, "..", "visuals"))
    os.makedirs(visuals_dir, exist_ok=True)

    plot_metrics("HGT", os.path.join(results_dir, "epoch_results_HGT.csv"), visuals_dir)
    plot_metrics("SAGE", os.path.join(results_dir, "epoch_results_SAGE.csv"), visuals_dir)

if __name__ == "__main__":
    main()
