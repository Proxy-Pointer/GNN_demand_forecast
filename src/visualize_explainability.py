
import matplotlib.pyplot as plt
import os
import seaborn as sns

# Data provided by user
data_first_7 = {
    "Feature Importance": {
        "Sales Lag (7d)": 0.8606,
        "Delivery": 0.7178,
        "Factory Issue": 0.7025,
        "Sales": 0.6648,
        "Production": 0.4672,
        "Rolling Mean (7d)": 0.3951
    },
    "Key Neighbors": {
        "SOS005L04P\n[storage]": 0.91,
        "SOS002L09P\n[subgroup]": 0.19,
        "ATN01K24P\n[storage]": 0.14,
        "SOS500M24P\n[subgroup]": 0.10,
        "SOS003L04P\n[subgroup]": 0.07
    }
}

data_last_7 = {
    "Feature Importance": {
        "Rolling Mean (7d)": 0.9043,
        "Delivery": 0.8716,
        "Sales": 0.8309,
        "Sales Lag (7d)": 0.7808,
        "Factory Issue": 0.6221,
        "Production": 0.3490
    },
    "Key Neighbors": {
        "SOS002L09P\n[subgroup]": 0.67,
        "SOS005L04P\n[subgroup]": 0.65,
        "POV001L24P\n[storage]": 0.14,
        "SOS500M24P\n[subgroup]": 0.09,
        "SOS003L04P\n[subgroup]": 0.08
    }
}

def get_base_key(key, key_type):
    """
    Extracts the base key for color matching.
    - Features: Returns key as is.
    - Neighbors: Returns first part of 'SKU\n[relation]'
    """
    if key_type == "Key Neighbors":
        return key.split('\n')[0]
    return key

def get_color_map(all_data, key_type):
    """
    Generates a consistent color map for a given key type (e.g. 'Feature Importance')
    across multiple datasets. Matches based on BASE key (SKU name for neighbors).
    """
    all_keys = set()
    for d in all_data:
        for k in d[key_type].keys():
            base_k = get_base_key(k, key_type)
            all_keys.update([base_k])
    
    unique_keys = sorted(list(all_keys))
    
    # Use consistent bright palette (Set2) for both
    # If more than 8 items, Set2 cycles, so let's use 'hls' for large unique sets if needed
    # But Set2 is what the user liked.
    if len(unique_keys) <= 8:
        palette = sns.color_palette('Set2', n_colors=len(unique_keys))
    else:
        palette = sns.color_palette('husl', n_colors=len(unique_keys)) # Fallback for many neighbors
        
    return dict(zip(unique_keys, palette))


def plot_pie(ax, data, title, key_type, color_map=None):
    labels = list(data.keys())
    values = list(data.values())
    
    # Determine colors
    if color_map:
        colors = []
        for lbl in labels:
            base_k = get_base_key(lbl, key_type)
            colors.append(color_map.get(base_k, 'gray')) # Fallback to gray if not found
    else:
        # Fallback
        colors = sns.color_palette('pastel')[0:len(labels)]
    
    wedges, texts, autotexts = ax.pie(
        values, 
        labels=labels, 
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        textprops=dict(color="black", fontweight='bold', fontsize=13)
    )
    
    ax.set_title(title, fontweight='bold', fontsize=14)
    
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(12)

def create_visualizations():
    OUTPUT_DIR = os.path.join("visuals")
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    # Build global color maps
    feature_color_map = get_color_map([data_first_7, data_last_7], "Feature Importance")
    neighbor_color_map = get_color_map([data_first_7, data_last_7], "Key Neighbors")

    # Create 2x2 Grid
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('HGT Explainability Analysis: First 7 vs Last 7 Days', fontweight='bold', fontsize=16)

    # Row 1: Feature Importance
    plot_pie(axs[0, 0], data_first_7["Feature Importance"], "Feature Importance (First 7 Days)", "Feature Importance", color_map=feature_color_map)
    plot_pie(axs[0, 1], data_last_7["Feature Importance"], "Feature Importance (Last 7 Days)", "Feature Importance", color_map=feature_color_map)

    # Row 2: Key Neighbors
    plot_pie(axs[1, 0], data_first_7["Key Neighbors"], "Key Neighbor Influence (First 7 Days)", "Key Neighbors", color_map=neighbor_color_map)
    plot_pie(axs[1, 1], data_last_7["Key Neighbors"], "Key Neighbor Influence (Last 7 Days)", "Key Neighbors", color_map=neighbor_color_map)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle
    
    save_path = os.path.join(OUTPUT_DIR, "explainability_pie_charts.png")
    plt.savefig(save_path)
    print(f"Explainability charts saved to {save_path}")

if __name__ == "__main__":
    create_visualizations()
