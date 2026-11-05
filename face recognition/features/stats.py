import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os

# --- CONFIGURATION (Matches your labeling.py) ---
WIDERFACE_DIR = Path(__file__).parent.parent / "data" / "widerface"
labels_csv = WIDERFACE_DIR / 'labels.csv'


def analyze_labels():
    if not os.path.exists(labels_csv):
        print(f"Error: {labels_csv} not found.")
        return

    # 1. Load the data
    df = pd.read_csv(labels_csv)

    # Identify attribute columns (everything except image_name)
    attributes = [col for col in df.columns if col != 'image_name']
    total_images = len(df)

    print(f"--- Dataset Analysis ---")
    print(f"Total Images Labeled: {total_images}\n")
    print(f"{'Attribute':<15} | {'Count':<6} | {'Percentage'}")
    print("-" * 40)

    stats = []
    for attr in attributes:
        count = df[attr].sum()
        percentage = (count / total_images) * 100
        print(f"{attr:<15} | {int(count):<6} | {percentage:>6.2f}%")
        stats.append({'Attribute': attr, 'Percentage': percentage})

    # 2. Visualization
    stats_df = pd.DataFrame(stats).sort_values(by='Percentage', ascending=False)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(stats_df['Attribute'], stats_df['Percentage'], color='skyblue', edgecolor='navy')

    # Add labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 1, f'{yval:.1f}%', ha='center', va='bottom',
                 fontweight='bold')

    plt.title(f'Distribution of Features (Total: {total_images} images)')
    plt.ylabel('Percentage of Positive Labels (%)')
    plt.ylim(0, 110)  # Leave room for labels
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Save chart
    output_path = WIDERFACE_DIR / 'feature_distribution.png'
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"\nChart saved to: {output_path}")
    plt.show()


if __name__ == "__main__":
    analyze_labels()