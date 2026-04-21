import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    raw_file = Path('/home/oussama/Desktop/reranking_project/data/bioasq/raw/training13b.json')
    
    if not raw_file.exists():
        print(f"Error: {raw_file} not found.")
        return

    print("Loading data...")
    with open(raw_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    questions = data.get('questions', [])
    snippet_counts = [len(q.get('snippets', [])) for q in questions]

    # Calculate queries with > X snippets
    thresholds = [3,5, 10, 15, 20]
    counts_above_threshold = []
    
    for t in thresholds:
        count = sum(1 for c in snippet_counts if c > t)
        counts_above_threshold.append(count)
        print(f"Queries with > {t} snippets: {count}")

    # Plotting
    sns.set_theme(style="whitegrid")
    plt.style.use('ggplot')
    
    plt.figure(figsize=(8, 6))
    
    # Create bar plot
    ax = sns.barplot(x=[f"> {t}" for t in thresholds], y=counts_above_threshold, palette="magma")
    
    # Add values on top of bars
    for i, p in enumerate(ax.patches):
        ax.annotate(f"{counts_above_threshold[i]}", 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom', fontsize=12, xytext=(0, 5), 
                    textcoords='offset points')

    plt.title('Number of Queries with More Than X Snippets', fontsize=14, pad=15)
    plt.xlabel('Snippet Count Threshold', fontsize=12)
    plt.ylabel('Number of Queries', fontsize=12)
    
    out_dir = Path('/home/oussama/Desktop/reranking_project/data/bioasq/processed/images')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    plot_path = out_dir / 'snippets_cumulative.png'
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    print(f"Cumulative plot saved to: {plot_path}")

if __name__ == '__main__':
    main()
