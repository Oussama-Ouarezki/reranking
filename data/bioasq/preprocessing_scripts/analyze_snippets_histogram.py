import json
from collections import Counter
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
    print(f"Total questions loaded: {len(questions)}")
    
    snippet_counts = []
    
    for q in questions:
        snippets = q.get('snippets', [])
        snippet_counts.append(len(snippets))
        
    print(f"\nTotal snippets across all questions: {sum(snippet_counts)}")
    print(f"Average snippets per question: {sum(snippet_counts) / len(questions):.2f}\n")
    
    # Plotting
    plt.style.use('ggplot')
    plt.figure(figsize=(10, 6))
    
    # Setting seaborn options for a nice look along with ggplot style
    sns.set_theme(style="whitegrid")
    plt.style.use('ggplot') # apply ggplot on top of seaborn defaults
    
    sns.histplot(snippet_counts, bins=list(range(0, max(snippet_counts) + 2)), discrete=True, color='steelblue', alpha=0.8)
    plt.title('Distribution of Snippets per Question in BioASQ', fontsize=14, pad=15)
    plt.xlabel('Number of Snippets', fontsize=12)
    plt.ylabel('Frequency (Number of Questions)', fontsize=12)
    
    # Set x-ticks to be integers
    plt.xticks(range(0, max(snippet_counts) + 1, max(1, max(snippet_counts) // 20)))
    
    # Save plot
    out_dir = Path('/home/oussama/Desktop/reranking_project/data/bioasq/processed')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    plot_path = out_dir / 'snippets_histogram.png'
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    print(f"Histogram saved as image to: {plot_path}")

if __name__ == '__main__':
    main()
