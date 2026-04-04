"""
This script loads and visualizes evaluation results from JSON files in a results directory.

It compares the performance of different RAG (Retrieval-Augmented Generation) systems:
- Agentic RAG: Uses an agentic approach for question answering.
- Standard RAG: Uses a standard retrieval-augmented approach.
- Vanilla LLM: The model's baseline performance without retrieval.

The results are loaded from JSON files, and a bar chart is generated showing the 
mean evaluation scores (percentage) as judged by an LLM judge.
"""
from pathlib import Path
from utils.results_manager import load_evaluation_results
import argparse
import matplotlib.pyplot as plt
import pandas as pd


def read_results(results_dir: str):
    """
    Read evaluation results from all JSON files in the specified directory.

    Args:
        results_dir: Directory to load the results from.
    """
    all_models_scores = []

    # Use glob to find all files ending in .json in the results_dir
    results_path = Path(results_dir)
    json_files = list(results_path.glob("*.json"))

    if not json_files:
        print(f"No JSON files found in {results_dir}")
        return all_models_scores

    for filepath in json_files:
        # load_evaluation_results handles the specific filename
        results = load_evaluation_results(results_dir, filepath.name)

        # We only want to process valid files with 'model_name'
        if "model_name" not in results:
            continue

        scores = {"model_name": results["model_name"]}

        for system_type in [
                "agentic_rag",
                "standard_rag",
                "standard",
        ]:
            if system_type in results:
                scores[system_type] = results[system_type][
                    'eval_score_LLM_judge_int'].mean() * 100

        print(f"Processed: {scores['model_name']}\n")
        all_models_scores.append(scores)

    return all_models_scores


def plot_scores(results_dir, scores_list):
    """
    Plot the evaluation scores using a bar chart.
    """
    if not scores_list:
        print("No scores to plot.")
        return

    df = pd.DataFrame(scores_list)

    # Append superscript star(s) for duplicate model names
    seen = {}
    new_names = []
    for name in df["model_name"]:
        seen[name] = seen.get(name, 0) + 1
        if seen[name] > 1:
            new_names.append(name + "*" * (seen[name] - 1))
        else:
            new_names.append(name)
    df["model_name"] = new_names

    df.set_index("model_name", inplace=True)
    df.rename(columns={
        "agentic_rag": "Agentic RAG",
        "standard_rag": "Standard RAG",
        "standard": "Vanilla LLM"
    },
              inplace=True)

    # Set background to black
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(max(8, len(df) * 2.2), 6))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    # A soft sage green (#8faa82), or warm gold (#d9b66c), or
    # Muted Purple/Mauve: '#9c849c' works well
    # harmoniously with icy blue (#c7dfff) and terracotta (#c56b46)
    colors = ['#c7dfff', '#c56b46', '#8faa82']

    # Plot bars
    ax = df.plot(kind="bar",
                 ax=ax,
                 rot=45,
                 color=colors,
                 edgecolor='white',
                 linewidth=1.5,
                 width=0.8)

    # Add small spaces between bars in the same group
    for container in ax.containers:
        for patch in container:
            current_width = patch.get_width()
            gap = current_width * 0.1  # 5% gap
            patch.set_width(current_width - gap)
            patch.set_x(patch.get_x() + gap / 2)

    # Formatting axes and labels
    ax.set_title("Performance of answering technical questions",
                 fontsize=14,
                 color='white',
                 pad=20)
    ax.set_xlabel("Model Name", fontsize=12, color='white')

    # Y-axis formatting to show Percentages
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'],
                       color='white',
                       fontsize=12)
    ax.tick_params(axis='x', colors='white', labelsize=12)

    # Hide the y-axis label and ticks to match style more closely if desired
    # ax.set_ylabel("")
    ax.tick_params(axis='y', left=False)  # remove tick marks

    # Hide spines (borders)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Add bar labels on top
    for container in ax.containers:
        labels = [
            f'{v.get_height():.1f}' if v.get_height() > 0 else ''
            for v in container
        ]
        ax.bar_label(container,
                     labels=labels,
                     padding=5,
                     color='white',
                     fontsize=10,
                     fontweight='bold',
                     rotation=45)

    legend = plt.legend(bbox_to_anchor=(0.7 + len(df) * 0.05, 1.05),
                        loc='upper left',
                        frameon=False,
                        fontsize=12)
    plt.setp(legend.get_texts(), color='white')
    if legend.get_title():
        plt.setp(legend.get_title(), color='white')

    plt.tight_layout()
    output_path = Path(results_dir) / "evaluation_scores.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to {output_path}")
    plt.show()


def main():
    """
    Main function to load and show evaluation results.
    """
    parser = argparse.ArgumentParser(description="Show evaluation results.")
    parser.add_argument("--results_dir", type=str, default="results")

    args = parser.parse_args()

    # Now we just pass the directory to read all files rather than a single file name
    all_scores = read_results(args.results_dir)
    plot_scores(args.results_dir, all_scores)


if __name__ == "__main__":
    main()
