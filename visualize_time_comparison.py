"""
Compare running / inference time between remote API calls and local vLLM
inference serving for selected models.

Generates a grouped bar chart (same colour palette as
visualize_rag_performance.py) with three model groups, each containing
six bars: Agentic-RAG (API / vLLM), Standard-RAG (API / vLLM), and
Vanilla LLM (API / vLLM).

Usage:
    python visualize_time_comparison.py MiniMax-M2.7 Qwen3.6-35B Qwen3.5-122B 
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------
VLLM_DIR = Path("results")
API_DIR = Path("results/api_call")

# ---------------------------------------------------------------------------
# Colour palette — reuses the palette from visualize_rag_performance.py
#   icy blue  (#c7dfff)  — Agentic RAG
#   terracotta(#c56b46)  — Standard RAG
#   sage green(#8faa82)  — Vanilla LLM
# For each task we use the *full* colour for API and a deeper tint
# for vLLM so pairs are visually associated.
# ---------------------------------------------------------------------------
COLORS_API = ['#c7dfff', '#c56b46', '#8faa82']  # API  (saturated)
COLORS_VLLM = ['#6fa8e6', '#8b3e24', '#5c7a52']  # vLLM (deeper tint)


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------
def _find_json(directory: Path, model_prefix: str) -> Path:
    """Find the first JSON file in *directory* whose name starts with
    *model_prefix* (case-sensitive).  Raises FileNotFoundError if no
    match is found.
    """
    candidates = sorted(directory.glob(f"{model_prefix}*.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No JSON file matching '{model_prefix}*' in {directory}")
    if len(candidates) > 1:
        print(f"  ⚠ Multiple matches for '{model_prefix}' in {directory}, "
              f"using {candidates[0].name}")
    return candidates[0]


def _load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Timing extraction
# ---------------------------------------------------------------------------
def extract_times(model_name: str):
    """Return (api_times, vllm_times) each as [agentic, standard, vanilla].

    Files are auto-discovered by globbing ``<model_name>*.json`` inside
    the vLLM and API result directories.
    """
    vllm_path = _find_json(VLLM_DIR, model_name)
    api_path = _find_json(API_DIR, model_name)
    print(f"  vLLM : {vllm_path.name}")
    print(f"  API  : {api_path.name}")

    # --- API ---
    api_data = _load(api_path)
    api_t = api_data["timing"]
    api_times = [
        api_t["agentic_rag_seconds"],
        api_t["standard_rag_seconds"],
        api_t["vanilla_seconds"],
    ]

    # --- vLLM ---
    vllm_data = _load(vllm_path)
    vllm_t = vllm_data["timing"]
    agentic_vllm = vllm_t.get("phase2", {}).get("agentic_batch_seconds", 0)
    standard_vllm = vllm_t.get("phase1", {}).get("rag_batch_seconds", 0)
    vanilla_vllm = vllm_t.get("phase1", {}).get("vanilla_batch_seconds", 0)
    vllm_times = [agentic_vllm, standard_vllm, vanilla_vllm]

    return api_times, vllm_times


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_time_comparison(models: list[str]):
    task_labels = ["Agentic RAG", "Standard RAG", "Vanilla LLM"]

    # Collect data -------------------------------------------------------
    api_data = []  # list of [agentic, standard, vanilla] per model
    vllm_data = []
    for m in models:
        print(f"\n→ {m}")
        a, v = extract_times(m)
        api_data.append(a)
        vllm_data.append(v)

    # Convert seconds → minutes for readability
    api_data = np.array(api_data) / 60.0
    vllm_data = np.array(vllm_data) / 60.0

    # Layout: N model groups, each with 6 bars (3 tasks × 2 sources)
    n_models = len(models)
    n_tasks = len(task_labels)
    n_bars_per_group = n_tasks * 2  # API + vLLM per task
    bar_width = 0.11
    group_width = n_bars_per_group * bar_width + bar_width  # add spacing

    # Style ---------------------------------------------------------------
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(max(8, n_models * 2.2), 6))
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    x_centres = np.arange(n_models) * group_width * 1.05  # spread groups

    # Collect legend handles from the first model group
    api_handles = []  # one per task
    vllm_handles = []  # one per task

    for i_model in range(n_models):
        cx = x_centres[i_model]
        offset = -(n_bars_per_group - 1) / 2 * bar_width
        for i_task in range(n_tasks):
            # API bar
            pos_api = cx + offset + (i_task * 2) * bar_width
            bar_api = ax.bar(
                pos_api,
                api_data[i_model, i_task],
                width=bar_width * 0.88,
                color=COLORS_API[i_task],
                edgecolor="white",
                linewidth=0.8,
            )
            # vLLM bar
            pos_vllm = pos_api + bar_width
            bar_vllm = ax.bar(
                pos_vllm,
                vllm_data[i_model, i_task],
                width=bar_width * 0.88,
                color=COLORS_VLLM[i_task],
                edgecolor="white",
                linewidth=0.8,
            )
            # Keep the first model's handles for the legend
            if i_model == 0:
                api_handles.append(bar_api)
                vllm_handles.append(bar_vllm)

            # Value labels on top
            for pos, val in [(pos_api, api_data[i_model, i_task]),
                             (pos_vllm, vllm_data[i_model, i_task])]:
                if val > 0:
                    ax.text(
                        pos,
                        val + 0.5,
                        f"{val:.1f}",
                        ha="center",
                        va="bottom",
                        color="white",
                        fontsize=8.5,
                        fontweight="bold",
                        rotation=45,
                    )

    # Axes ----------------------------------------------------------------
    ax.set_xticks(x_centres)
    ax.set_xticklabels(models, fontsize=12, color="white")
    ax.set_ylabel("Time (minutes)", fontsize=12, color="white")
    ax.set_title(
        "Inference Time: Remote API  vs  Local vLLM Serving",
        fontsize=14,
        color="white",
        pad=20,
    )
    ax.tick_params(axis="y", colors="white", labelsize=11)
    ax.tick_params(axis="x", colors="white")
    ax.tick_params(axis="y", left=False)

    # Hide spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Add a thin horizontal grid for readability
    ax.yaxis.grid(True, linestyle="--", alpha=0.15, color="white")
    ax.set_axisbelow(True)

    # Legend — API in the left column, vLLM in the right column -----------
    # Matplotlib fills legend columns top-to-bottom, left-to-right
    # (column-major).  So items [0,1,2] go in col-1, [3,4,5] in col-2.
    # → put all API handles first, then all vLLM handles.
    ordered_handles = api_handles + vllm_handles
    ordered_labels = list(task_labels) + list(task_labels)

    legend = ax.legend(
        ordered_handles,
        ordered_labels,
        loc="upper right",
        frameon=False,
        fontsize=8,
        ncol=2,
        columnspacing=1.5,
        title="  API                   vLLM",
        title_fontproperties={
            "size": 10,
            "weight": "bold"
        },
    )
    plt.setp(legend.get_texts(), color="white")
    plt.setp(legend.get_title(), color="white")

    plt.tight_layout()
    out = VLLM_DIR / "time_comparison.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to {out}")
    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Compare inference time: remote API vs local vLLM.", )
    parser.add_argument(
        "models",
        nargs="+",
        help="Model name prefixes to compare, e.g. MiniMax-M2.7 Qwen3.5-122B",
    )
    args = parser.parse_args()
    plot_time_comparison(args.models)


if __name__ == "__main__":
    main()
