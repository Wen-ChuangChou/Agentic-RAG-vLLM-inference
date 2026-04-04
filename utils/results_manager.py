import json
import pandas as pd
from datetime import datetime
from pathlib import Path


def save_evaluation_results(json_results: dict, results: dict,
                            RESULTS_DIR: str, eval_performance_filename: str):
    """
    Save evaluation results to a JSON file.
    If the file already exists, save as a new file with a timestamp suffix to avoid overwriting.

    Args:
        json_results: Dictionary to store the meta-data and results.
        results: Dictionary containing results (system_type: DataFrame).
        RESULTS_DIR: Directory to save the results.
        eval_performance_filename: Name of the file to save the results.
    """
    # Add a timestamp to the results metadata
    json_results["timestamp"] = datetime.now().strftime("%Y%m%d_%H%M")

    # Convert DataFrames in the results dictionary to JSON-serializable lists of records
    for system_type, df in results.items():
        json_results[system_type] = json.loads(df.to_json(orient='records'))

    save_path = Path(RESULTS_DIR) / eval_performance_filename

    # Check if file exists; if so, create a new path with a date timestamp suffix
    if save_path.exists():
        stem = save_path.stem
        suffix = save_path.suffix
        timestamp = datetime.now().strftime("%m%d")
        new_filename = f"{stem}_{timestamp}{suffix}"
        save_path = Path(RESULTS_DIR) / new_filename
        print(f"File already exists. Saving as: {save_path}")

    with open(save_path, 'w') as f:
        json.dump(json_results, f, indent=4)
    print(f"Results saved to {save_path}")


def load_evaluation_results(results_dir: str, eval_performance_filename: str):
    """
    Load evaluation results from a JSON file and reconstruct pandas DataFrames.

    Args:
        results_dir: Directory to load the results from.
        eval_performance_filename: Name of the file to load the results from.
    
    Returns:
        dict: A dictionary containing the original metadata and the results reconstructed as DataFrames.
    """
    # Load and parse the JSON results file
    with open(Path(results_dir) / eval_performance_filename, 'r') as f:
        json_results = json.load(f)

    # Reconstruct DataFrames from the JSON records while maintaining other metadata
    loaded_results = {}
    for key, records in json_results.items():
        # If the value is a list, treat it as a collection of records for a DataFrame
        if isinstance(records, list):
            loaded_results[key] = pd.DataFrame.from_records(records)
        else:
            loaded_results[key] = records

    print(
        f"Successfully loaded results from: {Path(results_dir) / eval_performance_filename}"
    )
    print(
        f"Evaluation results for systems implementing model: {json_results['model_name']}\n"
    )
    return loaded_results
