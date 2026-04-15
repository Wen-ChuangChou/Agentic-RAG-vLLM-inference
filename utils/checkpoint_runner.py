"""
General-purpose checkpoint runner for LLM evaluation.
Provides functions for running evaluations with atomic checkpointing and retry logic.
"""
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Union

from tqdm import tqdm


def save_checkpoint(checkpoint_file: Union[str, Path],
                    results: list,
                    next_idx: int,
                    model_name: str = "unknown",
                    prompt_name: str = "unknown") -> None:
    """
    Save checkpoint data to file using atomic write to prevent corruption.
    
    Args:
        checkpoint_file: Path to the JSON checkpoint file.
        results: List of processed results.
        next_idx: Index of the next item to process in the dataset.
        model_name: Identifier for the model being evaluated.
        prompt_name: Identifier for the prompt configuration used.
    """
    checkpoint_file = Path(checkpoint_file)
    checkpoint_data = {
        "model_name": model_name,
        "prompt_name": prompt_name,
        "results": results,
        "next_idx": next_idx,
        "timestamp": datetime.now().isoformat(),
    }

    # Ensure the directory exists
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)

    # Atomic write: write to temp file first, then rename
    temp_file = checkpoint_file.with_suffix(".tmp")
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)

    # Safely replace the old checkpoint file
    temp_file.replace(checkpoint_file)


def load_checkpoint(checkpoint_file: Union[str, Path]) -> dict:
    """
    Load checkpoint data from a JSON file.

    Args:
        checkpoint_file: Path to the checkpoint file.

    Returns:
        A dictionary containing checkpoint state, or empty dict if not found.
    """
    checkpoint_file = Path(checkpoint_file)
    if not checkpoint_file.exists():
        return {}

    try:
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Error loading checkpoint: {e}")
        print("Starting from beginning.")
        return {}


def run_with_checkpoint(
    eval_dataset,
    answer_fn: Callable[[str], str],
    checkpoint_file: Union[str, Path] = Path("checkpoints/checkpoint.json"),
    model_name: str = "unknown",
    prompt_name: str = "unknown",
    delay: float = 0,
) -> List[dict]:
    """
    Run LLM evaluation with checkpointing to allow resuming from
    interruptions (API errors, rate limits, keyboard interrupt, etc.).

    Args:
        eval_dataset: HuggingFace dataset with "question", "answer",
                      and "source_doc" columns.
        answer_fn: A callable that takes a question string and returns
                   an answer string. This is the core abstraction that
                   makes this function work for any backend (agentic RAG,
                   vanilla LLM, etc.).
        checkpoint_file: Path (or str) to JSON file for checkpoint data.
        model_name: Name of the LLM model (for metadata tracking).
        prompt_name: Name of the prompt config used (for metadata tracking).
        delay: Seconds to wait between API calls (rate limiting).

    Returns:
        List of result dicts, each containing:
        {question, true_answer, source_doc, generated_answer}
    """
    checkpoint_file = Path(checkpoint_file)
    results = []
    start_idx = 0

    # Load existing checkpoint if available
    checkpoint = load_checkpoint(checkpoint_file)
    if checkpoint:
        results = checkpoint.get("results", [])
        start_idx = checkpoint.get("next_idx", 0)
        prev_model = checkpoint.get("model_name", "unknown")
        prev_prompt = checkpoint.get("prompt_name", "unknown")
        print(
            f"Resuming from checkpoint at index {start_idx}/{len(eval_dataset)} "
            f"({len(results)} results already processed)")
        print(f"  Previous run: model={prev_model}, prompt={prev_prompt}")

    if start_idx >= len(eval_dataset):
        print("All questions already processed!")
        return results

    total = len(eval_dataset)
    pbar = tqdm(total=total, initial=start_idx, desc="Evaluating")

    try:
        for idx in range(start_idx, total):
            example = eval_dataset[idx]
            question = example["question"]

            try:
                answer = answer_fn(question)

                print(
                    "=======================================================")
                print(f"Question: {question}")
                print(f"Answer: {answer}")
                print(f'True answer: {example["answer"]}')

                result = {
                    "question": question,
                    "true_answer": example["answer"],
                    "source_doc": example["source_doc"],
                    "generated_answer": str(answer),
                }
                results.append(result)

                # Save checkpoint after each successful question
                save_checkpoint(checkpoint_file, results, idx + 1, model_name,
                                prompt_name)
                pbar.update(1)

                # Optional rate limiting
                if delay > 0:
                    time.sleep(delay)

            except Exception as e:
                print(f"\nError at question {idx}: {e}")
                save_checkpoint(checkpoint_file, results, idx, model_name,
                                prompt_name)
                print(f"Checkpoint saved. Re-run to resume from index {idx}.")
                raise

    except KeyboardInterrupt:
        print(f"\nInterrupted by user at question {idx}.")
        save_checkpoint(checkpoint_file, results, idx, model_name, prompt_name)
        print(f"Checkpoint saved. Re-run to resume from index {idx}.")

    finally:
        pbar.close()

    return results


def save_results(results_file: Union[str, Path], system_type: str,
                 outputs: list) -> None:
    """
    Save specific system results to a shared JSON results file.
    Does not overwrite results from other systems.

    Args:
        results_file: Path to the shared JSON file.
        system_type: Label for the system (e.g., 'agentic_rag').
        outputs: List of result dictionaries.
    """
    results_file = Path(results_file)

    # Load existing results (preserve other agents' data)
    all_results = load_results(results_file)

    # Update with this agent's outputs
    all_results[system_type] = outputs

    # Ensure directory exists
    results_file.parent.mkdir(parents=True, exist_ok=True)

    # Atomic write
    temp_file = results_file.with_suffix(".tmp")
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    temp_file.replace(results_file)

    print(f"Saved {len(outputs)} results for '{system_type}' "
          f"to {results_file}")


def load_results(results_file: Union[str, Path]) -> dict:
    """
    Load all system results from a shared JSON results file.

    Args:
        results_file: Path to the JSON results file.

    Returns:
        A dictionary where keys are system types and values are lists of results.
        Returns an empty dictionary if the file does not exist or is invalid.
    """
    results_file = Path(results_file)
    if not results_file.exists():
        return {}

    try:
        with open(results_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Error loading results: {e}")
        return {}


# ---------------------------------------------------------------------------
# Evaluation with checkpointing
# ---------------------------------------------------------------------------


def _extract_retry_delay(error_message: str) -> float | None:
    """
    Extract retry delay (in seconds) from an API error message.
    Example: 'rate limit exceeded, retry in 26.2s'
    """
    import re
    match = re.search(r'retry in ([\d.]+)s', str(error_message))
    if match:
        return float(match.group(1))
    return None


def _is_retryable_error(error_str: str) -> bool:
    """
    Determine if a given error message indicates a condition
    that should be retried (e.g., rate limits, service down).
    """
    retryable_codes = ["429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE"]
    return any(code in error_str for code in retryable_codes)


def evaluate_with_retry(evaluation_llm,
                        messages: list,
                        max_retries: int = 5) -> str | None:
    """
    Generate an evaluation response using the LLM with automatic retry logic.
    Retries on transient errors like rate limits (429) or service outages (503).

    Args:
        evaluation_llm: The LLM instance used for evaluation.
        messages: List of chat messages for the model.
        max_retries: Maximum number of retry attempts.

    Returns:
        The content of the generated response, or None if evaluation failed.
    """
    for attempt in range(max_retries):
        try:
            return evaluation_llm.generate(messages).content
        except Exception as e:
            error_str = str(e)

            if _is_retryable_error(error_str):
                retry_delay = _extract_retry_delay(error_str)
                if retry_delay is None:
                    retry_delay = min(2**attempt * 10, 120)

                if attempt < max_retries - 1:
                    print(f"\nRetryable error. Waiting {retry_delay:.0f}s "
                          f"(attempt {attempt + 1}/{max_retries})...")
                    time.sleep(retry_delay)
                else:
                    print(f"\nMax retries ({max_retries}) exhausted.")
                    raise
            else:
                print(f"\nNon-retryable error: {error_str}")
                raise

    return None


def run_evaluation_with_checkpoint(
    system_outputs: dict,
    evaluation_prompt: str,
    evaluation_llm,
    checkpoint_file: Union[str,
                           Path] = Path("checkpoints/eval_checkpoint.json"),
    delay: float = 5,
    max_retries: int = 5,
    max_consecutive_errors: int = 3,
) -> dict:
    """
    Evaluate multiple systems with LLM-as-judge, with checkpointing.

    If errors persist for max_consecutive_errors in a row, the loop STOPS
    and saves checkpoint (does not skip to the next evaluation).

    Args:
        system_outputs: Dict like
            {"agentic_rag": [...], "standard_rag": [...], "standard": [...]}
            where each value is a list of result dicts with keys:
            question, true_answer, source_doc, generated_answer.
        evaluation_prompt: Prompt template with {instruction}, {response},
            {reference_answer} placeholders.
        evaluation_llm: The LLM model instance for evaluation.
        checkpoint_file: Path to JSON checkpoint file.
        delay: Seconds between API calls.
        max_retries: Max retries per individual API call.
        max_consecutive_errors: Stop after this many consecutive failures.

    Returns:
        Dict of evaluated outputs (same structure as system_outputs but
        with eval_score_LLM_judge and eval_feedback_LLM_judge added).
    """
    checkpoint_file = Path(checkpoint_file)

    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_file)
    evaluated = checkpoint.get("results", {})
    progress = checkpoint.get("progress", {})

    for system_type, outputs in system_outputs.items():
        start_idx = progress.get(system_type, 0)

        # Initialize evaluated list for this system if needed
        if system_type not in evaluated:
            evaluated[system_type] = []

        if start_idx >= len(outputs):
            print(f"'{system_type}' already fully evaluated "
                  f"({len(outputs)} items). Skipping.")
            continue

        print(f"\n{'='*50}")
        print(f"Evaluating {system_type} "
              f"(starting from {start_idx}/{len(outputs)})")
        print(f"{'='*50}")

        consecutive_errors = 0
        pbar = tqdm(total=len(outputs), initial=start_idx, desc=system_type)

        try:
            for idx in range(start_idx, len(outputs)):
                experiment = outputs[idx].copy()
                eval_prompt = evaluation_prompt.format(
                    instruction=experiment["question"],
                    response=experiment["generated_answer"],
                    reference_answer=experiment["true_answer"],
                )
                messages = [
                    {
                        "role": "system",
                        "content": "You are a fair evaluator language model."
                    },
                    {
                        "role": "user",
                        "content": eval_prompt
                    },
                ]

                try:
                    eval_result = evaluate_with_retry(evaluation_llm, messages,
                                                      max_retries)

                    if eval_result:
                        # Use rsplit to split on the LAST "[RESULT]"
                        # occurrence, in case leaked chain-of-thought
                        # reasoning contains earlier mentions.
                        parts = eval_result.rsplit("[RESULT]", 1)
                        if len(parts) == 2:
                            feedback = parts[0].strip()
                            score = parts[1].strip()
                            experiment["eval_score_LLM_judge"] = score
                            experiment["eval_feedback_LLM_judge"] = feedback
                        else:
                            print(f"\nParsing failed: {eval_result}")
                            experiment["eval_score_LLM_judge"] = None
                            experiment["eval_feedback_LLM_judge"] = None

                    consecutive_errors = 0  # Reset on success

                except Exception as e:
                    consecutive_errors += 1
                    print(f"\nError at {system_type}[{idx}]: {e}")
                    experiment["eval_score_LLM_judge"] = None
                    experiment["eval_feedback_LLM_judge"] = None

                    if consecutive_errors >= max_consecutive_errors:
                        print(f"\n{consecutive_errors} consecutive errors. "
                              f"Stopping evaluation.")
                        # Save what we have so far
                        evaluated[system_type].append(experiment)
                        progress[system_type] = idx + 1
                        save_checkpoint(checkpoint_file,
                                        evaluated,
                                        0,
                                        model_name="eval")
                        # Also save progress
                        _save_eval_checkpoint(checkpoint_file, evaluated,
                                              progress)
                        pbar.close()
                        raise RuntimeError(
                            f"Stopped: {consecutive_errors} consecutive "
                            f"errors at {system_type}[{idx}]. "
                            f"Checkpoint saved. Re-run to resume.")

                evaluated[system_type].append(experiment)
                progress[system_type] = idx + 1

                # Save checkpoint after each success
                _save_eval_checkpoint(checkpoint_file, evaluated, progress)
                pbar.update(1)

                if delay > 0:
                    time.sleep(delay)

        except KeyboardInterrupt:
            progress[system_type] = idx
            _save_eval_checkpoint(checkpoint_file, evaluated, progress)
            pbar.close()
            print(f"\nInterrupted. Checkpoint saved at "
                  f"{system_type}[{idx}].")
            return evaluated

        finally:
            pbar.close()

        print(f"\nCompleted {system_type}: "
              f"{len(evaluated[system_type])} evaluated")

    print(f"\n{'='*50}")
    print("Evaluation complete!")
    print(f"{'='*50}\n")
    return evaluated


def _save_eval_checkpoint(checkpoint_file: Union[str, Path], evaluated: dict,
                          progress: dict) -> None:
    """
    Save evaluation progress to a checkpoint file.

    Args:
        checkpoint_file: Path to the JSON checkpoint file.
        evaluated: Dictionary containing results evaluated so far.
        progress: Dictionary tracking the current index for each system.
    """
    checkpoint_file = Path(checkpoint_file)
    checkpoint_data = {
        "results": evaluated,
        "progress": progress,
        "timestamp": datetime.now().isoformat(),
    }
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    temp_file = checkpoint_file.with_suffix(".tmp")
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
    temp_file.replace(checkpoint_file)
