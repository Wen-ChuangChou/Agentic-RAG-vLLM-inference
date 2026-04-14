"""
Offline batch inference using vllm.LLM for maximum GPU throughput.

Used for Phase 1 (Standard RAG + Vanilla) and Phase 3 (Judge evaluation).
vllm.LLM.generate() / .chat() processes all prompts in a single highly
optimised call, eliminating HTTP overhead and maximising GPU utilisation.
"""

import gc
import json
import re
import time
import torch
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm
from vllm import LLM, SamplingParams


# ---------------------------------------------------------------------------
# LLM lifecycle
# ---------------------------------------------------------------------------

def create_offline_llm(
    model_id: str,
    tensor_parallel_size: int = 2,
    gpu_memory_utilization: float = 0.92,
    max_model_len: int = 131072,
    trust_remote_code: bool = True,
    dtype: str = "auto",
    disable_custom_all_reduce: bool = True,
) -> LLM:
    """Create a vLLM offline LLM instance for batch processing."""
    return LLM(
        model=model_id,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        trust_remote_code=trust_remote_code,
        dtype=dtype,
        disable_custom_all_reduce=disable_custom_all_reduce,
    )


def release_offline_llm(llm: LLM) -> None:
    """
    Release GPU memory held by an offline LLM instance.
    Call between phases so the next phase can allocate freely.
    """
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    # Brief pause to let GPU memory fully release
    time.sleep(5)
    print("GPU memory released.")


def strip_thinking(text: str) -> str:
    """
    Remove ``<think>...</think>`` reasoning blocks from model output.

    Reasoning models like GLM-4.7-Flash wrap their chain-of-thought in
    ``<think>`` tags.  For evaluation we only want the final answer.
    """
    # If there's a closing </think> tag, take everything after it
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    # Also strip any remaining opening tags (edge case: no closing tag)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    return text.strip()


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_vanilla_prompts(eval_dataset) -> List[List[Dict[str, str]]]:
    """Build chat-format vanilla prompts for all questions."""
    conversations = []
    for example in eval_dataset:
        messages = [{"role": "user", "content": (
            "Answer the following question as clearly and concisely as possible.\n"
            "Use your own knowledge to respond.\n"
            "If the question is ambiguous or cannot be answered definitively, "
            "state so clearly.\n\n"
            f"Question:\n{example['question']}\n"
        )}]
        conversations.append(messages)
    return conversations


def build_rag_prompts(
    eval_dataset,
    retriever_tool,
) -> List[List[Dict[str, str]]]:
    """Pre-retrieve context on CPU and build Standard RAG prompts."""
    conversations = []
    print("Pre-retrieving context for all questions (CPU)...")
    for example in tqdm(eval_dataset, desc="Retrieving"):
        context = retriever_tool(example["question"], k=5)
        messages = [{"role": "user", "content": (
            "Given the question and supporting documents below, give a "
            "comprehensive answer to the question.\n"
            "Respond only to the question asked, response should be concise "
            "and relevant to the question.\n"
            "If the question is ambiguous or cannot be answered definitively, "
            "state so clearly.\n\n"
            f"Question:\n{example['question']}\n\n{context}"
        )}]
        conversations.append(messages)
    return conversations


def build_judge_prompts(
    system_outputs: Dict[str, list],
    evaluation_prompt: str,
) -> Tuple[List[List[Dict[str, str]]], List[Dict]]:
    """
    Build judge evaluation prompts for all system outputs.

    Returns:
        (conversations, metadata) where *metadata* tracks system_type
        and original index so results can be reassembled later.
    """
    conversations: List[List[Dict[str, str]]] = []
    metadata: List[Dict] = []

    for system_type, outputs in system_outputs.items():
        for idx, output in enumerate(outputs):
            prompt = evaluation_prompt.format(
                instruction=output["question"],
                response=output["generated_answer"],
                reference_answer=output["true_answer"],
            )
            messages = [
                {"role": "system",
                 "content": "You are a fair evaluator language model."},
                {"role": "user", "content": prompt},
            ]
            conversations.append(messages)
            metadata.append({"system_type": system_type, "idx": idx})

    return conversations, metadata


# ---------------------------------------------------------------------------
# Batch inference
# ---------------------------------------------------------------------------

def run_offline_batch(
    llm: LLM,
    conversations: List[List[Dict[str, str]]],
    sampling_params: Optional[SamplingParams] = None,
) -> List[str]:
    """
    Run batch chat inference using vLLM offline mode.

    Args:
        llm: vLLM LLM instance.
        conversations: List of chat conversations (list of message dicts).
        sampling_params: vLLM SamplingParams (defaults to temp=0.2, max_tokens=16384).

    Returns:
        List of generated response strings.
    """
    if sampling_params is None:
        sampling_params = SamplingParams(temperature=0.2, max_tokens=16384)

    print(f"Running offline batch inference on {len(conversations)} prompts...")
    outputs = llm.chat(conversations, sampling_params=sampling_params)
    return [strip_thinking(output.outputs[0].text) for output in outputs]


# ---------------------------------------------------------------------------
# Phase checkpoint helpers
# ---------------------------------------------------------------------------

def save_phase_results(
    results: list,
    checkpoint_file: Path,
    phase_name: str = "unknown",
) -> None:
    """Save phase results to a checkpoint file (atomic write)."""
    checkpoint_file = Path(checkpoint_file)
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "phase": phase_name,
        "results": results,
        "count": len(results),
        "timestamp": datetime.now().isoformat(),
    }
    temp_file = checkpoint_file.with_suffix(".tmp")
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    temp_file.replace(checkpoint_file)
    print(f"Saved {len(results)} results for '{phase_name}' → {checkpoint_file}")


def load_phase_results(checkpoint_file: Path) -> Optional[list]:
    """Load phase results from checkpoint. Returns None if not found."""
    checkpoint_file = Path(checkpoint_file)
    if not checkpoint_file.exists():
        return None
    try:
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        results = data.get("results", [])
        print(f"Loaded {len(results)} cached results from {checkpoint_file}")
        return results
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Error loading phase results: {e}")
        return None
