"""
Offline batch inference using vllm.LLM for maximum GPU throughput.

Used for Phase 1 (Standard RAG + Vanilla) and Phase 3 (Judge evaluation).
vllm.LLM.generate() / .chat() processes all prompts in a single highly
optimised call, eliminating HTTP overhead and maximising GPU utilisation.

Supports single-process offline LLM (TP >= 1, DP <= 1) and multi-process
data-parallel offline inference (DP > 1, EP enabled) based on the official
vLLM data_parallel_offline.py reference.
"""

import gc
import json
import os
import re
import time
import torch
from datetime import datetime
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm
from vllm import LLM, SamplingParams


# ---------------------------------------------------------------------------
# LLM lifecycle (Single-process)
# ---------------------------------------------------------------------------

def create_offline_llm(
    model_id: str,
    tensor_parallel_size: int = 2,
    gpu_memory_utilization: float = 0.92,
    max_model_len: int = 131072,
    trust_remote_code: bool = True,
    dtype: str = "auto",
    disable_custom_all_reduce: bool = True,
    enforce_eager: bool = False,
    quantization: Optional[str] = None,
    kv_cache_dtype: Optional[str] = None,
    max_num_seqs: Optional[int] = None,
    max_num_batched_tokens: Optional[int] = None,
    moe_backend: Optional[str] = None,
    attention_backend: Optional[str] = None,
    **kwargs_extra,
) -> LLM:
    """Create a vLLM offline LLM instance for batch processing.

    Args:
        max_num_seqs: Maximum number of sequences processed concurrently.
    """
    kwargs: dict = dict(
        model=model_id,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        trust_remote_code=trust_remote_code,
        dtype=dtype,
        disable_custom_all_reduce=disable_custom_all_reduce,
        enforce_eager=enforce_eager,
    )
    if quantization is not None:
        kwargs["quantization"] = quantization
    if kv_cache_dtype is not None:
        kwargs["kv_cache_dtype"] = kv_cache_dtype
    if max_num_seqs is not None:
        kwargs["max_num_seqs"] = max_num_seqs
    if max_num_batched_tokens is not None:
        kwargs["max_num_batched_tokens"] = max_num_batched_tokens
    if moe_backend is not None:
        kwargs["moe_backend"] = moe_backend
    if attention_backend is not None:
        kwargs["attention_backend"] = attention_backend
    return LLM(**kwargs)


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
    Remove reasoning / chain-of-thought blocks from model output.

    Handles three formats:
    1. ``<think>...</think>`` tags (e.g. GLM-4.7-Flash, Qwen3).
    2. Inline reasoning prefixed before ``Feedback:`` (e.g. GPT-OSS-120b
       which emits ``analysis...assistantfinalFeedback: ...``).
    3. Unstructured ``Thinking Process:`` blocks (e.g. Qwen3.5-122B).
    """
    # 1) Handle <think>...</think> tags
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    # Also strip any remaining opening tags (edge case: no closing tag)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)

    # 2) Handle unstructured "Thinking Process:" blocks
    if "Thinking Process:" in text:
        markers = [
            r'\*\s+\*(?:Final Answer|Revised Answer|Drafting the Answer|Drafting the Response|Final Plan|Final decision|Answer|Conclusion):\*',
            r'\n(?:\d+\.)?\s*\*\*Drafting the (?:Answer|Response):?\*\*',
            r'\n(?:\d+\.)?\s*\*(?:Final Answer|Revised Answer|Answer):?\*',
            r'\n(?:\d+\.)?\s*\*(?:Final Plan|Plan):\*',
            r'\*\s+(?:Final Answer|Answer):'
        ]
        best_match_idx = -1
        for marker in markers:
            matches = list(re.finditer(marker, text, re.IGNORECASE))
            if matches:
                idx = matches[-1].end()
                best_match_idx = max(best_match_idx, idx)
                
        if best_match_idx != -1:
            ans = text[best_match_idx:].strip()
        else:
            # Fallback: grab the last paragraph
            paragraphs = [p for p in text.split('\n\n') if p.strip()]
            if len(paragraphs) > 1:
                ans = paragraphs[-1].strip()
            else:
                ans = text[-600:].strip()
                
        # Clean up any leftover markdown bullet points from the thought process
        ans = re.sub(r'^\s*\*\s*', '', ans, flags=re.MULTILINE)
        text = re.sub(r'^\*(.*)\*$', r'\1', ans, flags=re.MULTILINE).strip()

    # 3) Handle inline reasoning (e.g. "analysis...assistantfinalFeedback:")
    if "Feedback:" in text:
        idx = text.rfind("Feedback:")
        text = text[idx:]

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
# Batch inference (Single-process)
# ---------------------------------------------------------------------------

def run_offline_batch(
    llm: LLM,
    conversations: List[List[Dict[str, str]]],
    sampling_params: Optional[SamplingParams] = None,
    chat_template_kwargs: Optional[dict] = None,
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
    outputs = llm.chat(
        conversations,
        sampling_params=sampling_params,
        chat_template_kwargs=chat_template_kwargs,
    )
    return [strip_thinking(output.outputs[0].text) for output in outputs]


# ---------------------------------------------------------------------------
# Multi-Process Data Parallel Batch Inference (from data_parallel_offline.py)
# ---------------------------------------------------------------------------

def _dp_offline_worker(
    global_dp_rank: int,
    local_dp_rank: int,
    dp_size: int,
    dp_master_ip: str,
    dp_master_port: int,
    engine_kwargs: dict,
    conversations_shard: list,
    sampling_params: SamplingParams,
    output_file: str,
    chat_template_kwargs: Optional[dict] = None,
):
    """Worker process for offline Data Parallel inference."""
    try:
        os.setpgrp()
    except Exception:
        pass
    os.environ["VLLM_DP_RANK"] = str(global_dp_rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(local_dp_rank)
    os.environ["VLLM_DP_SIZE"] = str(dp_size)
    os.environ["VLLM_DP_MASTER_IP"] = str(dp_master_ip)
    os.environ["VLLM_DP_MASTER_PORT"] = str(dp_master_port)

    import torch
    dev_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if dev_count < (local_dp_rank + 1):
        raise RuntimeError(
            f"[DP Rank {global_dp_rank} (local {local_dp_rank})] Requires device index {local_dp_rank}, "
            f"but only {dev_count} CUDA device(s) are accessible "
            f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<not set>')}). "
            f"Ensure the Slurm job is submitted with --mem=0 --gpus-per-node=4 --gres-flags=disable-binding."
        )

    print(f"  [DP Rank {global_dp_rank} (local {local_dp_rank})] Starting engine on GPU {local_dp_rank} "
          f"({len(conversations_shard)} prompts, {dev_count} devices accessible)...")
    llm = LLM(**engine_kwargs)
    outputs = llm.chat(
        conversations_shard,
        sampling_params=sampling_params,
        chat_template_kwargs=chat_template_kwargs,
    )    
    results = [strip_thinking(output.outputs[0].text) for output in outputs]

    # Persist shard results atomically
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)
    tmp_path.replace(out_path)

    print(f"  [DP Rank {global_dp_rank}] Completed {len(results)} outputs.")
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    time.sleep(1)


def run_offline_dp_batch(
    model_id: str,
    conversations: List[List[Dict[str, str]]],
    sampling_params: Optional[SamplingParams] = None,
    data_parallel_size: int = 1,
    data_parallel_size_local: Optional[int] = None,
    tensor_parallel_size: int = 1,
    enable_expert_parallel: bool = False,
    all2all_backend: Optional[str] = None,
    gpu_memory_utilization: float = 0.92,
    max_model_len: int = 16384,
    trust_remote_code: bool = True,
    dtype: str = "auto",
    disable_custom_all_reduce: bool = True,
    enforce_eager: bool = False,
    quantization: Optional[str] = None,
    kv_cache_dtype: Optional[str] = None,
    max_num_seqs: Optional[int] = None,
    max_num_batched_tokens: Optional[int] = None,
    moe_backend: Optional[str] = None,
    attention_backend: Optional[str] = None,
    dp_num_nodes: int = 1,
    dp_node_rank: int = 0,
    dp_master_addr: str = "127.0.0.1",
    dp_master_port: int = 13345,
    timeout: int = 3600,
    chat_template_kwargs: Optional[dict] = None,
) -> List[str]:
    """Run batch inference across multiple DP ranks following vLLM data_parallel_offline pattern.

    If data_parallel_size <= 1, delegates directly to single-process LLM.
    """
    if sampling_params is None:
        sampling_params = SamplingParams(temperature=0.2, max_tokens=16384)

    if data_parallel_size <= 1:
        # Standard single process offline inference
        llm = create_offline_llm(
            model_id=model_id,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=trust_remote_code,
            dtype=dtype,
            disable_custom_all_reduce=disable_custom_all_reduce,
            enforce_eager=enforce_eager,
            quantization=quantization,
            kv_cache_dtype=kv_cache_dtype,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            moe_backend=moe_backend,
            attention_backend=attention_backend,
        )
        try:
            return run_offline_batch(
                llm,
                conversations,
                sampling_params=sampling_params,
                chat_template_kwargs=chat_template_kwargs,
            )
        finally:
            release_offline_llm(llm)

    # Multi-process Data Parallel inference
    print(f"\nRunning offline DP batch inference: {len(conversations)} prompts across "
          f"{data_parallel_size} DP ranks ({dp_num_nodes} nodes, node_rank={dp_node_rank})...")

    engine_kwargs: dict = dict(
        model=model_id,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        trust_remote_code=trust_remote_code,
        dtype=dtype,
        disable_custom_all_reduce=disable_custom_all_reduce,
        enforce_eager=enforce_eager,
    )
    if enable_expert_parallel:
        engine_kwargs["enable_expert_parallel"] = True
    if all2all_backend:
        engine_kwargs["all2all_backend"] = all2all_backend
    if quantization is not None:
        engine_kwargs["quantization"] = quantization
    if kv_cache_dtype is not None:
        engine_kwargs["kv_cache_dtype"] = kv_cache_dtype
    if max_num_seqs is not None:
        engine_kwargs["max_num_seqs"] = max_num_seqs
    if max_num_batched_tokens is not None:
        engine_kwargs["max_num_batched_tokens"] = max_num_batched_tokens
    if moe_backend is not None:
        engine_kwargs["moe_backend"] = moe_backend
    if attention_backend is not None:
        engine_kwargs["attention_backend"] = attention_backend

    dp_per_node = data_parallel_size_local or (data_parallel_size // max(dp_num_nodes, 1))
    start_global_rank = dp_node_rank * dp_per_node
    end_global_rank = start_global_rank + dp_per_node

    # Shard slicing helper matching vllm data_parallel_offline.py
    floor = len(conversations) // data_parallel_size
    remainder = len(conversations) % data_parallel_size

    def get_shard_bounds(rank):
        return rank * floor + min(rank, remainder), (rank + 1) * floor + min(rank + 1, remainder)

    tmp_dir = Path(f"checkpoints/dp_tmp_{os.environ.get('SLURM_JOB_ID', os.getpid())}")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    mp_ctx = mp.get_context("spawn")
    procs = []
    output_files = {}
    for local_dp_rank, global_dp_rank in enumerate(range(start_global_rank, end_global_rank)):
        s_start, s_end = get_shard_bounds(global_dp_rank)
        shard = conversations[s_start:s_end]
        if len(shard) == 0:
            # Placeholder if rank has no prompts (vLLM DP requires all ranks active)
            shard = [[{"role": "user", "content": "Placeholder"}]]

        out_file = tmp_dir / f"dp_rank_{global_dp_rank}.json"
        output_files[global_dp_rank] = (out_file, s_start, s_end, len(conversations[s_start:s_end]))

        proc = mp_ctx.Process(
            target=_dp_offline_worker,
            args=(
                global_dp_rank,
                local_dp_rank,
                data_parallel_size,
                dp_master_addr,
                dp_master_port,
                engine_kwargs,
                shard,
                sampling_params,
                str(out_file),
                chat_template_kwargs,
            ),
        )
        proc.start()
        procs.append(proc)

    import signal

    def _terminate_proc_tree(p):
        try:
            pgid = os.getpgid(p.pid)
            os.killpg(pgid, signal.SIGTERM)
            time.sleep(0.5)
            os.killpg(pgid, signal.SIGKILL)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass

    for proc in procs:
        proc.join(timeout=timeout)
        if proc.exitcode is None:
            for p in procs:
                _terminate_proc_tree(p)
            raise TimeoutError(f"DP worker PID {proc.pid} timed out after {timeout}s.")
        elif proc.exitcode != 0:
            for p in procs:
                if p != proc and p.is_alive():
                    _terminate_proc_tree(p)
            raise RuntimeError(f"DP worker PID {proc.pid} exited with code {proc.exitcode}.")

    gc.collect()

    # Gather results
    if dp_node_rank == 0:
        all_results = [None] * len(conversations)
        wait_start = time.time()
        for r in range(data_parallel_size):
            r_file = tmp_dir / f"dp_rank_{r}.json"
            s_start, s_end = get_shard_bounds(r)
            expected_count = s_end - s_start
            while not r_file.exists():
                if time.time() - wait_start > timeout:
                    raise TimeoutError(f"Timed out waiting for rank {r} output {r_file}")
                time.sleep(1)

            with open(r_file, "r", encoding="utf-8") as f:
                rank_out = json.load(f)
            all_results[s_start:s_end] = rank_out[:expected_count]

        # Cleanup tmp files
        for r in range(data_parallel_size):
            r_file = tmp_dir / f"dp_rank_{r}.json"
            if r_file.exists():
                try:
                    r_file.unlink()
                except Exception:
                    pass
        try:
            tmp_dir.rmdir()
        except Exception:
            pass

        return all_results
    else:
        # Worker node completes its portion and returns empty list
        return []


# ---------------------------------------------------------------------------
# Phase checkpoint helpers
# ---------------------------------------------------------------------------

def save_phase_results(
    results: list,
    checkpoint_file: Path,
    phase_name: str = "unknown",
    elapsed_time: Optional[float] = None,
    extra_timing: Optional[dict] = None,
) -> None:
    """Save phase results to a checkpoint file (atomic write)."""
    checkpoint_file = Path(checkpoint_file)
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)

    timing_dict: dict = {}
    if elapsed_time is not None:
        timing_dict["elapsed_time"] = elapsed_time
    if extra_timing:
        timing_dict.update(extra_timing)

    data = {
        "phase": phase_name,
        "results": results,
        "count": len(results),
        "timestamp": datetime.now().isoformat(),
    }
    if timing_dict:
        data["timing"] = timing_dict

    temp_file = checkpoint_file.with_suffix(".tmp")
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    temp_file.replace(checkpoint_file)
    print(f"Saved {len(results)} results for '{phase_name}' → {checkpoint_file}")


def load_phase_results(
    checkpoint_file: Path,
) -> Optional[Tuple[list, dict]]:
    """Load phase results from checkpoint."""
    checkpoint_file = Path(checkpoint_file)
    if not checkpoint_file.exists():
        return None
    try:
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        results = data.get("results", [])
        if isinstance(results, dict):
            print(f"Warning: unexpected dict results in {checkpoint_file}, skipping.")
            return None
        timing_dict = data.get("timing", {})
        print(f"Loaded {len(results)} cached results from {checkpoint_file}")
        return results, timing_dict
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Error loading phase results: {e}")
        return None
