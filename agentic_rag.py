"""
Hybrid Agentic RAG Evaluation Pipeline
=======================================

Three-phase evaluation that maximises GPU utilisation on HPC:

  Phase 1 (Offline Batch):   Standard RAG + Vanilla LLM  →  vllm.LLM.generate()
  Phase 2 (Async Server):    Concurrent Agentic RAG      →  vLLM server + asyncio
  Phase 3 (Offline Batch):   Judge LLM evaluation        →  vllm.LLM.generate()

Usage:
    python agentic_rag.py --config recipes/GLM-4.7-Flash.yaml
    python agentic_rag.py --config recipes/GLM-4.7-Flash.yaml --test 5
"""

import argparse
import asyncio
import datasets
import gc
import os
import pandas as pd
import torch
import yaml
from dotenv import load_dotenv
from pathlib import Path
from vllm import SamplingParams

from utils.agent_tools import RetrieverTool
from utils.async_agentic_runner import run_agentic_batch
from utils.offline_runner import (
    create_offline_llm,
    release_offline_llm,
    build_vanilla_prompts,
    build_rag_prompts,
    build_judge_prompts,
    run_offline_batch,
    save_phase_results,
    load_phase_results,
)
from utils.vllm_server_manager import VLLMServerManager
from utils.results_manager import save_evaluation_results
from utils.vectordb_utils import load_or_create_vectordb

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------


def fill_score(x, default_score):
    """Convert a string score to an integer, falling back on *default_score*."""
    try:
        return int(x)
    except Exception:
        return default_score


def _resolve_model_path(config: dict) -> str:
    """
    Determine the model path to load.
    Priority: VLLM_MODEL_PATH env var  >  config model.model_path  >  config model.model_id
    """
    env_path = os.environ.get("VLLM_MODEL_PATH", "")
    cfg_path = config.get("model", {}).get("model_path", "")
    model_id = config["model"]["model_id"]
    return env_path or cfg_path or model_id


def _gpu_cleanup(label: str = "") -> None:
    """Force GPU memory release between phases."""
    import time
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    time.sleep(5)
    print(f"GPU memory cleaned up{' after ' + label if label else ''}.")


# ===================================================================
#  Phase 1  —  Offline Batch (Standard RAG + Vanilla)
# ===================================================================


def phase1_offline_batch(config, eval_dataset, retriever_tool,
                         checkpoints_dir):
    """
    Phase 1: Offline batch processing for Standard RAG + Vanilla LLM.
    Pre-retrieves all contexts on CPU, then runs a single
    ``vllm.LLM.chat()`` call for maximum throughput.
    """
    model_cfg = config["model"]
    model_id = _resolve_model_path(config)
    model_name = model_cfg["model_id"].split("/")[-1]

    rag_ckpt = checkpoints_dir / f"{model_name}_phase1_rag.json"
    vanilla_ckpt = checkpoints_dir / f"{model_name}_phase1_vanilla.json"

    cached_rag = load_phase_results(rag_ckpt)
    cached_vanilla = load_phase_results(vanilla_ckpt)

    if cached_rag is not None and cached_vanilla is not None:
        print("Phase 1: Using cached results (both RAG and vanilla).")
        return cached_rag, cached_vanilla

    print("\n" + "=" * 60)
    print(" Phase 1: Offline Batch (Standard RAG + Vanilla)")
    print("=" * 60)

    # 1a — Pre-retrieve contexts (CPU only, no GPU)
    rag_convos = (build_rag_prompts(eval_dataset, retriever_tool)
                  if cached_rag is None else None)
    vanilla_convos = (build_vanilla_prompts(eval_dataset)
                      if cached_vanilla is None else None)

    # 1b — Load model offline
    print(f"\nLoading model for offline inference: {model_id}")
    llm = create_offline_llm(
        model_id=model_id,
        tensor_parallel_size=model_cfg.get("tensor_parallel_size", 2),
        gpu_memory_utilization=model_cfg.get("gpu_memory_utilization", 0.92),
        max_model_len=model_cfg.get("max_model_len", 131072),
        trust_remote_code=model_cfg.get("trust_remote_code", True),
        dtype=model_cfg.get("dtype", "auto"),
    )

    sampling = SamplingParams(
        temperature=model_cfg.get("temperature", 0.2),
        max_tokens=model_cfg.get("max_tokens", 16384),
    )

    # 1c — Standard RAG batch
    rag_results = cached_rag
    if rag_results is None:
        print("\n--- Standard RAG batch inference ---")
        rag_answers = run_offline_batch(llm, rag_convos, sampling)
        rag_results = [{
            "question": eval_dataset[i]["question"],
            "true_answer": eval_dataset[i]["answer"],
            "source_doc": eval_dataset[i]["source_doc"],
            "generated_answer": str(a),
        } for i, a in enumerate(rag_answers)]
        save_phase_results(rag_results, rag_ckpt, "standard_rag")

    # 1d — Vanilla batch
    vanilla_results = cached_vanilla
    if vanilla_results is None:
        print("\n--- Vanilla LLM batch inference ---")
        vanilla_answers = run_offline_batch(llm, vanilla_convos, sampling)
        vanilla_results = [{
            "question": eval_dataset[i]["question"],
            "true_answer": eval_dataset[i]["answer"],
            "source_doc": eval_dataset[i]["source_doc"],
            "generated_answer": str(a),
        } for i, a in enumerate(vanilla_answers)]
        save_phase_results(vanilla_results, vanilla_ckpt, "standard")

    # 1e — Release GPU
    print("\nReleasing GPU memory after Phase 1 ...")
    release_offline_llm(llm)

    print(f"\nPhase 1 complete: "
          f"{len(rag_results)} RAG + {len(vanilla_results)} vanilla")
    return rag_results, vanilla_results


# ===================================================================
#  Phase 2  —  Async Agentic RAG
# ===================================================================


def phase2_agentic(config, eval_dataset, prompt_config, vectordb,
                   checkpoints_dir):
    """
    Phase 2: Concurrent Agentic RAG via vLLM server + asyncio.
    Starts vLLM server as a subprocess, runs N agents in parallel,
    then stops the server.
    """
    model_cfg = config["model"]
    server_cfg = config.get("server", {})
    async_cfg = config.get("async", {})
    model_id = _resolve_model_path(config)
    model_name = model_cfg["model_id"].split("/")[-1]

    agentic_ckpt = checkpoints_dir / f"{model_name}_phase2_agentic.json"

    print("\n" + "=" * 60)
    print(" Phase 2: Async Agentic RAG (Concurrent Agents)")
    print("=" * 60)

    port = server_cfg.get("port", 8000)
    api_key = server_cfg.get("api_key", "ai4all")

    with VLLMServerManager(
            model_id=model_id,
            served_model_name=model_cfg["model_id"],
            port=port,
            api_key=api_key,
            tensor_parallel_size=model_cfg.get("tensor_parallel_size", 2),
            gpu_memory_utilization=model_cfg.get("gpu_memory_utilization",
                                                 0.92),
            max_model_len=model_cfg.get("max_model_len", 131072),
            dtype=model_cfg.get("dtype", "auto"),
            trust_remote_code=model_cfg.get("trust_remote_code", True),
            enable_prefix_caching=True,
            extra_args=server_cfg.get("extra_args", []),
    ) as server:
        agent_model_config = {
            "model_id": model_cfg["model_id"],  # served-model-name
            "api_base": server.url,
            "api_key": api_key,
            "max_tokens": model_cfg.get("max_tokens", 16384),
            "temperature": model_cfg.get("temperature", 0.2),
        }

        concurrency = async_cfg.get("concurrency", 16)
        ckpt_interval = async_cfg.get("checkpoint_interval", 5)

        agentic_results = asyncio.run(
            run_agentic_batch(
                eval_dataset=eval_dataset,
                model_config=agent_model_config,
                prompt_config=prompt_config,
                vectordb=vectordb,
                concurrency=concurrency,
                checkpoint_file=agentic_ckpt,
                checkpoint_interval=ckpt_interval,
            ))

    # Server is automatically stopped by the context manager
    print(f"\nPhase 2 complete: {len(agentic_results)} agentic results")
    return agentic_results


# ===================================================================
#  Phase 3  —  Offline Judge Evaluation
# ===================================================================


def phase3_judge(config, all_outputs, evaluation_prompt, checkpoints_dir):
    """
    Phase 3: Offline batch Judge LLM evaluation.
    Loads the judge model, scores *all* system outputs in a single
    ``vllm.LLM.chat()`` call, and parses the scores.
    """
    eval_cfg = config.get("evaluation", {})
    model_cfg = config["model"]
    judge_model_id = eval_cfg.get("model_id", model_cfg["model_id"])
    model_name = model_cfg["model_id"].split("/")[-1]

    judge_ckpt = checkpoints_dir / f"{model_name}_phase3_judge.json"
    cached = load_phase_results(judge_ckpt)
    if cached is not None:
        print("Phase 3: Using cached judge results.")
        return cached

    print("\n" + "=" * 60)
    print(" Phase 3: Offline Judge Evaluation")
    print("=" * 60)

    # Build prompts
    judge_convos, judge_meta = build_judge_prompts(all_outputs,
                                                   evaluation_prompt)
    print(f"Judge model:   {judge_model_id}")
    print(f"Total prompts: {len(judge_convos)}")

    # Resolve judge model path
    judge_path = (os.environ.get("VLLM_JUDGE_MODEL_PATH", "")
                  or eval_cfg.get("model_path", "") or judge_model_id)

    llm = create_offline_llm(
        model_id=judge_path,
        tensor_parallel_size=model_cfg.get("tensor_parallel_size", 2),
        gpu_memory_utilization=model_cfg.get("gpu_memory_utilization", 0.92),
        # Judge prompts are short (~2K tokens); no need for 131072 context
        max_model_len=eval_cfg.get("max_model_len", 8192),
        trust_remote_code=model_cfg.get("trust_remote_code", True),
        dtype=model_cfg.get("dtype", "auto"),
    )

    sampling = SamplingParams(
        temperature=eval_cfg.get("temperature", 0.0),
        max_tokens=eval_cfg.get("max_tokens", 16384),
    )

    judge_answers = run_offline_batch(llm, judge_convos, sampling)
    release_offline_llm(llm)

    # Parse "[RESULT]" formatted responses
    judge_results = []
    for answer, meta in zip(judge_answers, judge_meta):
        try:
            feedback, score = [s.strip() for s in answer.split("[RESULT]")]
        except ValueError:
            feedback, score = answer, None
        judge_results.append({
            "system_type": meta["system_type"],
            "idx": meta["idx"],
            "eval_score_LLM_judge": score,
            "eval_feedback_LLM_judge": feedback,
        })

    save_phase_results(judge_results, judge_ckpt, "judge")
    print(f"\nPhase 3 complete: {len(judge_results)} evaluations")
    return judge_results


# ===================================================================
#  Main
# ===================================================================


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Hybrid Agentic RAG evaluation pipeline")
    parser.add_argument(
        "--config",
        default="recipes/GLM-4.7-Flash.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--test",
        type=int,
        nargs="?",
        const=5,
        default=None,
        metavar="N",
        help="Test mode: only process first N questions (default: 5)",
    )
    parser.add_argument(
        "--embedding-device",
        default="cuda",
        choices=["cpu", "cuda"],
        help="Device to use for embedding model (default: cuda)",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    dataset_name = "m-ric/huggingface_doc"
    RESULTS_DIR = Path("results")
    CHECKPOINTS_DIR = Path("checkpoints")
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model_cfg = config["model"]
    model_name = model_cfg["model_id"].split("/")[-1]

    print("=" * 60)
    print(" Hybrid Agentic RAG Evaluation Pipeline")
    print("=" * 60)
    print(f" Model:  {model_cfg['model_id']}")
    print(f" Judge:  {config.get('evaluation', {}).get('model_id', 'N/A')}")
    print(f" Config: {args.config}")
    if args.test:
        print(f" Mode:   TEST (first {args.test} questions only)")
    print("=" * 60 + "\n")

    # === Setup: vectordb + dataset ===
    vdb_cfg = config.get("vectordb", {})
    vectordb_config = {
        "batch_size": vdb_cfg.get("batch_size", 50),
        "max_workers": vdb_cfg.get("max_workers", 4),
        "doc_chunk_size": vdb_cfg.get("doc_chunk_size", 100),
        "text_chunk_size": vdb_cfg.get("text_chunk_size", 200),
        "text_chunk_overlap": vdb_cfg.get("text_chunk_overlap", 40),
        "force_rebuild": vdb_cfg.get("force_rebuild", False),
        "use_parallel": vdb_cfg.get("use_parallel", True),
    }
    vectordb = load_or_create_vectordb(dataset_name,
                                       embedding_device=args.embedding_device,
                                       **vectordb_config)
    eval_dataset = datasets.load_dataset("m-ric/huggingface_doc_qa_eval",
                                         split="train")

    # --- Test mode: slice dataset to first N questions ---
    if args.test:
        n = min(args.test, len(eval_dataset))
        eval_dataset = eval_dataset.select(range(n))
        print(f"TEST MODE: using {n} of {len(eval_dataset)} questions\n")

    retriever_tool = RetrieverTool(vectordb)

    # Load prompts
    prompt_path = Path("prompts") / "guide_agent_system_prompt.yaml"
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_config = yaml.safe_load(f)

    eval_prompt_path = Path("prompts") / "evaluation_prompt.yaml"
    with open(eval_prompt_path, "r", encoding="utf-8") as f:
        evaluation_prompt = yaml.safe_load(f)

    # ==================================================================
    #  Phase 1 — Offline Batch (Standard RAG + Vanilla)
    # ==================================================================
    rag_results, vanilla_results = phase1_offline_batch(
        config, eval_dataset, retriever_tool, CHECKPOINTS_DIR)

    # GPU cleanup between Phase 1 and Phase 2
    _gpu_cleanup("Phase 1")

    # ==================================================================
    #  Phase 2 — Async Agentic RAG
    # ==================================================================
    agentic_results = phase2_agentic(config, eval_dataset, prompt_config,
                                     vectordb, CHECKPOINTS_DIR)

    # GPU cleanup between Phase 2 and Phase 3
    _gpu_cleanup("Phase 2")

    # ==================================================================
    #  Phase 3 — Offline Judge Evaluation
    # ==================================================================
    all_outputs = {
        "agentic_rag": agentic_results,
        "standard_rag": rag_results,
        "standard": vanilla_results,
    }

    judge_results = phase3_judge(config, all_outputs,
                                 evaluation_prompt["prompt"], CHECKPOINTS_DIR)

    # ==================================================================
    #  Scoring & Results
    # ==================================================================
    print("\n" + "=" * 60)
    print(" Results")
    print("=" * 60)

    # Attach judge scores to system outputs
    evaluated = {}
    for sys_type, outputs in all_outputs.items():
        evaluated[sys_type] = [o.copy() for o in outputs]

    for jr in judge_results:
        sys_type = jr["system_type"]
        idx = jr["idx"]
        if idx < len(evaluated[sys_type]):
            evaluated[sys_type][idx]["eval_score_LLM_judge"] = (
                jr["eval_score_LLM_judge"])
            evaluated[sys_type][idx]["eval_feedback_LLM_judge"] = (
                jr["eval_feedback_LLM_judge"])

    # Calculate scores
    results = {}
    DEFAULT_SCORE = 2
    for sys_type in ["agentic_rag", "standard_rag", "standard"]:
        df = pd.DataFrame.from_dict(evaluated[sys_type])
        df = df.loc[~df["generated_answer"].str.contains("Error", na=False)]

        df["eval_score_LLM_judge_int"] = (df["eval_score_LLM_judge"].fillna(
            DEFAULT_SCORE).apply(lambda x: fill_score(x, DEFAULT_SCORE)).astype(float))
        df["eval_score_LLM_judge_int"] = (df["eval_score_LLM_judge_int"] -
                                          1) / 2

        avg = df["eval_score_LLM_judge_int"].mean() * 100
        print(f"  {sys_type:20s} → {avg:.1f}%")
        results[sys_type] = df

    print("=" * 60 + "\n")

    # Persist
    eval_cfg = config.get("evaluation", {})
    eval_model_name = eval_cfg.get("model_id", "unknown").split("/")[-1]
    TEMPERATURE = model_cfg.get("temperature", 0.2)

    meta_data = {
        "model_name": model_name,
        "model_id": model_cfg["model_id"],
        "prompt_filename": "guide_agent_system_prompt.yaml",
        "eval_model_name": eval_model_name,
        "eval_model_id": eval_cfg.get("model_id", "unknown"),
    }
    filename = (f"{model_name}_vect{vectordb_config['text_chunk_size']}"
                f"_t{TEMPERATURE}.json")
    save_evaluation_results(meta_data, results, RESULTS_DIR, filename)


if __name__ == "__main__":
    main()
