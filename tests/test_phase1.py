"""
Isolated Phase 1 unit test — Offline Batch (Standard RAG + Vanilla).

Tests ONLY the offline vLLM batch inference with reduced max_model_len
to verify that the TMA descriptor crash was caused by KV cache memory
exhaustion (num_gpu_blocks=0 override), NOT a FlashAttention bug.

Usage:
    python tests/test_phase1.py --config recipes/GLM-4.7-Flash.yaml
    python tests/test_phase1.py --config recipes/GLM-4.7-Flash.yaml --test 3
"""

import argparse
import datasets
import gc
import os
import sys
import time
import torch
import yaml
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vllm import SamplingParams
from utils.agent_tools import RetrieverTool
from utils.offline_runner import (
    create_offline_llm,
    release_offline_llm,
    build_vanilla_prompts,
    build_rag_prompts,
    run_offline_batch,
)
from utils.vectordb_utils import load_or_create_vectordb


def _resolve_model_path(config: dict) -> str:
    env_path = os.environ.get("VLLM_MODEL_PATH", "")
    cfg_path = config.get("model", {}).get("model_path", "")
    model_id = config["model"]["model_id"]
    return env_path or cfg_path or model_id


def main():
    parser = argparse.ArgumentParser(description="Phase 1 unit test")
    parser.add_argument("--config", default="recipes/GLM-4.7-Flash.yaml")
    parser.add_argument("--test", type=int, default=5,
                        help="Number of questions (default: 5)")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_cfg = config["model"]
    model_id = _resolve_model_path(config)

    print("=" * 60)
    print(" Phase 1 Unit Test — Offline Batch Inference")
    print("=" * 60)
    print(f" Model:           {model_cfg['model_id']}")
    print(f" Model path:      {model_id}")
    print(f" max_model_len:   {model_cfg.get('max_model_len', 131072)}")
    print(f" max_tokens:      {model_cfg.get('max_tokens', 16384)}")
    print(f" enforce_eager:   {model_cfg.get('enforce_eager', False)}")
    print(f" TP size:         {model_cfg.get('tensor_parallel_size', 2)}")
    print(f" GPU mem util:    {model_cfg.get('gpu_memory_utilization', 0.92)}")
    print(f" Test questions:  {args.test}")
    print("=" * 60 + "\n")

    # ---- Setup ----
    vdb_cfg = config.get("vectordb", {})
    vectordb = load_or_create_vectordb(
        "m-ric/huggingface_doc",
        batch_size=vdb_cfg.get("batch_size", 50),
        max_workers=vdb_cfg.get("max_workers", 4),
        doc_chunk_size=vdb_cfg.get("doc_chunk_size", 100),
        text_chunk_size=vdb_cfg.get("text_chunk_size", 200),
        text_chunk_overlap=vdb_cfg.get("text_chunk_overlap", 40),
        force_rebuild=vdb_cfg.get("force_rebuild", False),
        use_parallel=vdb_cfg.get("use_parallel", True),
    )

    eval_dataset = datasets.load_dataset(
        "m-ric/huggingface_doc_qa_eval", split="train"
    )
    n = min(args.test, len(eval_dataset))
    eval_dataset = eval_dataset.select(range(n))
    print(f"Using {n} questions\n")

    retriever_tool = RetrieverTool(vectordb)

    # ---- Build prompts ----
    print("Building RAG prompts...")
    rag_convos = build_rag_prompts(eval_dataset, retriever_tool)
    print("Building Vanilla prompts...")
    vanilla_convos = build_vanilla_prompts(eval_dataset)
    print(f"RAG prompts: {len(rag_convos)}, Vanilla prompts: {len(vanilla_convos)}\n")

    # ---- Load model ----
    print(f"Loading model: {model_id}")
    t0 = time.time()
    llm = create_offline_llm(
        model_id=model_id,
        tensor_parallel_size=model_cfg.get("tensor_parallel_size", 2),
        gpu_memory_utilization=model_cfg.get("gpu_memory_utilization", 0.92),
        max_model_len=model_cfg.get("max_model_len", 16384),
        trust_remote_code=model_cfg.get("trust_remote_code", True),
        dtype=model_cfg.get("dtype", "auto"),
        enforce_eager=model_cfg.get("enforce_eager", False),
    )
    print(f"Model loaded in {time.time() - t0:.1f}s\n")

    sampling = SamplingParams(
        temperature=model_cfg.get("temperature", 0.2),
        max_tokens=model_cfg.get("max_tokens", 8192),
    )

    # ---- RAG batch ----
    print("=" * 40)
    print(" Running RAG batch inference")
    print("=" * 40)
    t0 = time.time()
    rag_answers = run_offline_batch(llm, rag_convos, sampling)
    rag_time = time.time() - t0
    print(f"\nRAG batch completed in {rag_time:.1f}s")
    for i, ans in enumerate(rag_answers):
        q = eval_dataset[i]["question"]
        print(f"\n  Q{i}: {q[:80]}...")
        print(f"  A{i}: {ans[:120]}...")

    # ---- Vanilla batch ----
    print("\n" + "=" * 40)
    print(" Running Vanilla batch inference")
    print("=" * 40)
    t0 = time.time()
    vanilla_answers = run_offline_batch(llm, vanilla_convos, sampling)
    vanilla_time = time.time() - t0
    print(f"\nVanilla batch completed in {vanilla_time:.1f}s")
    for i, ans in enumerate(vanilla_answers):
        q = eval_dataset[i]["question"]
        print(f"\n  Q{i}: {q[:80]}...")
        print(f"  A{i}: {ans[:120]}...")

    # ---- Cleanup ----
    print("\n\nReleasing GPU memory...")
    release_offline_llm(llm)

    # ---- Summary ----
    print("\n" + "=" * 60)
    print(" Phase 1 Unit Test — PASSED")
    print("=" * 60)
    print(f"  RAG:     {len(rag_answers)} answers in {rag_time:.1f}s")
    print(f"  Vanilla: {len(vanilla_answers)} answers in {vanilla_time:.1f}s")
    print(f"  Config:  max_model_len={model_cfg.get('max_model_len')}, "
          f"enforce_eager={model_cfg.get('enforce_eager', False)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
