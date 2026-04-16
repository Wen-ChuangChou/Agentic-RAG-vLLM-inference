"""
Asynchronous runner for concurrent Agentic RAG evaluation (Phase 2).

Uses asyncio + threading to run multiple smolagents CodeAgent instances in
parallel.  Each agent sends requests to the vLLM server independently; vLLM's
continuous batching dynamically groups them on the GPU for near-100%
utilisation.

smolagents CodeAgent.run() is synchronous, so each call is wrapped in
asyncio.to_thread() and guarded by a semaphore to cap concurrency.
"""

import asyncio
import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from smolagents import OpenAIServerModel, CodeAgent
from smolagents.monitoring import LogLevel
from tqdm import tqdm

from utils.agent_tools import RetrieverTool


def _create_agent(model_config: dict, vectordb) -> CodeAgent:
    """Create a fresh CodeAgent instance for a single agentic run."""
    llm = OpenAIServerModel(
        model_id=model_config["model_id"],
        api_base=model_config["api_base"],
        api_key=model_config.get("api_key", "ai4all"),
        max_tokens=model_config.get("max_tokens", 16384),
        temperature=model_config.get("temperature", 0.2),
    )
    retriever = RetrieverTool(vectordb)
    agent = CodeAgent(
        tools=[retriever],
        model=llm,
        planning_interval=3,
        max_steps=12,
        verbosity_level=LogLevel.ERROR,
    )
    return agent


class AsyncAgenticRunner:
    """Thread-safe async runner for concurrent agentic RAG queries."""

    def __init__(
        self,
        eval_dataset,
        model_config: dict,
        prompt_config: dict,
        vectordb,
        concurrency: int = 16,
        checkpoint_file: Optional[Path] = None,
        checkpoint_interval: int = 5,
    ):
        self.eval_dataset = eval_dataset
        self.model_config = model_config
        self.prompt_config = prompt_config
        self.vectordb = vectordb
        self.concurrency = concurrency
        self.checkpoint_file = (Path(checkpoint_file)
                                if checkpoint_file else None)
        self.checkpoint_interval = checkpoint_interval

        # Thread-safe state
        self._lock = threading.Lock()
        self._results: Dict[int, dict] = {}
        self._completed = 0
        self._pbar = None

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def _load_checkpoint(self) -> set:
        """Load checkpoint and return set of already-completed indices."""
        if self.checkpoint_file is None or not self.checkpoint_file.exists():
            return set()
        try:
            with open(self.checkpoint_file, "r") as f:
                data = json.load(f)
            results = data.get("results", {})
            for str_idx, result in results.items():
                self._results[int(str_idx)] = result
            completed = set(int(k) for k in results.keys())
            print(f"Resumed agentic checkpoint: "
                  f"{len(completed)}/{len(self.eval_dataset)} done")
            return completed
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Error loading agentic checkpoint: {e}")
            return set()

    def _save_checkpoint(self) -> None:
        """Save current results to checkpoint (thread-safe)."""
        if self.checkpoint_file is None:
            return
        with self._lock:
            data = {
                "results": {
                    str(k): v
                    for k, v in self._results.items()
                },
                "completed": len(self._results),
                "total": len(self.eval_dataset),
                "timestamp": datetime.now().isoformat(),
            }
        self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        temp = self.checkpoint_file.with_suffix(".tmp")
        with open(temp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        temp.replace(self.checkpoint_file)

    # ------------------------------------------------------------------
    # Result recording
    # ------------------------------------------------------------------

    def _record_result(self, idx: int, result: dict) -> None:
        """Record a result and periodically save checkpoint."""
        with self._lock:
            self._results[idx] = result
            self._completed += 1
            completed = self._completed

        if self._pbar:
            self._pbar.update(1)

        if completed % self.checkpoint_interval == 0:
            self._save_checkpoint()

    # ------------------------------------------------------------------
    # Core async logic
    # ------------------------------------------------------------------

    async def _process_question(self, idx: int,
                                semaphore: asyncio.Semaphore) -> None:
        """Process a single question with an agent (runs in a thread)."""
        async with semaphore:
            example = self.eval_dataset[idx]
            question = example["question"]
            enhanced = self.prompt_config["prompt"].format(question=question)

            def _run_agent():
                agent = _create_agent(self.model_config, self.vectordb)
                return agent.run(enhanced)

            try:
                answer = await asyncio.to_thread(_run_agent)
                result = {
                    "question": question,
                    "true_answer": example["answer"],
                    "source_doc": example["source_doc"],
                    "generated_answer": str(answer),
                }
            except Exception as e:
                print(f"\nError at question {idx}: {e}")
                result = {
                    "question": question,
                    "true_answer": example["answer"],
                    "source_doc": example["source_doc"],
                    "generated_answer": f"Error: {str(e)}",
                }

            self._record_result(idx, result)

    async def run(self) -> List[dict]:
        """Run all agentic queries concurrently. Returns ordered results."""
        total = len(self.eval_dataset)
        completed_indices = self._load_checkpoint()
        remaining = [i for i in range(total) if i not in completed_indices]

        if not remaining:
            print("All agentic queries already completed!")
            return self._get_ordered_results()

        print(f"Running {len(remaining)} agentic queries "
              f"(concurrency={self.concurrency})...")

        semaphore = asyncio.Semaphore(self.concurrency)
        self._pbar = tqdm(
            total=total,
            initial=len(completed_indices),
            desc="Agentic RAG",
        )

        tasks = [self._process_question(idx, semaphore) for idx in remaining]
        await asyncio.gather(*tasks)

        self._pbar.close()
        self._save_checkpoint()
        return self._get_ordered_results()

    def _get_ordered_results(self) -> List[dict]:
        """Return results ordered by question index."""
        return [self._results[i] for i in sorted(self._results.keys())]


# ---------------------------------------------------------------------------
# Public convenience function
# ---------------------------------------------------------------------------


async def run_agentic_batch(
    eval_dataset,
    model_config: dict,
    prompt_config: dict,
    vectordb,
    concurrency: int = 16,
    checkpoint_file: Optional[Path] = None,
    checkpoint_interval: int = 5,
) -> List[dict]:
    """
    Run concurrent agentic RAG evaluation.

    Args:
        eval_dataset: HuggingFace dataset with question/answer/source_doc.
        model_config: Dict with model_id, api_base, api_key, etc.
        prompt_config: Dict with 'prompt' key containing the system prompt.
        vectordb: FAISS vectordb instance (shared read-only across threads).
        concurrency: Max simultaneous agentic queries.
        checkpoint_file: Path for resumable checkpointing.
        checkpoint_interval: Save checkpoint every N completions.

    Returns:
        Ordered list of result dicts.
    """
    runner = AsyncAgenticRunner(
        eval_dataset=eval_dataset,
        model_config=model_config,
        prompt_config=prompt_config,
        vectordb=vectordb,
        concurrency=concurrency,
        checkpoint_file=checkpoint_file,
        checkpoint_interval=checkpoint_interval,
    )
    return await runner.run()
