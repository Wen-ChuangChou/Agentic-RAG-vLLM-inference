"""
This script evaluates and compares three different Question-Answering (QA) systems:
1. Agentic RAG: Uses a Code Agent with access to a retriever tool.
2. Standard RAG: Uses a fixed-retrieval prompt with an LLM.
3. Vanilla LLM: Answers questions using the LLM's internal knowledge only.

The evaluation is performed using a Hugging Face dataset and an LLM-as-judge for scoring.
It uses checkpointing to allow resuming long-running evaluations.
"""
import argparse
import os
import datasets
import pandas as pd
import yaml
from dotenv import load_dotenv
from pathlib import Path
from smolagents import OpenAIServerModel, CodeAgent
from smolagents.monitoring import LogLevel
from utils.agent_tools import RetrieverTool
from utils.model_factory import create_llm
from utils.checkpoint_runner import run_with_checkpoint, run_evaluation_with_checkpoint
from utils.results_manager import save_evaluation_results
from utils.vectordb_utils import load_or_create_vectordb


def agentic_answer(question, agent, prompt_config):
    """
    Generate an answer using a CodeAgent with a tailored system prompt.
    
    Args:
        question (str): The user's query.
        agent (CodeAgent): The smolagents agent instance.
        prompt_config (dict): Configuration containing the system prompt template.
    
    Returns:
        str: The generated answer from the agent.
    """
    enhanced = prompt_config["prompt"].format(question=question)
    return agent.run(enhanced)


def rag_answer(question, retriever_tool, answer_llm):
    """
    Generate an answer using standard RAG (Retrieval-Augmented Generation).
    
    Args:
        question (str): The user's query.
        retriever_tool (RetrieverTool): Tool for fetching relevant documents.
        answer_llm (OpenAIServerModel): The LLM used for answering.
    
    Returns:
        str: The generated answer based on retrieved context.
    """
    context = retriever_tool(question, k=5)
    prompt = f"""Given the question and supporting documents below, give a comprehensive answer to the question.
Respond only to the question asked, response should be concise and relevant to the question.
If the question is ambiguous or cannot be answered definitively, state so clearly.

Question:
{question}

{context}
"""
    messages = [{"role": "user", "content": prompt}]
    return answer_llm.generate(messages).content


def vanilla_answer(question, answer_llm):
    """
    Generate an answer using only the LLM's pre-trained knowledge.
    
    Args:
        question (str): The user's query.
        answer_llm (OpenAIServerModel): The LLM used for answering.
    
    Returns:
        str: The generated answer.
    """
    prompt = f"""Answer the following question as clearly and concisely as possible.
Use your own knowledge to respond.
If the question is ambiguous or cannot be answered definitively, state so clearly.

Question:
{question}

"""
    messages = [{"role": "user", "content": prompt}]
    return answer_llm.generate(messages).content


def fill_score(x, default_score):
    """
    Convert a string score to an integer, falling back to a default if parsing fails.
    
    Args:
        x (any): The score value to convert.
        default_score (int): The fallback value.
    
    Returns:
        int: The converted score.
    """
    try:
        return int(x)
    except:
        return default_score


def main():
    # Initialize environment and paths
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Agentic RAG evaluation pipeline")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml)")
    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    dataset_name = "m-ric/huggingface_doc"
    RESULTS_DIR = Path("results")
    CHECKPOINTS_DIR = Path("checkpoints")

    # Configurable parameters for vector DB and processing
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

    # Load or create vector database for retrieval
    vectordb = load_or_create_vectordb(dataset_name, **vectordb_config)
    eval_dataset = datasets.load_dataset("m-ric/huggingface_doc_qa_eval",
                                         split="train")

    # Initialize LLM (vLLM or Blablador, based on config)
    answer_llm, model_fullname = create_llm(config, role="answer")
    model_name = model_fullname.split("/")[-1]  # Short name for filenames
    print(f"Backend: {config['backend']}")
    print(f"The agentic RAG uses the following model: {model_fullname}\n")

    TEMPERATURE = config.get(config["backend"], {}).get("temperature", 0.2)

    # Setup Agentic RAG components
    retriever_tool = RetrieverTool(vectordb)
    agent = CodeAgent(
        tools=[retriever_tool],
        model=answer_llm,
        planning_interval=5,
        max_steps=30,
        verbosity_level=LogLevel.ERROR,
    )

    # Load custom system prompt for the agent
    prompt_filename = "guide_agent_system_prompt.yaml"
    prompt_path = Path("prompts") / prompt_filename
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_config = yaml.safe_load(f)

    print(
        f"Evaluating agentic RAG performance using model: {model_fullname}\n")

    # 1. Evaluate Agentic RAG
    llm_outputs = {}
    llm_outputs["agentic_rag"] = run_with_checkpoint(
        eval_dataset,
        lambda q: agentic_answer(q, agent, prompt_config),
        checkpoint_file=CHECKPOINTS_DIR / f"{model_name}_agentic_rag.json",
        model_name=model_fullname,
        prompt_name="guide_agent_system_prompt",
    )

    print(f"\n{'='*50}")
    print("Agentic RAG evaluation completed.")
    print(f"{'='*50}\n")

    # 2. Evaluate Standard RAG
    print(
        f"Evaluating standard RAG performance using model: {model_fullname}\n")
    llm_outputs["standard_rag"] = run_with_checkpoint(
        eval_dataset,
        lambda q: rag_answer(q, retriever_tool, answer_llm),
        checkpoint_file=CHECKPOINTS_DIR / f"{model_name}_rag.json",
        model_name=model_fullname,
        prompt_name="none",
    )

    print(f"\n{'='*50}")
    print("RAG evaluation completed.")
    print(f"{'='*50}\n")

    # 3. Evaluate Vanilla LLM
    print(f"Utilizing {model_fullname} for vanilla question answering...\n")
    llm_outputs["standard"] = run_with_checkpoint(
        eval_dataset,
        lambda q: vanilla_answer(q, answer_llm),
        checkpoint_file=CHECKPOINTS_DIR / f"{model_name}_vallina.json",
        model_name=model_fullname,
        prompt_name="none",
    )

    print(f"\n{'='*50}")
    print("Vanilla LLM evaluation completed.")
    print(f"{'='*50}\n")

    # Setup LLM-as-judge for results evaluation
    eval_prompt_filename = "evaluation_prompt.yaml"
    eval_prompt_path = Path("prompts") / eval_prompt_filename
    with open(eval_prompt_path, "r", encoding="utf-8") as f:
        evaluation_prompt = yaml.safe_load(f)

    evaluation_llm, eval_model_fullname = create_llm(config,
                                                     role="evaluation")
    eval_model_name = eval_model_fullname.split("/")[-1]
    print(f"Performance Evaluation Model: {eval_model_fullname}\n")

    # Perform evaluation of all system outputs
    evaluated = run_evaluation_with_checkpoint(
        llm_outputs,
        evaluation_prompt["prompt"],
        evaluation_llm,
        checkpoint_file=CHECKPOINTS_DIR / f"{model_name}_eval.json",
        delay=5,
        max_retries=5,
        max_consecutive_errors=3,
    )

    # Process and clean evaluated results
    results = {}
    for system_type, outputs in evaluated.items():
        df = pd.DataFrame.from_dict(outputs)
        # Remove entries that failed with an Error message
        results[system_type] = df.loc[~df["generated_answer"].str.
                                      contains("Error", na=False)]

    # Calculate and display accuracy scores
    DEFAULT_SCORE_VAL = 2  # Average score used when scoring fails
    for system_type in ["agentic_rag", "standard_rag", "standard"]:
        # Convert text scores to numeric and normalize
        results[system_type]["eval_score_LLM_judge_int"] = (
            results[system_type]["eval_score_LLM_judge"].fillna(
                DEFAULT_SCORE_VAL).apply(
                    lambda x: fill_score(x, DEFAULT_SCORE_VAL)))
        # Scale to percentage (assuming original scale is 1-3)
        results[system_type]["eval_score_LLM_judge_int"] = (
            results[system_type]["eval_score_LLM_judge_int"] - 1) / 2

        avg_score = results[system_type]['eval_score_LLM_judge_int'].mean(
        ) * 100
        print(f"Average score for {system_type} : {avg_score:.1f}%")

    print(f"{'='*50}\n")

    # Persist final evaluation results to disk
    meta_data = {
        "model_name": model_name,
        "model_id": model_fullname,
        "prompt_filename": prompt_filename,
        "eval_model_name": eval_model_name,
        "eval_model_id": eval_model_fullname,
    }
    eval_performance_filename = f"{model_name}_vect{vectordb_config['text_chunk_size']}_t{TEMPERATURE}.json"
    save_evaluation_results(meta_data, results, RESULTS_DIR,
                            eval_performance_filename)


if __name__ == "__main__":
    main()
