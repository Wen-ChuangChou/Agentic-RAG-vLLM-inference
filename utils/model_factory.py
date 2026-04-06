"""
Model factory for creating LLM instances from different backends.

Supports:
  - "blablador": External Helmholtz Blablador API (existing behavior)
  - "vllm": Self-hosted vLLM OpenAI-compatible server on HPC

Both backends produce a smolagents.OpenAIServerModel, so the rest of the
codebase (CodeAgent, evaluation, etc.) works identically regardless of backend.
"""
import os
from smolagents import OpenAIServerModel
from utils.blablador_helper import BlabladorChatModel


def create_llm(config: dict,
               role: str = "answer") -> tuple:
    """
    Create an LLM instance based on configuration.

    Args:
        config: Parsed YAML configuration dictionary (from config.yaml).
        role: Which model to create:
              - "answer": The main QA / agent model.
              - "evaluation": The LLM-as-judge model.

    Returns:
        Tuple of (OpenAIServerModel instance, model_display_name: str).
    """
    if role == "evaluation":
        eval_cfg = config.get("evaluation", {})
        backend = eval_cfg.get("backend", config["backend"])
    else:
        backend = config["backend"]

    if backend == "vllm":
        return _create_vllm_model(config, role)
    elif backend == "blablador":
        return _create_blablador_model(config, role)
    else:
        raise ValueError(
            f"Unknown backend '{backend}'. "
            f"Supported backends: 'vllm', 'blablador'"
        )


def _create_vllm_model(config: dict, role: str) -> tuple:
    """Create an OpenAIServerModel pointing to a self-hosted vLLM server."""
    vllm_cfg = config.get("vllm", {})

    if role == "evaluation":
        section = config.get("evaluation", {})
        # Evaluation section can override vLLM defaults
        model_id = section.get("model_id", vllm_cfg["model_id"])
        api_base = section.get("api_base", vllm_cfg["api_base"])
        api_key = section.get("api_key", vllm_cfg.get("api_key", "EMPTY"))
        temperature = section.get("temperature", 0.0)
        max_tokens = section.get("max_tokens", vllm_cfg.get("max_tokens", 16384))
    else:
        model_id = vllm_cfg["model_id"]
        api_base = vllm_cfg["api_base"]
        api_key = vllm_cfg.get("api_key", "EMPTY")
        temperature = vllm_cfg.get("temperature", 0.2)
        max_tokens = vllm_cfg.get("max_tokens", 16384)

    llm = OpenAIServerModel(
        model_id=model_id,
        api_base=api_base,
        api_key=api_key,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return llm, model_id


def _create_blablador_model(config: dict, role: str) -> tuple:
    """Create an OpenAIServerModel pointing to the Blablador API."""
    api_key = os.getenv("Blablador_API_KEY")
    if not api_key:
        raise ValueError(
            "Blablador_API_KEY not found in environment variables. "
            "Please set it in your .env file."
        )

    blab_cfg = config.get("blablador", {})
    helper = BlabladorChatModel(api_key=api_key)

    if role == "evaluation":
        section = config.get("evaluation", {})
        model_name = section.get("model_name", blab_cfg["model_name"])
        temperature = section.get("temperature", 0.0)
        max_tokens = section.get("max_tokens", blab_cfg.get("max_tokens", 16384))
    else:
        model_name = blab_cfg["model_name"]
        temperature = blab_cfg.get("temperature", 0.2)
        max_tokens = blab_cfg.get("max_tokens", 16384)

    model_fullname = helper.get_model_fullname(model_name)
    api_base = blab_cfg.get(
        "api_base", "https://api.helmholtz-blablador.fz-juelich.de/v1"
    )

    llm = OpenAIServerModel(
        model_id=model_fullname,
        api_base=api_base,
        api_key=api_key,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return llm, model_fullname
