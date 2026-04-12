"""
LLM Manager
───────────
Centralized LLM configuration.

All LLMs are defined here once. Other modules import named instances
instead of creating their own ChatGroq objects scattered throughout
the codebase.

Benefits:
- One place to swap models or change providers
- One place to tune temperature per task
- Easy A/B testing (point a node at a different model and re-run eval)
- Cost tracking per task type
"""

import os
from enum import Enum
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
from langchain_groq import ChatGroq


load_dotenv()

# ══════════════════════════════════════════════════
# API KEY ROUTING
# ══════════════════════════════════════════════════
# Two keys: one for runtime traffic, one for evaluation.
# Keeps RAGAS from burning through runtime quota.

GROQ_API_KEY_RUNTIME = os.environ.get("GROQ_API_KEY_4")
GROQ_API_KEY_EVAL = os.environ.get("GROQ_API_KEY_3")

def get_api_key_for_task(task: "LLMTask") -> str:
    """
    Pick the right API key based on task type.

    - Eval tasks (RAGAS) → Key 2
    - Everything else (runtime) → Key 1

    This isolation prevents evaluation runs from eating
    into the runtime token budget.
    """
    eval_tasks = {
        LLMTask.RAGAS_JUDGE,
        #LLMTask.DEEPEVAL_JUDGE,
        }

    if task in eval_tasks:
        if not GROQ_API_KEY_EVAL:
            # Fallback to runtime key if eval key not configured
            return GROQ_API_KEY_RUNTIME
        return GROQ_API_KEY_EVAL

    return GROQ_API_KEY_RUNTIME


# ══════════════════════════════════════════════════
# MODEL REGISTRY
# ══════════════════════════════════════════════════
# Single source of truth for all model IDs.
# When Groq deprecates a model, only update it here.

class ModelID:
    """Available Groq models. Update IDs here when Groq deprecates."""

    # ── Production models ──
    LLAMA_3_3_70B = "llama-3.3-70b-versatile"
    LLAMA_3_1_8B_INSTANT = "llama-3.1-8b-instant"
    GPT_OSS_120B = "openai/gpt-oss-120b"
    GPT_OSS_20B = "openai/gpt-oss-20b"
    LLAMA_GUARD_4_12B = "meta-llama/llama-guard-4-12b"
    LLAMA_4_SCOUT = "meta-llama/llama-4-scout-17b-16e-instruct"
    QWEN3_32B = "qwen/qwen3-32b"

    # ── Specialized ──
    WHISPER_LARGE_V3 = "whisper-large-v3"


# ══════════════════════════════════════════════════
# TASK TYPES
# ══════════════════════════════════════════════════
# Named tasks that map to specific model + config combinations.
# Each task can have different temperature, max_tokens, etc.

class LLMTask(str, Enum):
    """Tasks the agent performs. Each task gets its own LLM config."""

    # ── RAG pipeline tasks ──
    GENERATION = "generation"           # Final answer generation (needs reasoning)
    QUERY_DECOMPOSITION = "decomp"      # Multi-hop query splitting (needs reasoning)
    QUERY_ROUTING = "routing"           # CHAT vs RAG (simple classification)
    DOCUMENT_GRADING = "grading"        # Relevance scoring (simple classification)
    GROUNDING_CHECK = "grounding"       # Hallucination detection (simple)
    CHAT = "chat"                       # Direct conversational replies

    # ── Evaluation tasks ──
    RAGAS_JUDGE = "ragas_judge"         # RAGAS evaluation LLM

    # ── Future tasks ──
    CONTENT_MODERATION = "moderation"   # Safety check on user input


# ══════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════
# Maps each task to its model and parameters.
# Change models per task without touching node code.

TASK_CONFIG = {
    # ── Tasks needing reasoning power → big model ──
    LLMTask.GENERATION: {
        "model": ModelID.LLAMA_3_3_70B,
        "temperature": 0,
        "max_retries": 3,
    },
    LLMTask.QUERY_DECOMPOSITION: {
        "model": ModelID.LLAMA_3_3_70B,
        "temperature": 0,
        "max_retries": 3,
    },
    LLMTask.CHAT: {
        "model": ModelID.GPT_OSS_20B,
        "temperature": 0.6,  # Slightly creative for chat
        "max_retries": 5,
    },

    # ── Fast classification tasks → small model ──
    LLMTask.QUERY_ROUTING: {
        "model": ModelID.LLAMA_4_SCOUT,
        "temperature": 0,
        "max_retries": 3,
    },
    LLMTask.DOCUMENT_GRADING: {
        "model": ModelID.LLAMA_3_3_70B,
        "temperature": 0,
        "max_retries": 3,
    },
    LLMTask.GROUNDING_CHECK: {
        "model": ModelID.LLAMA_3_1_8B_INSTANT,
        "temperature": 0,
        "max_retries": 3,
    },

    # ── Safety ──
    LLMTask.CONTENT_MODERATION: {
        "model": ModelID.LLAMA_GUARD_4_12B,
        "temperature": 0,
        "max_retries": 3,
    },

     LLMTask.RAGAS_JUDGE: {
        "model": ModelID.LLAMA_3_1_8B_INSTANT,
        "temperature": 0,
        "max_retries": 5,
    },
}

# ══════════════════════════════════════════════════
# MANAGER
# ══════════════════════════════════════════════════

class LLMManager:
    """
    Centralized LLM factory. Get LLM instances by task name.

    Usage:
        manager = LLMManager()
        generation_llm = manager.get_llm(LLMTask.GENERATION)
        grading_llm = manager.get_llm(LLMTask.DOCUMENT_GRADING)

    LLMs are cached so the same task returns the same instance,
    avoiding repeated client creation overhead.
    """
    def __init__(self):
        if not GROQ_API_KEY_RUNTIME:
            raise ValueError(
                "GROQ_API_KEY not found. Set it in your .env file."
            )
        if not GROQ_API_KEY_EVAL:
            print(
                "[LLM_MANAGER] Warning: GROQ_API_KEY_2 not set. "
                "Evaluation will use the runtime key (shared quota)."
            )

    @lru_cache(maxsize=None)
    def get_llm(self, task: LLMTask) -> ChatGroq:
        """
        Get the configured LLM for a specific task.

        Cached so each task creates only one ChatGroq instance per process.
        """
        if task not in TASK_CONFIG:
            raise ValueError(
                f"Unknown task: {task}."
                f"Available: {list(TASK_CONFIG.keys())}"
            )
        config = TASK_CONFIG[task]
        api_key = get_api_key_for_task(task)

        return ChatGroq(
            api_key=api_key,
            model=config["model"],
            temperature=config["temperature"],
            max_retries=config["max_retries"],
        )

    def get_model_id(self, task: LLMTask) -> str:
        """Get the model ID for a task (useful for logging)."""
        return TASK_CONFIG[task]["model"]

    def list_tasks(self) -> dict:
        """Return a summary of all task → model mappings (for debugging)."""
        return {
            task.value: config["model"]
            for task, config in TASK_CONFIG.items()
        }

# ══════════════════════════════════════════════════
# MODULE-LEVEL SINGLETON
# ══════════════════════════════════════════════════
# Convenience: import this directly instead of creating new managers.


_manager: Optional[LLMManager] = None

def get_manager() -> LLMManager:
    """Get the global LLMManager singleton."""
    global _manager
    if _manager is None:
        _manager = LLMManager()
    return _manager


def get_llm(task: LLMTask) -> ChatGroq:
    """
    Shortcut to get an LLM for a task.

    Usage:
        from agents.llm_manager import get_llm, LLMTask
        llm = get_llm(LLMTask.GENERATION)
    """
    return get_manager().get_llm(task)

# ══════════════════════════════════════════════════
# CLI HELPER
# ══════════════════════════════════════════════════
# Run this file directly to see your current model configuration.

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  LLM Manager — Current Configuration")
    print("=" * 70 + "\n")

    manager = get_manager()
    config = manager.list_tasks()

    print(f"  {'Task':<22} {'Model':<32} {'Key':<10}")
    print("  " + "─" * 64)

    for task_name, model_id in config.items():
        # Reconstruct the LLMTask enum to check key routing
        task = LLMTask(task_name)
        key = get_api_key_for_task(task)
        # Show which key is being used (mask the actual value)
        if key == GROQ_API_KEY_EVAL:
            key_label = "Key 2 (eval)"
        else:
            key_label = "Key 1 (runtime)"

        print(f"  {task_name:<22} {model_id:<32} {key_label:<10}")

    print("\n" + "=" * 70)
    print(f"  Runtime key configured: {bool(GROQ_API_KEY_RUNTIME)}")
    print(f"  Eval key configured:    {bool(GROQ_API_KEY_EVAL)}")
    print("=" * 70 + "\n")

        
