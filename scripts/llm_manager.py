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
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()

# ══════════════════════════════════════════════════
# API KEY ROUTING
# ══════════════════════════════════════════════════
# Two keys: one for runtime traffic, one for evaluation.
# Keeps RAGAS from burning through runtime quota.

GROQ_API_KEY_RUNTIME = os.environ.get("GROQ_API_KEY_4")
GROQ_API_KEY_EVAL = os.environ.get("GROQ_API_KEY_3")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")


class Provider(str, Enum):
    GROQ="groq"
    OPENAI="openai"
    GEMINI="gemini"

# ══════════════════════════════════════════════════
# MODEL REGISTRY
# ══════════════════════════════════════════════════
# Single source of truth for all model IDs.
# When Groq deprecates a model, only update it here.

class ModelID:
    """Available models across all providers."""

    # ── Production models ──
    LLAMA_3_3_70B = "llama-3.3-70b-versatile"
    LLAMA_3_1_8B_INSTANT = "llama-3.1-8b-instant"
    GPT_OSS_120B = "openai/gpt-oss-120b"
    GPT_OSS_20B = "openai/gpt-oss-20b"
    LLAMA_GUARD_4_12B = "meta-llama/llama-guard-4-12b"
    LLAMA_4_SCOUT = "meta-llama/llama-4-scout-17b-16e-instruct"
    QWEN3_32B = "qwen/qwen3-32b"

    # ── OpenAI ──
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4O = "gpt-4o"

    # ── Gemini ──
    GEMINI_2_5_FLASH = "gemini-2.5-flash"            # Best balance of speed + quality
    GEMINI_2_5_FLASH_LITE = "gemini-2.5-flash-lite"   # Cheapest, fastest, simple tasks
    GEMINI_2_5_PRO = "gemini-2.5-pro"                 # Complex reasoning (1M context)

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
        "provider": Provider.OPENAI,
        "model": ModelID.GPT_4O_MINI,
        "temperature": 0,
        "max_retries": 3,
    },
    LLMTask.QUERY_DECOMPOSITION: {
        "provider": Provider.OPENAI,
        "model": ModelID.GPT_4O_MINI,
        "temperature": 0,
        "max_retries": 3,
    },
    LLMTask.CHAT: {
        "provider": Provider.GEMINI,
        "model": ModelID.GEMINI_2_5_PRO,
        "temperature": 0.6,  # Slightly creative for chat
        "max_retries": 5,
    },

    # ── Fast classification tasks → small model ──
    LLMTask.QUERY_ROUTING: {
        "provider": Provider.GROQ,
        "model": ModelID.LLAMA_4_SCOUT,
        "temperature": 0,
        "max_retries": 3,
    },
    LLMTask.DOCUMENT_GRADING: {
        "provider": Provider.GEMINI,
        "model": ModelID.GEMINI_2_5_FLASH,
        "temperature": 0,
        "max_retries": 3,
    },
    LLMTask.GROUNDING_CHECK: {
        "provider": Provider.GEMINI,
        "model": ModelID.GEMINI_2_5_FLASH,
        "temperature": 0,
        "max_retries": 3,
    },

    # ── Safety ──
    LLMTask.CONTENT_MODERATION: {
        "provider": Provider.GROQ,
        "model": ModelID.LLAMA_GUARD_4_12B,
        "temperature": 0,
        "max_retries": 3,
    },

     LLMTask.RAGAS_JUDGE: {
        "provider": Provider.GEMINI,
        "model": ModelID.GEMINI_2_5_FLASH,
        "temperature": 0,
        "max_retries": 5,
    },
}


# ══════════════════════════════════════════════════
# LLM FACTORY
# ══════════════════════════════════════════════════

def _create_llm(provider: Provider, model: str, temperature:float, max_retries:int):
    """
    Create an LLM instance for given provider.
    All providers return langchain compatiable chat model.
    """
    if provider == Provider.GROQ:
        api_key = GROQ_API_KEY_RUNTIME
        return ChatGroq(
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_retries=max_retries,
        )
    elif provider == Provider.OPENAI:
        api_key = OPENAI_API_KEY
        return ChatOpenAI(
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_retries=max_retries,
        )
    elif provider == Provider.GEMINI:
        api_key = GOOGLE_API_KEY
        return ChatGoogleGenerativeAI(
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_retries=max_retries,
        )
    else:
        raise ValueError(f"Unknown Provider: {provider}")

# ══════════════════════════════════════════════════
# API KEY ROUTING (for Groq eval isolation)
# ══════════════════════════════════════════════════

def get_api_key_for_task(task: LLMTask) -> str:
    """Route Groq tasks to the right API key."""
    eval_tasks = {LLMTask.RAGAS_JUDGE}

    if task in eval_tasks:
        if not GROQ_API_KEY_EVAL:
            return GROQ_API_KEY_RUNTIME
        return GROQ_API_KEY_EVAL

    return GROQ_API_KEY_RUNTIME

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
        # Validate atleast one provider is configured.
        available = []
        if GROQ_API_KEY_RUNTIME:
            available.append("Groq")
        if OPENAI_API_KEY:
            available.append("OpenAI")
        if GOOGLE_API_KEY:
            available.append("Gemini")
        
        if not available:
            raise ValueError(
                "No LLM API keys found. Set atleast one of the keys: GROQ_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY "
            )
        print(f"[LLM_MANAGER] Available providers: {', '.join(available)}")

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
        provider = config["provider"]
        
        # For groq eval tasks use eval API key
        if provider == Provider.GROQ and task in {LLMTask.RAGAS_JUDGE}:
            api_key = GROQ_API_KEY_EVAL or GROQ_API_KEY_RUNTIME
            return ChatGroq(
                api_key=api_key,
                model=config["model"],
                temperature=config["temperature"],
                max_retries=config["max_retries"],
            )
        return _create_llm(
            provider = provider,
            model = config["model"],
            temperature=config["temperature"],
            max_retries=config["max_retries"],
        )

    def get_model_id(self, task: LLMTask) -> str:
        """Get the model ID for a task (useful for logging)."""
        return TASK_CONFIG[task]["model"]

    def list_tasks(self) -> dict:
        """Return a summary of all task → model mappings (for debugging)."""
        return {
            task.value: {
                "model": config["model"],
                "provider": config["provider"].value,
                }
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
 
    print(f"  {'Task':<22} {'Provider':<10} {'Model':<40}")
    print("  " + "─" * 64)

    for task_name, info in config.items():
        print(
            f"  {task_name:<22} "
            f"{info['provider']:<10} "
            f"{info['model']:<40}"
        )

    print("\n" + "=" * 75)
    print(f"  Groq Runtime:  {'configured' if GROQ_API_KEY_RUNTIME else 'missing'}")
    print(f"  Groq Eval:     {'configured' if GROQ_API_KEY_EVAL else 'missing'}")
    print(f"  OpenAI:        {'configured' if OPENAI_API_KEY else 'missing'}")
    print(f"  Gemini:        {'configured' if GOOGLE_API_KEY else 'missing'}")
    print("=" * 75 + "\n")