"""
LLM Manager
───────────
Centralized LLM configuration.

Production pipeline: All Groq — fast, reliable, no quota issues.
  - Heavy tasks (generation, decomp, chat) → 70B model, distributed across keys 1-3
  - Light tasks (routing, grading, grounding, moderation) → 8B model on key 4
  - Each task gets its own dedicated key to avoid rate limit contention

Evaluation (RAGAS): Gemini with Groq fallback — runs offline, separate key

LLMWithFallback extends LangChain's Runnable so it works transparently
with the pipe operator (|) and all LangChain chains.
"""
import os
from enum import Enum
from functools import lru_cache
from typing import Any, Iterator, List, Optional

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

load_dotenv()

# ══════════════════════════════════════════════════
# API KEYS
# ══════════════════════════════════════════════════

from core.secrets import get_secret

GROQ_API_KEY_1 = get_secret("GROQ_API_KEY")     # Generation — heaviest task
GROQ_API_KEY_2 = get_secret("GROQ_API_KEY_2")   # Decomposition — heavy task
GROQ_API_KEY_3 = get_secret("GROQ_API_KEY_3")   # Chat + RAGAS eval
GROQ_API_KEY_4 = get_secret("GROQ_API_KEY_4")   # Routing, grading, grounding, moderation
OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
GOOGLE_API_KEY = get_secret("GOOGLE_API_KEY")


# ══════════════════════════════════════════════════
# ENUMS
# ══════════════════════════════════════════════════

class Provider(str, Enum):
    GROQ = "groq"
    OPENAI = "openai"
    GEMINI = "gemini"


class ModelID:
    # Groq
    LLAMA_3_3_70B = "llama-3.3-70b-versatile"
    LLAMA_3_1_8B_INSTANT = "llama-3.1-8b-instant"
    LLAMA_GUARD_4_12B = "meta-llama/llama-guard-4-12b"

    # OpenAI
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4O = "gpt-4o"

    # Gemini
    GEMINI_FLASH = "gemini-flash-latest"
    GEMINI_2_5_FLASH_LITE = "gemini-2.5-flash-lite"


class LLMTask(str, Enum):
    GENERATION = "generation"
    QUERY_DECOMPOSITION = "decomp"
    QUERY_ROUTING = "routing"
    DOCUMENT_GRADING = "grading"
    GROUNDING_CHECK = "grounding"
    CHAT = "chat"
    RAGAS_JUDGE = "ragas_judge"
    CONTENT_MODERATION = "moderation"


# ══════════════════════════════════════════════════
# TASK CONFIG
# ══════════════════════════════════════════════════
# Production: all Groq, distributed across 4 keys by task weight.
# Test (CI): same config — Groq everywhere, no Gemini quota issues.
# RAGAS: Gemini with Groq fallback — runs offline, isolated key.

GROQ_FALLBACK_MODEL = ModelID.LLAMA_3_3_70B

TASK_CONFIG = {
    # ── Heavy tasks: 70B model, one key each ──────────────────────────

    LLMTask.GENERATION: {
        # Generates the final answer — most token-heavy call per request
        "provider": Provider.GROQ,
        "model": ModelID.LLAMA_3_3_70B,
        "api_key": GROQ_API_KEY_1,
        "temperature": 0,
        "max_retries": 3,
    },
    LLMTask.QUERY_DECOMPOSITION: {
        # Decomposes multi-hop questions — 70B for reasoning quality
        "provider": Provider.GROQ,
        "model": ModelID.LLAMA_3_3_70B,
        "api_key": GROQ_API_KEY_2,
        "temperature": 0,
        "max_retries": 3,
    },
    LLMTask.CHAT: {
        # Direct conversational responses — 70B for natural replies
        "provider": Provider.GROQ,
        "model": ModelID.LLAMA_3_3_70B,
        "api_key": GROQ_API_KEY_3,
        "temperature": 0.6,
        "max_retries": 3,
    },

    # ── Light tasks: 8B model, shared key 4 ──────────────────────────
    # These are binary classification tasks — 8B is more than sufficient

    LLMTask.QUERY_ROUTING: {
        # CHAT vs RAG classification — binary, fast
        "provider": Provider.GROQ,
        "model": ModelID.LLAMA_3_1_8B_INSTANT,
        "api_key": GROQ_API_KEY_4,
        "temperature": 0,
        "max_retries": 3,
    },
    LLMTask.DOCUMENT_GRADING: {
        # Relevance yes/no per document — binary, fast
        "provider": Provider.GROQ,
        "model": ModelID.LLAMA_3_1_8B_INSTANT,
        "api_key": GROQ_API_KEY_4,
        "temperature": 0,
        "max_retries": 3,
    },
    LLMTask.GROUNDING_CHECK: {
        # Grounded/ungrounded check — binary, fast
        "provider": Provider.GROQ,
        "model": ModelID.LLAMA_3_1_8B_INSTANT,
        "api_key": GROQ_API_KEY_4,
        "temperature": 0,
        "max_retries": 3,
    },
    LLMTask.CONTENT_MODERATION: {
        # Safety classification — specialized guard model
        "provider": Provider.GROQ,
        "model": ModelID.LLAMA_GUARD_4_12B,
        "api_key": GROQ_API_KEY_4,
        "temperature": 0,
        "max_retries": 3,
    },

    # ── Evaluation: Gemini with Groq fallback ─────────────────────────
    # Runs offline during eval pipeline, not in user-facing requests

    LLMTask.RAGAS_JUDGE: {
        "provider": Provider.GEMINI,
        "model": ModelID.GEMINI_2_5_FLASH_LITE,
        "api_key": None,                          # Uses GOOGLE_API_KEY
        "temperature": 0,
        "max_retries": 5,
        "fallback_model": GROQ_FALLBACK_MODEL,
        "fallback_api_key": GROQ_API_KEY_3,       # Shared with chat key
    },
}


# ══════════════════════════════════════════════════
# FACTORY HELPERS
# ══════════════════════════════════════════════════

def _create_groq(
    model: str,
    temperature: float,
    max_retries: int,
    api_key: str = None,
) -> ChatGroq:
    key = api_key or GROQ_API_KEY_4  # Default to key 4 if none specified
    return ChatGroq(
        api_key=key,
        model=model,
        temperature=temperature,
        max_retries=max_retries,
    )


def _create_openai(model: str, temperature: float, max_retries: int) -> ChatOpenAI:
    return ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model=model,
        temperature=temperature,
        max_retries=max_retries,
    )


def _create_gemini(model: str, temperature: float, max_retries: int) -> ChatGoogleGenerativeAI:
    try:
        return ChatGoogleGenerativeAI(
            api_key=GOOGLE_API_KEY,
            model=model,
            temperature=temperature,
            max_retries=max_retries,
        )
    except TypeError:
        return ChatGoogleGenerativeAI(
            api_key=GOOGLE_API_KEY,
            model=model,
            temperature=temperature,
        )


# ══════════════════════════════════════════════════
# FALLBACK WRAPPER
# ══════════════════════════════════════════════════

class LLMWithFallback(Runnable):
    """
    Wraps a primary LLM with a strict fallback.

    ANY exception from the primary triggers the fallback — no hardcoded
    error signatures. If the primary fails for any reason (429, 503,
    timeout, network error), the fallback handles it.

    Used for RAGAS judge (Gemini → Groq) and any future task
    that benefits from multi-provider resilience.

    Extends LangChain's Runnable so it works with the pipe operator (|)
    and with_structured_output() transparently.
    """

    def __init__(
        self,
        primary: BaseChatModel,
        fallback: BaseChatModel,
        task_name: str,
    ):
        self.primary = primary
        self.fallback = fallback
        self.task_name = task_name

    def with_structured_output(self, schema, **kwargs):
        """
        Override to preserve fallback through structured output chains.
        Without this, with_structured_output() delegates to primary via
        __getattr__, returning a chain that loses the fallback wrapper.
        """
        primary_structured = self.primary.with_structured_output(schema, **kwargs)
        fallback_structured = self.fallback.with_structured_output(schema, **kwargs)
        return LLMWithFallback(
            primary=primary_structured,
            fallback=fallback_structured,
            task_name=f"{self.task_name}:structured",
        )

    def invoke(
        self,
        input: Any,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Any:
        try:
            return self.primary.invoke(input, config=config, **kwargs)
        except Exception as e:
            print(
                f"[LLM_MANAGER] {self.task_name}: primary failed "
                f"({type(e).__name__}: {str(e)[:80]}) — using fallback"
            )
            return self.fallback.invoke(input, config=config, **kwargs)

    def stream(
        self,
        input: Any,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Iterator:
        try:
            yield from self.primary.stream(input, config=config, **kwargs)
        except Exception as e:
            print(
                f"[LLM_MANAGER] {self.task_name}: primary stream failed "
                f"({type(e).__name__}) — using fallback"
            )
            yield from self.fallback.stream(input, config=config, **kwargs)

    def batch(
        self,
        inputs: List[Any],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> List[Any]:
        try:
            return self.primary.batch(inputs, config=config, **kwargs)
        except Exception as e:
            print(
                f"[LLM_MANAGER] {self.task_name}: primary batch failed "
                f"({type(e).__name__}) — using fallback"
            )
            return self.fallback.batch(inputs, config=config, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.primary, name)


# ══════════════════════════════════════════════════
# MANAGER
# ══════════════════════════════════════════════════

class LLMManager:
    """
    Centralized LLM factory.

    Production pipeline: pure Groq — fast, reliable, rate limits spread
    across 4 keys. Each heavy task gets its own dedicated key.

    RAGAS evaluation: Gemini with Groq fallback — isolated from pipeline keys.
    """

    def __init__(self):
        available = []
        if any([GROQ_API_KEY_1, GROQ_API_KEY_2, GROQ_API_KEY_3, GROQ_API_KEY_4]):
            available.append("Groq")
        if OPENAI_API_KEY:
            available.append("OpenAI")
        if GOOGLE_API_KEY:
            available.append("Gemini")

        if not available:
            raise ValueError(
                "No LLM API keys found. Set at least one Groq key."
            )
        print(f"[LLM_MANAGER] Available providers: {', '.join(available)}")

        # Log which Groq keys are configured
        key_status = {
            "GROQ_API_KEY": bool(GROQ_API_KEY_1),
            "GROQ_API_KEY_2": bool(GROQ_API_KEY_2),
            "GROQ_API_KEY_3": bool(GROQ_API_KEY_3),
            "GROQ_API_KEY_4": bool(GROQ_API_KEY_4),
        }
        configured = [k for k, v in key_status.items() if v]
        missing = [k for k, v in key_status.items() if not v]
        if configured:
            print(f"[LLM_MANAGER] Groq keys configured: {', '.join(configured)}")
        if missing:
            print(f"[LLM_MANAGER] Groq keys missing: {', '.join(missing)}")

    @lru_cache(maxsize=None)
    def get_llm(self, task: LLMTask) -> Runnable:
        """
        Get the LLM for a task.

        Groq tasks: return model with dedicated API key for rate limit isolation.
        Gemini tasks (RAGAS): wrapped in LLMWithFallback for resilience.
        Cached per task — same instance reused across all requests.
        """
        if task not in TASK_CONFIG:
            raise ValueError(
                f"Unknown task: {task}. Available: {list(TASK_CONFIG.keys())}"
            )

        config = TASK_CONFIG[task]
        provider = config["provider"]
        model = config["model"]
        temperature = config["temperature"]
        max_retries = config["max_retries"]
        api_key = config.get("api_key")

        if provider == Provider.GROQ:
            return _create_groq(model, temperature, max_retries, api_key=api_key)

        elif provider == Provider.OPENAI:
            return _create_openai(model, temperature, max_retries)

        elif provider == Provider.GEMINI:
            primary = _create_gemini(model, temperature, max_retries)
            # Wrap with Groq fallback
            fallback_model = config.get("fallback_model", GROQ_FALLBACK_MODEL)
            fallback_key = config.get("fallback_api_key", GROQ_API_KEY_4)
            fallback = _create_groq(fallback_model, temperature, max_retries, api_key=fallback_key)
            return LLMWithFallback(
                primary=primary,
                fallback=fallback,
                task_name=task.value,
            )

        raise ValueError(f"Unknown provider: {provider}")

    def get_model_id(self, task: LLMTask) -> str:
        return TASK_CONFIG[task]["model"]

    def list_tasks(self) -> dict:
        return {
            task.value: {
                "model": config["model"],
                "provider": config["provider"].value,
                "api_key": "key_" + config.get("api_key", "")[-4:] if config.get("api_key") else "default",
                "has_fallback": "fallback_model" in config,
            }
            for task, config in TASK_CONFIG.items()
        }


# ══════════════════════════════════════════════════
# SINGLETON + SHORTCUTS
# ══════════════════════════════════════════════════

_manager: Optional[LLMManager] = None


def get_manager() -> LLMManager:
    global _manager
    if _manager is None:
        _manager = LLMManager()
    return _manager


def get_llm(task: LLMTask) -> Runnable:
    """
    Shortcut — get an LLM for a task.

    Usage:
        from scripts.llm_manager import get_llm, LLMTask
        llm = get_llm(LLMTask.GENERATION)
        response = llm.invoke("your prompt")
    """
    return get_manager().get_llm(task)


# ══════════════════════════════════════════════════
# CLI HELPER
# ══════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  LLM Manager — Current Configuration")
    print("=" * 70 + "\n")

    manager = get_manager()
    config = manager.list_tasks()

    print(f"  {'Task':<25} {'Provider':<10} {'Model':<38} {'Key':<12} {'Fallback'}")
    print("  " + "─" * 85)

    for task_name, info in config.items():
        fallback = "Groq" if info["has_fallback"] else "-"
        print(
            f"  {task_name:<25} "
            f"{info['provider']:<10} "
            f"{info['model']:<38} "
            f"{info['api_key']:<12} "
            f"{fallback}"
        )

    print("\n" + "=" * 70)
    print(f"  GROQ_API_KEY:    {'configured' if GROQ_API_KEY_1 else 'MISSING'}")
    print(f"  GROQ_API_KEY_2:  {'configured' if GROQ_API_KEY_2 else 'MISSING'}")
    print(f"  GROQ_API_KEY_3:  {'configured' if GROQ_API_KEY_3 else 'MISSING'}")
    print(f"  GROQ_API_KEY_4:  {'configured' if GROQ_API_KEY_4 else 'MISSING'}")
    print(f"  OpenAI:          {'configured' if OPENAI_API_KEY else 'missing'}")
    print(f"  Gemini:          {'configured' if GOOGLE_API_KEY else 'missing'}")
    print(f"  Environment:     {os.environ.get('ENVIRONMENT', 'prod')}")
    print("=" * 70 + "\n")