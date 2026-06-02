"""
LLM Manager
───────────
Centralized LLM configuration with strict Groq fallback.

If Gemini fails for ANY reason (rate limit, quota, 503, timeout, etc.),
the manager automatically retries with Groq. No hardcoded error signatures —
any exception from the primary triggers the fallback.

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

GROQ_API_KEY_RUNTIME = get_secret("GROQ_API_KEY_4")
GROQ_API_KEY_EVAL = get_secret("GROQ_API_KEY_3")
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
# In test environment: all Gemini tasks use Groq directly (no quota issues in CI).
# In production: Gemini primary with Groq fallback on any failure.

GROQ_FALLBACK_MODEL = ModelID.LLAMA_3_3_70B
GROQ_FAST_FALLBACK_MODEL = ModelID.LLAMA_3_1_8B_INSTANT

_IS_TEST = os.environ.get("ENVIRONMENT") == "test"

TASK_CONFIG = {
    LLMTask.GENERATION: {
        "provider": Provider.GROQ,
        "model": ModelID.LLAMA_3_3_70B,
        "temperature": 0,
        "max_retries": 3,
    },
    LLMTask.QUERY_DECOMPOSITION: {
        "provider": Provider.GROQ if _IS_TEST else Provider.GEMINI,
        "model": ModelID.LLAMA_3_3_70B if _IS_TEST else ModelID.GEMINI_2_5_FLASH_LITE,
        "temperature": 0,
        "max_retries": 3,
        "fallback_model": GROQ_FALLBACK_MODEL,
    },
    LLMTask.CHAT: {
        "provider": Provider.GROQ if _IS_TEST else Provider.GEMINI,
        "model": ModelID.LLAMA_3_3_70B if _IS_TEST else ModelID.GEMINI_FLASH,
        "temperature": 0.6,
        "max_retries": 3,
        "fallback_model": GROQ_FALLBACK_MODEL,
    },
    LLMTask.QUERY_ROUTING: {
        "provider": Provider.GROQ,
        "model": ModelID.LLAMA_3_1_8B_INSTANT,
        "temperature": 0,
        "max_retries": 3,
    },
    LLMTask.DOCUMENT_GRADING: {
        "provider": Provider.GROQ,
        "model": ModelID.LLAMA_3_1_8B_INSTANT,
        "temperature": 0,
        "max_retries": 3,
    },
    LLMTask.GROUNDING_CHECK: {
        "provider": Provider.GROQ if _IS_TEST else Provider.GEMINI,
        "model": ModelID.LLAMA_3_1_8B_INSTANT if _IS_TEST else ModelID.GEMINI_FLASH,
        "temperature": 0,
        "max_retries": 3,
        "fallback_model": GROQ_FAST_FALLBACK_MODEL,
    },
    LLMTask.CONTENT_MODERATION: {
        "provider": Provider.GROQ,
        "model": ModelID.LLAMA_GUARD_4_12B,
        "temperature": 0,
        "max_retries": 3,
    },
    LLMTask.RAGAS_JUDGE: {
        "provider": Provider.GEMINI,
        "model": ModelID.GEMINI_2_5_FLASH_LITE,
        "temperature": 0,
        "max_retries": 5,
        "fallback_model": GROQ_FALLBACK_MODEL,
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
    return ChatGroq(
        api_key=api_key or GROQ_API_KEY_RUNTIME,
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


def _create_llm(
    provider: Provider,
    model: str,
    temperature: float,
    max_retries: int,
) -> BaseChatModel:
    if provider == Provider.GROQ:
        return _create_groq(model, temperature, max_retries)
    elif provider == Provider.OPENAI:
        return _create_openai(model, temperature, max_retries)
    elif provider == Provider.GEMINI:
        return _create_gemini(model, temperature, max_retries)
    raise ValueError(f"Unknown provider: {provider}")


# ══════════════════════════════════════════════════
# FALLBACK WRAPPER
# ══════════════════════════════════════════════════

class LLMWithFallback(Runnable):
    """
    Wraps a primary LLM (Gemini) with a strict Groq fallback.

    ANY exception from the primary triggers the fallback — no hardcoded
    error signatures. If Gemini fails for any reason (429, 503, timeout,
    network error, etc.), Groq handles it instead.

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
        __getattr__, returning a pure Gemini chain that loses the fallback.
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
                f"({type(e).__name__}: {str(e)[:80]}) — using Groq fallback"
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
                f"({type(e).__name__}) — using Groq fallback"
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
                f"({type(e).__name__}) — using Groq fallback"
            )
            return self.fallback.batch(inputs, config=config, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.primary, name)


# ══════════════════════════════════════════════════
# MANAGER
# ══════════════════════════════════════════════════

class LLMManager:
    """
    Centralized LLM factory. Gemini tasks get strict Groq fallback —
    any Gemini failure (quota, 503, timeout) automatically uses Groq.
    """

    def __init__(self):
        available = []
        if GROQ_API_KEY_RUNTIME:
            available.append("Groq")
        if OPENAI_API_KEY:
            available.append("OpenAI")
        if GOOGLE_API_KEY:
            available.append("Gemini")

        if not available:
            raise ValueError(
                "No LLM API keys found. Set at least one of: "
                "GROQ_API_KEY_4, OPENAI_API_KEY, GOOGLE_API_KEY"
            )
        print(f"[LLM_MANAGER] Available providers: {', '.join(available)}")

    @lru_cache(maxsize=None)
    def get_llm(self, task: LLMTask) -> Runnable:
        """
        Get the LLM for a task.
        Gemini tasks are wrapped in LLMWithFallback — any failure uses Groq.
        Groq/OpenAI tasks return the model directly.
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

        # RAGAS eval uses dedicated eval API key
        if provider == Provider.GROQ and task == LLMTask.RAGAS_JUDGE:
            api_key = GROQ_API_KEY_EVAL or GROQ_API_KEY_RUNTIME
            return _create_groq(model, temperature, max_retries, api_key=api_key)

        primary = _create_llm(provider, model, temperature, max_retries)

        # Wrap Gemini tasks with strict Groq fallback
        if provider == Provider.GEMINI and GROQ_API_KEY_RUNTIME:
            fallback_model = config.get("fallback_model", GROQ_FALLBACK_MODEL)
            fallback = _create_groq(fallback_model, temperature, max_retries)
            return LLMWithFallback(
                primary=primary,
                fallback=fallback,
                task_name=task.value,
            )

        return primary

    def get_model_id(self, task: LLMTask) -> str:
        return TASK_CONFIG[task]["model"]

    def list_tasks(self) -> dict:
        return {
            task.value: {
                "model": config["model"],
                "provider": config["provider"].value,
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
        llm = get_llm(LLMTask.CHAT)
        chain = prompt | llm | StrOutputParser()
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

    print(f"  {'Task':<25} {'Provider':<10} {'Model':<38} {'Fallback'}")
    print("  " + "─" * 75)

    for task_name, info in config.items():
        fallback = "Groq" if info["has_fallback"] else "-"
        print(
            f"  {task_name:<25} "
            f"{info['provider']:<10} "
            f"{info['model']:<38} "
            f"{fallback}"
        )

    print("\n" + "=" * 70)
    print(f"  Groq Runtime:  {'configured' if GROQ_API_KEY_RUNTIME else 'MISSING'}")
    print(f"  Groq Eval:     {'configured' if GROQ_API_KEY_EVAL else 'missing'}")
    print(f"  OpenAI:        {'configured' if OPENAI_API_KEY else 'missing'}")
    print(f"  Gemini:        {'configured' if GOOGLE_API_KEY else 'missing'}")
    print(f"  Environment:   {os.environ.get('ENVIRONMENT', 'prod')}")
    print("=" * 70 + "\n")