"""
Policy Agent Pipeline
─────────────────────
Class-based wrapper around the LangGraph graph.
Uses functools.partial for dependency injection.
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

import os
from functools import partial
from pathlib import Path

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph

from agents.nodes import (
    chat_node,
    check_grounding_node,
    generate_node,
    grade_documents_node,
    retrieve_node,
    route_query_node,
    transform_query,
)
from agents.schemas import PolicyAgentState

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class PolicyAgentPipeline:
    """LangGraph-based agentic RAG pipeline for HR policy questions."""

    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        self._graph = None
        # ── Nothing heavy in __init__ ──
        # LLMs, retriever, and graph are built lazily in create_agent()

    def create_agent(self):
        """Build and compile the LangGraph agent. Cached after first call."""
        if self._graph is not None:
            return self._graph

        # ── Import heavy dependencies only when first needed ──
        from scripts.llm_manager import LLMTask, get_llm

        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # ── LangSmith tracing (only if key exists) ──
        if os.environ.get("LANGSMITH_API_KEY") and os.environ.get("LANGSMITH_TRACING", "true").lower() == "true":
            os.environ["LANGSMITH_TRACING_V2"] = "true"
            os.environ["LANGSMITH_PROJECT"] = os.environ.get("LANGSMITH_PROJECT", "hragent")

        print("[PIPELINE] Loading LLMs...")
        routing_llm = get_llm(LLMTask.QUERY_ROUTING)
        chat_llm = get_llm(LLMTask.CHAT)
        decomp_llm = get_llm(LLMTask.QUERY_DECOMPOSITION)
        grading_llm = get_llm(LLMTask.DOCUMENT_GRADING)
        generation_llm = get_llm(LLMTask.GENERATION)
        grounding_llm = get_llm(LLMTask.GROUNDING_CHECK)

        print("[PIPELINE] Building LangGraph...")
        workflow = StateGraph(PolicyAgentState)

        workflow.add_node("route_query", partial(route_query_node, base_llm=routing_llm))
        workflow.add_node("chat_node", partial(chat_node, base_llm=chat_llm))
        workflow.add_node("transform_query", partial(transform_query, base_llm=decomp_llm))
        workflow.add_node("retrieve", retrieve_node)
        workflow.add_node("grade_documents", partial(grade_documents_node, base_llm=grading_llm))
        workflow.add_node("generate", partial(generate_node, base_llm=generation_llm))
        workflow.add_node("check_grounding", partial(check_grounding_node, base_llm=grounding_llm))

        workflow.add_edge(START, "route_query")

        memory = MemorySaver()
        graph = workflow.compile(checkpointer=memory)

        self._save_graph_image(graph)

        self._graph = graph
        print("[PIPELINE] Agent ready")
        return self._graph

    def _save_graph_image(self, graph):
        """Save graph visualization only in dev when explicitly enabled."""
        should_generate = (
            os.environ.get("ENVIRONMENT", "dev") == "dev"
            and os.environ.get("GENERATE_GRAPH_VIZ", "false") == "true"
        )
        if not should_generate:
            return

        try:
            outputs_dir = PROJECT_ROOT / "outputs"
            outputs_dir.mkdir(exist_ok=True)
            image_path = outputs_dir / "policy_agent_graph.png"
            png_bytes = graph.get_graph().draw_mermaid_png()
            with open(image_path, "wb") as f:
                f.write(png_bytes)
            print(f"[GRAPH] Saved visualization: {image_path}")
        except Exception as e:
            print(f"[GRAPH] Could not save visualization: {e}")

    def run(
        self,
        query: str,
        thread_id: str = "default",
        metadata: dict = None,
        tags: list = None,
    ) -> str:
        """Run a single query through the agent."""
        graph = self.create_agent()
        config = {
            "configurable": {"thread_id": thread_id},
            "metadata": metadata or {},
            "tags": tags or [],
        }
        final_state = graph.invoke({"question": query}, config=config)
        return final_state["answer"]