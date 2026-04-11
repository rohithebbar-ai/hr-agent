"""
Policy Agent Pipeline
─────────────────────
Class-based wrapper around the LangGraph graph.
Uses functools.partial for dependency injection.
"""

import os
from functools import partial
from pathlib import Path

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph

from scripts.llm_manager import LLMTask, get_llm
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
from rag.retriever import COLLECTION_POLICY_AWARE, get_retriever

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

class PolicyAgentPipeline:
    """LangGraph-based agentic RAG pipeline for HR policy questions."""
    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        
        # ── Get LLMs from the manager──
        self.routing_llm = get_llm(LLMTask.QUERY_ROUTING)
        self.chat_llm = get_llm(LLMTask.CHAT)
        self.decomp_llm = get_llm(LLMTask.QUERY_DECOMPOSITION)
        self.grading_llm = get_llm(LLMTask.DOCUMENT_GRADING)
        self.generation_llm = get_llm(LLMTask.GENERATION)
        self.grounding_llm = get_llm(LLMTask.GROUNDING_CHECK)

        # ── Retriever ──
        self.retriever = get_retriever(
            collection=COLLECTION_POLICY_AWARE,
            search_type="mmr",
            k=8,
        )

        self._graph = None

    def create_agent(self):
        """Build and compile the LangGraph agent."""
        if self._graph is not None:
            return self._graph

        if os.environ.get("LANGSMITH_API_KEY"):
            os.environ["LANGSMITH_TRACING_V2"] = "true"
            os.environ["LANGSMITH_PROJECT"] = os.environ.get(
                "LANGSMITH_PROJECT", "vanaciretain"
            )

        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        workflow = StateGraph(PolicyAgentState)

        # -- Each node gets its task-specific LLM --
        workflow.add_node(
            "route_query",
            partial(route_query_node, base_llm=self.routing_llm),
        )
        workflow.add_node(
            "chat_node",
            partial(chat_node, base_llm=self.chat_llm),
        )
        workflow.add_node(
            "transform_query",
            partial(transform_query, base_llm=self.decomp_llm),
        )
        workflow.add_node(
            "retrieve",
            partial(retrieve_node, retriever=self.retriever),
        )
        workflow.add_node(
            "grade_documents",
            partial(grade_documents_node, base_llm=self.grading_llm),
        )
        workflow.add_node(
            "generate",
            partial(generate_node, base_llm=self.generation_llm),
        )
        workflow.add_node(
            "check_grounding",
            partial(check_grounding_node, base_llm=self.grounding_llm),
        )

        workflow.add_edge(START, "route_query")

        memory = MemorySaver()
        graph = workflow.compile(checkpointer=memory)

        self._save_graph_image(graph)

        self._graph = graph
        return graph

    def _save_graph_image(self, graph):
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

    def run(self, query: str, thread_id: str = "default") -> str:
        graph = self.create_agent()
        config = {"configurable": {"thread_id": thread_id}}
        final_state = graph.invoke(
            {"question": query},
            config=config,
        )
        return final_state["answer"]