"""
Streamlit UI for the Policy Agent.
 
Features:
- Chat interface with conversation memory
- Thread ID per browser session
- Graph visualization in sidebar
"""
 
import uuid
from pathlib import Path
 
import streamlit as st
 
from agents.pipeline import PolicyAgentPipeline
 
 
# ── Page config ──
st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="📋",
    layout="centered",
)
 
st.title("📋 HR Policy Assistant")
st.caption("Powered by LangGraph Agentic RAG · VanaciPrime")
st.divider()
 
 
# ── Session state ──
if "messages" not in st.session_state:
    st.session_state.messages = []
 
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
 
if "pipeline" not in st.session_state:
    with st.spinner(
        "Initializing agent… (loading model, connecting to Qdrant)"
    ):
        st.session_state.pipeline = PolicyAgentPipeline(top_k=5)
        # Pre-compile graph
        st.session_state.pipeline.create_agent()
 
 
# ── Render conversation ──
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
 
 
# ── Chat input ──
question = st.chat_input("Ask an HR policy question…")
 
if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })
    with st.chat_message("user"):
        st.markdown(question)
 
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                answer = st.session_state.pipeline.run(
                    query=question,
                    thread_id=st.session_state.thread_id,
                )
            except Exception as exc:
                answer = f"⚠️ Error: {exc}"
 
        st.markdown(answer)
 
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
    })
 
 
# ── Sidebar ──
with st.sidebar:
    st.header("Controls")
 
    if st.button("🗑️ New conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()
 
    st.divider()
 
    # ── Graph visualization ──
    graph_path = Path("outputs/policy_agent_graph.png")
    if graph_path.exists():
        st.markdown("**Agent Architecture**")
        st.image(str(graph_path))
 
    st.divider()
 
    st.markdown("**Main Model:** `llama-3.1-70b-versatile`")
    st.markdown("**Grading Model:** `llama-3.1-8b-instant`")
    st.markdown("**Retrieval:** Policy-Aware MMR (k=5)")
    st.markdown("**Vector DB:** Qdrant")
    st.markdown("**Memory:** Per-session thread")
 
    st.divider()
 
    st.caption(
        "💡 Try multi-hop questions:\n\n"
        "- How does FMLA differ from personal leave?\n"
        "- If I exhaust PTO and sick leave, what options do I have?\n"
        "- What's the difference between harassment policies?"
    )
 