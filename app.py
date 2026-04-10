"""
HR Policy Assistant — Streamlit UI
───────────────────────────────────
Wraps the policy-aware RAG chain. Single-turn for now —
proper multi-turn support comes with the LangGraph pipeline.
"""

import streamlit as st

from rag.policy_aware_rag import build_chain


# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="📋",
    layout="centered",
)

st.title("📋 HR Policy Assistant")
st.caption("Powered by Policy-Aware RAG · VanaciPrime")
st.divider()


# ── Session state ────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chain" not in st.session_state:
    with st.spinner(
        "Loading model and connecting to Qdrant… "
        "(first load may take a minute)"
    ):
        st.session_state.chain = build_chain()


# ── Render existing conversation ─────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ── Chat input ───────────────────────────────────────────────
question = st.chat_input("Ask an HR policy question…")

if question:
    # Show user message
    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })
    with st.chat_message("user"):
        st.markdown(question)

    # Stream the answer — send ONLY the current question
    # Conversation history is NOT used here because the chain
    # doesn't separate retrieval from generation. Sending history
    # would contaminate the vector search.
    with st.chat_message("assistant"):
        placeholder = st.empty()
        answer_chunks = []

        try:
            for chunk in st.session_state.chain.stream(question):
                answer_chunks.append(chunk)
                placeholder.markdown("".join(answer_chunks) + "▌")
            answer = "".join(answer_chunks)
            placeholder.markdown(answer)
        except Exception as exc:
            answer = f"⚠️ Error: {exc}"
            placeholder.markdown(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
    })


# ── Sidebar controls ─────────────────────────────────────────
with st.sidebar:
    st.header("Controls")

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.markdown("**Model:** `llama-3.1-70b-versatile`")
    st.markdown("**Retrieval:** Policy-Aware MMR (k=5)")
    st.markdown("**Embeddings:** `all-MiniLM-L6-v2`")
    st.markdown("**Vector DB:** Qdrant")

    st.divider()
    st.caption(
        "ℹ️ Single-turn mode. Each question is independent. "
        "Multi-turn conversation memory will be added with the "
        "LangGraph agentic pipeline."
    )
    st.divider()
    st.caption(
        "💡 Try asking:\n\n"
        "- How many vacation days do I get?\n"
        "- What is the drug testing policy?\n"
        "- How does FMLA work?"
    )