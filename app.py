"""
VanaciRetain HR Assistant — Streamlit UI
─────────────────────────────────────────
Multi-turn conversation with the LangGraph agentic RAG pipeline.
Captures LangSmith run IDs for thumbs up/down feedback.

Usage:
    streamlit run app.py
"""

import os
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_core.tracers.context import collect_runs
from langsmith import Client

from agents.pipeline import PolicyAgentPipeline

load_dotenv()

# ══════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════

st.set_page_config(
    page_title="VanaciRetain HR Assistant",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ══════════════════════════════════════════════════
# RESOURCES (cached singletons)
# ══════════════════════════════════════════════════

@st.cache_resource
def get_pipeline():
    """Build the agent pipeline once and reuse across messages."""
    pipeline = PolicyAgentPipeline()
    pipeline.create_agent()
    return pipeline


@st.cache_resource
def get_langsmith_client():
    """LangSmith client for pushing user feedback."""
    if not os.environ.get("LANGSMITH_API_KEY"):
        return None
    try:
        return Client()
    except Exception as e:
        print(f"[WARN] LangSmith client init failed: {e}")
        return None


# ══════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════

def init_session_state():
    """Initialize session state on first load."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "thread_id" not in st.session_state:
        # Unique thread per browser session for conversation memory
        st.session_state.thread_id = f"streamlit_{uuid.uuid4().hex[:8]}"

    if "feedback_given" not in st.session_state:
        # Track which messages have already received feedback
        # (so users can't double-vote on the same answer)
        st.session_state.feedback_given = set()


# ══════════════════════════════════════════════════
# FEEDBACK HANDLER
# ══════════════════════════════════════════════════

def submit_feedback(run_id: str, score: int, comment: str):
    """Push user thumbs up/down to LangSmith as feedback."""
    client = get_langsmith_client()

    if client is None:
        st.warning("LangSmith not configured — feedback not recorded")
        return False

    if not run_id:
        st.warning("No run ID available for this message")
        return False

    try:
        client.create_feedback(
            run_id=run_id,
            key="user_rating",
            score=score,
            comment=comment,
        )
        return True
    except Exception as e:
        st.error(f"Failed to submit feedback: {e}")
        return False


# ══════════════════════════════════════════════════
# AGENT CALL
# ══════════════════════════════════════════════════

def query_agent(question: str) -> tuple[str, str | None]:
    """
    Run the agent on a query and capture the LangSmith run ID.

    Returns:
        (answer, run_id) tuple. run_id is None if tracing is off.
    """
    pipeline = get_pipeline()

    try:
        with collect_runs() as cb:
            answer = pipeline.run(
                query=question,
                thread_id=st.session_state.thread_id,
            )

        # Capture the top-level LangGraph run ID for feedback
        run_id = (
            str(cb.traced_runs[0].id)
            if cb.traced_runs else None
        )

        return answer, run_id

    except Exception as e:
        return f"I encountered an error: {e}", None


# ══════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:
        st.title("VanaciRetain")
        st.caption("HR Policy Assistant")

        st.divider()

        st.subheader("About")
        st.write(
            "Ask questions about HR policies, benefits, leave, "
            "workplace conduct, and more. The assistant uses a "
            "LangGraph agentic RAG pipeline with query "
            "decomposition, document grading, and grounding "
            "checks."
        )

        st.divider()

        st.subheader("Try asking")
        example_questions = [
            "How many vacation days do I get?",
            "What is the probationary period?",
            "How does FMLA differ from personal leave?",
            "Does VanaciPrime conduct drug testing?",
            "What are the standard working hours?",
        ]
        for q in example_questions:
            if st.button(q, key=f"example_{q}", use_container_width=True):
                st.session_state.pending_question = q
                st.rerun()

        st.divider()

        # Session info
        with st.expander("Session Info"):
            st.code(f"Thread ID:\n{st.session_state.thread_id}")
            st.write(f"Messages: {len(st.session_state.messages)}")

            ls_client = get_langsmith_client()
            if ls_client:
                st.success("LangSmith: Connected")
            else:
                st.info("LangSmith: Not configured")

        st.divider()

        # Reset conversation
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.feedback_given = set()
            st.session_state.thread_id = (
                f"streamlit_{uuid.uuid4().hex[:8]}"
            )
            st.rerun()


# ══════════════════════════════════════════════════
# MESSAGE RENDERING
# ══════════════════════════════════════════════════

def render_message(message: dict, idx: int):
    """Render a single chat message with optional feedback buttons."""
    role = message["role"]
    content = message["content"]

    with st.chat_message(role):
        st.markdown(content)

        # Only show feedback buttons on assistant messages with run_id
        if role == "assistant" and message.get("run_id"):
            run_id = message["run_id"]
            already_voted = run_id in st.session_state.feedback_given

            if already_voted:
                st.caption("✓ Feedback recorded")
            else:
                col1, col2, col3 = st.columns([1, 1, 8])

                with col1:
                    if st.button(
                        "👍",
                        key=f"up_{idx}_{run_id}",
                        help="This answer was helpful",
                    ):
                        success = submit_feedback(
                            run_id=run_id,
                            score=1,
                            comment="User: helpful",
                        )
                        if success:
                            st.session_state.feedback_given.add(run_id)
                            st.toast("Thanks for your feedback!", icon="✅")
                            st.rerun()

                with col2:
                    if st.button(
                        "👎",
                        key=f"down_{idx}_{run_id}",
                        help="This answer was not helpful",
                    ):
                        success = submit_feedback(
                            run_id=run_id,
                            score=0,
                            comment="User: not helpful",
                        )
                        if success:
                            st.session_state.feedback_given.add(run_id)
                            st.toast(
                                "Feedback recorded — we'll improve",
                                icon="📝",
                            )
                            st.rerun()


# ══════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════

def main():
    init_session_state()
    render_sidebar()

    st.title("HR Policy Assistant")
    st.caption(
        "Ask about VanaciPrime HR policies, benefits, and procedures"
    )

    # Render conversation history
    for idx, message in enumerate(st.session_state.messages):
        render_message(message, idx)

    # Handle pending question from sidebar example buttons
    pending = st.session_state.pop("pending_question", None)

    # Chat input
    user_input = pending or st.chat_input(
        "Ask a question about HR policies..."
    )

    if user_input:
        # Add user message to history
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
        })

        # Render the new user message immediately
        with st.chat_message("user"):
            st.markdown(user_input)

        # Get agent response with spinner
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, run_id = query_agent(user_input)

            st.markdown(answer)

            # Show feedback buttons immediately for the new message
            if run_id:
                col1, col2, col3 = st.columns([1, 1, 8])

                idx = len(st.session_state.messages)

                with col1:
                    if st.button(
                        "👍",
                        key=f"up_new_{idx}_{run_id}",
                    ):
                        success = submit_feedback(
                            run_id=run_id,
                            score=1,
                            comment="User: helpful",
                        )
                        if success:
                            st.session_state.feedback_given.add(run_id)
                            st.toast("Thanks!", icon="✅")

                with col2:
                    if st.button(
                        "👎",
                        key=f"down_new_{idx}_{run_id}",
                    ):
                        success = submit_feedback(
                            run_id=run_id,
                            score=0,
                            comment="User: not helpful",
                        )
                        if success:
                            st.session_state.feedback_given.add(run_id)
                            st.toast("Feedback recorded", icon="📝")

        # Add assistant message to history with run_id
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "run_id": run_id,
        })


if __name__ == "__main__":
    main()