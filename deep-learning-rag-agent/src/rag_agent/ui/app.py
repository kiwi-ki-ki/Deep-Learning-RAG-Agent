"""
app.py
======
Streamlit user interface for the Deep Learning RAG Interview Prep Agent.

Three-panel layout:
  - Left sidebar: Document ingestion and corpus browser
  - Centre: Document viewer
  - Right: Chat interface

API contract with the backend (agree this with Pipeline Engineer
before building anything):

  ingest(file_paths: list[Path]) -> IngestionResult
  list_documents() -> list[dict]
  get_document_chunks(source: str) -> list[DocumentChunk]
  chat(query: str, history: list[dict], filters: dict) -> AgentResponse

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

from pathlib import Path

from tempfile import TemporaryDirectory

from langchain_core.messages import HumanMessage

import streamlit as st

from rag_agent.agent.graph import get_compiled_graph
from rag_agent.agent.state import AgentResponse
from rag_agent.config import get_settings
from rag_agent.corpus.chunker import DocumentChunker
from rag_agent.vectorstore.store import VectorStoreManager


# ---------------------------------------------------------------------------
# Cached Resources
# ---------------------------------------------------------------------------
# Use st.cache_resource for objects that should persist across reruns
# and be shared across all user sessions. This prevents re-initialising
# ChromaDB and reloading the embedding model on every button click.


@st.cache_resource
def get_vector_store() -> VectorStoreManager:
    """
    Return the singleton VectorStoreManager.

    Cached so ChromaDB connection is initialised once per application
    session, not on every Streamlit rerun.
    """
    return VectorStoreManager()


@st.cache_resource
def get_chunker() -> DocumentChunker:
    """Return the singleton DocumentChunker."""
    return DocumentChunker()


@st.cache_resource
def get_graph():
    """Return the compiled LangGraph agent."""
    return get_compiled_graph()


# ---------------------------------------------------------------------------
# Session State Initialisation
# ---------------------------------------------------------------------------


def initialise_session_state() -> None:
    """
    Initialise all st.session_state keys on first run.

    Must be called at the top of main() before any UI is rendered.
    Without this, state keys referenced in callbacks will raise KeyError.

    Interview talking point: Streamlit reruns the entire script on every
    user interaction. session_state is the mechanism for persisting data
    (chat history, ingestion results) across reruns.
    """
    defaults = {
        "chat_history": [],           # list of {"role": "user"|"assistant", "content": str}
        "ingested_documents": [],     # list of dicts from list_documents()
        "selected_document": None,    # source filename currently in viewer
        "last_ingestion_result": None,
        "thread_id": "default-session",  # LangGraph conversation thread
        "topic_filter": None,
        "difficulty_filter": None,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


# ---------------------------------------------------------------------------
# Ingestion Panel (Sidebar)
# ---------------------------------------------------------------------------


def render_ingestion_panel(
    store: VectorStoreManager,
    chunker: DocumentChunker,
) -> None:
    """
    Render the document ingestion panel in the sidebar.

    Allows multi-file upload of PDF and Markdown files. Displays
    ingestion results (chunks added, duplicates skipped, errors).
    Updates the ingested documents list after successful ingestion.

    Parameters
    ----------
    store : VectorStoreManager
    chunker : DocumentChunker
    """
    st.sidebar.header("📂 Corpus Ingestion")

    # TODO: implement
    uploaded_files = st.sidebar.file_uploader(
        "Upload study materials",
        type=["pdf", "md"],
        accept_multiple_files=True
        )
    #
    # 2. "Ingest Documents" button — only enabled when files are selected
    #
    # 3. On button click:
    #    a. Save uploaded files to a temp directory
    #    b. chunker.chunk_files(file_paths)
    #    c. store.ingest(chunks) → IngestionResult
    #    d. Display result: st.success / st.warning / st.error
    #       Show: "{result.ingested} chunks added, {result.skipped} duplicates skipped"
    #    e. Refresh ingested documents list in session_state
    #
    # 4. Render ingested documents list below the uploader
    #    For each document: show source name, topic, chunk count
    #    Add a small "🗑 Remove" button per document that calls store.delete_document()

    local_corpus_files = sorted(Path(get_settings().corpus_dir).glob("*.md")) + sorted(
        Path(get_settings().corpus_dir).glob("*.pdf")
    )

    ingest_uploaded = st.sidebar.button(
        "Ingest Uploaded Files",
        disabled=not uploaded_files,
        use_container_width=True,
    )

    ingest_local = st.sidebar.button(
        "Ingest data/corpus Folder",
        disabled=not local_corpus_files,
        use_container_width=True,
    )

    result = None

    if ingest_uploaded and uploaded_files:
        with TemporaryDirectory() as tmpdir:
            file_paths = []
            for uploaded in uploaded_files:
                temp_path = Path(tmpdir) / uploaded.name
                temp_path.write_bytes(uploaded.getvalue())
                file_paths.append(temp_path)

            chunks = chunker.chunk_files(file_paths)
            result = store.ingest(chunks)

    if ingest_local and local_corpus_files:
        chunks = chunker.chunk_files(local_corpus_files)
        result = store.ingest(chunks)

    if result is not None:
        st.session_state["last_ingestion_result"] = result
        st.session_state["ingested_documents"] = store.list_documents()

        if result.errors:
            st.sidebar.error(
                f"{result.ingested} chunks added, "
                f"{result.skipped} duplicates skipped, "
                f"{len(result.errors)} errors"
            )
            for err in result.errors[:5]:
                st.sidebar.caption(err)
        elif result.ingested > 0:
            st.sidebar.success(
                f"{result.ingested} chunks added, "
                f"{result.skipped} duplicates skipped"
            )
        else:
            st.sidebar.warning(
                f"No new chunks added. {result.skipped} duplicates skipped."
            )

    st.sidebar.divider()
    st.sidebar.subheader("Ingested Documents")

    docs = store.list_documents()
    st.session_state["ingested_documents"] = docs

    if not docs:
        st.sidebar.info("No documents ingested yet.")
        return

    for doc in docs:
        col1, col2 = st.sidebar.columns([4, 1])
        with col1:
            st.caption(
                f"{doc['source']} | {doc['topic']} | {doc['chunk_count']} chunks"
            )
        with col2:
            if st.button("🗑", key=f"delete_{doc['source']}"):
                deleted = store.delete_document(doc["source"])
                st.sidebar.success(f"Deleted {deleted} chunks from {doc['source']}")
                st.session_state["ingested_documents"] = store.list_documents()
                st.rerun()


def render_corpus_stats(store: VectorStoreManager) -> None:
    """
    Render a compact corpus health summary in the sidebar.

    Shows total chunks, topics covered, and whether bonus topics
    are present. Used during Hour 3 to demonstrate corpus completeness.

    Parameters
    ----------
    store : VectorStoreManager
    """
    # TODO: implement
    # stats = store.get_collection_stats()
    # st.sidebar.metric("Total Chunks", stats["total_chunks"])
    # st.sidebar.write("Topics:", ", ".join(stats["topics"]))
    # if stats["bonus_topics_present"]:
    #     st.sidebar.success("✅ Bonus topics present")
    # else:
    #     st.sidebar.warning("⚠️ No bonus topics yet")
    stats = store.get_collection_stats()

    st.sidebar.divider()
    st.sidebar.subheader("📊 Corpus Stats")
    st.sidebar.metric("Total Chunks", stats["total_chunks"])

    if stats["topics"]:
        st.sidebar.write("Topics:", ", ".join(stats["topics"]))
    else:
        st.sidebar.write("Topics: none yet")

    if stats["bonus_topics_present"]:
        st.sidebar.success("✅ Bonus topics present")
    else:
        st.sidebar.warning("⚠️ No bonus topics yet")


# ---------------------------------------------------------------------------
# Document Viewer Panel (Centre)
# ---------------------------------------------------------------------------


def render_document_viewer(store: VectorStoreManager) -> None:
    """
    Render the document viewer in the main centre column.

    Displays a selectable list of ingested documents. When a document
    is selected, renders its chunk content in a scrollable pane.

    Parameters
    ----------
    store : VectorStoreManager
    """
    st.subheader("📄 Document Viewer")

    # TODO: implement
    # 1. If no documents ingested: show placeholder message
    #
    # 2. st.selectbox("Select document", options=[doc["source"] for doc in docs])
    #    Store selection in st.session_state["selected_document"]
    #
    # 3. On selection change: store.get_document_chunks(selected_source)
    #
    # 4. Render chunks in a scrollable container (st.container with fixed height)
    #    For each chunk:
    #    - Show metadata badge: topic | difficulty | type
    #    - Show chunk text
    #    - Show similarity score if this chunk was used in last response
    #
    # 5. Display chunk count and coverage summary below viewer
    docs = st.session_state.get("ingested_documents", [])
    if not docs:
        st.info("Ingest documents using the sidebar to view content here.")
        return

    options = [doc["source"] for doc in docs]

    if (
        st.session_state["selected_document"] is None
        or st.session_state["selected_document"] not in options
    ):
        st.session_state["selected_document"] = options[0]

    selected_source = st.selectbox(
        "Select document",
        options=options,
        index=options.index(st.session_state["selected_document"]),
    )
    st.session_state["selected_document"] = selected_source

    chunks = store.get_document_chunks(selected_source)

    st.caption(f"{len(chunks)} chunks in {selected_source}")

    viewer = st.container(height=500)
    with viewer:
        for i, chunk in enumerate(chunks, start=1):
            st.markdown(
                f"**Chunk {i}** — "
                f"`{chunk.metadata.topic}` | "
                f"`{chunk.metadata.difficulty}` | "
                f"`{chunk.metadata.type}`"
            )
            st.write(chunk.chunk_text)
            st.divider()


# ---------------------------------------------------------------------------
# Chat Interface Panel (Right)
# ---------------------------------------------------------------------------


def render_chat_interface(graph) -> None:
    """
    Render the chat interface in the right column.

    Supports multi-turn conversation with the LangGraph agent.
    Displays source citations with every response.
    Shows a clear "no relevant context" indicator when the
    hallucination guard fires.

    Parameters
    ----------
    graph : CompiledStateGraph
        The compiled LangGraph agent from get_compiled_graph().
    """
    st.subheader("💬 Interview Prep Chat")

    docs = st.session_state.get("ingested_documents", [])
    topic_options = ["All"] + sorted({doc["topic"] for doc in docs}) if docs else ["All"]
    diff_options = ["All", "beginner", "intermediate", "advanced"]

    col_topic, col_diff = st.columns(2)
    with col_topic:
        selected_topic = st.selectbox("Topic", topic_options)
        st.session_state["topic_filter"] = None if selected_topic == "All" else selected_topic

    with col_diff:
        selected_diff = st.selectbox("Difficulty", diff_options, index=2)
        st.session_state["difficulty_filter"] = None if selected_diff == "All" else selected_diff

    chat_container = st.container(height=400)
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("sources"):
                    with st.expander("📎 Sources"):
                        for source in message["sources"]:
                            st.caption(source)
                if message.get("no_context_found"):
                    st.warning("⚠️ No relevant content found in corpus.")

    query = st.chat_input("Ask about a deep learning topic...")

    if query:
        st.session_state.chat_history.append(
            {"role": "user", "content": query}
        )

        graph_input = {
            "messages": [HumanMessage(content=query)],
            "topic_filter": st.session_state["topic_filter"],
            "difficulty_filter": st.session_state["difficulty_filter"],
        }
        config = {"configurable": {"thread_id": st.session_state["thread_id"]}}

        try:
            result = graph.invoke(graph_input, config=config)
            response: AgentResponse = result["final_response"]

            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": response.answer,
                    "sources": response.sources,
                    "no_context_found": response.no_context_found,
                }
            )
        except Exception as e:
            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": f"An error occurred while generating a response: {str(e)}",
                    "sources": [],
                    "no_context_found": True,
                }
            )

        st.rerun()


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------


def main() -> None:
    """
    Application entry point.

    Sets page config, initialises session state, instantiates shared
    resources, and renders all UI panels.

    Run with: uv run streamlit run src/rag_agent/ui/app.py
    """
    settings = get_settings()

    st.set_page_config(
        page_title=settings.app_title,
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title(f"🧠 {settings.app_title}")
    st.caption(
        "RAG-powered interview preparation — built with LangChain, LangGraph, and ChromaDB"
    )

    initialise_session_state()

    # Instantiate shared backend resources
    store = get_vector_store()
    chunker = get_chunker()
    graph = get_graph()

    # Sidebar
    render_ingestion_panel(store, chunker)
    render_corpus_stats(store)

    # Main content area — two columns
    viewer_col, chat_col = st.columns([1, 1], gap="large")

    with viewer_col:
        render_document_viewer(store)

    with chat_col:
        render_chat_interface(graph)


if __name__ == "__main__":
    main()
