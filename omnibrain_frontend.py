"""
OmniBrain Streamlit frontend.

Run:
    streamlit run omnibrain_frontend.py

Start the backend separately:
    python omnibrain_backend.py
"""

from typing import Any, Dict, List

import requests
import streamlit as st


st.set_page_config(
    page_title="OMNIBRAIN - AGENTIC MULTI-MODAL RAG ORCHESTRATOR",
    page_icon="🧠",
    layout="wide",
)


# Small amount of CSS for a cleaner dashboard.
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.1rem;
    }

    .muted {
        color: #777;
        margin-bottom: 1rem;
    }

    .citation-box {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 12px;
        margin: 8px 0;
        background: #f8f9fa;
        color: #222;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Values stored here stay available during the Streamlit session.
if "api_url" not in st.session_state:
    st.session_state.api_url = "http://127.0.0.1:8000"

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def api_get(path: str, timeout: int = 10):
    """Send a GET request to the backend."""
    return requests.get(
        f"{st.session_state.api_url}{path}",
        timeout=timeout,
    )


def api_post(path: str, **kwargs):
    """Send a POST request to the backend."""
    return requests.post(
        f"{st.session_state.api_url}{path}",
        timeout=120,
        **kwargs,
    )


def backend_health() -> Dict[str, Any] | None:
    try:
        response = api_get("/health")

        if response.ok:
            return response.json()

    except requests.RequestException:
        pass

    return None


def upload_document(uploaded_file) -> Dict[str, Any] | None:
    try:
        response = api_post(
            "/upload",
            files={
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    "application/pdf",
                )
            },
        )

        if response.ok:
            return response.json()

        st.error(response.text)

    except requests.RequestException as exc:
        st.error(f"Upload failed: {exc}")

    return None


def get_documents() -> List[Dict[str, Any]]:
    try:
        response = api_get("/documents")

        if response.ok:
            return response.json()

    except requests.RequestException:
        pass

    return []


def get_stats() -> Dict[str, Any]:
    try:
        response = api_get("/stats")

        if response.ok:
            return response.json()

    except requests.RequestException:
        pass

    return {}


def query_backend(
    query: str,
    document_ids: List[str],
    include_reasoning: bool,
    threshold: float,
) -> Dict[str, Any] | None:
    try:
        response = api_post(
            "/query",
            json={
                "query": query,
                "document_ids": document_ids,
                "include_reasoning": include_reasoning,
                "confidence_threshold": threshold,
            },
        )

        if response.ok:
            return response.json()

        st.error(response.text)

    except requests.RequestException as exc:
        st.error(f"Query failed: {exc}")

    return None


# -------------------------
# Sidebar
# -------------------------

with st.sidebar:
    st.header("⚙️ Settings")

    st.session_state.api_url = st.text_input(
        "Backend URL",
        value=st.session_state.api_url,
        help="For local development use http://127.0.0.1:8000",
    ).rstrip("/")

    page = st.radio(
        "Page",
        ["Chat", "Documents", "Stats", "About"],
        index=0,
    )

    st.divider()

    health = backend_health()

    if health:
        st.success("Backend online")
        st.caption(
            f"v{health.get('version', '?')} · "
            f"{health.get('documents_loaded', 0)} documents"
        )

        if health.get("semantic_embeddings"):
            st.caption("Semantic embeddings: ON")
        else:
            st.warning(
                "Semantic embeddings: OFF. Install sentence-transformers "
                "for better retrieval."
            )
    else:
        st.error("Backend offline")
        st.caption("Start with: python omnibrain_backend.py")

    st.divider()
    st.caption("OmniBrain 2.0 · Local development build")


# -------------------------
# Main title
# -------------------------

st.markdown(
    '<div class="main-title">🧠 OmniBrain</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="muted">Agentic Multi-Modal RAG Document Intelligence</div>',
    unsafe_allow_html=True,
)


# -------------------------
# Chat page
# -------------------------

if page == "Chat":
    st.subheader("Ask your documents")

    documents = get_documents()

    if not documents:
        st.info("Upload a PDF from the Documents section to begin.")

    else:
        document_options = {
            document["filename"]: document["document_id"]
            for document in documents
        }

        selected_names = st.multiselect(
            "Search in",
            list(document_options.keys()),
            default=list(document_options.keys()),
        )

        selected_ids = [
            document_options[name]
            for name in selected_names
        ]

        threshold = st.slider(
            "Confidence threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.50,
            step=0.05,
        )

        show_trace = st.checkbox(
            "Show execution trace",
            value=False,
            help="Shows observable agent actions, not hidden reasoning.",
        )

        for item in st.session_state.chat_history:
            with st.chat_message(item["role"]):
                st.markdown(item["content"])

        prompt = st.chat_input(
            "Ask a question about the selected PDFs..."
        )

        if prompt:
            st.session_state.chat_history.append({
                "role": "user",
                "content": prompt,
            })

            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Retrieving and checking evidence..."):
                    result = query_backend(
                        prompt,
                        selected_ids,
                        show_trace,
                        threshold,
                    )

                if result:
                    st.markdown(result["answer"])

                    col1, col2, col3 = st.columns(3)
                    col1.metric(
                        "Confidence",
                        f"{result['confidence']:.0%}",
                    )
                    col2.metric(
                        "Time",
                        f"{result['processing_time']:.2f}s",
                    )
                    col3.metric(
                        "Citations",
                        len(result["citations"]),
                    )

                    if result.get("guardrails", {}).get("is_grounded"):
                        st.success("Grounding check passed.")
                    else:
                        st.warning(
                            "Grounding check needs verification."
                        )

                    if result.get("citations"):
                        st.markdown("### 📚 Sources")

                        for number, citation in enumerate(
                            result["citations"],
                            start=1,
                        ):
                            st.markdown(
                                f"""
                                <div class="citation-box">
                                <b>Source {number}</b><br>
                                <b>{citation.get('document_name', 'document.pdf')}</b>
                                · Page {citation.get('page', '?')}
                                · Relevance {citation.get('similarity', 0):.0%}
                                <br><br>
                                {citation.get('content', '')}
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                    if show_trace and result.get("execution_trace"):
                        st.markdown("### 🔎 Execution trace")

                        for step in result["execution_trace"]:
                            st.write(
                                f"**{step.get('step', 'step')}** — "
                                f"{step.get('action', '')}"
                            )

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": result["answer"],
                    })


# -------------------------
# Documents page
# -------------------------

elif page == "Documents":
    st.subheader("📄 Document ingestion")

    uploaded_file = st.file_uploader(
        "Upload a corporate financial PDF",
        type=["pdf"],
        help="The backend extracts the PDF before it becomes searchable.",
    )

    if uploaded_file is not None:
        if st.button("Process PDF", type="primary"):
            with st.spinner("Extracting PDF text and images..."):
                result = upload_document(uploaded_file)

            if result:
                st.success(
                    f"Processed {result['filename']} — "
                    f"{result['chunks']} text chunks, "
                    f"{result['images']} images."
                )

    st.divider()

    documents = get_documents()

    if documents:
        st.markdown("### Indexed documents")

        for document in documents:
            with st.expander(f"📄 {document['filename']}"):
                col1, col2, col3, col4 = st.columns(4)

                col1.metric("Pages", document.get("page_count", 0))
                col2.metric("Chunks", document.get("chunk_count", 0))
                col3.metric("Images", document.get("image_count", 0))
                col4.metric(
                    "Status",
                    document.get("status", "unknown"),
                )

                st.caption(
                    f"ID: {document['document_id']}"
                )

                st.caption(
                    f"Processing time: "
                    f"{document.get('processing_time', 0):.2f}s"
                )
    else:
        st.info("No documents indexed yet.")


# -------------------------
# Stats page
# -------------------------

elif page == "Stats":
    st.subheader("📊 System statistics")

    stats = get_stats()

    if stats:
        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Documents", stats.get("documents", 0))
        col2.metric("Chunks", stats.get("chunks", 0))
        col3.metric("Images", stats.get("images", 0))
        col4.metric("Queries", stats.get("queries", 0))

        if stats.get("semantic_embeddings"):
            st.success("Semantic retrieval is enabled.")
        else:
            st.warning(
                "Semantic retrieval is using the basic fallback. "
                "Install sentence-transformers and restart the backend."
            )

    else:
        st.info("Backend statistics are unavailable.")


# -------------------------
# About page
# -------------------------

else:
    st.subheader("About OmniBrain")

    st.markdown(
        """
        **OmniBrain** is a local document intelligence project with:

        - Agent-based query routing
        - Page-aware PDF ingestion
        - Semantic text retrieval
        - Basic structured-data analysis
        - PDF image detection
        - Self-RAG retry for weak retrieval
        - Grounding and citation checks
        - Page-level source information
        - FastAPI backend + Streamlit frontend

        **Note:** the current version does not invent chart values.
        A vision model is required before numerical chart or table
        interpretation can be claimed.
        """
    )
