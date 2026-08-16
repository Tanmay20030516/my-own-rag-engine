"""
Streamlit chat UI for the RAG engine.

Flow: upload one or more documents -> "Build index" chunks, embeds and
indexes them -> ask questions in the chat, with each answer showing the
passages it was grounded in.

Run with:
    streamlit run frontend/app.py

Needs a reachable Ollama server (OLLAMA_HOST, default http://localhost:11434)
with the embedding and chat models pulled.
"""

import sys
import time
from pathlib import Path

import streamlit as st

# The app lives in frontend/ but imports the engine from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from frontend import rag_service as svc  # noqa: E402
from frontend.loaders import SUPPORTED_EXTENSIONS, DocumentLoadError, load_document  # noqa: E402

st.set_page_config(page_title="RAG Engine", page_icon="📚", layout="wide")

WELCOME = (
    "Upload a document in the sidebar and build the index — then ask me anything about it."
)


def init_state() -> None:
    st.session_state.setdefault("settings", svc.Settings())
    st.session_state.setdefault("documents", {})  # filename -> extracted text
    st.session_state.setdefault("pipeline", None)
    st.session_state.setdefault("report", None)
    st.session_state.setdefault("messages", [])  # {role, content, sources?}
    st.session_state.setdefault("indexed_signature", None)


def current_signature(settings: svc.Settings) -> tuple:
    """Identifies the (documents + retrieval settings) an index was built from, to detect staleness."""
    return (
        tuple(sorted(st.session_state.documents)),
        settings.index_kind,
        settings.chunker_kind,
        settings.metric,
        settings.chunk_size,
        settings.chunk_overlap,
        settings.max_sentences,
        settings.embed_model,
    )


def reset_chat() -> None:
    st.session_state.messages = []


# --------------------------------------------------------------------------
# Sidebar: upload, ingest, and engine settings
# --------------------------------------------------------------------------


def render_sidebar() -> svc.Settings:
    settings: svc.Settings = st.session_state.settings

    with st.sidebar:
        st.title("📚 RAG Engine")
        st.caption("Vector search from scratch, answers from a local LLM.")

        st.subheader("1 · Documents")
        uploads = st.file_uploader(
            "Upload files",
            type=SUPPORTED_EXTENSIONS,
            accept_multiple_files=True,
            help="Text, markdown, PDF, or Word documents.",
        )

        if uploads:
            for upload in uploads:
                if upload.name in st.session_state.documents:
                    continue
                try:
                    st.session_state.documents[upload.name] = load_document(
                        upload.name, upload.getvalue()
                    )
                except DocumentLoadError as exc:
                    st.error(f"**{upload.name}** — {exc}")

        if st.session_state.documents:
            for name, text in st.session_state.documents.items():
                cols = st.columns([5, 1])
                cols[0].markdown(f"📄 `{name}` · {len(text.split()):,} words")
                if cols[1].button("✕", key=f"rm-{name}", help="Remove"):
                    del st.session_state.documents[name]
                    st.rerun()

        st.subheader("2 · Index")
        settings.index_kind = st.selectbox(
            "Index type",
            svc.INDEX_CHOICES,
            index=svc.INDEX_CHOICES.index(settings.index_kind),
            help="Flat is exact; the others trade recall for speed or memory.",
        )
        settings.chunker_kind = st.selectbox(
            "Chunking",
            svc.CHUNKER_CHOICES,
            index=svc.CHUNKER_CHOICES.index(settings.chunker_kind),
            help="Semantic chunking costs one embedding call per sentence.",
        )

        if settings.chunker_kind == "Fixed size":
            settings.chunk_size = st.slider("Chunk size (words)", 64, 1024, settings.chunk_size, 32)
            settings.chunk_overlap = st.slider(
                "Overlap (words)", 0, max(1, settings.chunk_size - 1), settings.chunk_overlap, 8
            )
        elif settings.chunker_kind == "Sentence windows":
            settings.max_sentences = st.slider(
                "Sentences per chunk", 1, 20, settings.max_sentences
            )

        build = st.button(
            "🔨 Build index",
            type="primary",
            use_container_width=True,
            disabled=not st.session_state.documents,
        )

        with st.expander("Retrieval & models"):
            settings.top_k = st.slider("Passages retrieved (k)", 1, 12, settings.top_k)
            settings.metric = st.selectbox(
                "Distance metric", ["cosine", "l2", "ip"], index=["cosine", "l2", "ip"].index(settings.metric)
            )
            settings.condense_followups = st.toggle(
                "Rewrite follow-up questions",
                value=settings.condense_followups,
                help="Turns 'what about the second one?' into a standalone query before searching.",
            )
            settings.embed_model = st.text_input("Embedding model", settings.embed_model)
            settings.llm_model = st.text_input("Chat model", settings.llm_model)
            settings.host = st.text_input("Ollama host", settings.host)

        render_connection_status(settings)

        if st.session_state.messages and st.button("Clear conversation", use_container_width=True):
            reset_chat()
            st.rerun()

    if build:
        run_ingest(settings)

    return settings


def render_connection_status(settings: svc.Settings) -> None:
    """Show whether Ollama is reachable and whether the configured models are pulled."""
    try:
        models = svc.list_models(settings.host)
    except Exception as exc:
        st.error(f"Ollama unreachable at `{settings.host}`.")
        st.caption(str(exc))
        return

    missing = [
        m
        for m in (settings.embed_model, settings.llm_model)
        # Ollama reports "name:tag"; a bare name in settings should still match.
        if not any(name == m or name.split(":")[0] == m.split(":")[0] for name in models)
    ]
    if missing:
        st.warning(f"Not pulled: {', '.join(f'`{m}`' for m in missing)}")
        st.code("\n".join(f"ollama pull {m}" for m in missing), language="bash")
    else:
        st.success(f"Ollama connected · {len(models)} models")


def run_ingest(settings: svc.Settings) -> None:
    """Chunk, embed, and index every uploaded document, replacing any previous index."""
    docs = st.session_state.documents
    status = st.status("Building index…", expanded=True)

    try:
        with status:
            st.write(f"Chunking and embedding {len(docs)} document(s)…")
            started = time.perf_counter()
            pipeline, report = svc.build_pipeline(settings, docs)
            elapsed = time.perf_counter() - started
            st.write(f"Indexed **{report.chunks:,} chunks** in {elapsed:.1f}s.")
    except Exception as exc:
        status.update(label="Indexing failed", state="error")
        st.sidebar.error(f"{type(exc).__name__}: {exc}")
        return

    status.update(label=f"Indexed {report.chunks:,} chunks in {elapsed:.1f}s", state="complete")

    st.session_state.pipeline = pipeline
    st.session_state.report = report
    st.session_state.indexed_signature = current_signature(settings)
    reset_chat()


# --------------------------------------------------------------------------
# Main pane: chat
# --------------------------------------------------------------------------


def render_sources(hits: list) -> None:
    if not hits:
        return
    with st.expander(f"📎 {len(hits)} source passage(s)"):
        for n, hit in enumerate(hits, start=1):
            st.markdown(f"**[{n}]** · chunk `{hit.id}` · distance `{hit.distance:.4f}`")
            st.caption(hit.text)


def render_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            render_sources(message.get("sources", []))


def handle_question(question: str, settings: svc.Settings) -> None:
    pipeline = st.session_state.pipeline

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        # History excluding the question just appended -- that goes in the prompt separately.
        history = st.session_state.messages[:-1]

        with st.spinner("Searching…"):
            search_query = question
            if settings.condense_followups and history:
                search_query = svc.condense_question(history, question, settings)
            hits = pipeline.retrieve(search_query, k=settings.top_k)

        if not hits:
            answer = "I couldn't find anything relevant in the indexed documents."
            st.markdown(answer)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": []}
            )
            return

        if search_query != question:
            st.caption(f"🔎 searched for: _{search_query}_")

        try:
            answer = st.write_stream(svc.stream_answer(question, hits, history, settings))
        except Exception as exc:
            answer = f"⚠️ Generation failed: `{type(exc).__name__}: {exc}`"
            st.error(answer)

        render_sources(hits)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": hits}
    )


def render_main(settings: svc.Settings) -> None:
    report = st.session_state.report

    if report:
        cols = st.columns(3)
        cols[0].metric("Documents", len(report.documents))
        cols[1].metric("Chunks indexed", f"{report.chunks:,}")
        cols[2].metric("Index", settings.index_kind.split(" ")[0])
        for warning in report.warnings:
            st.warning(warning, icon="⚠️")

        if st.session_state.indexed_signature != current_signature(settings):
            st.info(
                "Documents or chunking settings changed since the last build — "
                "rebuild the index to use them.",
                icon="🔄",
            )

    if st.session_state.pipeline is None:
        st.info(WELCOME, icon="👋")
        st.markdown(
            "**How it works** — your file is split into chunks, each chunk is embedded "
            "with a local Ollama model, and the vectors go into an index built from "
            "scratch in this repo (`vectordb/`). Your question is embedded the same way, "
            "the nearest chunks are retrieved, and a local LLM answers from them."
        )
        return

    render_history()

    if question := st.chat_input("Ask a question about your documents…"):
        handle_question(question, settings)


def main() -> None:
    init_state()
    settings = render_sidebar()
    render_main(settings)


main()
