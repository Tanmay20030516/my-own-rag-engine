"""
Glue between the Streamlit UI and the RAG engine.

Owns three things the pipeline deliberately doesn't: building an index
from UI settings (including the sizing rules IVF/PQ need on small
documents), rewriting follow-up questions into standalone ones so
retrieval works mid-conversation, and streaming the answer token by token
so the UI can render it as it arrives.

Every Ollama call here goes through an explicit Client bound to
OLLAMA_HOST, so the same code works against a local server or the
`ollama` service in docker-compose.
"""

import os
import re
from collections.abc import Iterator
from dataclasses import dataclass, field

import ollama

from rag.chunker import chunk_fixed, chunk_semantic, chunk_sentences
from rag.embedder import OllamaEmbedder
from rag.pipeline import RAGPipeline, RetrievedChunk
from vectordb.index import FlatIndex, HNSWIndex, IVFIndex, PQIndex

DEFAULT_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_EMBED_MODEL = os.environ.get("RAG_EMBED_MODEL", "nomic-embed-text")
DEFAULT_LLM_MODEL = os.environ.get("RAG_LLM_MODEL", "qwen3:8b")

INDEX_CHOICES = ["Flat (exact)", "HNSW (graph)", "IVF (clustered)", "PQ (compressed)"]
CHUNKER_CHOICES = ["Sentence windows", "Fixed size", "Semantic"]

ANSWER_SYSTEM_PROMPT = (
    "You are a document question-answering assistant. Answer using only the "
    "numbered context passages provided. Cite the passages you used inline as "
    "[1], [2], and so on. If the context does not contain the answer, say so "
    "plainly instead of guessing. Keep answers concise and factual."
)

CONDENSE_SYSTEM_PROMPT = (
    "Rewrite the user's latest message into a single standalone search query "
    "that makes sense without the conversation history. Resolve pronouns and "
    "references to earlier turns. Reply with the rewritten query only -- no "
    "preamble, no quotes, no explanation."
)


@dataclass
class Settings:
    """Everything the sidebar can change about how retrieval and generation behave."""

    index_kind: str = INDEX_CHOICES[0]
    chunker_kind: str = CHUNKER_CHOICES[0]
    metric: str = "cosine"
    top_k: int = 4
    chunk_size: int = 256
    chunk_overlap: int = 32
    max_sentences: int = 5
    embed_model: str = DEFAULT_EMBED_MODEL
    llm_model: str = DEFAULT_LLM_MODEL
    host: str = DEFAULT_OLLAMA_HOST
    condense_followups: bool = True


@dataclass
class IngestReport:
    """What one ingest run produced, for the UI to summarise."""

    chunks: int = 0
    documents: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_THINK_OPEN, _THINK_CLOSE = "<think>", "</think>"


def get_client(host: str) -> ollama.Client:
    return ollama.Client(host=host)


def strip_thinking(text: str) -> str:
    """Drop <think>...</think> spans that reasoning models like qwen3 emit."""
    return _THINK_BLOCK_RE.sub("", text).strip()


def _without_thinking(tokens: Iterator[str]) -> Iterator[str]:
    """
    Filter <think>...</think> out of a token stream.

    The tags can straddle token boundaries, so text is held back whenever it
    could still turn out to be the start of a tag, and released once it
    can't. Suppressing the reasoning trace here rather than passing
    `think=False` to Ollama keeps this working on models that reject the
    option entirely.
    """
    buffer = ""
    inside = False

    for token in tokens:
        buffer += token
        while buffer:
            if inside:
                end = buffer.find(_THINK_CLOSE)
                if end == -1:
                    # Keep only what could still be a partial closing tag.
                    buffer = buffer[-len(_THINK_CLOSE) :]
                    break
                buffer = buffer[end + len(_THINK_CLOSE) :]
                inside = False
                continue

            start = buffer.find(_THINK_OPEN)
            if start != -1:
                if start:
                    yield buffer[:start]
                buffer = buffer[start + len(_THINK_OPEN) :]
                inside = True
                continue

            # No tag in hand: emit everything except a possible partial "<think".
            safe = len(buffer)
            for n in range(1, min(len(_THINK_OPEN), len(buffer)) + 1):
                if _THINK_OPEN.startswith(buffer[-n:]):
                    safe = len(buffer) - n
                    break
            if safe:
                yield buffer[:safe]
            buffer = buffer[safe:]
            break

    if buffer and not inside:
        yield buffer


def list_models(host: str) -> list[str]:
    """Names of models the server has pulled. Raises if the server is unreachable."""
    response = get_client(host).list()
    return sorted(m.model for m in response.models if m.model)


def _build_index(settings: Settings, n_chunks: int, warnings: list[str]):
    """
    Construct the index the user picked, sized for this document.

    IVF and PQ fit centroids over the corpus, so their parameters can't
    exceed what the data supports: k-means needs at least one vector per
    cluster. Rather than failing on a short upload, clamp and tell the user.
    """
    kind = settings.index_kind
    metric = settings.metric

    if kind.startswith("Flat"):
        return FlatIndex(metric=metric)

    if kind.startswith("HNSW"):
        return HNSWIndex(M=16, ef_construction=200, ef_search=64, metric=metric)

    if kind.startswith("IVF"):
        # A handful of clusters over a small document keeps lists non-empty;
        # sqrt(n) is the usual rule of thumb for how many to use.
        nlist = max(1, min(int(n_chunks**0.5), n_chunks))
        nprobe = max(1, min(4, nlist))
        if nlist < 100:
            warnings.append(
                f"IVF sized down to nlist={nlist}, nprobe={nprobe} -- only {n_chunks} chunks to cluster."
            )
        return IVFIndex(nlist=nlist, nprobe=nprobe, metric=metric)

    if kind.startswith("PQ"):
        # Ks centroids per subspace need >= Ks training vectors to be meaningful.
        ks = max(1, min(256, n_chunks))
        if ks < 256:
            warnings.append(
                f"PQ sized down to Ks={ks} -- only {n_chunks} chunks available to fit codebooks. "
                "Compression stats are not meaningful at this scale."
            )
        return PQIndex(M=8, Ks=ks, metric=metric)

    raise ValueError(f"unknown index kind {kind!r}")


def _build_chunker(settings: Settings, embedder: OllamaEmbedder):
    """
    Build the configured chunker, memoised on the document text.

    build_pipeline() chunks once to size the index and the pipeline chunks
    again to ingest. Semantic chunking costs one embedding call per sentence,
    so without the cache that whole pass would run twice.
    """
    kind = settings.chunker_kind

    if kind == "Fixed size":
        def chunk(text: str) -> list[str]:
            return chunk_fixed(text, chunk_size=settings.chunk_size, overlap=settings.chunk_overlap)
    elif kind == "Semantic":
        def chunk(text: str) -> list[str]:
            return chunk_semantic(text, embedder=embedder)
    else:
        def chunk(text: str) -> list[str]:
            return chunk_sentences(text, max_sentences=settings.max_sentences)

    cache: dict[str, list[str]] = {}

    def cached(text: str) -> list[str]:
        if text not in cache:
            cache[text] = chunk(text)
        return cache[text]

    return cached


def build_pipeline(settings: Settings, documents: dict[str, str]) -> tuple[RAGPipeline, IngestReport]:
    """
    Chunk, embed, and index every document, returning a ready-to-query pipeline.

    A fresh pipeline is built per ingest: index parameters depend on the
    corpus size, and the trained indexes can't be re-fitted in place.
    """
    report = IngestReport(documents=list(documents))
    if not documents:
        return _empty_pipeline(settings), report

    embedder = OllamaEmbedder(model=settings.embed_model, host=settings.host)
    chunker = _build_chunker(settings, embedder)

    # Count chunks up front so index parameters can be sized to the corpus
    # before anything is embedded.
    texts = list(documents.values())
    n_chunks = sum(len(chunker(text)) for text in texts)
    if n_chunks == 0:
        report.warnings.append("The uploaded files produced no chunks.")
        return _empty_pipeline(settings), report

    index = _build_index(settings, n_chunks, report.warnings)
    pipeline = RAGPipeline(
        index=index, embedder=embedder, chunker=chunker, llm_model=settings.llm_model
    )
    pipeline.index_documents(texts)

    report.chunks = len(pipeline._chunks)
    return pipeline, report


def _empty_pipeline(settings: Settings) -> RAGPipeline:
    return RAGPipeline(
        index=FlatIndex(metric=settings.metric),
        embedder=OllamaEmbedder(model=settings.embed_model, host=settings.host),
        llm_model=settings.llm_model,
    )


def condense_question(history: list[dict], question: str, settings: Settings) -> str:
    """
    Rewrite a follow-up into a standalone query.

    "What about the second one?" embeds to nothing useful on its own; the
    retriever needs the subject carried over from earlier turns. Falls back
    to the raw question if the rewrite fails or comes back empty, since a
    degraded query still beats no answer.
    """
    if not history:
        return question

    transcript = "\n".join(f"{turn['role']}: {turn['content']}" for turn in history[-6:])
    try:
        response = get_client(settings.host).chat(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": CONDENSE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Conversation so far:\n{transcript}\n\nLatest message: {question}",
                },
            ],
        )
    except Exception:
        return question

    rewritten = strip_thinking(response["message"]["content"])
    return rewritten or question


def format_context(hits: list[RetrievedChunk]) -> str:
    """Number the passages so the model can cite them as [1], [2], ..."""
    return "\n\n".join(f"[{n}] {hit.text}" for n, hit in enumerate(hits, start=1))


def stream_answer(
    question: str,
    hits: list[RetrievedChunk],
    history: list[dict],
    settings: Settings,
) -> Iterator[str]:
    """Yield answer text as it arrives, grounded in `hits` and aware of prior turns."""
    messages = [{"role": "system", "content": ANSWER_SYSTEM_PROMPT}]
    messages += [{"role": t["role"], "content": t["content"]} for t in history[-6:]]
    messages.append(
        {
            "role": "user",
            "content": f"Context passages:\n{format_context(hits)}\n\nQuestion: {question}",
        }
    )

    stream = get_client(settings.host).chat(
        model=settings.llm_model, messages=messages, stream=True
    )
    tokens = (part["message"]["content"] for part in stream if part["message"]["content"])
    yield from _without_thinking(tokens)
