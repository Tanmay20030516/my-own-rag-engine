"""
Tests for the Streamlit app's non-UI logic.

Covers the two pieces with real edge cases behind them: the <think> filter,
which has to cope with tags split across streamed tokens, and the index
sizing rules that keep IVF/PQ from asking k-means for more clusters than
there are chunks. Document loading is checked for its dispatch and error
paths. Nothing here needs a live Ollama server.
"""

import numpy as np
import pytest

from frontend.loaders import DocumentLoadError, load_document
from frontend.rag_service import Settings, _build_index, _without_thinking, strip_thinking
from vectordb.index import FlatIndex, HNSWIndex, IVFIndex, PQIndex


def _filter(chunks: list[str]) -> str:
    return "".join(_without_thinking(iter(chunks)))


class TestThinkingFilter:

    def test_passes_through_text_without_tags(self):
        assert _filter(["Hello ", "world"]) == "Hello world"

    def test_removes_a_whole_think_block(self):
        assert _filter(["<think>reasoning</think>", "Answer"]) == "Answer"

    def test_removes_a_block_split_across_tokens(self):
        assert _filter(["<th", "ink>hmm</thi", "nk>Answer"]) == "Answer"

    def test_removes_multiple_blocks(self):
        assert _filter(["a", "<think>x</think>", "b", "<think>y</think>", "c"]) == "abc"

    def test_unclosed_block_suppresses_the_rest(self):
        assert _filter(["<think>never closed"]) == ""

    def test_keeps_a_lone_angle_bracket(self):
        assert _filter(["2 < 3 and ", "4 > 1"]) == "2 < 3 and 4 > 1"

    def test_strip_thinking_handles_multiline_blocks(self):
        assert strip_thinking("<think>a\nb</think>  Final") == "Final"


class TestIndexSizing:

    @pytest.mark.parametrize(
        "kind,expected",
        [
            ("Flat (exact)", FlatIndex),
            ("HNSW (graph)", HNSWIndex),
            ("IVF (clustered)", IVFIndex),
            ("PQ (compressed)", PQIndex),
        ],
    )
    def test_builds_the_selected_index_type(self, kind, expected):
        index = _build_index(Settings(index_kind=kind), n_chunks=500, warnings=[])
        assert isinstance(index, expected)

    def test_ivf_never_asks_for_more_clusters_than_chunks(self):
        warnings = []
        index = _build_index(Settings(index_kind="IVF (clustered)"), n_chunks=3, warnings=warnings)
        assert index.nlist <= 3
        assert index.nprobe <= index.nlist
        assert warnings, "clamping should be reported to the user"

    def test_pq_never_asks_for_more_centroids_than_chunks(self):
        warnings = []
        index = _build_index(Settings(index_kind="PQ (compressed)"), n_chunks=10, warnings=warnings)
        assert index.Ks <= 10
        assert warnings

    def test_full_size_pq_is_not_clamped_or_warned(self):
        warnings = []
        index = _build_index(Settings(index_kind="PQ (compressed)"), n_chunks=5000, warnings=warnings)
        assert index.Ks == 256
        assert warnings == []

    def test_clamped_ivf_index_still_trains_and_searches(self):
        vectors = np.random.default_rng(0).normal(size=(4, 16))
        index = _build_index(Settings(index_kind="IVF (clustered)"), n_chunks=4, warnings=[])
        index.train(vectors)
        index.add(vectors, np.arange(4))

        _, ids = index.search(vectors[0], k=2)
        assert len(ids) > 0


class TestDocumentLoading:

    def test_reads_plain_text(self):
        assert load_document("notes.txt", b"hello") == "hello"

    def test_reads_markdown(self):
        assert load_document("notes.md", b"# title") == "# title"

    def test_falls_back_when_utf8_decoding_fails(self):
        assert load_document("odd.txt", b"caf\xe9") == "caf\xe9"

    def test_rejects_unsupported_extensions(self):
        with pytest.raises(DocumentLoadError, match="Unsupported file type"):
            load_document("archive.zip", b"data")

    def test_rejects_files_with_no_text(self):
        with pytest.raises(DocumentLoadError, match="no extractable text"):
            load_document("blank.txt", b"   \n  ")

    def test_reports_unreadable_pdfs(self):
        with pytest.raises(DocumentLoadError, match="Could not read this PDF"):
            load_document("broken.pdf", b"not a pdf at all")
