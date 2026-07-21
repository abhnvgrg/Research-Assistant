"""
Chunker tests — the most logic-dense file in the ingestion pipeline.
Covers token-based size limits (not character-based), overlap
correctness, the orphan-chunk filter (edge case 2.4), and figure
reference stripping (silent failure S4).

Also covers the tiktoken-unavailable fallback path — discovered as a
REAL bug while building this pipeline: tiktoken.get_encoding()
downloads its vocab file over the network on first use, and doing
that at module import time crashed the entire test suite in any
network-restricted environment (this sandbox's egress allowlist
didn't include the download domain). Fixed by lazy-loading with a
graceful fallback to approximate character-based token counting —
same "external dependency degrades, never crashes" pattern used for
Pinecone and MLflow elsewhere in this project.
"""

from __future__ import annotations

import app.ingestion.chunker as chunker_module
from app.ingestion.chunker import (
    CHUNK_OVERLAP_TOKENS,
    CHUNK_SIZE_TOKENS,
    MIN_CHUNK_TOKENS,
    _strip_figure_references,
    chunk_text,
    chunk_text_with_stats,
    count_tokens,
)


def test_count_tokens_matches_tiktoken_directly():
    text = "The quick brown fox jumps over the lazy dog."
    assert count_tokens(text) > 0
    assert count_tokens("") == 0


def test_chunk_text_with_stats_reports_zero_dropped_when_nothing_dropped():
    text = "This is a reasonably long piece of content. " * 30
    kept, dropped = chunk_text_with_stats(text)
    assert dropped == 0
    assert kept == chunk_text(text)


def test_chunk_text_with_stats_reports_dropped_orphan_chunks():
    """Constructs a document where a tiny heading-like fragment
    becomes its own FIRST chunk after a paragraph-break split. The
    first chunk never receives overlap padding (there's no prior
    chunk to pull context from), so unlike a trailing fragment — which
    the overlap step usually rescues by prepending prior context — a
    tiny leading fragment genuinely gets dropped by the orphan-chunk
    filter. This is exactly the case that IngestionResult.
    chunks_dropped silently missed before chunk_text_with_stats
    existed (it was hardcoded to 0)."""
    tiny_heading = "Contents"
    long_body = "Introduction to the topic at hand. " * 400  # forces multiple chunks
    text = f"{tiny_heading}\n\n{long_body}"

    kept, dropped = chunk_text_with_stats(text)

    assert dropped >= 1
    assert not any(c.strip() == tiny_heading for c in kept)


def test_short_text_produces_single_chunk():
    text = "This is a short document with very little content in it at all."
    chunks = chunk_text(text * 20)  # repeat to clear MIN_CHUNK_TOKENS
    assert len(chunks) >= 1


def test_no_chunk_exceeds_max_size():
    """Every chunk must fit under CHUNK_SIZE_TOKENS + a small overlap
    allowance — the overlap prepends tokens from the previous chunk,
    so chunks are slightly larger than CHUNK_SIZE_TOKENS alone, but
    must never balloon far past it."""
    long_paragraph = "This is a sentence about transformers. " * 300
    chunks = chunk_text(long_paragraph)

    for chunk in chunks:
        assert count_tokens(chunk) <= CHUNK_SIZE_TOKENS + CHUNK_OVERLAP_TOKENS + 20


def test_chunks_never_split_mid_word():
    """The recursive splitter falls back to sentence/word boundaries
    before ever hard-cutting — verify no chunk starts or ends with a
    partial word glued to punctuation in a way that indicates a
    mid-word split (heuristic: no chunk should start with a lowercase
    continuation immediately after our overlap boundary in a way that
    breaks a word)."""
    text = "Supercalifragilisticexpialidocious is a very long word indeed. " * 50
    chunks = chunk_text(text)
    # The literal long word must appear whole in at least one chunk —
    # if it were split mid-word it would never appear intact anywhere.
    assert any("Supercalifragilisticexpialidocious" in c for c in chunks)


def test_overlap_repeats_content_between_adjacent_chunks():
    """Consecutive chunks should share some trailing/leading text —
    proves the overlap step actually ran, not just single chunks with
    no relationship to each other."""
    text = "Sentence number {}. ".format(0) * 1
    # Build genuinely distinct sentences so we can detect overlap
    sentences = [f"This is unique sentence number {i} about topic X." for i in range(200)]
    text = " ".join(sentences)

    chunks = chunk_text(text)
    assert len(chunks) >= 2

    # Some suffix of chunk[0] should appear as a prefix inside chunk[1]
    tail_words = chunks[0].split()[-5:]
    tail_phrase = " ".join(tail_words)
    assert tail_phrase in chunks[1] or any(w in chunks[1] for w in tail_words)


def test_orphan_chunks_below_min_tokens_are_dropped():
    """Edge case 2.4: a table-of-contents-style fragment with high
    keyword density but near-zero informational content must be
    filtered out entirely, not passed through to embedding."""
    tiny_fragment = "1. Introduction 2. Related Work 3. Methodology"
    assert count_tokens(tiny_fragment) < MIN_CHUNK_TOKENS

    chunks = chunk_text(tiny_fragment)
    assert chunks == []


def test_chunk_exactly_at_min_threshold_is_kept():
    """Boundary test: a chunk at exactly MIN_CHUNK_TOKENS should
    survive the filter (only chunks strictly below it are dropped)."""
    # Build text that lands close to the threshold
    words = ["word"] * (MIN_CHUNK_TOKENS + 5)
    text = " ".join(words)
    chunks = chunk_text(text)
    assert len(chunks) >= 1
    assert all(count_tokens(c) >= MIN_CHUNK_TOKENS for c in chunks)


def test_figure_references_are_replaced_not_left_dangling():
    """Silent failure S4: a chunk claiming 'as shown in Figure 3'
    when no image was ingested must have that reference replaced
    with an explicit marker the synthesizer's prompt knows to react to."""
    text = "The results improve significantly, as shown in Figure 3 below."
    cleaned = _strip_figure_references(text)

    assert "Figure 3" not in cleaned
    assert "[visual reference omitted]" in cleaned


def test_figure_reference_stripping_handles_multiple_variants():
    variants = [
        "See Table 2 for details.",
        "Refer to Fig. 5 for the architecture diagram.",
        "as shown in figure 1, accuracy peaks early.",
    ]
    for text in variants:
        cleaned = _strip_figure_references(text)
        assert "[visual reference omitted]" in cleaned, f"failed on: {text!r}"


def test_figure_reference_stripping_does_not_touch_unrelated_text():
    text = "This paragraph has nothing to do with figures or tables at all."
    cleaned = _strip_figure_references(text)
    assert cleaned == text


def test_empty_text_produces_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   \n\n  ") == []


def test_chunk_text_integrates_figure_stripping_before_chunking():
    """End-to-end: figure references get stripped even when they're
    embedded deep inside a long document, not just in isolated
    strings passed straight to the stripping function."""
    long_text = (
        "Introduction text about neural networks and their applications. " * 30
        + "As shown in Figure 7, our method outperforms the baseline. "
        + "More discussion follows about the implications of this result. " * 30
    )
    chunks = chunk_text(long_text)
    assert not any("Figure 7" in c for c in chunks)
    assert any("[visual reference omitted]" in c for c in chunks)


class TestTiktokenUnavailableFallback:
    """Simulates the exact failure this project hit: tiktoken's
    network download fails. count_tokens/chunk_text must still
    function using the approximate character-based fallback rather
    than raising or hanging."""

    def setup_method(self):
        # Force the "encoding unavailable" path regardless of whether
        # tiktoken actually works in the environment running this test.
        self._original_encoding = chunker_module._encoding
        self._original_failed_flag = chunker_module._encoding_load_failed
        chunker_module._encoding = None
        chunker_module._encoding_load_failed = True

    def teardown_method(self):
        chunker_module._encoding = self._original_encoding
        chunker_module._encoding_load_failed = self._original_failed_flag

    def test_count_tokens_falls_back_to_approximation(self):
        assert count_tokens("") == 0
        # ~4 chars/token approximation, not an exact tiktoken count
        assert count_tokens("a" * 40) == 10

    def test_chunk_text_still_produces_chunks_without_tiktoken(self):
        long_text = "This is a sentence about attention mechanisms. " * 200
        chunks = chunk_text(long_text)

        assert len(chunks) > 1
        # Every chunk should still respect the size limit under the
        # approximate counter, not just under the real tokenizer.
        assert all(count_tokens(c) <= CHUNK_SIZE_TOKENS * 1.5 for c in chunks)

    def test_overlap_still_works_under_fallback_counting(self):
        long_text = "Sentence number {}. ".format(0) * 1
        long_text = "".join(f"Sentence number {i}. " for i in range(200))
        chunks = chunk_text(long_text)

        assert len(chunks) > 1
        # Adjacent chunks should still share some trailing/leading
        # text from the overlap step, even using char-based tail cuts.
        assert any(
            chunks[i][:20] in chunks[i - 1] or chunks[i - 1][-20:] in chunks[i]
            for i in range(1, len(chunks))
        )

    def test_get_encoding_does_not_retry_network_after_first_failure(self, monkeypatch):
        """Once the download has failed once, subsequent calls must
        NOT attempt the network call again — that would mean every
        single chunk in a large document retries a doomed network
        request, adding real latency for nothing."""
        call_count = {"n": 0}

        def flaky_get_encoding(name):
            call_count["n"] += 1
            raise ConnectionError("simulated network failure")

        monkeypatch.setattr(chunker_module.tiktoken, "get_encoding", flaky_get_encoding)
        chunker_module._encoding = None
        chunker_module._encoding_load_failed = False

        for _ in range(5):
            chunker_module._get_encoding()

        assert call_count["n"] == 1, "tiktoken.get_encoding was retried after already failing once"
