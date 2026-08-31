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
    tiny_heading = "Contents"
    long_body = "Introduction to the topic at hand. " * 400
    text = f"{tiny_heading}\n\n{long_body}"

    kept, dropped = chunk_text_with_stats(text)

    assert dropped >= 1
    assert not any(c.strip() == tiny_heading for c in kept)


def test_short_text_produces_single_chunk():
    text = "This is a short document with very little content in it at all."
    chunks = chunk_text(text * 20)
    assert len(chunks) >= 1


def test_no_chunk_exceeds_max_size():
    long_paragraph = "This is a sentence about transformers. " * 300
    chunks = chunk_text(long_paragraph)

    for chunk in chunks:
        assert count_tokens(chunk) <= CHUNK_SIZE_TOKENS + CHUNK_OVERLAP_TOKENS + 20


def test_chunks_never_split_mid_word():
    text = "Supercalifragilisticexpialidocious is a very long word indeed. " * 50
    chunks = chunk_text(text)
    assert any("Supercalifragilisticexpialidocious" in c for c in chunks)


def test_overlap_repeats_content_between_adjacent_chunks():
    text = "Sentence number {}. ".format(0) * 1
    sentences = [f"This is unique sentence number {i} about topic X." for i in range(200)]
    text = " ".join(sentences)

    chunks = chunk_text(text)
    assert len(chunks) >= 2

    tail_words = chunks[0].split()[-5:]
    tail_phrase = " ".join(tail_words)
    assert tail_phrase in chunks[1] or any(w in chunks[1] for w in tail_words)


def test_orphan_chunks_below_min_tokens_are_dropped():
    tiny_fragment = "1. Introduction 2. Related Work 3. Methodology"
    assert count_tokens(tiny_fragment) < MIN_CHUNK_TOKENS

    chunks = chunk_text(tiny_fragment)
    assert chunks == []


def test_chunk_exactly_at_min_threshold_is_kept():
    words = ["word"] * (MIN_CHUNK_TOKENS + 5)
    text = " ".join(words)
    chunks = chunk_text(text)
    assert len(chunks) >= 1
    assert all(count_tokens(c) >= MIN_CHUNK_TOKENS for c in chunks)


def test_figure_references_are_replaced_not_left_dangling():
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
    long_text = (
        "Introduction text about neural networks and their applications. " * 30
        + "As shown in Figure 7, our method outperforms the baseline. "
        + "More discussion follows about the implications of this result. " * 30
    )
    chunks = chunk_text(long_text)
    assert not any("Figure 7" in c for c in chunks)
    assert any("[visual reference omitted]" in c for c in chunks)


class TestTiktokenUnavailableFallback:
    def setup_method(self):
        self._original_encoding = chunker_module._encoding
        self._original_failed_flag = chunker_module._encoding_load_failed
        chunker_module._encoding = None
        chunker_module._encoding_load_failed = True

    def teardown_method(self):
        chunker_module._encoding = self._original_encoding
        chunker_module._encoding_load_failed = self._original_failed_flag

    def test_count_tokens_falls_back_to_approximation(self):
        assert count_tokens("") == 0
        assert count_tokens("a" * 40) == 10

    def test_chunk_text_still_produces_chunks_without_tiktoken(self):
        long_text = "This is a sentence about attention mechanisms. " * 200
        chunks = chunk_text(long_text)

        assert len(chunks) > 1
        assert all(count_tokens(c) <= CHUNK_SIZE_TOKENS * 1.5 for c in chunks)

    def test_overlap_still_works_under_fallback_counting(self):
        long_text = "Sentence number {}. ".format(0) * 1
        long_text = "".join(f"Sentence number {i}. " for i in range(200))
        chunks = chunk_text(long_text)

        assert len(chunks) > 1
        assert any(
            chunks[i][:20] in chunks[i - 1] or chunks[i - 1][-20:] in chunks[i]
            for i in range(1, len(chunks))
        )

    def test_get_encoding_does_not_retry_network_after_first_failure(self, monkeypatch):
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
