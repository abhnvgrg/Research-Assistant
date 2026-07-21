"""
Tests for _extract_and_validate_citations — the post-generation
guard against edge case 4.1 (citation hallucination): the LLM cites
[N] for an N that doesn't correspond to any real graded chunk.
"""

from __future__ import annotations

from app.graph.nodes import _extract_and_validate_citations
from tests.conftest import make_chunk


def _graded(n: int) -> list[dict]:
    return [
        {**make_chunk(f"chunk-{i}", source=f"https://source-{i}.com"), "relevant": True, "grade_score": 0.9, "grade_reason": "ok"}
        for i in range(n)
    ]


def test_valid_citations_are_preserved():
    graded = _graded(3)
    answer = "Fact A is true [1]. Fact B follows from this [2] and [3]."

    cleaned, citations = _extract_and_validate_citations(answer, graded)

    assert cleaned == answer  # nothing stripped
    assert citations == {
        "1": "https://source-0.com",
        "2": "https://source-1.com",
        "3": "https://source-2.com",
    }


def test_hallucinated_citation_index_is_stripped():
    """The core edge-case-4.1 test: only 3 chunks exist, but the
    LLM cites [4]. [4] must be removed from the answer text and
    must never appear in the citations map."""
    graded = _graded(3)
    answer = "Fact A is well established [1][2][4]."

    cleaned, citations = _extract_and_validate_citations(answer, graded)

    assert "[4]" not in cleaned
    assert "4" not in citations
    assert citations == {"1": "https://source-0.com", "2": "https://source-1.com"}


def test_citation_map_only_includes_indices_actually_cited():
    """If graded has 5 chunks but the answer only cites [1] and
    [3], the citation map should not include 2, 4, 5 — no point
    showing sources the answer never referenced."""
    graded = _graded(5)
    answer = "Point one [1]. Point two [3]."

    _, citations = _extract_and_validate_citations(answer, graded)

    assert set(citations.keys()) == {"1", "3"}


def test_no_citations_in_answer_returns_empty_map():
    graded = _graded(2)
    answer = "This answer has no citations at all."

    cleaned, citations = _extract_and_validate_citations(answer, graded)

    assert cleaned == answer
    assert citations == {}


def test_empty_graded_list_strips_all_citations():
    """Defensive: if somehow called with graded=[] (shouldn't happen
    since synthesizer_node guards this earlier), every [N] is
    invalid by definition and must be stripped."""
    answer = "This claims something [1]."

    cleaned, citations = _extract_and_validate_citations(answer, [])

    assert "[1]" not in cleaned
    assert citations == {}
