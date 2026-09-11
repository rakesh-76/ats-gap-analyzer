"""Unit tests for pipeline.semantic (Stage 4: semantic gap analysis).

Uses a fake encoder with hand-picked, pre-normalized vectors instead of
the real sentence-transformers model, so cosine similarities are exact
and known in advance. This tests the actual classification and matrix
logic deterministically, without requiring network access to download
the real model — see the module docstring in semantic.py for why that
verification has to happen separately, on a machine with normal
internet access.
"""

from __future__ import annotations

import numpy as np
import pytest

from ats_gap_analyzer.pipeline.extract import Bullet, ExtractedDocument
from ats_gap_analyzer.pipeline.semantic import (
    STRONG_THRESHOLD,
    WEAK_THRESHOLD,
    MatchType,
    SemanticModel,
    analyze_gap,
)


class _FakeEncoder:
    """Maps known strings to fixed vectors instead of computing real
    embeddings, so tests can assert on exact, hand-computed similarities."""

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self._vectors = vectors

    def encode(self, sentences: list[str], **kwargs: object) -> "np.ndarray":
        return np.array([self._vectors[s] for s in sentences])


def _doc(*bullets_with_heading: tuple[str, bool]) -> ExtractedDocument:
    bullets = [Bullet(text=t, is_heading=h) for t, h in bullets_with_heading]
    return ExtractedDocument(full_text="", bullets=bullets, sections={})


def _plain_doc(*texts: str) -> ExtractedDocument:
    return _doc(*[(t, False) for t in texts])


def test_identical_vectors_are_classified_as_strong() -> None:
    vectors = {"req": [1.0, 0.0], "bullet": [1.0, 0.0]}
    model = SemanticModel(encoder=_FakeEncoder(vectors))

    results = analyze_gap(_plain_doc("bullet"), _plain_doc("req"), model)

    assert len(results) == 1
    assert results[0].requirement == "req"
    assert results[0].best_bullet == "bullet"
    assert results[0].similarity == pytest.approx(1.0)
    assert results[0].match_type == MatchType.STRONG


def test_moderately_similar_vectors_are_classified_as_weak() -> None:
    # cos(60 degrees) = 0.5, which sits between WEAK_THRESHOLD (0.35)
    # and STRONG_THRESHOLD (0.6).
    vectors = {"req": [1.0, 0.0], "bullet": [0.5, 0.8660254]}
    model = SemanticModel(encoder=_FakeEncoder(vectors))

    results = analyze_gap(_plain_doc("bullet"), _plain_doc("req"), model)

    assert results[0].similarity == pytest.approx(0.5)
    assert WEAK_THRESHOLD <= results[0].similarity < STRONG_THRESHOLD
    assert results[0].match_type == MatchType.WEAK


def test_orthogonal_vectors_are_classified_as_missing() -> None:
    vectors = {"req": [1.0, 0.0], "bullet": [0.0, 1.0]}
    model = SemanticModel(encoder=_FakeEncoder(vectors))

    results = analyze_gap(_plain_doc("bullet"), _plain_doc("req"), model)

    assert results[0].similarity == pytest.approx(0.0)
    assert results[0].match_type == MatchType.MISSING
    # Best bullet is still reported even when missing, for transparency
    # about what the closest (but insufficient) match was.
    assert results[0].best_bullet == "bullet"


def test_best_matching_bullet_is_selected_among_several() -> None:
    vectors = {
        "req": [1.0, 0.0],
        "unrelated": [0.0, 1.0],
        "somewhat": [0.5, 0.8660254],
        "best": [1.0, 0.0],
    }
    model = SemanticModel(encoder=_FakeEncoder(vectors))
    resume = _plain_doc("unrelated", "somewhat", "best")

    results = analyze_gap(resume, _plain_doc("req"), model)

    assert results[0].best_bullet == "best"
    assert results[0].match_type == MatchType.STRONG


def test_heading_bullets_are_excluded_from_both_sides() -> None:
    """Regression guard for the whole reason Stage 2 classified headings
    in the first place: a job-title line or a JD's internal label must
    never be compared as if it were real content."""
    vectors = {
        "job title heading": [1.0, 0.0],
        "real requirement": [1.0, 0.0],
        "jd label heading": [1.0, 0.0],
        "real achievement": [0.0, 1.0],
    }
    model = SemanticModel(encoder=_FakeEncoder(vectors))
    resume = _doc(("job title heading", True), ("real achievement", False))
    jd = _doc(("jd label heading", True), ("real requirement", False))

    results = analyze_gap(resume, jd, model)

    # Only the non-heading requirement should be compared, and only
    # against the non-heading resume bullet -- even though the heading
    # bullets would have been a "perfect" match if wrongly included.
    assert len(results) == 1
    assert results[0].requirement == "real requirement"
    assert results[0].best_bullet == "real achievement"
    assert results[0].match_type == MatchType.MISSING  # orthogonal vectors


def test_empty_job_description_returns_no_results() -> None:
    model = SemanticModel(encoder=_FakeEncoder({}))
    resume = _plain_doc("something")
    jd = ExtractedDocument(full_text="", bullets=[], sections={})

    assert analyze_gap(resume, jd, model) == []


def test_empty_resume_marks_every_requirement_missing_with_no_bullet() -> None:
    model = SemanticModel(encoder=_FakeEncoder({"req": [1.0, 0.0]}))
    resume = ExtractedDocument(full_text="", bullets=[], sections={})

    results = analyze_gap(resume, _plain_doc("req"), model)

    assert len(results) == 1
    assert results[0].match_type == MatchType.MISSING
    assert results[0].best_bullet is None
    assert results[0].similarity == 0.0


def test_resume_with_only_heading_bullets_behaves_like_empty_resume() -> None:
    model = SemanticModel(encoder=_FakeEncoder({"req": [1.0, 0.0], "title only": [1.0, 0.0]}))
    resume = _doc(("title only", True))

    results = analyze_gap(resume, _plain_doc("req"), model)

    assert results[0].match_type == MatchType.MISSING
    assert results[0].best_bullet is None
