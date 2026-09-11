"""Unit tests for pipeline.keywords (Stage 3: keyword extraction)."""

from __future__ import annotations

import pytest

from ats_gap_analyzer.pipeline.extract import Bullet, ExtractedDocument
from ats_gap_analyzer.pipeline.keywords import (
    Keyword,
    extract_keywords,
    load_keyword_extractor,
)


@pytest.fixture(scope="module")
def extractor():
    # Loading the spaCy model is the expensive part of this stage, so it's
    # loaded once per test module rather than once per test.
    return load_keyword_extractor()


def _document(*bullet_texts: str) -> ExtractedDocument:
    bullets = [Bullet(text=t, is_heading=False) for t in bullet_texts]
    return ExtractedDocument(full_text="\n".join(bullet_texts), bullets=bullets, sections={})


def _by_text(keywords: list[Keyword]) -> dict[str, Keyword]:
    return {k.text: k for k in keywords}


class TestGazetteerMatching:
    def test_exact_match_is_found(self, extractor) -> None:
        doc = _document("Built a pipeline using Python and SQL.")

        keywords = extract_keywords(doc, extractor)
        gazetteer_texts = {k.text for k in keywords if k.is_gazetteer_match}

        assert "Python" in gazetteer_texts
        assert "SQL" in gazetteer_texts

    def test_matching_is_case_insensitive_with_canonical_casing(self, extractor) -> None:
        doc = _document("Experience with pytorch and TENSORFLOW.")

        keywords = _by_text([k for k in extract_keywords(doc, extractor) if k.is_gazetteer_match])

        # Canonical casing comes from the gazetteer file, not the source
        # text, so "pytorch"/"TENSORFLOW" as written still normalize to
        # the same string a differently-cased mention elsewhere would.
        assert "PyTorch" in keywords
        assert "TensorFlow" in keywords

    def test_multi_word_gazetteer_term_is_matched_as_one_keyword(self, extractor) -> None:
        doc = _document("Applied machine learning to solve the problem.")

        keywords = {k.text for k in extract_keywords(doc, extractor) if k.is_gazetteer_match}

        assert "Machine Learning" in keywords

    def test_same_keyword_across_bullets_is_deduplicated(self, extractor) -> None:
        doc = _document("Used Python for scripting.", "Also used Python for automation.")

        keywords = _by_text([k for k in extract_keywords(doc, extractor) if k.is_gazetteer_match])

        assert len(keywords["Python"].source_bullets) == 2
        assert "Used Python for scripting." in keywords["Python"].source_bullets
        assert "Also used Python for automation." in keywords["Python"].source_bullets

    def test_heading_bullets_are_still_scanned_for_keywords(self, extractor) -> None:
        """Regression guard: heading/title bullets (e.g. a project title
        listing its tech stack) must still contribute gazetteer keywords,
        even though they're excluded from later semantic matching."""
        doc = ExtractedDocument(
            full_text="",
            bullets=[Bullet(text="Side Project | Python, Docker", is_heading=True)],
            sections={},
        )

        keywords = {k.text for k in extract_keywords(doc, extractor) if k.is_gazetteer_match}

        assert "Python" in keywords
        assert "Docker" in keywords


class TestNounPhraseExtraction:
    def test_plain_noun_phrase_is_captured_as_candidate(self, extractor) -> None:
        doc = _document("The query performance issue was resolved.")

        keywords = {k.text for k in extract_keywords(doc, extractor) if not k.is_gazetteer_match}

        assert "query performance issue" in keywords

    def test_leading_determiner_is_stripped(self, extractor) -> None:
        doc = _document("Led the database migration project.")

        keywords = {k.text for k in extract_keywords(doc, extractor) if not k.is_gazetteer_match}

        assert "database migration project" in keywords
        assert "the database migration project" not in keywords

    def test_pure_stopword_chunk_is_dropped(self, extractor) -> None:
        doc = _document("It was a great success for the team.")

        keywords = [k.text for k in extract_keywords(doc, extractor) if not k.is_gazetteer_match]

        assert "it" not in keywords

    def test_gazetteer_span_is_not_duplicated_as_noun_phrase(self, extractor) -> None:
        """A noun chunk that IS a gazetteer match (e.g. "machine learning")
        should not also appear as a separate, lower-confidence noun-phrase
        candidate for the same span."""
        doc = _document("Applied machine learning to the dataset.")

        noun_phrase_texts = [
            k.text for k in extract_keywords(doc, extractor) if not k.is_gazetteer_match
        ]

        assert "machine learning" not in noun_phrase_texts

    def test_stray_symbols_are_stripped_from_noun_phrases(self, extractor) -> None:
        """Regression test for a real bug: spaCy's parser sometimes tags a
        stray symbol like "|" or "+" as NOUN/NUM rather than PUNCT (e.g. in
        "... | Coursera" or "5+ experience"), which must not leak into the
        cleaned keyword text."""
        doc = _document("Google Certificate | Coursera", "5+ years of experience.")

        noun_phrase_texts = [
            k.text for k in extract_keywords(doc, extractor) if not k.is_gazetteer_match
        ]

        assert not any("|" in text for text in noun_phrase_texts)
        assert not any("+" in text for text in noun_phrase_texts)


class TestEdgeCases:
    def test_empty_document_returns_no_keywords(self, extractor) -> None:
        doc = ExtractedDocument(full_text="", bullets=[], sections={})

        assert extract_keywords(doc, extractor) == []

    def test_bullet_with_no_recognizable_content_returns_no_keywords(self, extractor) -> None:
        doc = _document("ok")

        assert extract_keywords(doc, extractor) == []
