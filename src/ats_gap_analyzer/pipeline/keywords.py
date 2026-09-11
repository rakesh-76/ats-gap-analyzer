"""Stage 3 of the ATS Gap Analyzer pipeline: keyword extraction.

Extracts skill/keyword candidates from an ExtractedDocument's bullets
(see pipeline.extract), producing a set of Keyword objects for the
resume and, separately, for the job description. Contains no
Streamlit-specific code so it can be developed, reasoned about, and
tested independently of the UI layer.

Design note: generic spaCy NER (trained for people, places,
organizations, dates, etc.) doesn't actually recognize skills or tools
as a category, so it isn't used here — a company name isn't a "skill"
either side is trying to match. Instead, this stage leads with a
curated skills gazetteer (matched via spaCy's PhraseMatcher) as the
high-confidence signal, and uses noun-phrase extraction as a lower-
confidence supplementary net for skills the gazetteer doesn't know
about.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ats_gap_analyzer.pipeline.extract import ExtractedDocument

if TYPE_CHECKING:
    from spacy.language import Language
    from spacy.matcher import PhraseMatcher
    from spacy.tokens import Span

__all__ = ["Keyword", "KeywordExtractor", "load_keyword_extractor", "extract_keywords"]

_GAZETTEER_PATH = Path(__file__).parent / "data" / "skills_gazetteer.txt"

# Determiners/possessives stripped from the front of a noun-phrase
# candidate before it's considered as a keyword — spaCy's is_stop
# already covers most of these, but pronouns like "our"/"their" aren't
# always flagged as stop words depending on the model version.
_LEADING_WORDS_TO_STRIP = {"a", "an", "the", "this", "that", "these", "those", "our", "their", "your", "its"}

_MAX_NOUN_PHRASE_TOKENS = 5
_MIN_NOUN_PHRASE_LENGTH = 3


@dataclass(frozen=True)
class Keyword:
    """A single extracted skill/keyword candidate.

    Attributes:
        text: The keyword text — the gazetteer's canonical casing for a
            high-confidence match (e.g. "Scikit-learn"), or lowercased
            text for a noun-phrase candidate.
        source_bullets: Every bullet's text (from the originating
            document) that this keyword was found in, for traceability
            back to where it came from.
        is_gazetteer_match: True for a high-confidence curated skill
            match; False for a lower-confidence noun-phrase candidate
            not found in the gazetteer.
    """

    text: str
    source_bullets: tuple[str, ...]
    is_gazetteer_match: bool


@dataclass(frozen=True)
class KeywordExtractor:
    """Bundles the loaded spaCy model and skill matcher.

    Loading the model and building the matcher are both expensive
    relative to per-document extraction, so callers (e.g. app.py via
    st.cache_resource) should create this once and reuse it across
    calls to extract_keywords.
    """

    nlp: "Language"
    matcher: "PhraseMatcher"


def _load_gazetteer_terms() -> list[str]:
    if not _GAZETTEER_PATH.exists():
        return []
    lines = _GAZETTEER_PATH.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]


_GAZETTEER_TERMS = _load_gazetteer_terms()
_GAZETTEER_CANONICAL: dict[str, str] = {term.lower(): term for term in _GAZETTEER_TERMS}


def load_keyword_extractor(spacy_model: str = "en_core_web_sm") -> KeywordExtractor:
    """Load the spaCy model and build the gazetteer PhraseMatcher.

    This is the expensive setup step — call it once and reuse the
    returned KeywordExtractor for every extract_keywords call.
    """
    import spacy
    from spacy.matcher import PhraseMatcher

    nlp = spacy.load(spacy_model)
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    if _GAZETTEER_TERMS:
        patterns = [nlp.make_doc(term) for term in _GAZETTEER_TERMS]
        matcher.add("SKILL", patterns)
    return KeywordExtractor(nlp=nlp, matcher=matcher)


def extract_keywords(document: ExtractedDocument, extractor: KeywordExtractor) -> list[Keyword]:
    """Extract keyword candidates from every bullet in a document.

    Runs over all bullets, including headings (Bullet.is_heading) —
    project and certification titles often list real technologies
    (e.g. "... | Python, Scikit-Learn, Pandas") that would otherwise be
    lost, even though those same heading bullets are meant to be
    excluded from later sentence-level semantic matching.

    Args:
        document: The ExtractedDocument to extract keywords from (a
            resume or a job description).
        extractor: A KeywordExtractor from load_keyword_extractor().

    Returns:
        A deduplicated list of Keyword objects — gazetteer matches
        first, then noun-phrase candidates — each linked back to every
        bullet it was found in.
    """
    gazetteer_hits: dict[str, list[str]] = {}
    noun_phrase_hits: dict[str, list[str]] = {}

    for bullet in document.bullets:
        doc = extractor.nlp(bullet.text)

        matched_spans: list[tuple[int, int]] = []
        for _, start, end in extractor.matcher(doc):
            span = doc[start:end]
            canonical = _GAZETTEER_CANONICAL.get(span.text.lower(), span.text)
            gazetteer_hits.setdefault(canonical, [])
            if bullet.text not in gazetteer_hits[canonical]:
                gazetteer_hits[canonical].append(bullet.text)
            matched_spans.append((start, end))

        for chunk in doc.noun_chunks:
            if _overlaps_any(chunk, matched_spans):
                continue
            cleaned = _clean_noun_chunk(chunk)
            if cleaned is None:
                continue
            noun_phrase_hits.setdefault(cleaned, [])
            if bullet.text not in noun_phrase_hits[cleaned]:
                noun_phrase_hits[cleaned].append(bullet.text)

    keywords = [
        Keyword(text=text, source_bullets=tuple(bullets), is_gazetteer_match=True)
        for text, bullets in gazetteer_hits.items()
    ]
    keywords += [
        Keyword(text=text, source_bullets=tuple(bullets), is_gazetteer_match=False)
        for text, bullets in noun_phrase_hits.items()
    ]
    return keywords


def _overlaps_any(span: "Span", ranges: list[tuple[int, int]]) -> bool:
    return any(span.start < end and start < span.end for start, end in ranges)


def _clean_noun_chunk(chunk: "Span") -> str | None:
    """Normalize a noun chunk into a keyword candidate, or None to drop it."""
    # Filtering on is_punct alone isn't reliable: spaCy's parser sometimes
    # tags stray symbols like "|" or "+" as NOUN/NUM rather than PUNCT
    # (e.g. in "... | Coursera" or "5+ experience"), which would otherwise
    # leak them into the output. Requiring at least one alphanumeric
    # character catches those regardless of the POS tag assigned.
    tokens = [t for t in chunk if any(c.isalnum() for c in t.text)]

    while tokens and (tokens[0].is_stop or tokens[0].text.lower() in _LEADING_WORDS_TO_STRIP):
        tokens = tokens[1:]

    if not tokens or all(t.is_stop for t in tokens):
        return None
    if len(tokens) > _MAX_NOUN_PHRASE_TOKENS:
        return None

    text = " ".join(t.text for t in tokens).strip().lower()
    if len(text) < _MIN_NOUN_PHRASE_LENGTH:
        return None
    return text
