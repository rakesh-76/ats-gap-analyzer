"""Stage 4 of the ATS Gap Analyzer pipeline: semantic gap analysis.

Compares job description requirements against resume bullets using
sentence embeddings, to catch cases where a resume expresses a skill in
different words than the job description uses — something pure keyword
matching (Stage 3) would miss entirely. Contains no Streamlit-specific
code so it can be developed, reasoned about, and tested independently
of the UI layer.

Naming note: the spec describes classifying each requirement as an
"exact match, a semantic (weak) match, or missing." Since this stage is
entirely embedding-based (there is no literal string-equality check
here — that's what Stage 3's gazetteer matching is for), the top tier
is named STRONG rather than "exact" to avoid implying literal text
equality; it means "very likely the same point, differently worded."
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Protocol

from ats_gap_analyzer.pipeline.extract import ExtractedDocument

if TYPE_CHECKING:
    import numpy as np

__all__ = [
    "MatchType",
    "RequirementMatch",
    "SemanticModel",
    "STRONG_THRESHOLD",
    "WEAK_THRESHOLD",
    "load_semantic_model",
    "analyze_gap",
]

# Cosine similarity thresholds for all-MiniLM-L6-v2. STRONG_THRESHOLD is
# still an untuned default (see the module docstring above). WEAK_THRESHOLD
# went through two rounds of real-world tuning: 0.35 -> 0.42 after finding
# clearly-absent requirements (e.g. "mentor junior data scientists" against
# a resume with no mentoring content) scoring 0.37-0.40 from shared topical
# vocabulary alone; then 0.42 -> 0.41 after that same jump caught a
# genuine partial match as a side effect ("a collaborative mindset..." at
# 0.41, matched against a resume bullet that also explicitly says
# "collaborative" — real, if partial, overlap that shouldn't count as
# missing). 0.41 is the value that correctly separates all three
# real examples found so far.
STRONG_THRESHOLD = 0.6
WEAK_THRESHOLD = 0.41


class MatchType(str, Enum):
    STRONG = "strong"
    WEAK = "weak"
    MISSING = "missing"


@dataclass(frozen=True)
class RequirementMatch:
    """The best resume match found for a single job description requirement.

    Attributes:
        requirement: The job description bullet this result is for.
        match_type: STRONG, WEAK, or MISSING based on the thresholds above.
        best_bullet: The resume bullet with the highest similarity to this
            requirement — populated even for MISSING, so a caller can show
            "closest we found" for transparency, rather than nothing.
            None only when the resume has no non-heading bullets at all.
        similarity: The cosine similarity score for best_bullet, in [-1, 1]
            (in practice close to [0, 1] for real sentence pairs).
    """

    requirement: str
    match_type: MatchType
    best_bullet: str | None
    similarity: float


class _Encoder(Protocol):
    """Structural type for anything with sentence-transformers' .encode()
    signature — lets tests substitute a fast fake without downloading the
    real model."""

    def encode(self, sentences: list[str], **kwargs: object) -> "np.ndarray": ...


@dataclass(frozen=True)
class SemanticModel:
    """Wraps the loaded sentence embedding model.

    Loading the model is expensive relative to per-document analysis, so
    callers (e.g. app.py via st.cache_resource) should create this once
    and reuse it across calls to analyze_gap.
    """

    encoder: _Encoder


def load_semantic_model(model_name: str = "all-MiniLM-L6-v2") -> SemanticModel:
    """Load the sentence-transformers model used for semantic matching."""
    from sentence_transformers import SentenceTransformer

    return SemanticModel(encoder=SentenceTransformer(model_name))


def analyze_gap(
    resume_doc: ExtractedDocument, jd_doc: ExtractedDocument, semantic_model: SemanticModel
) -> list[RequirementMatch]:
    """Compare every JD requirement against every resume bullet semantically.

    Heading/title bullets (Bullet.is_heading) are excluded from both
    documents before comparison — a job title line or a JD's internal
    label like "What You'll Do:" isn't a requirement or an achievement,
    and comparing it as one would only dilute the real signal. This is
    exactly why Stage 2 classified headings in the first place.

    Args:
        resume_doc: The candidate's extracted resume.
        jd_doc: The extracted job description.
        semantic_model: A SemanticModel from load_semantic_model().

    Returns:
        One RequirementMatch per non-heading JD bullet, in the same
        order they appear in jd_doc.
    """
    import numpy as np

    resume_bullets = [b.text for b in resume_doc.bullets if not b.is_heading]
    jd_requirements = [b.text for b in jd_doc.bullets if not b.is_heading]

    if not jd_requirements:
        return []
    if not resume_bullets:
        return [
            RequirementMatch(requirement=r, match_type=MatchType.MISSING, best_bullet=None, similarity=0.0)
            for r in jd_requirements
        ]

    resume_embeddings = np.asarray(
        semantic_model.encoder.encode(resume_bullets, normalize_embeddings=True)
    )
    jd_embeddings = np.asarray(
        semantic_model.encoder.encode(jd_requirements, normalize_embeddings=True)
    )

    # Embeddings are L2-normalized, so the dot product between any pair
    # is exactly their cosine similarity.
    similarity_matrix = jd_embeddings @ resume_embeddings.T

    results: list[RequirementMatch] = []
    for i, requirement in enumerate(jd_requirements):
        row = similarity_matrix[i]
        best_idx = int(np.argmax(row))
        best_score = float(row[best_idx])
        results.append(
            RequirementMatch(
                requirement=requirement,
                match_type=_classify(best_score),
                best_bullet=resume_bullets[best_idx],
                similarity=best_score,
            )
        )
    return results


def _classify(similarity: float) -> MatchType:
    if similarity >= STRONG_THRESHOLD:
        return MatchType.STRONG
    if similarity >= WEAK_THRESHOLD:
        return MatchType.WEAK
    return MatchType.MISSING
