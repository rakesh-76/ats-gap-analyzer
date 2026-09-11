"""Stage 5 of the ATS Gap Analyzer pipeline: scoring & suggestions.

Combines Stage 3's exact keyword sets and Stage 4's semantic match
results into a single overall score, matched/missing skill lists, and
rule/template-based phrasing suggestions. Contains no Streamlit-specific
code so it can be developed, reasoned about, and tested independently
of the UI layer. Unlike Stages 3 and 4, this stage needs no external
model — it's pure combination logic over their outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ats_gap_analyzer.pipeline.keywords import Keyword
from ats_gap_analyzer.pipeline.semantic import MatchType, RequirementMatch

__all__ = [
    "Suggestion",
    "SuggestionKind",
    "GapAnalysisResult",
    "EXACT_MATCH_WEIGHT",
    "SEMANTIC_MATCH_WEIGHT",
    "WEAK_MATCH_CREDIT",
    "score_gap",
]

# Equal weighting between exact skill coverage (Stage 3) and semantic
# requirement coverage (Stage 4). A reasonable starting point, not an
# empirically tuned value — same caveat as Stage 4's thresholds.
EXACT_MATCH_WEIGHT = 0.5
SEMANTIC_MATCH_WEIGHT = 0.5

# A weak semantic match counts as partial credit toward the semantic
# match rate: it represents some real coverage, just not strong enough
# to call the requirement clearly met.
WEAK_MATCH_CREDIT = 0.5


class SuggestionKind(Enum):
    """What a Suggestion is about, for grouping in the UI.

    Lets app.py split one flat suggestions list into labeled groups
    (missing requirements vs. weak matches vs. missing skills) without
    guessing from wording or from whether related_requirement is set.
    """

    SKILL = "skill"
    MISSING_REQUIREMENT = "missing_requirement"
    WEAK_REQUIREMENT = "weak_requirement"


@dataclass(frozen=True)
class Suggestion:
    """A single actionable, rule/template-based suggestion.

    Attributes:
        text: The suggestion text shown to the user.
        kind: What this suggestion is about — see SuggestionKind.
        related_requirement: The JD requirement this suggestion is
            about, if it came from a Stage 4 semantic gap rather than a
            missing exact skill (which isn't tied to one specific
            requirement sentence).
    """

    text: str
    kind: SuggestionKind
    related_requirement: str | None = None


@dataclass(frozen=True)
class GapAnalysisResult:
    """The combined result of Stages 3-5.

    Attributes:
        overall_score: 0-100, a weighted combination of exact skill
            coverage and semantic requirement coverage.
        exact_match_rate: Fraction (0.0-1.0) of the JD's recognized
            skills found in the resume's recognized skills. 1.0 if the
            JD has no recognized skills at all (nothing to be missing).
        semantic_match_rate: Fraction (0.0-1.0) of non-heading JD
            requirements at least weakly covered, with weak matches
            counted as WEAK_MATCH_CREDIT. 1.0 if there are no
            requirements to compare at all.
        matched_skills: JD skills found in the resume, in JD order.
        missing_skills: JD skills not found in the resume, in JD order.
        suggestions: Missing-skill suggestions first, then one per
            non-strong Stage 4 requirement, in that order.
    """

    overall_score: int
    exact_match_rate: float
    semantic_match_rate: float
    matched_skills: list[str]
    missing_skills: list[str]
    suggestions: list[Suggestion]


def score_gap(
    resume_keywords: list[Keyword],
    jd_keywords: list[Keyword],
    semantic_results: list[RequirementMatch],
) -> GapAnalysisResult:
    """Combine Stage 3 and Stage 4 output into a single scored result.

    Only gazetteer-matched keywords (Keyword.is_gazetteer_match) are
    used for the exact-skill comparison — noun-phrase candidates are
    too noisy to treat as either "required" or "have."

    Note: a missing skill and a missing/weak semantic requirement can
    both fire for the same underlying gap (e.g. "Python" as a missing
    skill, and the JD sentence mentioning it as a missing requirement),
    producing two related suggestions rather than one merged one. That's
    still acceptable — the two are different granularities of the same
    feedback, not a contradiction. What *was* fixed after real-world
    testing: missing skills used to get one suggestion line each (very
    repetitive with several missing skills), and consecutive
    missing/weak requirement suggestions all used identical template
    wording. See _build_suggestions for the fix.

    Args:
        resume_keywords: extract_keywords() output for the resume.
        jd_keywords: extract_keywords() output for the job description.
        semantic_results: analyze_gap() output.

    Returns:
        A GapAnalysisResult with the overall score, matched/missing
        skills, and suggestions.
    """
    resume_skill_set = {k.text for k in resume_keywords if k.is_gazetteer_match}
    jd_skills = [k.text for k in jd_keywords if k.is_gazetteer_match]

    matched_skills = [s for s in jd_skills if s in resume_skill_set]
    missing_skills = [s for s in jd_skills if s not in resume_skill_set]

    exact_match_rate = len(matched_skills) / len(jd_skills) if jd_skills else 1.0
    semantic_match_rate = _semantic_match_rate(semantic_results)

    overall_score = round(
        100 * (EXACT_MATCH_WEIGHT * exact_match_rate + SEMANTIC_MATCH_WEIGHT * semantic_match_rate)
    )

    return GapAnalysisResult(
        overall_score=overall_score,
        exact_match_rate=exact_match_rate,
        semantic_match_rate=semantic_match_rate,
        matched_skills=matched_skills,
        missing_skills=missing_skills,
        suggestions=_build_suggestions(missing_skills, semantic_results),
    )


def _semantic_match_rate(results: list[RequirementMatch]) -> float:
    if not results:
        return 1.0
    points = sum(
        1.0
        if r.match_type == MatchType.STRONG
        else WEAK_MATCH_CREDIT
        if r.match_type == MatchType.WEAK
        else 0.0
        for r in results
    )
    return points / len(results)


# Phrasing pools for non-strong requirement suggestions, rotated through
# independently per match type (see _build_suggestions) so that several
# consecutive missing/weak suggestions don't read as identical templated
# text — the concrete complaint from real-world testing (see Streamlit_v8
# screenshot review). Order matters: index 0 is used first, then wraps
# around via modulo once a type's suggestions outnumber its pool.
_MISSING_REQUIREMENT_PHRASINGS = [
    'Your resume doesn\'t address this requirement: "{requirement}". '
    "Add a bullet for it if it applies to you.",
    'Nothing in your resume speaks to "{requirement}" right now. '
    "Add a bullet if it's part of your background.",
    'This requirement isn\'t covered anywhere in your resume: "{requirement}". '
    "Add a bullet if it's true for you.",
]

_WEAK_REQUIREMENT_PHRASINGS = [
    '"{requirement}" is only loosely covered. Your closest bullet is: '
    '"{best_bullet}". Consider rewording it to address this more directly.',
    '"{requirement}" is only partially reflected in your resume. The closest '
    'match is "{best_bullet}", which could be sharpened to address this '
    "more directly.",
    'Coverage for "{requirement}" is weak right now. The nearest bullet is '
    '"{best_bullet}". Reword it to align more closely if that fits.',
]


def _consolidated_missing_skills_suggestion(missing_skills: list[str]) -> Suggestion | None:
    """One suggestion covering all missing skills, not one per skill.

    A resume missing 7 skills used to produce 7 near-identical "Add X"
    lines — real-world testing showed this reads as noise rather than
    signal. related_requirement is None either way, same as before: a
    missing skill isn't tied to one specific JD requirement sentence.
    """
    if not missing_skills:
        return None
    if len(missing_skills) == 1:
        text = (
            f'"{missing_skills[0]}" is listed in the job description but not '
            "in your resume. Add it if it applies to you."
        )
    else:
        skills_text = ", ".join(f'"{s}"' for s in missing_skills)
        text = (
            "These skills are listed in the job description but not in your "
            f"resume: {skills_text}. Add any that apply to you."
        )
    return Suggestion(text=text, kind=SuggestionKind.SKILL)


def _build_suggestions(
    missing_skills: list[str], semantic_results: list[RequirementMatch]
) -> list[Suggestion]:
    suggestions = []

    consolidated = _consolidated_missing_skills_suggestion(missing_skills)
    if consolidated is not None:
        suggestions.append(consolidated)

    # Separate counters per match type: rotation is independent, so an
    # alternating sequence of missing/weak requirements doesn't burn
    # through one shared pool and leave the other type stuck on index 0.
    missing_seen = 0
    weak_seen = 0
    for r in semantic_results:
        if r.match_type == MatchType.STRONG:
            continue
        if r.match_type == MatchType.MISSING:
            phrasing = _MISSING_REQUIREMENT_PHRASINGS[missing_seen % len(_MISSING_REQUIREMENT_PHRASINGS)]
            text = phrasing.format(requirement=r.requirement)
            missing_seen += 1
            kind = SuggestionKind.MISSING_REQUIREMENT
        else:  # WEAK
            phrasing = _WEAK_REQUIREMENT_PHRASINGS[weak_seen % len(_WEAK_REQUIREMENT_PHRASINGS)]
            text = phrasing.format(requirement=r.requirement, best_bullet=r.best_bullet)
            weak_seen += 1
            kind = SuggestionKind.WEAK_REQUIREMENT
        suggestions.append(Suggestion(text=text, kind=kind, related_requirement=r.requirement))

    return suggestions
