"""Unit tests for pipeline.scoring (Stage 5: scoring & suggestions).

Pure logic over Stage 3/4 output — no external model needed, so these
tests construct Keyword and RequirementMatch objects directly and run
fast and deterministically.
"""

from __future__ import annotations

from ats_gap_analyzer.pipeline.keywords import Keyword
from ats_gap_analyzer.pipeline.scoring import (
    _MISSING_REQUIREMENT_PHRASINGS,
    _WEAK_REQUIREMENT_PHRASINGS,
    SuggestionKind,
    score_gap,
)
from ats_gap_analyzer.pipeline.semantic import MatchType, RequirementMatch


def _kw(text: str, gazetteer: bool = True) -> Keyword:
    return Keyword(text=text, source_bullets=("some bullet",), is_gazetteer_match=gazetteer)


def _match(requirement: str, match_type: MatchType, best_bullet: str | None = "a bullet") -> RequirementMatch:
    similarity = {MatchType.STRONG: 0.8, MatchType.WEAK: 0.5, MatchType.MISSING: 0.1}[match_type]
    return RequirementMatch(
        requirement=requirement, match_type=match_type, best_bullet=best_bullet, similarity=similarity
    )


class TestExactMatchRate:
    def test_all_skills_matched(self) -> None:
        resume_kw = [_kw("Python"), _kw("SQL")]
        jd_kw = [_kw("Python"), _kw("SQL")]

        result = score_gap(resume_kw, jd_kw, [])

        assert result.exact_match_rate == 1.0
        assert result.matched_skills == ["Python", "SQL"]
        assert result.missing_skills == []

    def test_all_skills_missing(self) -> None:
        resume_kw = [_kw("Java")]
        jd_kw = [_kw("Python"), _kw("SQL")]

        result = score_gap(resume_kw, jd_kw, [])

        assert result.exact_match_rate == 0.0
        assert result.matched_skills == []
        assert result.missing_skills == ["Python", "SQL"]

    def test_partial_skill_overlap(self) -> None:
        resume_kw = [_kw("Python")]
        jd_kw = [_kw("Python"), _kw("SQL"), _kw("R")]

        result = score_gap(resume_kw, jd_kw, [])

        assert result.exact_match_rate == 1 / 3
        assert result.matched_skills == ["Python"]
        assert result.missing_skills == ["SQL", "R"]

    def test_noun_phrase_keywords_are_ignored(self) -> None:
        """Only gazetteer matches count toward exact skill comparison —
        noisy noun-phrase candidates shouldn't count as "required" or
        "have", in either direction."""
        resume_kw = [_kw("some noun phrase", gazetteer=False)]
        jd_kw = [_kw("Python"), _kw("another noun phrase", gazetteer=False)]

        result = score_gap(resume_kw, jd_kw, [])

        assert result.missing_skills == ["Python"]
        assert result.exact_match_rate == 0.0

    def test_no_jd_skills_defaults_to_full_credit(self) -> None:
        result = score_gap([_kw("Python")], [], [])

        assert result.exact_match_rate == 1.0
        assert result.missing_skills == []


class TestSemanticMatchRate:
    def test_all_strong_gives_full_credit(self) -> None:
        results = [_match("req1", MatchType.STRONG), _match("req2", MatchType.STRONG)]

        result = score_gap([], [], results)

        assert result.semantic_match_rate == 1.0

    def test_all_missing_gives_no_credit(self) -> None:
        results = [_match("req1", MatchType.MISSING), _match("req2", MatchType.MISSING)]

        result = score_gap([], [], results)

        assert result.semantic_match_rate == 0.0

    def test_weak_gives_half_credit(self) -> None:
        results = [_match("req1", MatchType.STRONG), _match("req2", MatchType.WEAK)]

        result = score_gap([], [], results)

        assert result.semantic_match_rate == (1.0 + 0.5) / 2

    def test_no_requirements_defaults_to_full_credit(self) -> None:
        result = score_gap([], [], [])

        assert result.semantic_match_rate == 1.0


class TestOverallScore:
    def test_perfect_match_scores_100(self) -> None:
        resume_kw = [_kw("Python")]
        jd_kw = [_kw("Python")]
        results = [_match("req1", MatchType.STRONG)]

        result = score_gap(resume_kw, jd_kw, results)

        assert result.overall_score == 100

    def test_total_mismatch_scores_0(self) -> None:
        resume_kw = [_kw("Java")]
        jd_kw = [_kw("Python")]
        results = [_match("req1", MatchType.MISSING)]

        result = score_gap(resume_kw, jd_kw, results)

        assert result.overall_score == 0

    def test_mixed_result_is_weighted_average(self) -> None:
        # exact_match_rate = 1.0 (Python matched), semantic_match_rate = 0.5
        # (one weak match) -> overall = 100 * (0.5*1.0 + 0.5*0.5) = 75
        resume_kw = [_kw("Python")]
        jd_kw = [_kw("Python")]
        results = [_match("req1", MatchType.WEAK)]

        result = score_gap(resume_kw, jd_kw, results)

        assert result.overall_score == 75


class TestSuggestions:
    def test_missing_skill_produces_suggestion(self) -> None:
        result = score_gap([], [_kw("Python")], [])

        assert len(result.suggestions) == 1
        assert "Python" in result.suggestions[0].text
        assert result.suggestions[0].kind == SuggestionKind.SKILL
        assert result.suggestions[0].related_requirement is None

    def test_strong_match_produces_no_suggestion(self) -> None:
        results = [_match("req1", MatchType.STRONG)]

        result = score_gap([], [], results)

        assert result.suggestions == []

    def test_missing_requirement_produces_suggestion(self) -> None:
        results = [_match("Do the thing", MatchType.MISSING, best_bullet=None)]

        result = score_gap([], [], results)

        assert len(result.suggestions) == 1
        assert "Do the thing" in result.suggestions[0].text
        assert result.suggestions[0].kind == SuggestionKind.MISSING_REQUIREMENT
        assert result.suggestions[0].related_requirement == "Do the thing"

    def test_weak_requirement_suggestion_references_closest_bullet(self) -> None:
        results = [_match("Do the thing", MatchType.WEAK, best_bullet="Did something close")]

        result = score_gap([], [], results)

        assert len(result.suggestions) == 1
        assert "Do the thing" in result.suggestions[0].text
        assert "Did something close" in result.suggestions[0].text
        assert result.suggestions[0].kind == SuggestionKind.WEAK_REQUIREMENT

    def test_suggestions_order_is_missing_skills_then_requirements(self) -> None:
        resume_kw: list[Keyword] = []
        jd_kw = [_kw("Python")]
        results = [_match("Do the thing", MatchType.MISSING, best_bullet=None)]

        result = score_gap(resume_kw, jd_kw, results)

        assert len(result.suggestions) == 2
        assert "Python" in result.suggestions[0].text
        assert "Do the thing" in result.suggestions[1].text

    def test_multiple_missing_skills_are_consolidated_into_one_suggestion(self) -> None:
        """Real-world testing showed one "Add X" line per missing skill
        reads as repetitive noise with several missing skills — they
        should be consolidated into a single suggestion instead."""
        resume_kw: list[Keyword] = []
        jd_kw = [_kw("Python"), _kw("SQL"), _kw("PyTorch")]

        result = score_gap(resume_kw, jd_kw, [])

        assert len(result.suggestions) == 1
        text = result.suggestions[0].text
        assert "Python" in text
        assert "SQL" in text
        assert "PyTorch" in text
        assert result.suggestions[0].related_requirement is None

    def test_repeated_missing_requirements_use_varied_phrasing(self) -> None:
        """Several consecutive missing-requirement suggestions shouldn't
        all read as identical templated text — verify the phrasing pool
        rotates (and wraps around once results outnumber the pool)."""
        requirements = ["Req A", "Req B", "Req C", "Req D"]
        results = [_match(r, MatchType.MISSING, best_bullet=None) for r in requirements]

        result = score_gap([], [], results)

        expected = [
            _MISSING_REQUIREMENT_PHRASINGS[i % len(_MISSING_REQUIREMENT_PHRASINGS)].format(requirement=r)
            for i, r in enumerate(requirements)
        ]
        assert [s.text for s in result.suggestions] == expected

    def test_repeated_weak_requirements_use_varied_phrasing(self) -> None:
        requirements = ["Req A", "Req B", "Req C", "Req D"]
        bullets = ["Bullet A", "Bullet B", "Bullet C", "Bullet D"]
        results = [_match(r, MatchType.WEAK, best_bullet=b) for r, b in zip(requirements, bullets)]

        result = score_gap([], [], results)

        expected = [
            _WEAK_REQUIREMENT_PHRASINGS[i % len(_WEAK_REQUIREMENT_PHRASINGS)].format(
                requirement=r, best_bullet=b
            )
            for i, (r, b) in enumerate(zip(requirements, bullets))
        ]
        assert [s.text for s in result.suggestions] == expected

    def test_missing_and_weak_phrasing_rotation_are_independent(self) -> None:
        """Each match type rotates through its own pool independently —
        an alternating sequence shouldn't advance the other type's
        counter or leave either stuck on the same phrasing."""
        results = [
            _match("M1", MatchType.MISSING, best_bullet=None),
            _match("W1", MatchType.WEAK, best_bullet="B1"),
            _match("M2", MatchType.MISSING, best_bullet=None),
            _match("W2", MatchType.WEAK, best_bullet="B2"),
        ]

        result = score_gap([], [], results)

        assert result.suggestions[0].text == _MISSING_REQUIREMENT_PHRASINGS[0].format(requirement="M1")
        assert result.suggestions[1].text == _WEAK_REQUIREMENT_PHRASINGS[0].format(
            requirement="W1", best_bullet="B1"
        )
        assert result.suggestions[2].text == _MISSING_REQUIREMENT_PHRASINGS[1].format(requirement="M2")
        assert result.suggestions[3].text == _WEAK_REQUIREMENT_PHRASINGS[1].format(
            requirement="W2", best_bullet="B2"
        )
