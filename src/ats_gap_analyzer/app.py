"""Streamlit entry point for the ATS Gap Analyzer.

Layout state machine (see `st.session_state.phase` / `pending_analysis`):
  - phase "input"  — resume upload, JD paste, and Analyze live centered in
    the main area, as a single hero card. This is the first thing anyone
    sees.
  - phase "result" — the same input widgets move into the sidebar (so
    they stay editable for a re-run) and the main area is dedicated to
    the progress bar, then the results.

Every click of Analyze (not just the first) sets `pending_analysis = True`
and immediately calls `st.rerun()` *before* running the pipeline, rather
than running it in the same script run as the click. Two reasons:
  1. Streamlit has no concept of a widget animating between two different
     layout containers — the sidebar and the main area are separate DOM
     subtrees, so a widget cannot visually "travel" between them; it can
     only be absent from one and freshly mounted in the other. The rerun
     lets that fresh mount (and its CSS animation — see _inject_theme)
     happen in a render where the input card is already gone from the
     main area, instead of overlapping with a progress bar appearing in
     the same pass.
  2. It lets the Analyze button be rendered with `disabled=True` for the
     entire run that actually does the work: `disabled` has to be decided
     before the button is drawn, so the run that computes the result has
     to already know `pending_analysis` is true *before* it renders the
     sidebar — which only works if that flag was set on a *previous* run.
A second `st.rerun()` after the pipeline finishes clears that disabled
state immediately, rather than leaving it stale until the next unrelated
interaction.

Visual theme: colors and font come from .streamlit/config.toml (the
supported, version-stable way to theme a Streamlit app). The CSS block
below layers on top only for what config.toml can't do — the animations,
default borders on the input widgets (Streamlit's own default border is
nearly invisible against our card background), and scoping a couple of
custom card styles to specific containers via their `key=`.
"""

from __future__ import annotations

import time

import streamlit as st

from ats_gap_analyzer.pipeline.extract import (
    ExtractionError,
    extract_job_description,
    extract_resume_text,
)
from ats_gap_analyzer.pipeline.keywords import (
    KeywordExtractor,
    extract_keywords,
    load_keyword_extractor,
)
from ats_gap_analyzer.pipeline.scoring import GapAnalysisResult, SuggestionKind, score_gap
from ats_gap_analyzer.pipeline.semantic import (
    SemanticModel,
    analyze_gap,
    load_semantic_model,
)

st.set_page_config(page_title="ATS Gap Analyzer", layout="wide")


def _inject_theme() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        .stApp { font-family: 'Inter', 'Segoe UI', sans-serif; }

        /* Streamlit's default block padding leaves more room than this
           single-column layout needs, which was producing an empty
           scrollbar on short pages. */
        .block-container { padding-top: 2rem; padding-bottom: 2rem; }

        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(14px); }
            to   { opacity: 1; transform: translateY(0); }
        }
        @keyframes slideInLeft {
            from { opacity: 0; transform: translateX(-28px); }
            to   { opacity: 1; transform: translateX(0); }
        }

        /* Sidebar and progress bar only exist once analysis has started —
           their first mount plays these animations automatically. */
        [data-testid="stSidebar"] { animation: slideInLeft 0.45s ease-out; }
        [data-testid="stProgress"] { animation: fadeInUp 0.4s ease-out; }

        /* Default state for the JD textarea and the resume dropzone: a
           visible neutral border. Without this they're invisible, since
           their default background matches our card background exactly.
           The colored focus ring Streamlit already draws on top of this
           (from primaryColor) is untouched — this only fills in the gap
           for the *unfocused* state. */
        [data-testid="stTextArea"] textarea,
        [data-testid="stFileUploaderDropzone"] {
            border: 1px solid rgba(255,255,255,0.18) !important;
            border-radius: 8px !important;
        }

        /* Custom card containers, scoped via st.container(key=...). */
        .st-key-input-card {
            max-width: 700px;
            margin: 0.5rem auto 1.5rem auto;
            padding: 2rem 2.25rem;
            background: #141a21;
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 16px;
        }
        .st-key-results-card {
            animation: fadeInUp 0.5s ease-out;
        }
        .st-key-suggestions-card {
            padding: 1.25rem 1.5rem;
            background: #141a21;
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 16px;
        }

        /* Score color-coding — a red-to-sage progression by value.
           Applied to whichever bucket container actually wraps the
           metric, so only one of these ever matches. */
        .st-key-score-good [data-testid="stMetricValue"] { color: #6B8E4E; }
        .st-key-score-medium [data-testid="stMetricValue"] { color: #E3B341; }
        .st-key-score-low [data-testid="stMetricValue"] { color: #E5484D; }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def _get_keyword_extractor() -> KeywordExtractor:
    # Loading the spaCy model and building the gazetteer matcher is
    # expensive; cache_resource ensures it happens once per session
    # rather than on every Streamlit rerun.
    return load_keyword_extractor()


@st.cache_resource
def _get_semantic_model() -> SemanticModel:
    # Same reasoning as above — the sentence-transformers model is
    # loaded (and downloaded, on first run) once per session.
    return load_semantic_model()


def _score_bucket(score: float) -> str:
    if score >= 70:
        return "score-good"
    if score >= 40:
        return "score-medium"
    return "score-low"


def _render_score_summary(result: GapAnalysisResult) -> None:
    with st.container(key=_score_bucket(result.overall_score)):
        st.metric("Overall Match Score", f"{result.overall_score}%")

    col1, col2 = st.columns(2)
    col1.metric("Exact Skill Coverage", f"{result.exact_match_rate:.0%}")
    col2.metric("Semantic Requirement Coverage", f"{result.semantic_match_rate:.0%}")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Matched skills ({len(result.matched_skills)})**")
        st.write(", ".join(result.matched_skills) if result.matched_skills else "None.")
    with col2:
        st.write(f"**Missing skills ({len(result.missing_skills)})**")
        st.write(", ".join(result.missing_skills) if result.missing_skills else "None.")

    if not result.suggestions:
        st.success("No suggestions — strong coverage across the board!")
        return

    st.divider()
    st.subheader("Suggestions")

    # Grouped by kind rather than guessed from wording or from whether
    # related_requirement is set — a labeled group per suggestion type,
    # not one flat undifferentiated list.
    skill_suggestions = [s for s in result.suggestions if s.kind == SuggestionKind.SKILL]
    missing_requirement_suggestions = [
        s for s in result.suggestions if s.kind == SuggestionKind.MISSING_REQUIREMENT
    ]
    weak_requirement_suggestions = [
        s for s in result.suggestions if s.kind == SuggestionKind.WEAK_REQUIREMENT
    ]

    for s in skill_suggestions:
        st.info(s.text)

    if missing_requirement_suggestions or weak_requirement_suggestions:
        with st.container(key="suggestions-card"):
            if missing_requirement_suggestions:
                st.write(f"**Missing requirements ({len(missing_requirement_suggestions)})**")
                for i, s in enumerate(missing_requirement_suggestions, start=1):
                    st.markdown(f"{i}. {s.text}")
            if weak_requirement_suggestions:
                st.write(f"**Weak matches ({len(weak_requirement_suggestions)})**")
                for i, s in enumerate(weak_requirement_suggestions, start=1):
                    st.markdown(f"{i}. {s.text}")


def _render_input_widgets(disabled: bool) -> tuple:
    # Resume uses a stable key deliberately — after "Start over" it should
    # stay attached, since the common case is re-running the same resume
    # against a different JD. The JD textarea is the one thing "Start
    # over" should actually flush, and versioning its key is what makes
    # that reliable: popping a text_area's session_state entry alone can
    # take an extra rerun to visibly clear (see module docstring history
    # of the file_uploader version of this same issue) — a fresh key
    # guarantees it's empty immediately, on the very next render.
    resume_file = st.file_uploader(
        "Upload your resume", type=["pdf", "docx"], key="resume_upload", disabled=disabled
    )
    job_description_text = st.text_area(
        "Paste the job description",
        height=220,
        key=f"jd_text_{st.session_state.jd_version}",
        disabled=disabled,
    )
    # Deliberately not disabled based on whether both inputs are present: a
    # disabled HTML button can't receive the click that would otherwise
    # commit a just-pasted job description (Streamlit's text_area only
    # sends its value to the backend on blur/Ctrl+Enter). Validating after
    # the click lets the same click do both — commit the paste and trigger
    # analysis — instead of forcing an extra commit step first. `disabled`
    # here is only ever true while analysis is actually running (see
    # module docstring), never because inputs are missing.
    submitted = st.button(
        "Analyze", key="analyze_button", type="primary", use_container_width=True, disabled=disabled
    )
    return resume_file, job_description_text, submitted


def _run_analysis(resume_file, job_description_text: str) -> GapAnalysisResult | None:
    # Streamlit's uploader restricts the file picker to .pdf/.docx, but the
    # extension is re-derived here rather than trusted blindly, since a
    # mismatched or corrupt file still needs to surface a clear error.
    file_type = resume_file.name.rsplit(".", 1)[-1]
    file_bytes = resume_file.getvalue()

    # A staged progress bar rather than a single spinner: each step below
    # reflects real pipeline work finishing (not a fake timer), so the
    # bar's pace naturally matches how long each stage actually takes.
    progress = st.progress(0, text="Reading your resume...")

    try:
        resume_doc = extract_resume_text(file_bytes, file_type)
    except ExtractionError as exc:
        progress.empty()
        st.error(f"Couldn't read your resume: {exc}")
        return None

    progress.progress(15, text="Reading the job description...")
    jd_doc = extract_job_description(job_description_text)

    progress.progress(30, text="Identifying skills and keywords...")
    keyword_extractor = _get_keyword_extractor()
    resume_keywords = extract_keywords(resume_doc, keyword_extractor)
    jd_keywords = extract_keywords(jd_doc, keyword_extractor)

    progress.progress(
        50, text="Loading the semantic matching model (first run downloads ~90MB)..."
    )
    semantic_model = _get_semantic_model()

    progress.progress(75, text="Comparing your resume against each requirement...")
    semantic_results = analyze_gap(resume_doc, jd_doc, semantic_model)

    progress.progress(92, text="Calculating your match score...")
    gap_result = score_gap(resume_keywords, jd_keywords, semantic_results)

    progress.progress(100, text="Done!")
    time.sleep(0.4)  # let the completed bar register before it disappears
    progress.empty()

    return gap_result


_inject_theme()

if "phase" not in st.session_state:
    st.session_state.phase = "input"  # "input" | "result"
if "jd_version" not in st.session_state:
    st.session_state.jd_version = 0

processing = st.session_state.get("pending_analysis", False)

if st.session_state.phase == "input":
    with st.container(key="input-card"):
        st.title("ATS Gap Analyzer")
        st.caption("See how well your resume matches a job description, skill by skill.")
        resume_file, job_description_text, submitted = _render_input_widgets(processing)
else:
    st.title("ATS Gap Analyzer")
    with st.sidebar:
        st.header("Your inputs")
        resume_file, job_description_text, submitted = _render_input_widgets(processing)
        if st.button(
            "Start over", key="start_over_button", use_container_width=True, disabled=processing
        ):
            for key in ("phase", "pending_analysis", "gap_result", "analyzed_resume_name"):
                st.session_state.pop(key, None)
            st.session_state.jd_version += 1
            st.rerun()

if submitted:
    if not resume_file or not job_description_text.strip():
        location_hint = "in the sidebar" if st.session_state.phase == "result" else "above"
        st.warning(
            f"Please upload a resume and paste a job description {location_hint} "
            "before analyzing."
        )
    else:
        # Every analysis (first or repeat) reruns once before doing any
        # work — see module docstring for why.
        st.session_state.phase = "result"
        st.session_state.pending_analysis = True
        st.rerun()

if st.session_state.get("pending_analysis"):
    result = _run_analysis(resume_file, job_description_text)
    st.session_state.pending_analysis = False
    if result is not None:
        st.session_state["gap_result"] = result
        st.session_state["analyzed_resume_name"] = resume_file.name
    # Refresh immediately so the Analyze/Start over buttons re-enable
    # right away, instead of appearing stuck until some unrelated
    # interaction happens to trigger the next rerun.
    st.rerun()

if "gap_result" in st.session_state:
    with st.container(key="results-card"):
        st.caption(f"Results for **{st.session_state['analyzed_resume_name']}**")
        _render_score_summary(st.session_state["gap_result"])
