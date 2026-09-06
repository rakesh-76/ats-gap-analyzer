"""Streamlit entry point for the ATS Gap Analyzer.

Currently wires up Stage 1 (input ingestion) and Stage 2 (text
extraction) only. Stages 3-6 (keyword extraction, semantic analysis,
scoring, and the real results UI) will be added incrementally on top of
this. The extraction preview below is a temporary debug view used to
verify Stage 2 against real uploaded files — it will be replaced by the
actual Stage 6 results display once scoring and suggestions exist.

UI styling/theming is intentionally left at Streamlit's defaults for
now; a deliberate styling pass is deferred until just before deployment.
"""

from __future__ import annotations

import streamlit as st

from ats_gap_analyzer.pipeline.extract import (
    Bullet,
    ExtractionError,
    extract_job_description,
    extract_resume_text,
)

st.title("ATS Gap Analyzer")

resume_file = st.file_uploader("Upload your resume", type=["pdf", "docx"])
job_description_text = st.text_area("Paste the job description", height=250)

has_both_inputs = bool(resume_file) and bool(job_description_text.strip())
submitted = st.button("Analyze", disabled=not has_both_inputs)

if not has_both_inputs:
    st.caption("Upload a resume and paste a job description to enable analysis.")


def _render_bullet(bullet: Bullet) -> None:
    if bullet.is_heading:
        # Rendered as plain bold text rather than a bulleted list item —
        # a heading/title line isn't really "a bullet point" conceptually,
        # so it shouldn't look like one.
        st.markdown(f"**{bullet.text}**")
    else:
        st.write(f"- {bullet.text}")


if submitted:
    # Streamlit's uploader restricts the file picker to .pdf/.docx, but the
    # extension is re-derived here rather than trusted blindly, since a
    # mismatched or corrupt file still needs to surface a clear error.
    file_type = resume_file.name.rsplit(".", 1)[-1]
    file_bytes = resume_file.getvalue()

    try:
        resume_doc = extract_resume_text(file_bytes, file_type)
    except ExtractionError as exc:
        st.error(f"Couldn't read your resume: {exc}")
    else:
        jd_doc = extract_job_description(job_description_text)

        # --- Temporary debug view for verifying Stage 2 end to end --- #
        st.subheader("Extracted resume (debug preview)")
        st.caption("Bold, no bullet = heading/title line, excluded from later semantic matching")
        st.write(
            f"{len(resume_doc.bullets)} bullet(s) across "
            f"{len(resume_doc.sections)} section(s)"
        )
        for section, bullets in resume_doc.sections.items():
            with st.expander(section.title()):
                for bullet in bullets:
                    _render_bullet(bullet)

        st.subheader("Extracted job description (debug preview)")
        st.write(f"{len(jd_doc.bullets)} bullet(s)")
        for bullet in jd_doc.bullets:
            _render_bullet(bullet)
