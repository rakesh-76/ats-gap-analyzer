# ATS Gap Analyzer

A free, ML-powered tool that analyzes how well a resume matches a job description, using keyword extraction and semantic similarity to catch skills expressed in different words rather than just counting exact keyword matches.

**Live app:** https://ats-gap-analyzer.streamlit.app/

## Features

- Resume upload (PDF, DOCX) and job description paste
- Text extraction and cleaning
- Skill and keyword extraction using spaCy (NER, noun-phrase extraction, and a curated skills gazetteer)
- Semantic gap analysis using sentence-transformer embeddings
- Overall match score combining exact skill coverage and semantic requirement coverage
- Matched and missing skills, plus suggestions split into missing requirements and weak matches

## Tech Stack

- **App framework:** Streamlit
- **Resume parsing:** pdfplumber (PDF), python-docx (DOCX)
- **NLP/NER:** spaCy (`en_core_web_sm`)
- **Semantic embeddings:** sentence-transformers (`all-MiniLM-L6-v2`)
- **Hosting:** Streamlit Community Cloud

## Project Status

Live and deployed. See `docs/` for the original project specification and documentation.

## Local Development

```powershell
uv sync
uv run python -m spacy download en_core_web_sm
uv run streamlit run src/ats_gap_analyzer/app.py
```

## Testing

```powershell
uv run pytest
```

## License

TBD
