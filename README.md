# ATS Gap Analyzer

A free, ML-powered tool that analyzes how well a resume matches a job description — going beyond simple keyword counting by using semantic similarity to catch skills expressed in different words.

## Features

- Resume upload (PDF, DOCX) and job description paste
- Text extraction and cleaning
- Skill/keyword extraction (spaCy NER + noun-phrase extraction)
- Semantic gap analysis (sentence-transformer embeddings)
- ATS-style keyword coverage score (exact + semantic match %)
- Actionable output: missing/weak skills, phrasing suggestions, overall match score

## Tech Stack

- **App framework:** Streamlit
- **Resume parsing:** pdfplumber (PDF), python-docx (DOCX)
- **NLP/NER:** spaCy (`en_core_web_sm`)
- **Semantic embeddings:** sentence-transformers (`all-MiniLM-L6-v2`)
- **Hosting:** Hugging Face Spaces (Docker SDK)

## Project Status

🚧 In development — see `docs/` for the full project specification and documentation.

## Local Development

```powershell
uv sync
uv run streamlit run src/ats_gap_analyzer/app.py
```

## License

TBD