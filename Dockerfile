# syntax=docker/dockerfile:1
FROM python:3.12-slim

# curl is only needed transiently, if a wheel build ever falls back to
# source; removed again in the same layer to keep the image lean.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# uv itself is copied in from its official image rather than pip-installed
# — Astral's own recommended Docker pattern, and it avoids needing Python
# already set up just to install the tool that manages Python deps.
COPY --from=ghcr.io/astral-sh/uv:0.12.11 /uv /uvx /usr/local/bin/

WORKDIR /app

# Dependencies are installed from just the lockfile + pyproject first,
# before the rest of the source is copied in. Editing app.py or the
# pipeline later won't invalidate this layer, which is the expensive one
# (sentence-transformers, spacy all live here).
#
# torch is deliberately excluded here (--no-install-package) and installed
# separately below via the CPU-only index directly, bypassing PyPI's
# default (CUDA-bundled) wheel entirely — pinning it via tool.uv.sources
# in pyproject.toml (the documented, "correct" way to do this) kept
# silently falling back to the CUDA wheel across several attempts, so
# this forces it unambiguously instead.
#
# --no-install-package torch only skips torch itself, not its now-orphaned
# CUDA runtime dependencies (nvidia-*, cuda-toolkit, cuda-bindings,
# triton) — those still get installed even though nothing will ever use
# them, since our replacement CPU torch doesn't need them. The pattern
# sweep afterward, chained with && into the same layer, actually removes
# them from the image rather than just hiding them in a later layer.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project --no-install-package torch \
    && uv pip list --format=freeze | grep -iE '^(nvidia-|cuda-|triton)' | cut -d= -f1 | xargs -r uv pip uninstall -y
RUN uv pip install torch --index-url https://download.pytorch.org/whl/cpu

# spaCy's model isn't a regular pip dependency — it has to be fetched
# separately. Baked into the image at build time (not lazily at runtime)
# so a freshly started container doesn't stall a user's first request.
# --no-project: at this point app.py etc. don't exist yet (COPY . . is
# below), so `uv run` shouldn't try to install the local project package.
RUN uv run --no-project python -m spacy download en_core_web_sm

# Same reasoning for the sentence-transformers model: baking it in here
# means the very first real analysis on a fresh container doesn't have to
# wait through the ~90MB download the app's own progress bar warns about
# — that message is about local `uv run streamlit run`, not this image.
RUN uv run --no-project python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Now the rest of the source, and a final sync to install the project
# package itself (skipped above via --no-install-project since app.py
# etc. didn't exist yet at that point). torch is excluded again here too
# — without this, this second sync would "correct" the environment back
# to exactly match uv.lock, silently reinstalling the CUDA torch we just
# replaced. Same CUDA-cruft sweep afterward, for the same reason as above
# — this second sync can just as easily re-add the orphaned packages.
COPY . .
# --inexact: uv sync's default "exact sync" removes anything installed
# that isn't tracked in uv.lock. The spaCy model (installed via `spacy
# download` above, entirely outside uv's tracking) was getting stripped
# right back out by this exact step — present at build time, gone by the
# time the container actually ran. --inexact leaves untracked packages
# alone instead of pruning them.
RUN uv sync --frozen --no-dev --inexact --no-install-package torch \
    && uv pip list --format=freeze | grep -iE '^(nvidia-|cuda-|triton)' | cut -d= -f1 | xargs -r uv pip uninstall -y

# Hugging Face Spaces' Docker SDK expects the app on port 7860, reached
# through HF's own reverse proxy. --server.enableCORS/XsrfProtection are
# turned off because that proxy already handles cross-origin/embedding
# concerns — leaving Streamlit's own protections on is a known cause of
# a Space silently failing to load.
#
# --no-sync here for the same reason as above: plain `uv run` re-syncs
# the environment against uv.lock before running, which would reinstall
# CUDA torch (and its cruft) on every single container start. --no-sync
# runs directly against whatever's already installed, which by this
# point is exactly the environment we deliberately built.
EXPOSE 7860

CMD ["uv", "run", "--no-sync", "streamlit", "run", "src/ats_gap_analyzer/app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false"]
