# DEMAE Phase-0 ETL — pinned environment so the corpus build reproduces on a
# rented vast.ai box (the cloud-burst escape hatch: same-region eu-central-1 reads
# to cut transatlantic COG-open latency; see docs/adr/0001 + the wayfinder map #1).
#
# This image reproduces the *ETL build* only (CPU, no GPU/torch needed — the loader
# imports torch lazily and the pipeline never touches it). Model training (Phase 1+)
# gets its own CUDA image later.
#
# Reproducibility anchor: uv + Python are pinned by the base image tag, and every
# Python dependency (rasterio wheels bundle GDAL — no system GDAL) is pinned by
# uv.lock via `uv sync --frozen`. rasterio's wheels carry their own GDAL, so no
# apt-get gdal is required.
#
#   Build:   docker build -t demae-etl .
#   Sample:  docker run --rm -v "$PWD/data:/data" demae-etl --source synthetic --out-dir /data/sample
#   Live:    docker run --rm -v "$PWD/data:/data" demae-etl --source live --out-dir /data/corpus
#   Probe:   docker run --rm --entrypoint python demae-etl -m dataset.throughput  # cloud-burst verdict

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# AWS_NO_SIGN_REQUEST: the Copernicus bucket is anonymous S3; GDAL/rasterio must
# skip request signing to read it (recon §1, config.AWS_NO_SIGN_ENV).
# The UV_* flags give a self-contained, bytecode-compiled venv at a fixed path.
ENV AWS_NO_SIGN_REQUEST=YES \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# 1) Install locked runtime deps first, without the project, so this heavy layer
#    is cached and only re-runs when pyproject.toml / uv.lock change. --no-dev
#    keeps torch/jupyter/matplotlib out: the ETL build is lean and CPU-only.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# 2) Install the project itself against the now-cached deps.
COPY src ./src
COPY README.md ./
RUN uv sync --frozen --no-dev

# The build CLI is the reproducible "run the pipeline" surface: it smoke-gates a
# tiny tier end-to-end, then ships the full debug corpus. Mount a volume at /data.
ENTRYPOINT ["python", "-m", "dataset.build"]
CMD ["--source", "live", "--out-dir", "/data/corpus"]
