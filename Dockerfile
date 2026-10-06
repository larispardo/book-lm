# Inference image: CPU-only PyTorch, models mounted read-only from ./artifacts.
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    BOOKLM_ARTIFACTS_DIR=/app/artifacts \
    BOOKLM_DEVICE=cpu

WORKDIR /app

# CPU wheels keep the image ~1 GB instead of ~5 GB with CUDA.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY booklm ./booklm
RUN pip install .

RUN useradd --create-home --uid 10001 booklm
USER booklm

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(urllib.request.urlopen('http://127.0.0.1:8000/readyz').status != 200)"

CMD ["booklm-serve"]
