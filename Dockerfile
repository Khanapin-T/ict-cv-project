FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# CPU-only torch/torchvision (the default Linux wheels pull in CUDA, GBs). Their pins come from
# requirements.txt; X+cpu satisfies torch==X, so the second install keeps the CPU builds.
COPY requirements.txt .
RUN pip install $(grep -E '^(torch|torchvision)==' requirements.txt | tr -d '\r') --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt

RUN useradd --create-home appuser
COPY app/ app/
# models/best.pth and models/classes.json are baked in if present; otherwise mount them with -v.
COPY models/ models/
USER appuser

EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
