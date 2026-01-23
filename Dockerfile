FROM python:3.12-slim-bookworm

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml .python-version uv.lock ./
RUN uv sync --locked --no-install-project

ENV PATH="/app/.venv/bin:$PATH"

COPY src/ ./src/
COPY main.py ./

COPY model/ ./model/

EXPOSE 8080

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]