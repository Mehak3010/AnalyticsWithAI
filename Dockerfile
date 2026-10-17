FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
RUN pip install .

COPY . .

ENV PYTHONPATH=/app/src

ENTRYPOINT ["python", "-m", "anomaly_agent.cli"]
