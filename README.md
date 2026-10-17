# Business Metrics Anomaly Agent

A sector-agnostic agent that monitors business metrics in spreadsheets, detects anomalies, compares actuals with forecasts, attributes causes to segments, and sends email alerts.

## Features
- **Ingestion & Validation**: Handles CSV/Excel, missing dates, and data quality checks.
- **Metric Engine**: Computes additive and ratio metrics correctly.
- **Anomaly Detection**: Seasonality-aware detection using STL decomposition and robust Z-scores.
- **Forecasting**: ETS (Holt-Winters) forecasting with 95% prediction intervals.
- **Attribution**: Root-cause analysis identifying segment drivers.
- **Narration**: LLM-driven plain language explanations using Groq API.
- **Alerting**: Email digest alerts via Mailtrap.
- **Evaluation Harness**: Synthetic anomaly injection and performance scoring.

## Setup

1. **Clone the repository**:
   ```bash
   git clone <repo-url>
   cd anomaly-agent
   ```

2. **Install dependencies**:
   ```bash
   pip install .
   ```

3. **Configure environment**:
   Create a `.env` file with:
   ```env
   GROQ_API_KEY=your_key
   MAILTRAP_HOST=sandbox.smtp.mailtrap.io
   MAILTRAP_PORT=587
   MAILTRAP_USERNAME=your_username
   MAILTRAP_PASSWORD=your_password
   ```

4. **Prepare data**:
   ```bash
   python data/prep_dataco.py
   ```

## Usage

Run the agent:
```bash
python -m anomaly_agent.cli --config config/dataco_supply_chain.yaml
```

Run in dry-run mode (no emails):
```bash
python -m anomaly_agent.cli --config config/dataco_supply_chain.yaml --dry-run
```

## Architecture
The project follows a modular pipeline:
1. `ingestion.py`: Data loading and cleaning.
2. `metrics.py`: Metric aggregation.
3. `detection.py`: Anomaly flagging.
4. `attribution.py`: Root cause analysis.
5. `narration/`: LLM-based summary generation.
6. `alerting/`: Email digest delivery.
7. `store.py`: SQLite-based alert deduplication.
