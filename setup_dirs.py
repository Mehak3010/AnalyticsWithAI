import os

directories = [
    "config",
    "data/committed",
    "src/anomaly_agent/narration",
    "src/anomaly_agent/alerting",
    "src/anomaly_agent/alerting/templates",
    "evaluation",
    "docs",
    "tests",
    "app",
    ".github/workflows",
    "reports"
]

for directory in directories:
    os.makedirs(directory, exist_ok=True)
    print(f"Created directory: {directory}")
