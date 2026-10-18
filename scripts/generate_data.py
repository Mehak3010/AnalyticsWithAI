import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def generate_sample_data(file_path):
    # Set seed for reproducibility
    np.random.seed(42)

    # Generate dates for the last 30 days
    dates = [datetime.now() - timedelta(days=x) for x in range(30, -1, -1)]

    data = {
        "Date": dates,
        "Revenue": np.random.normal(1000, 100, len(dates)),
        "Orders": np.random.poisson(50, len(dates)),
        "Traffic": np.random.normal(5000, 500, len(dates)),
        "Conversion Rate": np.random.uniform(0.01, 0.03, len(dates)),
        "Cost": np.random.normal(400, 50, len(dates)),
        "Refunds": np.random.poisson(2, len(dates)),
    }

    df = pd.DataFrame(data)

    # Introduce some anomalies in the last few days
    # Day -2: Sharp Revenue drop
    df.loc[len(df) - 3, "Revenue"] = 450.0
    # Day -1: Sharp Orders increase
    df.loc[len(df) - 2, "Orders"] = 120
    # Day 0 (today): High Refunds
    df.loc[len(df) - 1, "Refunds"] = 15

    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Save to Excel
    df.to_excel(file_path, index=False)
    print(f"Sample data generated at {file_path}")


if __name__ == "__main__":
    path = os.path.join("data", "business_data.xlsx")
    generate_sample_data(path)
